# Author: KrorngAI org.
# Date: February 2026


from typing import Sequence, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .tokenizer import get_tokenizer
from .nn_utils import (
    precompute_rotary_emb,
    DerfWrapper,
    norm,
    apply_rotary_emb,
    MLP,
    LinearWrapper,
    KVCache
)
from .common import print_banner


class Conv2d(nn.Conv2d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


class PatchEmbedding(nn.Module):
    def __init__(self, n_channels, patch_size, n_embed):
        super().__init__()
        self.patch_embed = Conv2d(n_channels, n_embed, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, n_embed, n_head, dropout=0.0, bias=True):
        super().__init__()
        self.n_embed = n_embed
        self.n_head = n_head
        self.dropout = nn.Dropout(dropout)
        self.q_proj = LinearWrapper(n_embed, n_embed, bias=bias)
        self.k_proj = LinearWrapper(n_embed, n_embed, bias=bias)
        self.v_proj = LinearWrapper(n_embed, n_embed, bias=bias)
        self.out_proj = LinearWrapper(n_embed, n_embed, bias=bias)

    def forward(self, query, key, value, attn_mask, cos_sin=None, kv_cache=None):
        b, t, _ = query.size()
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = q.view(*q.shape[:2], self.n_head, -1)
        k = k.view(*k.shape[:2], self.n_head, -1)
        v = v.view(*v.shape[:2], self.n_head, -1)
        q, k = norm(q), norm(k)
        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(0, k, v)
        qkv = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p, attn_mask=attn_mask, is_causal=False)
        qkv = qkv.permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.dropout(self.out_proj(qkv)), _


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.0, bias=True):
        super().__init__()
        self.n_embed = d_model
        self.n_head = nhead
        self.head_dim = d_model // nhead
        self.sa = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias)
        self.sa_norm = DerfWrapper(d_model)
        self.mlp = MLP(d_model, dim_feedforward, dropout, bias)
        self.mlp_norm = DerfWrapper(d_model)

    def forward(self, x, img_enc=None, src_mask=None, cos_sin=None, kv_cache=None):
        """
        x: query/hidden state (b, L, n_embed)
        img_enc: image encoding (b, n_patch, n_embed)
        src_mask: attention mask (L, n_patch + L)
        """
        # x_norm = norm(x)
        x_norm = self.sa_norm(x)
        if img_enc is None:
            memory = x_norm
        else:
            # memory = torch.cat([norm(img_enc), x_norm], dim=1)
            memory = torch.cat([self.sa_norm(img_enc), x_norm], dim=1)

        x = x + self.sa(x_norm, memory, memory, src_mask, cos_sin, kv_cache)[0]
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TrorYongOCR(nn.Module):
    """
    Receive image and output characters
    tokenizer must be given so that the model can handle bos, eos, and pad tokens.
    """

    def __init__(self, config, tokenizer) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        self.n_patch = (config.img_size[0] // config.patch_size[0]) * (config.img_size[1] // config.patch_size[1])
        self.img_embed = PatchEmbedding(config.n_channel, config.patch_size, config.n_embed)
        self.head_dim = self.config.n_embed // self.config.n_head
        cos, sin = precompute_rotary_emb(self.n_patch, self.head_dim, device=self.img_embed.patch_embed.weight.device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.tok_embed = nn.Embedding(config.vocab_size, config.n_embed, padding_idx=tokenizer.pad_id)
        self.pos_embed = nn.Parameter(torch.Tensor(1, config.block_size, config.n_embed))
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    d_model=config.n_embed,
                    nhead=config.n_head,
                    dim_feedforward=2*config.n_embed,
                    dropout=config.dropout,
                    bias=config.bias)
            for _ in range(config.n_layer-1)]
        )
        self.txt_decoder = ResidualAttentionBlock(
            d_model=config.n_embed,
            nhead=config.n_head,
            dim_feedforward=4*config.n_embed,
            dropout=config.dropout,
            bias=config.bias
        )
        self.lm_head = LinearWrapper(config.n_embed, config.vocab_size)

        mask = torch.tril(torch.ones((self.n_patch+config.block_size, self.n_patch+config.block_size), dtype=torch.bool))
        self.register_buffer("mask", mask)
        self.apply(self.init_weights)
        nn.init.trunc_normal_(self.pos_embed, std=1.0)

    def forward(self, img, x, targets=None):
        """
        x including bos
        img image tensor (b, 3, 32, 128)
        """
        b, L = x.size()

        img_emb = self.dropout(self.img_embed(img))
        cos_sin = self.cos[:, :self.n_patch], self.sin[:, :self.n_patch]
        for block in self.blocks:
            img_emb = block(img_emb, cos_sin=cos_sin) # regular encoding layer to encode image

        # decoding layer
        S = self.n_patch + L
        # all character tokens must communicate with all image encoding
        mask = self.mask[self.n_patch:self.n_patch + L, :S]

        query = self.dropout(self.tok_embed(x) + self.pos_embed[:, :L])
        query = self.txt_decoder(query, img_enc=img_emb, src_mask=mask)
        if targets is not None:
            query = norm(query)
            logits = self.lm_head(query).float() # (b, L, n_vocab)
            loss = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.tokenizer.pad_id)
            return logits, loss
        else:
            # inference mode
            return self.lm_head(norm(query[:, [-1], :])).float(), None

    @classmethod
    def from_pretrained(cls):
        tokenizer = get_tokenizer()
        config = TrorYongConfig(
            img_size=(32, 128),
            patch_size=(4, 8),
            n_channel=3,
            vocab_size=len(tokenizer), # exclude pad and unk tokens
            block_size=192,
            n_layer=4,
            n_head=6,
            n_embed=384,
            dropout=0.1,
            bias=True,
        )
        model = cls(config, tokenizer)
        state_dict = torch.hub.load_state_dict_from_url('https://huggingface.co/KrorngAI/PARSeqForKhmer/resolve/main/best_model-90epoch.pt', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        return model

    @torch.inference_mode()
    def decode(self, img_tensor: Tensor, max_tokens: int, temperature=1.0, top_k=None):
        """
        img_tensor: (3, 32, 128)
        max_tokens: int
        """
        img_tensor = img_tensor.unsqueeze(0)
        img_emb = self.dropout(self.img_embed(img_tensor))
        cos_sin = self.cos[:, :self.n_patch], self.sin[:, :self.n_patch]
        for block in self.blocks:
            img_emb = block(img_emb, cos_sin=cos_sin)

        seq_len = self.mask.shape[0]
        assert max_tokens <= seq_len, "too long sequence generation, consider lower max_tokens"
        kv_cache = KVCache(1, self.config.n_head, seq_len, self.head_dim, 1)

        idx = torch.full((1,1), fill_value=self.tokenizer.bos_id, dtype=torch.long, device=img_tensor.device)
        idx_next = None
        for i in range(1, max_tokens):
            mask = self.mask[self.n_patch + i-1:self.n_patch + i, :self.n_patch + i]
            if i == 1:
                query = self.dropout(self.tok_embed(idx) + self.pos_embed[:, i-1:i])
                query = self.txt_decoder(query, img_enc=img_emb, src_mask=mask, kv_cache=kv_cache)
            else:
                query = self.dropout(self.tok_embed(idx_next) + self.pos_embed[:, i-1:i])
                query = self.txt_decoder(query, src_mask=mask, kv_cache=kv_cache)
            logits = self.lm_head(norm(query[:, [-1], :])).float()

            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() == self.tokenizer.eos_id:
                break
        return idx

    @torch.no_grad()
    def init_weights(self, module: nn.Module, name: str = '', exclude: Sequence[str] = ('')):
        """Initialize the weights using the typical initialization schemes used in SOTA models."""
        if any(map(name.startswith, exclude)):
            return
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        if self.tok_embed.weight.device.type == 'cuda':
            self.tok_embed.weight = self.tok_embed.weight.to(torch.bfloat16)
            self.pos_embed = self.pos_embed.to(torch.bfloat16)


@dataclass
class TrorYongConfig:
    img_size: Sequence[int]
    patch_size: Sequence[int]
    n_channel: int
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embed: int
    dropout: float = 0.0
    bias: bool = True
