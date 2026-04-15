# Author: KrorngAI org.
# Date: March 2026


from typing import Sequence, Optional, List
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model
import math
from .nn_utils import (
    precompute_rotary_emb,
    norm,
    apply_rotary_emb,
    MLP,
    LinearWrapper,
    KVCache
)
from .common import print_banner
try:
    from whisper.model import AudioEncoder
except (ImportError, ModuleNotFoundError):
    print("You need to install openai-whisper package: pip install git+https://github.com/openai/whisper.git")
    raise


PRETRAINED_MODEL = [
    "KrorngAI/tror-yong-asr-tiny",
    "KrorngAI/tror-yong-asr-small"
]


class MultiheadAttention(nn.Module):
    def __init__(self, n_embed, n_head, dropout=0.0, bias=True, self_attn=True):
        super().__init__()
        self.n_embed = n_embed
        self.n_head = n_head
        self.dropout = nn.Dropout(dropout)
        self.q_proj = LinearWrapper(n_embed, n_embed, bias=bias)
        self.k_proj = LinearWrapper(n_embed, n_embed, bias=bias)
        self.v_proj = LinearWrapper(n_embed, n_embed, bias=bias)
        self.out_proj = LinearWrapper(n_embed, n_embed, bias=bias)
        self.self_attn = self_attn
        # self.max_logits = None

    def q_projection(self, query, cos_sin):
        """
        Compute and return the projection of query, key, and value
        """
        q = self.q_proj(query)
        q = q.view(*q.shape[:2], self.n_head, -1)
        if cos_sin is not None:
            cos, sin = cos_sin
            if q.shape[1] == 1:
                # due to my specific design, there is a shift for cos_sin in q
                q = apply_rotary_emb(q, cos[:, [1]], sin[:, [1]])
            else:
                q = apply_rotary_emb(q, cos, sin)
        q = q.permute(0, 2, 1, 3)
        q = norm(q)
        return q

    def kv_projection(self, key, value, cos_sin=None, kv_cache=None):
        """
        Compute and return the projection of key and value
        """
        k = self.k_proj(key)
        k = k.view(*k.shape[:2], self.n_head, -1)
        if cos_sin is not None:
            cos, sin = cos_sin
            if k.shape[1] == 1:
                # due to my specific design, there is a shift for cos_sin in q
                k = apply_rotary_emb(k, cos[:, [0]], sin[:, [0]])
            else:
                # in PARSeq's architecture, the first `key token` is null context (sot) and required no `position token`
                # for RoPE, the very first position contains `theta=0` (also, no rotation)
                # So, shift `cos` and `sin` to apply no rotation to null context
                k_cos = torch.concatenate([cos[:, [0]], cos[:, :-1]], dim=1)
                k_sin = torch.concatenate([sin[:, [0]], sin[:, :-1]], dim=1)
                k = apply_rotary_emb(k, k_cos, k_sin)
        k = k.permute(0, 2, 1, 3)
        k = norm(k)

        v = self.v_proj(value)
        v = v.view(*v.shape[:2], self.n_head, -1)
        v = v.permute(0, 2, 1, 3)
        if kv_cache is not None: # only self_attn use kv_cache
            k, v = kv_cache.insert_kv(0, k, v) # there is only one layer, so layer_idx=0
        return k, v

    def qkv_projection(self, query, key, value, cos_sin, kv_cache=None):
        """
        Compute and return the projection of query, key, and value
        """
        q_proj = self.q_projection(query, cos_sin)
        k_proj, v_proj = self.kv_projection(key, value, cos_sin, kv_cache)
        return q_proj, k_proj, v_proj

    def attention_forward(self, q_proj, k_proj, v_proj, attn_mask=None, key_padding_mask=None):
        """
        Given the q_proj, k_proj, v_proj projections and attention mask, return the attention output
        NOTE: we separate this function to save time as PARSeq use different attention masks
        for the same query, key, and value
        """

        b, _, lk, _ = k_proj.size()
        if key_padding_mask is not None and attn_mask is not None:
            attn_mask_expanded = attn_mask.view(1, 1, lk, lk).expand(b, self.n_head, -1, -1)
            key_mask_expanded = key_padding_mask.view(b, 1, 1, lk).expand(-1, self.n_head, -1, -1)
            attn_mask = attn_mask_expanded + key_mask_expanded

        qkv = F.scaled_dot_product_attention(q_proj, k_proj, v_proj, dropout_p=self.dropout.p, attn_mask=attn_mask, is_causal=False)

        if self.self_attn:
            if q_proj.shape == v_proj.shape:
                vn = F.normalize(v_proj, dim=-1, eps=1e-9)
            else:  # handle KVCache in decoding, q_proj=(b, n_head, 1, n_embed) != v_proj=(b, n_head, lk, n_embed)
                vn = F.normalize(v_proj[:, :, [-1], :], dim=-1, eps=1e-9)
            qkv = qkv - torch.sum(qkv * vn, dim=-1, keepdim=True) * vn

        qkv = qkv.permute(0, 2, 1, 3).flatten(start_dim=2)        # (b, lq, n_embed)
        out = self.out_proj(qkv)

        # tracking maximal logit, qk
        # for debugging only
        # scale = (self.n_embed // self.n_head) ** -0.5
        # with torch.no_grad():
        #     qk = scale * q_proj @ k_proj.transpose(-1, -2)      # (b, n_head, lq, lk)
        #     new_max = qk.amax(dim=(0, 2, 3))                    # (n_head)
        #     if self.max_logits is None:
        #         self.max_logits = new_max
        #     else:
        #         self.max_logits = torch.where(self.max_logits < new_max, new_max, self.max_logits)

        return self.dropout(out), None

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, cos_sin=None, kv_cache=None):
        """
        query, key, value are not projected yet and will be projected before
        attention mechanism
        """
        q_proj, k_proj, v_proj = self.qkv_projection(query, key, value, cos_sin, kv_cache)
        return self.attention_forward(q_proj, k_proj, v_proj, attn_mask, key_padding_mask)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, bias) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout, bias)
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout, bias, self_attn=False)
        self.mlp = MLP(d_model, dim_feedforward, dropout, bias)

    def forward_self_attn(self, q, q_proj, k_proj, v_proj, attn_mask, key_padding_mask=None):
        """
        q must be unnormalized in order to do residual connection
        key_padding_mask is None during inference
        """
        query, sa_weights = self.self_attn.attention_forward(q_proj, k_proj, v_proj, attn_mask, key_padding_mask)
        return q.clone() + query

    def forward_cross_attn_and_mlp(self, res, q_proj, k_proj, v_proj):
        """
        res must be unnormalized in order to do residual connection
        """
        query, cross_weights = self.cross_attn.attention_forward(q_proj, k_proj, v_proj)
        res = res + query
        res = res + self.mlp(norm(res))
        return res


@dataclass
class TrorYongASRConfig:
    vocab_size: int  # use the tokenizer's vocab size
    n_mels: int
    n_audio_ctx: int
    n_text_ctx: int
    n_embed: int
    n_head: int
    n_layer: int
    dropout: float = 0.0
    bias: bool = True
    pad_id: int = 0
    eot_id: int = 2
    tie_word_embeddings: bool = True


class TrorYongASRModel(nn.Module):
    def __init__(self, config, verbose=True) -> None:
        super().__init__()
        self.config = config

        self.encoder = AudioEncoder(
            config.n_mels,
            config.n_audio_ctx,
            config.n_embed,
            config.n_head,
            config.n_layer
        )
        self.n_head = 2 * config.n_head
        self.head_dim = config.n_embed // self.n_head
        cos, sin = precompute_rotary_emb(10 * config.n_text_ctx, self.head_dim, device=self.device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Make vocab_size multiple of 64
        self.vocab_size = config.vocab_size
        self.opt_vocab_size = (config.vocab_size + 63) // 64 * 64
        self.tok_embed = nn.Embedding(self.opt_vocab_size, config.n_embed, padding_idx=self.config.pad_id)
        self.pos_embed = nn.Parameter(torch.Tensor(1, config.n_text_ctx, config.n_embed))
        self.dropout = nn.Dropout(config.dropout)
        self.decoder = DecoderBlock(config.n_embed, self.n_head, config.n_embed * 4, config.dropout, config.bias)
        self.lm_head = LinearWrapper(config.n_embed, self.opt_vocab_size, bias=False)

        # weight tying
        if config.tie_word_embeddings:
            self.tok_embed.weight = self.lm_head.weight

        # When using F.scaled_dot_product, True means counted in attention, False means masked
        mask = torch.empty(config.n_text_ctx, config.n_text_ctx).fill_(-float('inf')).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
        self.apply(self.init_weights)
        nn.init.trunc_normal_(self.pos_embed, std=1.0)
        self.prefix = None
        if verbose:
            print_banner()

    def get_token_embedding(self, token_ids):
        return self.tok_embed(token_ids) * math.sqrt(self.config.n_embed)

    def encode(self, mels: Tensor) -> Tensor:
        return self.encoder(mels)

    def forward(self, mels, input_ids, target_ids=None):

        """
        mels: log-mel spectrum from audio array
        input_ids including sot sequence
        """

        # decoding layer
        b, L = input_ids.size()
        assert L <= self.config.n_text_ctx, "input_ids is too long"

        aud = self.encoder(mels)
        aud = norm(aud)
        aud_k_proj, aud_v_proj = self.decoder.cross_attn.kv_projection(aud, aud)

        q = self.dropout(self.pos_embed[:, :L].expand(b, -1, -1))
        ctx = self.get_token_embedding(input_ids)
        ctx = torch.concatenate([ctx[:, [0]], ctx[:, 1:] + self.pos_embed[:, :L-1]], dim=1)
        ctx = self.dropout(ctx)
        ctx = norm(ctx)

        cos_sin = self.cos[:, :L], self.sin[:, :L]
        q_proj, k_proj, v_proj = self.decoder.self_attn.qkv_projection(norm(q), ctx, ctx, cos_sin)
        mask = self.mask[:L, :L]
        key_padding_mask = torch.zeros((b, L), device=self.device)
        key_padding_mask.masked_fill_(input_ids == self.config.pad_id, -float('inf'))

        res = self.decoder.forward_self_attn(q, q_proj, k_proj, v_proj, mask, key_padding_mask)
        res_q_proj = self.decoder.cross_attn.q_projection(norm(res), cos_sin)
        res = self.decoder.forward_cross_attn_and_mlp(res, res_q_proj, aud_k_proj, aud_v_proj)
        if target_ids is not None:
            logits = self.lm_head(norm(res))  # (b, L, n_vocab)
            loss = F.cross_entropy(logits.flatten(end_dim=1).float(), target_ids.flatten(), ignore_index=-100, reduction='mean')
            return logits, loss
        else:
            # do not compute loss
            return self.lm_head(norm(res)).float(), None

    @property
    def device(self):
        return self.encoder.conv1.weight.device

    @classmethod
    def from_pretrained(cls, model_id: str = "KrorngAI/tror-yong-asr-tiny", verbose=True, trust_remote_code: Optional[bool]=None):
        assert model_id in PRETRAINED_MODEL, f"Not supported repo, please check again. {model_id}"
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        model = cls(config, verbose=verbose)
        file_path = hf_hub_download(model_id, filename="model.safetensors")
        load_model(model, file_path)
        return model

    def set_prefix(self, prefix: List[int]):
        """
        Set bos token, language token, and task token
        This is used in decode function
        """
        self.prefix = prefix

    @torch.inference_mode()
    def decode(self, mels: Tensor, max_tokens: int, temperature=1.0, top_k=None, seed=168):
        """
        mels: (n_mels,)
        max_tokens: int
        """
        if self.prefix is None:
            raise SyntaxError("Please provide bos, language, and task tokens as prefix using set_prefix before calling decode.")

        seq_len = self.mask.shape[0]
        assert max_tokens <= seq_len, "too long sequence generation, consider lower max_tokens"
        kv_cache = KVCache(1, self.n_head, seq_len, self.head_dim, 1) # batch_size=1, num_layers=1
        device = self.device
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        mels = mels.unsqueeze(0)
        aud = norm(self.encoder(mels))
        aud_k_proj, aud_v_proj = self.decoder.cross_attn.kv_projection(aud, aud)

        idx = torch.as_tensor([self.prefix], dtype=torch.long, device=mels.device)
        n = idx.shape[1]
        idx_next = None
        for i in range(n, n+max_tokens):
            if i == n:
                q = self.pos_embed[:, :i]
                ctx = self.get_token_embedding(idx)
                ctx = torch.concatenate([ctx[:, [0]], ctx[:, 1:] + self.pos_embed[:, :i-1]], dim=1)
                cos_sin = self.cos[:, :i], self.sin[:, :i]
                mask = self.mask[:i, :i]
            else:
                q = self.pos_embed[:, [i-1]]
                ctx = self.get_token_embedding(idx_next) + self.pos_embed[:, [i-2]]
                cos_sin = self.cos[:, i-2:i], self.sin[:, i-2:i]
                mask = self.mask[[i-1], :i]
            ctx = norm(ctx)
            q_proj, k_proj, v_proj = self.decoder.self_attn.qkv_projection(norm(q), ctx, ctx, cos_sin, kv_cache=kv_cache)
            res = self.decoder.forward_self_attn(q, q_proj, k_proj, v_proj, mask)
            res_q_proj = self.decoder.cross_attn.q_projection(norm(res), cos_sin)
            res = self.decoder.forward_cross_attn_and_mlp(res, res_q_proj, aud_k_proj, aud_v_proj)

            logits = self.lm_head(norm(res[:, [-1], :])).float()
            logits[:, -1, self.vocab_size:] = -float('inf') # suppress the pad tokens

            if temperature > 0:
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                idx_next = logits.argmax(dim=-1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() == self.config.eot_id:
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

        if self.tok_embed.weight.device.type == 'cuda':
            self.tok_embed.weight = self.tok_embed.weight.to(torch.bfloat16)
            self.pos_embed = self.pos_embed.to(torch.bfloat16)
