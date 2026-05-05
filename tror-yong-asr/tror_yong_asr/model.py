# Author: KrorngAI org.
# Date: March 2026


from typing import Optional, List
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig
from transformers.modeling_outputs import CausalLMOutput
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model
import math
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
from .whisper_audio_encoder import AudioEncoder


PRETRAINED_MODEL = [
    "KrorngAI/TrorYongASR-tiny",
    "KrorngAI/TrorYongASR-small",
    "Kimang18/tror-yong-asr-tiny",  # for testing TODO: remove this before publish
    "Kimang18/tror-yong-asr-small"  # for testing TODO: remove this before publish
]


class MultiheadAttention(nn.Module):
    def __init__(self, n_embed, n_head, dropout=0.0, bias=True, is_ca=False):
        super().__init__()
        self.n_embed = n_embed
        self.n_head = n_head
        self.dropout = nn.Dropout(dropout)
        if is_ca:  # in this implementation self_attn does not need q_proj
            self.q_proj = LinearWrapper(n_embed, n_embed, bias=bias)
        self.k_proj = LinearWrapper(n_embed, n_embed, bias=bias)
        self.v_proj = LinearWrapper(n_embed, n_embed, bias=bias)
        self.out_proj = LinearWrapper(n_embed, n_embed, bias=bias)
        self.is_ca = is_ca
        # self.max_logits = None

    def q_projection(self, q, cos_sin=None):
        """
        Compute and return the projection of query
        """
        if self.is_ca:  # in this implementation, self MHA does not need q projection
            q = self.q_proj(q)
        q = norm(q.view(*q.shape[:2], self.n_head, -1))
        q = q.permute(0, 2, 1, 3)
        if cos_sin is not None:
            cos, sin = cos_sin
            cos, sin = cos[:, 1:].permute(0, 2, 1, 3), sin[:, 1:].permute(0, 2, 1, 3)
            # in this implementation, query has one rotation faster than key
            q = apply_rotary_emb(q, cos, sin)
        return q

    def kv_projection(self, k, v, cos_sin=None, kv_cache=None):
        """
        Compute and return the projection of key and value
        """
        k = self.k_proj(k)
        k = norm(k.view(*k.shape[:2], self.n_head, -1))
        k = k.permute(0, 2, 1, 3)
        if cos_sin is not None:  # cross attention does not apply rotary embedding
            cos, sin = cos_sin
            cos, sin = cos[:, :-1].permute(0, 2, 1, 3), sin[:, :-1].permute(0, 2, 1, 3)
            # in this implementation, key has one rotation slower than query
            k = apply_rotary_emb(k, cos, sin)
        v = self.v_proj(v)
        v = v.view(*v.shape[:2], self.n_head, -1)
        v = v.permute(0, 2, 1, 3)
        if kv_cache is not None:  # only self_attn use kv_cache
            k, v = kv_cache.insert_kv(0, k, v)  # there is only one layer, so layer_idx=0
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
            # attn_mask is None for cross-attention layer
            # WARN: attn_mask cannot be None for self-attention layer
            # key_mask is used during pre-training for self-attention layer
            attn_mask_expanded = attn_mask.view(1, 1, lk, lk).expand(b, self.n_head, -1, -1)
            key_mask_expanded = key_padding_mask.view(b, 1, 1, lk).expand(-1, self.n_head, -1, -1)
            attn_mask = attn_mask_expanded + key_mask_expanded

        qkv = F.scaled_dot_product_attention(q_proj, k_proj, v_proj, dropout_p=self.dropout.p, attn_mask=attn_mask, is_causal=False)

        if not self.is_ca:  # perform 'exclusive self-attention'
            if qkv.shape == v_proj.shape:
                vn = F.normalize(v_proj, dim=-1, eps=1e-9)
            else:  # when decoding, qkv=(b, n_head, 1, n_embed) != v_proj=(b, n_head, lk, n_embed)
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

    def forward(self, query, key, value, cos_sin=None, attn_mask=None, key_padding_mask=None, kv_cache=None):
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
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout, bias, is_ca=True)
        self.ca_norm = DerfWrapper(d_model)
        self.mlp = MLP(d_model, dim_feedforward, dropout, bias)
        self.mlp_norm = DerfWrapper(d_model)

    def forward_self_attn(self, q, q_proj, k_proj, v_proj, attn_mask, key_padding_mask=None):
        """
        WARN: q must be unnormalized in order to do residual connection
        key_padding_mask is None during inference
        """
        query, sa_weights = self.self_attn.attention_forward(q_proj, k_proj, v_proj, attn_mask, key_padding_mask)
        return q.clone() + query

    def forward_cross_attn_and_mlp(self, res, q_proj, k_proj, v_proj):
        """
        WARN: res must be unnormalized in order to do residual connection
        """
        query, ca_weights = self.cross_attn.attention_forward(q_proj, k_proj, v_proj)
        res = res + query
        res = res + self.mlp(self.mlp_norm(res))
        return res

    def forward(self, q, ctx, aud, cos_sin, attn_mask, key_padding_mask=None, kv_cache=None):
        """
        regular forward pass
        WARN: q must be unnormalized in order to do residual connection
        """
        # NOTE: q is not normalized when passing to self_attn because it is q_projection already, and I only apply rotation.
        # query, sa_weights = self.self_attn(norm(q), ctx, ctx, cos_sin, attn_mask, key_padding_mask, kv_cache)
        query, sa_weights = self.self_attn(q, ctx, ctx, cos_sin, attn_mask, key_padding_mask, kv_cache)
        res = q.clone() + query
        query, ca_weights = self.cross_attn(self.ca_norm(res), aud, aud)
        res = res + query
        res = res + self.mlp(self.mlp_norm(res))
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
        # decoder's head is twice the head of encoder
        self.n_head = 2 * config.n_head
        self.head_dim = config.n_embed // self.n_head
        self.rotary_seq_len = 10 * config.n_text_ctx
        cos, sin = precompute_rotary_emb(self.rotary_seq_len, self.head_dim, device=self.device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Make vocab_size multiple of 64
        self.vocab_size = config.vocab_size
        self.pad_vocab_size = (config.vocab_size + 63) // 64 * 64
        self.tok_embed = nn.Embedding(self.pad_vocab_size, config.n_embed, padding_idx=self.config.pad_id)
        self.pos_basis = nn.Parameter(torch.Tensor(config.n_embed))
        self.dropout = nn.Dropout(config.dropout)
        self.decoder = DecoderBlock(config.n_embed, self.n_head, config.n_embed * 4, config.dropout, config.bias)
        self.lm_head = LinearWrapper(config.n_embed, self.pad_vocab_size, bias=False)

        # weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embed.weight

        mask = torch.empty(config.n_text_ctx, config.n_text_ctx).fill_(-float('inf')).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
        self.prefix = None
        self.init_weights()
        if verbose:
            print_banner()

    def get_token_embedding(self, token_ids):
        return self.tok_embed(token_ids) * math.sqrt(self.config.n_embed)

    def encode(self, mels: Tensor) -> Tensor:
        return self.encoder(mels)

    def forward(self, mels, input_ids, target_ids=None):
        """
        mels: log-mel spectrum from audio array
        input_ids: input token ids for decoder (inclu. sot sequence)
        target_ids: target token ids to compute cross_entropy loss
        """

        # decoding layer
        b, L = input_ids.size()
        assert L <= self.config.n_text_ctx, "input_ids is too long"

        # NOTE: in Whisper Audio Encoder, the audio encoding is already normalized by ln_post()
        aud = self.encoder(mels)

        q = self.pos_basis.view(1, 1, -1).expand(b, L, -1)
        ctx = self.get_token_embedding(input_ids)
        ctx = self.dropout(ctx)
        ctx = norm(ctx)

        cos_sin = self.cos[:, :L+1], self.sin[:, :L+1]

        attn_mask = self.mask[:L, :L]
        res = self.decoder(q, ctx, aud, cos_sin, attn_mask)
        logits = self.lm_head(norm(res))  # (b, L, pad_vocab_size)
        if target_ids is not None:
            loss = F.cross_entropy(logits.flatten(end_dim=1).float(), target_ids.flatten(), ignore_index=-100, reduction='mean')
            return CausalLMOutput(logits=logits, loss=loss)
        else:
            # do not compute loss
            return CausalLMOutput(logits=logits)

    @property
    def device(self):
        return self.encoder.conv1.weight.device

    @classmethod
    def from_pretrained(cls, model_id: str = "KrorngAI/TrorYongASR-tiny", verbose=True, trust_remote_code: Optional[bool]=None):
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
    def detect_language(self, mels: Tensor, tokenizer):
        """
        Predicting the language of audio
        NOTE: I could have called `forward` but there is a careful attention required for cos, sin
        """
        mels = mels.unsqueeze(0)
        aud = self.encoder(mels)

        idx = torch.as_tensor([[tokenizer.sot]], dtype=torch.long, device=self.device)
        logits = self.forward(mels, idx).logits

        # suppress non-language tokens
        mask = torch.ones(logits.shape[-1], dtype=torch.bool)
        mask[list(tokenizer.all_language_tokens)] = False
        logits[:, -1, mask] = -float('inf')
        language_tokens = logits.argmax(dim=-1)
        language_token_probs = F.softmax(logits, dim=-1)
        language_probs = [
            {
                c: language_token_probs[0, 0, j].item()
                for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
            }
        ]

        return language_tokens[0], language_probs[0]

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
        aud = self.encoder(mels)
        aud_k_proj, aud_v_proj = self.decoder.cross_attn.kv_projection(aud, aud)

        idx = torch.as_tensor([self.prefix], dtype=torch.long, device=mels.device)
        n = idx.shape[1]
        idx_next = None
        pos_query = self.pos_basis.view(1, 1, -1)
        for i in range(n, n+max_tokens):
            if i == n:
                q = pos_query.expand(-1, i, -1)
                ctx = self.get_token_embedding(idx)
                cos_sin = self.cos[:, :i+1], self.sin[:, :i+1]
                mask = self.mask[:i, :i]
            else:
                q = pos_query
                ctx = self.get_token_embedding(idx_next)
                cos_sin = self.cos[:, i-1:i+1], self.sin[:, i-1:i+1]
                mask = self.mask[[i-1], :i]
            ctx = norm(ctx)

            # q_proj, k_proj, v_proj = self.decoder.self_attn.qkv_projection(norm(q), ctx, ctx, cos_sin, kv_cache=kv_cache)
            # res = self.decoder.forward_self_attn(q, q_proj, k_proj, v_proj, mask)
            # NOTE: The 2 lines below are equivalent to the 2 lines above
            # q is not normalized because it is q_projectioin already and I only apply rotation.
            query, sa_weights = self.decoder.self_attn(q, ctx, ctx, cos_sin, mask, kv_cache=kv_cache)
            res = q.clone() + query

            res_q_proj = self.decoder.cross_attn.q_projection(self.decoder.ca_norm(res))
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
    def init_weights(self):
        nn.init.normal_(self.encoder.conv1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.encoder.conv2.weight, mean=0.0, std=0.02)

        n_embed = self.config.n_embed
        s = 3**0.5 * n_embed**-0.5
        for block in self.encoder.blocks:
            nn.init.uniform_(block.attn.query.weight, -s, s)
            nn.init.uniform_(block.attn.key.weight, -s, s)
            nn.init.uniform_(block.attn.value.weight, -s, s)
            nn.init.zeros_(block.attn.out.weight)
            nn.init.uniform_(block.mlp[0].weight, -s * 0.4, s * 0.4)
            nn.init.zeros_(block.mlp[2].weight)

        nn.init.uniform_(self.decoder.self_attn.k_proj.weight, -s, s)
        nn.init.uniform_(self.decoder.self_attn.v_proj.weight, -s, s)
        nn.init.zeros_(self.decoder.self_attn.out_proj.weight)
        nn.init.uniform_(self.decoder.cross_attn.q_proj.weight, -s, s)
        nn.init.uniform_(self.decoder.cross_attn.k_proj.weight, -s, s)
        nn.init.uniform_(self.decoder.cross_attn.v_proj.weight, -s, s)
        nn.init.zeros_(self.decoder.cross_attn.out_proj.weight)
        nn.init.uniform_(self.decoder.mlp.gate_proj.weight, -s * 0.4, s * 0.4)
        nn.init.uniform_(self.decoder.mlp.up_proj.weight, -s * 0.4, s * 0.4)
        nn.init.zeros_(self.decoder.mlp.down_proj.weight)

        # Embedding
        nn.init.normal_(self.tok_embed.weight, mean=0.0, std=0.001)
        # tie token embedding
        self.lm_head.weight = self.tok_embed.weight

        nn.init.trunc_normal_(self.pos_basis, std=1.0)

        # Rotary embeddings
        cos, sin = precompute_rotary_emb(self.rotary_seq_len, self.head_dim, device=self.device)
        self.cos, self.sin = cos, sin

        self.mask = torch.empty(self.config.n_text_ctx, self.config.n_text_ctx).fill_(-float('inf')).triu_(1)

        if self.tok_embed.weight.device.type == 'cuda':
            self.tok_embed.weight = self.tok_embed.weight.to(torch.bfloat16)
            self.pos_basis = self.pos_basis.to(torch.bfloat16)
