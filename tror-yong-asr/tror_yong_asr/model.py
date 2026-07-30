# Author: KrorngAI org.
# Date: March 2026
"""
Features:
- exclusive self-attention
- text decoder with a single transformer block
- RoPE in text decoder
- Dynamic Error Function instead of normalization layer in text decoder
- SiLU gate in MLP of transformer
- QK-normalization before attention mechanism
"""


from typing import List, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.modeling_outputs import CausalLMOutput
from huggingface_hub import PyTorchModelHubMixin
import math
from .nn_utils import (
    precompute_rotary_emb,
    DerfWrapper,
    norm,
    apply_rotary_emb,
    MLP,
    LinearWrapper,
    Conv1dWrapper,
    KVCache,
)
from .common import print_banner
from .whisper_audio_encoder import AudioEncoder


class MultiheadAttention(nn.Module):
    def __init__(self, n_embed, n_head, dropout=0.0, bias=True, is_ca=False):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embed // n_head
        self.dropout = nn.Dropout(dropout)
        if is_ca:  # in this implementation self_attn does not need q_proj
            self.q_proj = LinearWrapper(n_embed, n_embed, bias=bias)
        self.k_proj = LinearWrapper(n_embed, n_embed, bias=not is_ca)
        self.v_proj = LinearWrapper(n_embed, n_embed, bias=bias)
        self.out_proj = LinearWrapper(n_embed, n_embed, bias=bias)
        self.q_norm = DerfWrapper(self.head_dim)
        self.k_norm = DerfWrapper(self.head_dim)
        self.is_ca = is_ca

    def q_projection(self, q, cos, sin):
        """
        Compute and return the projection of query
        """
        if self.is_ca:  # in this implementation, self MHA does not need q projection
            q = self.q_proj(q)
        q = q.view(*q.shape[:2], self.n_head, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        q = apply_rotary_emb(q, cos, sin)
        return q

    def kv_projection(self, k, v, cos, sin):
        """
        Compute and return the projection of key and value
        """
        k = self.k_proj(k)
        k = k.view(*k.shape[:2], self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_norm(k)
        k = apply_rotary_emb(k, cos, sin)
        v = self.v_proj(v)
        v = v.view(*v.shape[:2], self.n_head, self.head_dim).transpose(1, 2)
        return k, v

    def qkv_projection(self, query, key, value, cos_sin, kv_cache=None):
        """
        Compute and return the projection of query, key, and value
        """
        # apply LARoPE
        q_cos, q_sin, k_cos, k_sin = cos_sin
        q_proj = self.q_projection(query, q_cos, q_sin)
        if kv_cache is None:
            k_proj, v_proj = self.kv_projection(key, value, k_cos, k_sin)
        else:
            if not self.is_ca or 'key' not in kv_cache:
                k_proj, v_proj = self.kv_projection(key, value, k_cos, k_sin)
                if self.is_ca:
                    kv_cache['key'] = k_proj
                    kv_cache['value'] = v_proj
                else:
                    k_proj, v_proj = kv_cache.insert_kv(0, k_proj, v_proj)  # there is only one layer, so layer_idx=0
            else:
                k_proj, v_proj = kv_cache['key'], kv_cache['value']
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

        qkv = qkv.transpose(1, 2).flatten(start_dim=2)        # (b, lq, n_embed)
        out = self.out_proj(qkv)

        return self.dropout(out), None

    def forward(self, query, key, value, cos_sin, attn_mask=None, key_padding_mask=None, kv_cache=None):
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
        return q + query

    def forward_cross_attn_and_mlp(self, res, q_proj, k_proj, v_proj):
        """
        WARN: res must be unnormalized in order to do residual connection
        """
        query, ca_weights = self.cross_attn.attention_forward(q_proj, k_proj, v_proj)
        res = res + query
        res = res + self.mlp(self.mlp_norm(res))
        return res

    def forward(self, q, ctx, aud, sa_cos_sin, attn_mask, ca_cos_sin, key_padding_mask=None, kv_cache=(None, None)):
        """
        regular forward pass
        WARN: q must be unnormalized in order to do residual connection
        NOTE: try LARoPE implementation - apply length-aware RoPE on q_proj and k_proj
        right before attention forward
        """
        # NOTE: q is not normalized when passing to self_attn because it is q_projection already, and I only apply rotation.
        query, sa_weights = self.self_attn(q, ctx, ctx, sa_cos_sin, attn_mask, key_padding_mask, kv_cache[0])
        res = q + query
        query, ca_weights = self.cross_attn(self.ca_norm(res), aud, aud, ca_cos_sin, kv_cache=kv_cache[1])
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
    gamma: float = 10.0 # scaling factor for LARoPE
    pad_id: int = 0
    eot_id: int = 2
    tie_word_embeddings: bool = True


class TrorYongASRModel(
    nn.Module,
    PyTorchModelHubMixin
):
    def __init__(self, config: TrorYongASRConfig, verbose: bool=True) -> None:
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

        self.pos_basis = nn.Parameter(Tensor(config.n_embed))
        self.dropout = nn.Dropout(config.dropout)
        mlp_dim = int(8 * config.n_embed // 3)
        self.decoder = DecoderBlock(config.n_embed, self.n_head, mlp_dim, config.dropout, config.bias)

        # Make vocab_size multiple of 32
        self.vocab_size = config.vocab_size
        self.pad_vocab_size = (config.vocab_size + 31) // 32 * 32
        self.tok_embed = nn.Embedding(self.pad_vocab_size, config.n_embed, padding_idx=self.config.pad_id)
        self.lm_head = LinearWrapper(config.n_embed, self.pad_vocab_size, bias=False)

        # weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embed.weight

        # LARoPE
        self.gamma = config.gamma  # LaRoPE gamma rescaling factor
        base = 1e4  # base frequency default to 1e4
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Sequence Length Predictor, used for LARoPE's inference
        self.conv3 = Conv1dWrapper(config.n_embed, config.n_embed, kernel_size=5, stride=5)
        self.lp_head = LinearWrapper(config.n_embed, config.n_text_ctx) # sequence length predictor specific to LARoPE
        self.lp_factor = 0.5  # weight of length prediciton loss

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

    def logits(self, aud_enc: Tensor, input_ids: Tensor, pred_seq_len: int=0, kv_cache=(None, None)):
        T0 = 0 if kv_cache[0] is None else kv_cache[0].get_pos()

        # decoding layer
        b, L = input_ids.size()
        T1 = T0 + L
        assert T1 <= self.config.n_text_ctx, "input_ids is too long"

        q = self.pos_basis.expand(b, L, -1)
        ctx = self.get_token_embedding(input_ids)
        if L > 1:  # always True during training. During inference, True for the first token
            ctx = ctx + F.pad(q[:, T0:T1-1], (0, 0, 1, 0, 0, 0))
        elif T0 > 0:  # During inference, from the second token onward
            ctx = ctx + q
        ctx = self.dropout(ctx)
        ctx = norm(ctx)

        # T1+1 because Q is one-phase faster than K
        sa_cos_sin = self.cos[:, :, T0+1:T1+1], self.sin[:, :, T0+1:T1+1], self.cos[:, :, T0:T1], self.sin[:, :, T0:T1]

        if pred_seq_len == 0:
            conditions = input_ids != self.config.pad_id
            key_padding_mask = torch.zeros((b, L), device=self.device)
            key_padding_mask.masked_fill_(~conditions, -float('inf'))
            actual_seq_len = torch.sum(conditions, dim=-1, dtype=torch.float32) # (b, )
            seq_len = L
        else:
            actual_seq_len = None
            key_padding_mask = None
            seq_len = pred_seq_len if T1 < pred_seq_len else T1

        q_cos, q_sin = self.compute_cos_sin(seq_len, self.device, actual_seq_len)
        k_cos, k_sin = self.compute_cos_sin(aud_enc.shape[1], self.device)
        ca_cos_sin = q_cos[:, :, T0:T1], q_sin[:, :, T0:T1], k_cos, k_sin

        attn_mask = self.mask[T0:T0+L, T0:T0+L]
        res = self.decoder(q, ctx, aud_enc, sa_cos_sin, attn_mask, ca_cos_sin, key_padding_mask, kv_cache)
        return self.lm_head(norm(res))  # (b, L, pad_vocab_size)

    def len_logits(self, aud_enc: Tensor):
        x = aud_enc.detach().transpose(1, 2)
        x = F.adaptive_avg_pool1d(F.gelu(self.conv3(x)), 1).squeeze(-1)
        return self.lp_head(x)

    def forward(self, mels, input_ids, target_ids=None, kv_cache=(None, None)):
        """
        mels: log-mel spectrum from audio array
        input_ids: input token ids for decoder (inclu. sot sequence)
        target_ids: target token ids to compute cross_entropy loss
        """
        # TODO: rethinking about seq-len prediction as it varies over tasks

        # decoding layer
        b, L = input_ids.size()
        assert L <= self.config.n_text_ctx, "input_ids is too long"

        # NOTE: in Whisper Audio Encoder, the audio encoding is already normalized by ln_post()
        aud = self.encoder(mels)

        q = self.pos_basis.expand(b, L, -1)
        ctx = self.get_token_embedding(input_ids)
        ctx = ctx + F.pad(q[:, :L-1], (0, 0, 1, 0, 0, 0))
        ctx = self.dropout(ctx)
        ctx = norm(ctx)

        # L+1 because Q is one-phase faster than K
        sa_cos_sin = self.cos[:, :, 1:L+1], self.sin[:, :, 1:L+1], self.cos[:, :, :L], self.sin[:, :, :L]

        conditions = input_ids != self.config.pad_id
        key_padding_mask = torch.zeros((b, L), device=self.device)
        key_padding_mask.masked_fill_(~conditions, -float('inf'))
        actual_seq_len = torch.sum(conditions, dim=-1, dtype=torch.float32) # (b, )

        q_cos, q_sin = self.compute_cos_sin(L, self.device, actual_seq_len)
        k_cos, k_sin = self.compute_cos_sin(self.config.n_audio_ctx, self.device)
        ca_cos_sin = q_cos, q_sin, k_cos, k_sin

        attn_mask = self.mask[:L, :L]
        res = self.decoder(q, ctx, aud, sa_cos_sin, attn_mask, ca_cos_sin, key_padding_mask, kv_cache)
        logits = self.lm_head(norm(res))  # (b, L, pad_vocab_size)
        if target_ids is not None:
            loss = F.cross_entropy(logits.flatten(end_dim=1).float(), target_ids.flatten(), ignore_index=-100, reduction='mean')

            # predict sequence length
            x = aud.detach().transpose(1, 2)  #(b, n_embed, n_audio_ctx)
            x = F.adaptive_avg_pool1d(F.gelu(self.conv3(x)), 1).squeeze(-1) #(b, n_embed)
            len_logits = self.lp_head(x)
            target_len = actual_seq_len.detach().clone().to(torch.long)

            loss += self.lp_factor * F.cross_entropy(len_logits, target_len, label_smoothing=0.05)

            return CausalLMOutput(logits=logits, loss=loss)
        else:
            # do not compute loss
            return CausalLMOutput(logits=logits)

    def compute_cos_sin(
            self,
            seq_len: int,
            device: torch.device,
            actual_seq_len: Optional[Tensor] = None
        ) -> tuple[Tensor, Tensor]:
        # seq_len may include pad tokens, actual_seq_len does not
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        if actual_seq_len is None:
            # cos_sin for key projection, i.e., audio encoding
            pos = self.gamma * pos / float(seq_len)
            freqs = torch.outer(pos, self.inv_freq).unsqueeze(0).unsqueeze(0)
        else:
            # cos_sin for query projection
            pos = (self.gamma * pos).unsqueeze(0) / actual_seq_len.unsqueeze(-1)  # normalize position and rescale
            freqs = torch.einsum('bi,bj->bij', pos, self.inv_freq.unsqueeze(0))  # (b, l, head_dim / 2)
            freqs = freqs.unsqueeze(1)  # expand for head dimension

        cos, sin = freqs.cos(), freqs.sin()
        return cos.bfloat16(), sin.bfloat16()

    @property
    def device(self):
        return self.encoder.conv1.weight.device

    def set_prefix(self, prefix: List[int]):
        """
        Set bos token, language token, and task token
        This is used in decode function
        """
        self.prefix = prefix

    @torch.inference_mode()
    def detect_language(self, mels: Tensor, tokenizer, kv_cache):
        """
        Predicting the language of audio
        NOTE: I could have called `forward` but there is a careful attention required for cos, sin
        """
        mels = mels.unsqueeze(0)
        aud_enc = self.encoder(mels)

        # predict seq_len
        len_logits = self.len_logits(aud_enc)
        len_logits[:, :2] = -float('inf') # no sequence has length shorter than 2
        pred_seq_len = len_logits.argmax(dim=-1).item()

        idx = torch.as_tensor([[tokenizer.sot]], dtype=torch.long, device=self.device)
        logits = self.logits(aud_enc, idx, pred_seq_len)

        # suppress non-language tokens
        mask = torch.ones(logits.shape[-1], dtype=torch.bool)
        mask[list(tokenizer.all_language_tokens) + [tokenizer.no_speech]] = False
        logits[:, -1, mask] = -float('inf')
        language_tokens = logits.argmax(dim=-1)
        language_token_probs = F.softmax(logits, dim=-1)
        language_probs = [
            {
                c: language_token_probs[0, 0, j].item()
                for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
            }
        ]
        language_probs[0]['no_speech'] = language_token_probs[0, 0, tokenizer.no_speech].item()

        return language_tokens[0], language_probs[0]

    @torch.inference_mode()
    def decode(self, mels: Tensor, max_tokens: int, temperature=1.0, top_k=None, seed=168):
        """
        mels: (n_mels,)
        max_tokens: int
        """
        if self.prefix is None:
            raise SyntaxError("Please provide bos, language, and task tokens as prefix using set_prefix before calling decode.")

        max_seq_len = self.config.n_text_ctx
        assert max_tokens <= max_seq_len, "too long sequence generation, consider lower max_tokens"
        kv_cache = [
            KVCache(1, self.n_head, max_seq_len, self.head_dim, 1), # batch_size=1, num_layers=1
            dict()
        ]
        device = self.device
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        mels = mels.unsqueeze(0)
        aud_enc = self.encoder(mels)

        # predict seq_len
        len_logits = self.len_logits(aud_enc)
        len_logits[:, :2] = -float('inf') # no sequence has length shorter than 2
        pred_seq_len = len_logits.argmax(dim=-1).item()

        idx = torch.as_tensor([self.prefix], dtype=torch.long, device=mels.device)
        n = idx.shape[1]
        idx_next = idx

        for i in range(n, max_tokens):
            logits = self.logits(aud_enc, idx_next, pred_seq_len, kv_cache)
            logits[:, -1, self.vocab_size:] = -float('inf') # suppress the pad tokens

            if temperature > 0:
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                idx_next = logits[:, [-1], :].argmax(dim=-1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() == self.config.eot_id:
                break
        return idx

    @torch.no_grad()
    def init_weights(self):
        nn.init.normal_(self.encoder.conv1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.encoder.conv2.weight, mean=0.0, std=0.02)

        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        nn.init.trunc_normal_(self.lp_head.weight, mean=0.0, std=0.02)

        s = 3**0.5 * self.config.n_embed**-0.5
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
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embed.weight

        nn.init.trunc_normal_(self.pos_basis, std=1.0)

        # Rotary embeddings
        cos, sin = precompute_rotary_emb(self.rotary_seq_len, self.head_dim, device=self.device)
        self.cos, self.sin = cos, sin

        self.mask = torch.empty(self.config.n_text_ctx, self.config.n_text_ctx).fill_(-float('inf')).triu_(1)

        if self.tok_embed.weight.device.type == 'cuda':
            self.tok_embed.weight = self.tok_embed.weight.to(torch.bfloat16)
            self.pos_basis = self.pos_basis.to(torch.bfloat16)
