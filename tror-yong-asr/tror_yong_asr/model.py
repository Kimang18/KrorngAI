# Author: KrorngAI org.
# Date: March 2026
"""
Features:
- auditory-lingual decoder with a single Transformer
- gated multi-head attention
- exclusive self-attention
- p-RoPE in self-attention
- p-LARoPE in cross-attention
- Derf instead of normalization layer in decoder
- head-specific QK-normalization with Derf before SDP attn
- SiLU gate in MLP of transformer
- during inference, accepts audio array of arbitrary length <= 30 * 16_000
"""


from typing import List, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import CausalLMOutput
from transformers import GenerationConfig
from huggingface_hub import PyTorchModelHubMixin
from .nn_utils import (
    precompute_rotary_emb,
    Qwen3RMSNorm,
    Dynamic_erf,
    apply_rotary_emb,
    MLP,
    LinearWrapper,
    Conv1dWrapper,
    KVCache,
)
from .common import print_banner
from .whisper_audio_encoder import AudioEncoder


@dataclass
class CustomCausalLMOutput(CausalLMOutput):
    logits: torch.FloatTensor = None
    loss: torch.FloatTensor = None
    dec_loss: torch.FloatTensor = None
    len_loss: torch.FloatTensor = None


class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention with Gated mechanism of Qwen3
    Base class of SelfAttention and CrossAttention
    Common methods are implemented here.
    """

    def __init__(self, n_embed, n_head, dropout=0.0, bias=True):
        super().__init__()
        self.n_head = n_head
        self.dropout = nn.Dropout(dropout)
        self.gate = LinearWrapper(n_embed, n_head, bias=bias)
        self.kv_proj = LinearWrapper(n_embed, 2 * n_embed, bias=bias)
        self.out_proj = LinearWrapper(n_embed, n_embed, bias=bias)
        self.q_norm = Dynamic_erf(n_embed)
        self.k_norm = Dynamic_erf(n_embed)

    def q_projection(self, query: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """
        Compute and return query vector
        """
        q = (
            self.q_norm(self.q_proj(query))
            .view(*query.shape[:2], self.n_head, -1)
            .transpose(1, 2)
        )
        q = apply_rotary_emb(q, cos, sin)
        return q

    def kv_projection(self, content: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute and return key and value vectors
        """
        k, v = self.kv_proj(content).chunk(2, dim=-1)
        k = (
            self.k_norm(k)
            .view(*k.shape[:2], self.n_head, -1)
            .transpose(1, 2)
        )
        k = apply_rotary_emb(k, cos, sin)
        v = v.view(*v.shape[:2], self.n_head, -1).transpose(1, 2)
        return k, v

    def qkv_projection(self, query, content, cos_sin, kv_cache=None) -> tuple[tuple[Tensor, Tensor], Tensor, Tensor]:
        """
        Compute and return query, key, and value vectors
        query and content must be **normalized**
        cos_sin is a tuple [q_cos, q_sin, k_cos, k_sin]
        """
        raise NotImplementedError()

    def exclude_self_value(self, qkv, v_proj) -> Tensor:
        """
        For Exclusive Self Attention, project qkv into the hyperplane
        of self-value
        """
        raise NotImplementedError()

    def attention_forward(self, q_proj: tuple[Tensor, Tensor], k_proj: Tensor, v_proj: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        Given the q_proj, k_proj, v_proj vectors and attention mask, return the attention output
        NOTE: we separate this function to save time as we use different attention masks
        for the same query, key, and value, and qkv projection requires compute and time
        NOTE: q_proj is a tuple of tensors, query vector and gate scores
        """
        qkv = F.scaled_dot_product_attention(q_proj[0], k_proj, v_proj, dropout_p=self.dropout.p, attn_mask=attn_mask, is_causal=False)
        qkv = self.exclude_self_value(qkv, v_proj)

        # apply gate score
        qkv = qkv * torch.sigmoid(q_proj[1])

        qkv = qkv.transpose(1, 2).flatten(start_dim=2)      # (b, lq, n_embed)
        return self.dropout(self.out_proj(qkv))


class SelfMultiHeadAttention(MultiHeadAttention):
    def __init__(self, n_embed, n_head, dropout=0.0, bias=True):
        super().__init__(n_embed, n_head, dropout, bias)
        # Specific to this design, self-attention does not require q_proj
        self.q_proj = nn.Identity()

    def kv_projection(self, content: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        in SelfAttention, compute gate score from content
        """
        k, v = super().kv_projection(content, cos, sin)

        g_score = (
            self.gate(content)  # (b, lk, n_head)
            .transpose(1, 2)
            .unsqueeze(-1)  # (b, n_head, lk, 1)
        )
        return k, v, g_score

    def qkv_projection(
            self,
            query: Tensor,
            content: Tensor,
            cos_sin: tuple[Tensor, Tensor, Tensor, Tensor],
            kv_cache: Optional[KVCache] = None) -> tuple[tuple[Tensor, Tensor], Tensor, Tensor]:
        """
        kv_cache stores key and value of previous positions.
        We compute key and value of current position
        and concatenate with the previous ones by kv_cache.insert_kv
        """
        # apply RoPE
        q_proj = self.q_projection(query, *cos_sin[:2])  # (b, n_head, lq, head_dim)
        k_proj, v_proj, g_score = self.kv_projection(content, *cos_sin[2:])
        if kv_cache is not None:
            k_proj, v_proj = kv_cache.insert_kv(0, k_proj, v_proj)  # there is only one layer, so layer_idx=0
        return (q_proj, g_score), k_proj, v_proj

    def exclude_self_value(self, qkv: Tensor, v_proj: Tensor) -> Tensor:
        """
        Project qkv to subspace that is orthogonal to v_proj
        to perform exclusive self-attention
        https://arxiv.org/pdf/2603.09078
        """
        if qkv.shape == v_proj.shape:
            vn = F.normalize(v_proj, dim=-1, eps=1e-6)
        else:  # when decoding, qkv=(b, n_head, 1, n_embed) != v_proj=(b, n_head, lk, head_dim)
            vn = F.normalize(v_proj[:, :, [-1], :], dim=-1, eps=1e-6)
        return qkv - torch.sum(qkv * vn, dim=-1, keepdim=True) * vn

    def attention_forward(self, q_proj: Tensor, k_proj: Tensor, v_proj: Tensor, attn_mask: Optional[Tensor] = None, key_mask: Optional[Tensor] = None) -> Tensor:
        """
        If key_mask is not None, then attn_mask is not None.
        So, construct the combined attention mask for SDA
        # key_mask: (b, 1, 1, lk), True means valid token, False means pad token
        """
        if key_mask is not None:  # always True during pre-training
            # assert attn_mask.shape == key_mask.shape
            attn_mask = key_mask & attn_mask
        return super().attention_forward(q_proj, k_proj, v_proj, attn_mask)

    def forward(
            self,
            query: Tensor,
            content: Tensor,
            cos_sin: tuple[Tensor, Tensor, Tensor, Tensor],
            attn_mask: Optional[Tensor]=None,
            key_mask: Optional[Tensor]=None,
            kv_cache: Optional[KVCache]=None) -> Tensor:
        """
        During training, attn_mask is generated by permutation.
        """
        q_proj, k_proj, v_proj = self.qkv_projection(query, content, cos_sin, kv_cache)
        return self.attention_forward(q_proj, k_proj, v_proj, attn_mask, key_mask)


class CrossMultiHeadAttention(MultiHeadAttention):
    def __init__(self, n_embed, n_head, dropout=0.0, bias=True):
        super().__init__(n_embed, n_head, dropout, bias)
        self.q_proj = LinearWrapper(n_embed, n_embed, bias=bias)

    def q_projection(self, query: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
        """
        in CrossAttention, compute gate score from query
        """
        q_proj = super().q_projection(query, cos, sin)

        g_score = (
            self.gate(query)  # (b, lq, n_head)
            .transpose(1, 2)
            .unsqueeze(-1)  # (b, n_head, lq, 1)
        )
        return (q_proj, g_score)

    def qkv_projection(
            self,
            query: Tensor,
            content: Tensor,
            cos_sin: tuple[Tensor, Tensor, Tensor, Tensor],
            kv_cache: Optional[dict] = None) -> tuple[tuple[Tensor, Tensor], Tensor, Tensor]:
        """
        cos_sin is length-aware rotation
        kv_cache is a dict that stores projections of audio encoding
        """
        # apply LARoPE
        (q_proj, g_score) = self.q_projection(query, *cos_sin[:2])
        if kv_cache is None:
            k_proj, v_proj = self.kv_projection(content, *cos_sin[2:])
        else:
            k_proj = kv_cache.get('key', None)
            if k_proj is None:
                k_proj, v_proj = self.kv_projection(content, *cos_sin[2:])
                kv_cache['key'] = k_proj
                kv_cache['value'] = v_proj
            else:
                v_proj = kv_cache['value']
        return (q_proj, g_score), k_proj, v_proj

    def exclude_self_value(self, qkv: Tensor, v_proj: Tensor) -> Tensor:
        """
        Cross Attention does not Exclude Self Value
        """

        return qkv

    def forward(
            self,
            query: Tensor,
            content: Tensor,
            cos_sin: tuple[Tensor, Tensor, Tensor, Tensor],
            key_mask: Optional[Tensor]=None,
            kv_cache: Optional[dict]=None) -> Tensor:
        """
        CrossAttention does not receive attn_mask argument
        key_mask: (b, 1, 1, lk), True means valid frame, False means pad frame
        """
        q_proj, k_proj, v_proj = self.qkv_projection(query, content, cos_sin, kv_cache)
        return self.attention_forward(q_proj, k_proj, v_proj, key_mask)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, bias) -> None:
        super().__init__()
        self.self_attn = SelfMultiHeadAttention(d_model, nhead, dropout, bias)
        self.ca_norm = Dynamic_erf(d_model)
        self.cross_attn = CrossMultiHeadAttention(d_model, nhead, dropout, bias)
        self.mlp_norm = Dynamic_erf(d_model)
        self.mlp = MLP(d_model, dim_feedforward, dropout, bias)

    def forward(
            self,
            q: Tensor,
            ctx: Tensor,
            aud: Tensor,
            sa_cos_sin: tuple[Tensor, Tensor, Tensor, Tensor],
            attn_mask: Tensor,
            ca_cos_sin: tuple[Tensor, Tensor, Tensor, Tensor],
            tok_mask: Optional[Tensor] = None,
            aud_mask: Optional[Tensor] = None,
            kv_cache: Optional[tuple[Optional[KVCache], Optional[dict]]] = (None, None)
            ) -> Tensor:
        """
        regular forward pass
        WARN: ctx and aud must be already normalized
        NOTE: sa_cos_sin for RoPE and ca_cos_sin for LARoPE
        """
        # NOTE: q is not normalized when passing to self_attn cuz it is used as q_proj, and I only apply rotation.
        hidden = q + self.self_attn(q, ctx, sa_cos_sin, attn_mask, tok_mask, kv_cache[0])
        hidden = hidden + self.cross_attn(self.ca_norm(hidden), aud, ca_cos_sin, aud_mask, kv_cache[1])
        hidden = hidden + self.mlp(self.mlp_norm(hidden))
        return hidden


class LengthPredictor(nn.Module):
    def __init__(self, n_text_ctx: int, pad_token_id: int, eot_token_id: int, transcribe_id: int, n_embed: int, dropout: float, bias: bool):
        super().__init__()
        self.half_text_ctx = 0.5 * n_text_ctx
        self.pad_token_id = pad_token_id
        self.eot_token_id = eot_token_id
        self.transcribe_id = transcribe_id
        self.no_speech_id = transcribe_id + 2  # this is due to TrorYongASR tokenizer design
        self.conv3 = Conv1dWrapper(n_embed, n_embed, kernel_size=5, stride=5)
        self.task_embed = nn.Embedding(3, n_embed)  # transcribe, translate, no_speech
        self.dropout = nn.Dropout(dropout)
        # Feature-wise Linear Modulation
        self.ab_proj = LinearWrapper(n_embed, 2 * n_embed, bias=False)  # alpha, beta projections
        # Output head
        self.head = LinearWrapper(n_embed, 1, bias=bias)

    def forward(self, aud_enc: Tensor, task_ids: Tensor, aud_mask: Optional[Tensor] = None) -> Tensor:
        """
        aud_enc: (b, n_mels, n_frames)
        task_ids: (b, 1)
        """
        task_ids.masked_fill_((task_ids == self.pad_token_id) | (task_ids == self.eot_token_id), self.no_speech_id)  # for no_speech case, replace eot_token_id or pad_token_id by no_speech_id
        task_ids = task_ids - self.transcribe_id  # convert to class: 0 (transcribe) or 1 (translate) or 2 (no_speech)
        task = self.dropout(self.task_embed(task_ids).squeeze(1))  # (b, n_embed)
        # Apply FiLM on the audio encoding
        alpha, beta = (
            self.ab_proj(task)
            .view(task.shape[0], 2, -1, 1)  # the last dim is 1 for broadcasting at the feature dimension of x
            .unbind(1)
        )
        x = aud_enc.transpose(1, 2).contiguous()    # (b, n_embed, lk)

        x = self.conv3(x)
        x = alpha * x + beta
        if aud_mask is not None:  # True during training
            aud_mask = aud_mask.unsqueeze(1).float()  # (b, 1, lk)
            # reduce mask using the same setting of conv3
            mask = F.max_pool1d(aud_mask, kernel_size=5, stride=5)
            x = F.gelu(mask * x)  # suppress bias on padded

            # masked_global_avg_pool1d
            x = x.sum(dim=-1, keepdim=True)  # (b, n_embed, 1)
            valid_counts = mask.sum(dim=-1, keepdim=True)  # (b, 1, 1)
            x = x / (valid_counts + 1e-6)    # (b, n_embed, 1)
        else:
            x = F.adaptive_avg_pool1d(F.gelu(x), 1)  # (b, n_embed, 1)

        # Output logits
        logits = self.head(x.squeeze(-1)).squeeze(-1)
        # use softcap and shift output to [0, 2*half_text_ctx]
        return self.half_text_ctx * torch.tanh(logits / self.half_text_ctx) + self.half_text_ctx


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
    gamma: float = 10.0  # scaling factor for LARoPE
    rope_percentage: float = 0.75  # for p-RoPE and p-LARoPE
    pad_token_id: int = 0
    eot_token_id: int = 2
    transcribe_id: int = 48002
    tie_word_embeddings: bool = True


class TrorYongASRModel(
    nn.Module,
    PyTorchModelHubMixin
):
    def __init__(self, config: TrorYongASRConfig, verbose: bool=True) -> None:
        super().__init__()
        self.config = config
        self.generation_config = GenerationConfig()
        self.generation_config.pad_token_id = self.config.pad_token_id
        self.generation_config.eos_token_id = self.config.eot_token_id
        self.generation_config.lang_to_id = {"km": 48000, "en": 48001}
        self.generation_config.task_to_id = {"transcribe": 48002, "translate": 48003}

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
        cos, sin = precompute_rotary_emb(config.n_text_ctx, self.head_dim, device=self.device, rope_percentage=config.rope_percentage)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.pos_basis = nn.Parameter(Tensor(config.n_embed))
        self.dropout = nn.Dropout(config.dropout)
        mlp_dim = int(8 * config.n_embed // 3)
        self.decoder = DecoderBlock(config.n_embed, self.n_head, mlp_dim, config.dropout, config.bias)

        # Make vocab_size multiple of 32
        self.vocab_size = config.vocab_size
        self.pad_vocab_size = (config.vocab_size + 31) // 32 * 32
        self.tok_embed = nn.Embedding(self.pad_vocab_size, config.n_embed, padding_idx=self.config.pad_token_id)
        self.aud_norm = Qwen3RMSNorm(config.n_embed)
        self.ctx_norm = Qwen3RMSNorm(config.n_embed)
        self.lm_norm = Qwen3RMSNorm(config.n_embed)
        self.lm_head = LinearWrapper(config.n_embed, self.pad_vocab_size, bias=False)

        # weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embed.weight

        # LARoPE
        base = 1e4  # base frequency default to 1e4
        rope_angles = int(config.rope_percentage * self.head_dim // 2)
        # Generate frequencies for the number of pairs we are actually rotating
        # 'NoPE' part gets '0.0', so sin(0)=0, cos(0)=1
        # NOTE: gamma is included here to save computation in compute_cos_sin
        # i.e., instead of gamma * pos * inv_freq, we do pos * gamma_inv_freq (outer product)
        gamma_inv_freq = torch.zeros(self.head_dim // 2, dtype=torch.float32)
        gamma_inv_freq[:rope_angles] = config.gamma / (
            base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32, device=self.device) / self.head_dim)
        )

        self.register_buffer("gamma_inv_freq", gamma_inv_freq, persistent=False)

        # Sequence Length Predictor, used for LARoPE during inference
        self.len_predictor = LengthPredictor(
            config.n_text_ctx,
            config.pad_token_id,
            config.eot_token_id,
            config.transcribe_id,
            config.n_embed,
            config.dropout,
            config.bias
        )  # sequence length predictor specific to LARoPE

        mask = torch.tril(torch.ones((self.config.n_text_ctx, self.config.n_text_ctx), dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)
        self.prefix = None
        self.init_weights()
        if verbose:
            print_banner()

    def get_token_embedding(self, token_ids: Tensor) -> Tensor:
        return self.tok_embed(token_ids)

    def encode(self, mels: Tensor) -> Tensor:
        return self.encoder.encode(mels)

    def logits(
            self,
            aud_enc: Tensor,
            ctx: Tensor,
            aud_mask: Optional[Tensor] = None,
            tok_mask: Optional[Tensor] = None,
            pred_seq_len: Optional[float] = 0.0,
            kv_cache: Optional[tuple[Optional[KVCache], Optional[dict]]] = (None, None)
            ) -> Tensor:
        """
        tok_mask: True means valid token, False means pad token
        aud_mask: True means valid frame, False means pad frame
        pred_seq_len: predicted number of tokens for LARoPE
        NOTE: ctx and aud_enc must be already normalized
        WARN: when pred_seq_len != 0, ignore tok_mask, aud_mask
        """
        T0 = 0 if kv_cache[0] is None else kv_cache[0].get_pos()

        # decoding layer
        b, L, _ = ctx.size()
        T1 = T0 + L
        # assert T1 <= self.config.n_text_ctx, "input_ids is too long"

        q = self.pos_basis.expand(b, L, -1)

        attn_mask = None
        if L > 1:  # always True during training. During inference, True for the initial tokens
            attn_mask = self.mask[T0:T1, T0:T1].view(1, 1, L, L)

        # T1+1 because Q is one-phase faster than K
        sa_cos_sin = self.cos[:, :, T0+1:T1+1], self.sin[:, :, T0+1:T1+1], self.cos[:, :, T0:T1], self.sin[:, :, T0:T1]

        if pred_seq_len == 0.0:  # always True during training
            actual_seq_len = torch.sum(tok_mask, dim=-1, dtype=torch.float32)  # (b, )
            actual_aud_len = torch.sum(aud_mask, dim=-1, dtype=torch.float32)  # (b, )

            aud_mask = aud_mask.view(b, 1, 1, -1)  # (b, 1, 1, lk)
            tok_mask = tok_mask.view(b, 1, 1, -1)  # (b, 1, 1, L)
            seq_len = L
        else:  # not training, ignore tok_mask, and aud_mask
            actual_seq_len = None
            actual_aud_len = None
            seq_len = pred_seq_len if T1 < pred_seq_len else T1

        # LARoPE
        q_cos, q_sin = self.compute_cos_sin(seq_len, self.device, actual_seq_len, q.dtype)
        k_cos, k_sin = self.compute_cos_sin(aud_enc.shape[1], self.device, actual_aud_len, q.dtype)
        ca_cos_sin = q_cos[:, :, T0:T1], q_sin[:, :, T0:T1], k_cos, k_sin

        res = self.decoder(q, ctx, aud_enc, sa_cos_sin, attn_mask, ca_cos_sin, tok_mask, aud_mask, kv_cache)
        logits = self.lm_head(self.lm_norm(res))  # (b, L, pad_vocab_size)
        return logits[..., :self.vocab_size]

    def forward(self, mels: Tensor, mel_masks: Tensor, input_ids: Tensor, labels: Optional[Tensor] = None) -> CausalLMOutput:
        """
        mels: log-mel spectrogram
        mel_masks: True means valid frame, False means pad frame
        input_ids: input token ids for decoder (inclu. sot sequence)
        labels: target token ids to compute cross_entropy loss
        """

        # NOTE: in Whisper Audio Encoder, the audio encoding is not normalized by ln_post()
        aud_enc, aud_mask = self.encoder(mels, mel_masks)
        tok_mask = input_ids != self.config.pad_token_id
        ctx = self.pos_basis.view(1, 1, -1) * self.get_token_embedding(input_ids)
        ctx = self.dropout(ctx)
        if labels is None:
            logits = self.logits(self.aud_norm(aud_enc), self.ctx_norm(ctx), aud_mask, tok_mask)  # (b, L, vocab_size)
            # do not compute loss
            return CausalLMOutput(logits=logits)
        else:
            # predict sequence length
            # pred_len is bounded on (0, n_text_ctx) via softcap and shift
            pred_len = self.len_predictor(aud_enc.detach(), input_ids[:, [2]], aud_mask)
            target_len = torch.sum(tok_mask, dim=-1, dtype=torch.float32)
            len_loss = F.smooth_l1_loss(pred_len.float(), target_len, beta=1.0)

            logits = self.logits(self.aud_norm(aud_enc), self.ctx_norm(ctx), aud_mask, tok_mask)  # (b, L, vocab_size)
            dec_loss = F.cross_entropy(logits.reshape(-1, self.vocab_size).float(), labels.reshape(-1), ignore_index=-100, reduction='mean')

            return CustomCausalLMOutput(logits=logits, loss=dec_loss+len_loss, dec_loss=dec_loss, len_loss=len_loss)

    def compute_cos_sin(
            self,
            seq_len: float,
            device: torch.device,
            actual_seq_len: Optional[Tensor] = None,
            dtype=torch.float32
        ) -> tuple[Tensor, Tensor]:
        # Different from RoPE by normalize pos with the seq_len
        # seq_len may include pad tokens, actual_seq_len does not
        pos = torch.arange(int(seq_len), device=device, dtype=torch.float32)
        if actual_seq_len is None:
            # cos_sin for a single sequence
            pos = pos / seq_len
            freqs = torch.outer(pos.float(), self.gamma_inv_freq.float()).unsqueeze(0).unsqueeze(0)
        else:
            # cos_sin for batched sequences
            pos = pos.unsqueeze(0) / actual_seq_len.unsqueeze(-1)  # normalize position and rescale
            freqs = torch.einsum('bi,bj->bij', pos.float(), self.gamma_inv_freq.unsqueeze(0).float())  # (b, l, head_dim / 2)
            freqs = freqs.unsqueeze(1)  # expand for head dimension

        cos, sin = freqs.cos(), freqs.sin()
        return cos.to(dtype), sin.to(dtype)

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
    def identify_language(self, mel: Tensor, tokenizer):
        """
        Predicting the language of audio
        """
        mels = mel.unsqueeze(0)
        aud_enc = self.encode(mels)

        idx = torch.as_tensor([[tokenizer.sot]], dtype=torch.long, device=self.device)
        ctx = self.pos_basis.view(1, 1, -1) * self.get_token_embedding(idx)
        logits = self.logits(self.aud_norm(aud_enc), self.ctx_norm(ctx), pred_seq_len=1.0)

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

    def generate(self, mels=None, mel_masks=None, generation_max_length=64, **kwargs):
        if mels is None:
            raise ValueError("input_ids must be provided for generation.")

        lang_id = self.generation_config.lang_to_id[self.generation_config.language]
        task_id = self.generation_config.task_to_id[self.generation_config.task]
        self.set_prefix([1, lang_id, task_id]) # 1 is sot_token_id
        b = mels.shape[0]
        generated_seq = []

        for i in range(b):
            mel = mels[i, :, mel_masks[i]]
            decode_ids = self.decode(mel, max_tokens=generation_max_length)
            generated_seq.append(decode_ids)
        result = pad_sequence(generated_seq, batch_first=True, padding_value=self.generation_config.pad_token_id)
        return result

    @torch.inference_mode()
    def decode(self, mel: Tensor, max_tokens: int, temperature=1.0, top_k=None, seed=168) -> Tensor:
        """
        mel: (n_mels, n_frames)
        max_tokens: int
        return: list of token ids
        """
        if self.prefix is None:
            raise SyntaxError("Please provide bos, language, and task tokens as prefix using set_prefix before calling decode.")

        max_seq_len = self.config.n_text_ctx
        assert max_tokens <= max_seq_len, "too long sequence generation, consider lower max_tokens"
        kv_cache = [
            KVCache(1, self.n_head, max_seq_len, self.head_dim, 1),  # batch_size=1, num_layers=1
            dict()
        ]
        device = self.device
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        mels = mel.unsqueeze(0)
        aud_enc = self.encode(mels)

        idx = torch.as_tensor([self.prefix], dtype=torch.long, device=mels.device)
        ctx = self.pos_basis.view(1, 1, -1) * self.get_token_embedding(idx)
        n = idx.shape[1]

        # predict seq_len
        pred_seq_len = self.len_predictor(aud_enc, idx[:, [2]])

        aud_enc = self.aud_norm(aud_enc)  # normalize outside the for-loop to save compute
        for i in range(n, max_tokens):
            logits = self.logits(aud_enc, self.ctx_norm(ctx), pred_seq_len=pred_seq_len, kv_cache=kv_cache)

            if temperature > 0:
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                probs = F.softmax(logits, dim=-1)
                id_next = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                id_next = logits[:, [-1], :].argmax(dim=-1)
            idx = torch.cat((idx, id_next), dim=1)
            if id_next.item() == self.config.eot_token_id:
                break
            ctx = self.pos_basis.view(1, 1, -1) * self.get_token_embedding(id_next)
        return idx[0]

    @torch.no_grad()
    def init_weights(self):
        nn.init.normal_(self.encoder.conv1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.encoder.conv2.weight, mean=0.0, std=0.02)

        s = 3**0.5 * self.config.n_embed**-0.5
        for block in self.encoder.blocks:
            nn.init.uniform_(block.attn.qkv_proj.weight, -s, s)
            nn.init.zeros_(block.attn.out.weight)
            nn.init.uniform_(block.mlp[0].weight, -s * 0.4, s * 0.4)
            nn.init.zeros_(block.mlp[2].weight)

        nn.init.uniform_(self.decoder.self_attn.gate.weight, -s, s)
        nn.init.uniform_(self.decoder.self_attn.kv_proj.weight, -s, s)
        nn.init.zeros_(self.decoder.self_attn.out_proj.weight)

        nn.init.uniform_(self.decoder.cross_attn.q_proj.weight, -s, s)
        nn.init.uniform_(self.decoder.cross_attn.gate.weight, -s, s)
        nn.init.uniform_(self.decoder.cross_attn.kv_proj.weight, -s, s)
        nn.init.zeros_(self.decoder.cross_attn.out_proj.weight)

        nn.init.uniform_(self.decoder.mlp.gate_proj.weight, -s * 0.4, s * 0.4)
        nn.init.uniform_(self.decoder.mlp.up_proj.weight, -s * 0.4, s * 0.4)
        nn.init.zeros_(self.decoder.mlp.down_proj.weight)

        nn.init.kaiming_normal_(self.len_predictor.conv3.weight, mode='fan_out')
        nn.init.trunc_normal_(self.len_predictor.task_embed.weight, std=1.0)
        nn.init.normal_(self.len_predictor.head.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.len_predictor.ab_proj.weight)
        if self.len_predictor.ab_proj.bias is not None:
            nn.init.ones_(self.len_predictor.ab_proj.bias)

        # Embedding
        nn.init.normal_(self.tok_embed.weight, mean=0.0, std=0.001)
        # tie token embedding
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embed.weight

        nn.init.trunc_normal_(self.pos_basis, std=1.0)

        # Rotary embeddings
        self.cos, self.sin = precompute_rotary_emb(self.config.n_text_ctx, self.head_dim, device=self.device, rope_percentage=self.config.rope_percentage)

        base = 1e4  # base frequency default to 1e4
        rope_angles = int(self.config.rope_percentage * self.head_dim // 2)
        # Generate frequencies for the number of pairs we are actually rotating
        # 'NoPE' part gets '0.0', so sin(0)=0, cos(0)=1
        self.gamma_inv_freq = torch.zeros(self.head_dim // 2, dtype=torch.float32)
        self.gamma_inv_freq[:rope_angles] = self.config.gamma / (
            base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / self.head_dim)
        )

        self.mask = torch.tril(torch.ones((self.config.n_text_ctx, self.config.n_text_ctx), dtype=torch.bool))

        if self.tok_embed.weight.device.type == 'cuda':
            self.tok_embed.weight = self.tok_embed.weight.to(torch.bfloat16)
            self.pos_basis = self.pos_basis.to(torch.bfloat16)
