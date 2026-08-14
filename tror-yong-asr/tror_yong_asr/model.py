# Author: KrorngAI org.
# Date: March 2026
"""
Features:
- exclusive self-attention
- text decoder with a single transformer block
- RoPE in text decoder
- Dynamic Error Function instead of normalization layer in text decoder
- SiLU gate in MLP of transformer
- QK-normalization with Derf before SDP
- LARoPE for Cross-Attention
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


class MultiHeadAttention(nn.Module):
    """
    Base class of SelfAttention and CrossAttention
    Common methods are implemented here.
    """

    def __init__(self, n_embed, n_head, dropout=0.0, bias=True):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embed // n_head
        self.dropout = nn.Dropout(dropout)
        self.k_proj = LinearWrapper(n_embed, n_embed, bias=False)
        self.v_proj = LinearWrapper(n_embed, n_embed, bias=bias)
        self.out_proj = LinearWrapper(n_embed, n_embed, bias=bias)
        self.q_norm = DerfWrapper(self.head_dim)
        self.k_norm = DerfWrapper(self.head_dim)

    def q_projection(self, query: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """
        Compute and return the projection of query
        """
        q = self.q_proj(query)
        q = q.view(*q.shape[:2], self.n_head, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        q = apply_rotary_emb(q, cos, sin)
        return q

    def kv_projection(self, content: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """
        Compute and return the projection of key and value
        """
        k = self.k_proj(content)
        k = k.view(*k.shape[:2], self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_norm(k)
        k = apply_rotary_emb(k, cos, sin)
        v = self.v_proj(content)
        v = v.view(*v.shape[:2], self.n_head, self.head_dim).transpose(1, 2)
        return k, v

    def qkv_projection(self, query, content, cos_sin, kv_cache=None) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute and return the projection of query, key, and value
        """
        raise NotImplementedError()

    def exclude_self_value(self, qkv, v_proj) -> Tensor:
        """
        For Exclusive Self Attention, project qkv into the hyperplane
        of self-value
        """
        raise NotImplementedError()

    def attention_forward(self, q_proj: Tensor, k_proj: Tensor, v_proj: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        Given the q_proj, k_proj, v_proj projections and attention mask, return the attention output
        NOTE: we separate this function to save time as we use different attention masks
        for the same query, key, and value and qkv projection requires compute, and time
        """
        qkv = F.scaled_dot_product_attention(q_proj, k_proj, v_proj, dropout_p=self.dropout.p, attn_mask=attn_mask, is_causal=False)
        qkv = self.exclude_self_value(qkv, v_proj)

        qkv = qkv.transpose(1, 2).flatten(start_dim=2)      # (b, lq, n_embed)
        out = self.out_proj(qkv)
        return self.dropout(out)


class SelfMultiHeadAttention(MultiHeadAttention):
    def __init__(self, n_embed, n_head, dropout=0.0, bias=True):
        super().__init__(n_embed, n_head, dropout, bias)
        # Specific to this design, self-attention does not require q_proj
        self.q_proj_bias = nn.Parameter(torch.zeros(n_embed))
        self.q_proj = lambda x: x + self.q_proj_bias

    def qkv_projection(
            self,
            query: Tensor,
            content: Tensor,
            cos_sin: tuple[Tensor, Tensor, Tensor, Tensor],
            kv_cache: Optional[KVCache] = None) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute and return the projection of query, key, and value
        """
        # apply LARoPE
        q_cos, q_sin, k_cos, k_sin = cos_sin
        q_proj = self.q_projection(query, q_cos, q_sin)
        k_proj, v_proj = self.kv_projection(content, k_cos, k_sin)
        if kv_cache is not None:
            k_proj, v_proj = kv_cache.insert_kv(0, k_proj, v_proj)  # there is only one layer, so layer_idx=0
        return q_proj, k_proj, v_proj

    def exclude_self_value(self, qkv: Tensor, v_proj: Tensor) -> Tensor:
        """
        Project qkv to subspace that is orthogonal to v_proj
        to perform exclusive self-attention
        """
        if qkv.shape == v_proj.shape:
            vn = F.normalize(v_proj, dim=-1, eps=1e-9)
        else:  # when decoding, qkv=(b, n_head, 1, n_embed) != v_proj=(b, n_head, lk, n_embed)
            vn = F.normalize(v_proj[:, :, [-1], :], dim=-1, eps=1e-9)
        return qkv - torch.sum(qkv * vn, dim=-1, keepdim=True) * vn

    def attention_forward(self, q_proj: Tensor, k_proj: Tensor, v_proj: Tensor, attn_mask: Optional[Tensor] = None, key_mask: Optional[Tensor] = None) -> Tensor:
        """
        """
        if key_mask is not None:  # always True during pre-training
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0) + key_mask
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
        """
        q_proj, k_proj, v_proj = self.qkv_projection(query, content, cos_sin, kv_cache)
        return self.attention_forward(q_proj, k_proj, v_proj, attn_mask, key_mask)


class CrossMultiHeadAttention(MultiHeadAttention):
    def __init__(self, n_embed, n_head, dropout=0.0, bias=True):
        super().__init__(n_embed, n_head, dropout, bias)
        self.q_proj = LinearWrapper(n_embed, n_embed, bias=bias)

    def qkv_projection(
            self,
            query: Tensor,
            content: Tensor,
            cos_sin: tuple[Tensor, Tensor, Tensor, Tensor],
            kv_cache: Optional[dict] = None) -> tuple[Tensor, Tensor, Tensor]:
        # apply LARoPE
        q_cos, q_sin, k_cos, k_sin = cos_sin
        q_proj = self.q_projection(query, q_cos, q_sin)
        if kv_cache is None:
            k_proj, v_proj = self.kv_projection(content, k_cos, k_sin)
        else:
            k_proj = kv_cache.get('key', None)
            if k_proj is None:
                k_proj, v_proj = self.kv_projection(content, k_cos, k_sin)
                kv_cache['key'] = k_proj
                kv_cache['value'] = v_proj
            else:
                v_proj = kv_cache['value']
        return q_proj, k_proj, v_proj

    def exclude_self_value(self, qkv: Tensor, v_proj: Tensor) -> Tensor:
        """
        Cross Attention does not Exclude Self Value
        """

        return qkv

    def attention_forward(self, q_proj: Tensor, k_proj: Tensor, v_proj: Tensor, key_mask: Optional[Tensor] = None) -> Tensor:
        """
        """
        return super().attention_forward(q_proj, k_proj, v_proj, attn_mask=key_mask)

    def forward(
            self,
            query: Tensor,
            content: Tensor,
            cos_sin: tuple[Tensor, Tensor, Tensor, Tensor],
            key_mask: Optional[Tensor]=None,
            kv_cache: Optional[dict]=None) -> Tensor:
        """
        """
        q_proj, k_proj, v_proj = self.qkv_projection(query, content, cos_sin, kv_cache)
        return self.attention_forward(q_proj, k_proj, v_proj, key_mask)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, bias) -> None:
        super().__init__()
        self.self_attn = SelfMultiHeadAttention(d_model, nhead, dropout, bias)
        self.cross_attn = CrossMultiHeadAttention(d_model, nhead, dropout, bias)
        self.ca_norm = DerfWrapper(d_model)
        self.mlp = MLP(d_model, dim_feedforward, dropout, bias)
        self.mlp_norm = DerfWrapper(d_model)

    def forward_self_attn(self, q: Tensor, q_proj: Tensor, k_proj: Tensor, v_proj: Tensor, attn_mask: Tensor, tok_mask: Optional[Tensor] = None) -> Tensor:
        """
        WARN: q must be unnormalized in order to do residual connection
        key_padding_mask is None during inference
        """
        query = self.self_attn.attention_forward(q_proj, k_proj, v_proj, attn_mask, tok_mask)
        return q + query

    def forward_cross_attn_and_mlp(self, res: Tensor, q_proj: Tensor, k_proj: Tensor, v_proj: Tensor, aud_mask: Optional[Tensor] = None) -> Tensor:
        """
        WARN: res must be unnormalized in order to do residual connection
        """
        query = self.cross_attn.attention_forward(q_proj, k_proj, v_proj, aud_mask)
        res = res + query
        res = res + self.mlp(self.mlp_norm(res))
        return res

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
        WARN: q must be unnormalized in order to do residual connection
        NOTE: try LARoPE implementation in cross attention - apply
        length-aware RoPE on q_proj and k_proj right before attention forward
        """
        # NOTE: q is not normalized when passing to self_attn because it is q_projection already, and I only apply rotation.
        query = self.self_attn(q, ctx, sa_cos_sin, attn_mask, tok_mask, kv_cache[0])
        res = q + query
        query = self.cross_attn(self.ca_norm(res), aud, ca_cos_sin, aud_mask, kv_cache[1])
        res = res + query
        res = res + self.mlp(self.mlp_norm(res))
        return res


class LengthPredictor(nn.Module):
    def __init__(self, transcribe_id:int, n_embed: int, dropout: float, bias: bool):
        super().__init__()
        self.transcribe_id = transcribe_id
        self.nospeech_id = transcribe_id + 2  # this is due to my tokenizer design
        self.dropout = dropout
        self.conv3 = Conv1dWrapper(n_embed, n_embed, kernel_size=5, stride=5)
        self.task_embed = nn.Embedding(4, n_embed) # transcribe, translate, nospeech, neutral
        # Feature-wise Linear Modulation
        self.alpha_proj = LinearWrapper(n_embed, n_embed, bias=False)
        self.beta_proj = LinearWrapper(n_embed, n_embed, bias=True)
        # Output head
        self.head = LinearWrapper(n_embed, 1, bias=bias)

    def forward(self, aud_enc: Tensor, task_ids: Tensor, aud_mask:Tensor) -> Tensor:
        x = aud_enc.transpose(1, 2).contiguous()    # (b, n_embed, lk)
        x = self.conv3(x)
        if aud_mask is not None:  # True during training
            aud_mask = aud_mask.unsqueeze(1).float()  # (b, 1, lk)
            # reduce mask using the same setting of conv3
            mask = F.max_pool1d(aud_mask, kernel_size=5, stride=5)
            x = mask * x  # suppress bias on padded

            # masked_global_avg_pool1d
            x = x.sum(dim=-1, keepdim=True)  # (b, n_embed, 1)
            valid_counts = mask.sum(dim=-1, keepdim=True)  # (b, 1, 1)
            x = x / (valid_counts + 1e-9)    # (b, n_embed, 1)
        else:
            x = F.adaptive_avg_pool1d(F.gelu(x), 1)

        task_ids.masked_fill_(task_ids == 2, self.nospeech_id)  # for no_speech case, replace eot_id by nospeech_id
        task_ids = task_ids - self.transcribe_id  # convert to class: 0 (transcribe) or 1 (translate) or 2 (nospeech)
        # Apply class-free guidance condition dropout during training
        if self.training and self.dropout > 0.0:
            drop_mask = torch.rand(task_ids.shape, device=task_ids.device) < self.dropout
            task_ids = torch.where(drop_mask, torch.tensor(3, device=task_ids.device), task_ids)
        task = self.task_embed(task_ids).squeeze(1)
        # Apply FiLM on the average of audio encoding
        alpha = self.alpha_proj(task)
        beta = self.beta_proj(task)
        x = alpha * x.squeeze(-1) + beta
        # Output logits
        return torch.tanh(self.head(x).squeeze(-1))


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
    gamma: float = 100.0  # scaling factor for LARoPE
    pad_id: int = 0
    eot_id: int = 2
    transcribe_id: int = 48002
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
        cos, sin = precompute_rotary_emb(config.n_text_ctx, self.head_dim, device=self.device)
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
        self.len_predictor = LengthPredictor(
            config.transcribe_id,
            config.n_embed,
            config.dropout,
            config.bias
        )  # sequence length predictor specific to LARoPE

        mask = torch.empty(config.n_text_ctx, config.n_text_ctx).fill_(-float('inf')).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
        self.prefix = None
        self.init_weights()
        if verbose:
            print_banner()

    def get_token_embedding(self, token_ids: Tensor) -> Tensor:
        return self.tok_embed(token_ids) * math.sqrt(self.config.n_embed)

    def encode(self, mels: Tensor) -> Tensor:
        return self.encoder.encode(mels)

    def logits(
            self,
            aud_enc: Tensor,
            tok_embed: Tensor,
            aud_mask: Optional[Tensor] = None,
            tok_mask: Optional[Tensor] = None,
            pred_seq_len: Optional[float]=0.0,
            kv_cache: Optional[tuple[Optional[KVCache], Optional[dict]]]=(None, None)
            ) -> Tensor:
        """
        tok_mask: True means valid token, False means pad token
        aud_mask: True means valid frame, False means pad frame
        pred_seq_len: predicted number of tokens for LARoPE
        """
        T0 = 0 if kv_cache[0] is None else kv_cache[0].get_pos()

        # decoding layer
        b, L, _ = tok_embed.size()
        T1 = T0 + L
        # assert T1 <= self.config.n_text_ctx, "input_ids is too long"

        attn_mask = None
        q = self.pos_basis.expand(b, L, -1)
        ctx = tok_embed
        if L > 1:  # always True during training. During inference, True for the initial tokens
            attn_mask = self.mask[T0:T1, T0:T1]
            ctx = ctx + F.pad(q[:, T0:T1-1], (0, 0, 1, 0, 0, 0))
        elif T0 > 0:  # During inference, from the second token onward
            ctx = ctx + q
        ctx = self.dropout(ctx)
        ctx = norm(ctx)

        # T1+1 because Q is one-phase faster than K
        sa_cos_sin = self.cos[:, :, T0+1:T1+1], self.sin[:, :, T0+1:T1+1], self.cos[:, :, T0:T1], self.sin[:, :, T0:T1]

        if pred_seq_len == 0.0:  # always True during training
            actual_seq_len = torch.sum(tok_mask, dim=-1, dtype=torch.float32) # (b, )
            actual_aud_len = torch.sum(aud_mask, dim=-1, dtype=torch.float32) # (b, )
            # convert boolean masks to float masks
            tok_mask = torch.where(tok_mask, 0.0, -float('inf'))
            tok_mask = tok_mask.view(b, 1, 1, -1).expand(-1, self.n_head, L, -1)
            aud_mask = torch.where(aud_mask, 0.0, -float('inf'))
            aud_mask = aud_mask.view(b, 1, 1, -1).expand(-1, self.n_head, L, -1)
            seq_len = L
        else:
            actual_seq_len = None
            actual_aud_len = None
            seq_len = pred_seq_len if T1 < pred_seq_len else T1

        # LARoPE
        q_cos, q_sin = self.compute_cos_sin(seq_len, self.device, actual_seq_len)
        k_cos, k_sin = self.compute_cos_sin(aud_enc.shape[1], self.device, actual_aud_len)
        ca_cos_sin = q_cos[:, :, T0:T1], q_sin[:, :, T0:T1], k_cos, k_sin

        res = self.decoder(q, ctx, aud_enc, sa_cos_sin, attn_mask, ca_cos_sin, tok_mask, aud_mask, kv_cache)
        return self.lm_head(norm(res))  # (b, L, pad_vocab_size)

    def forward(self, mels: Tensor, input_ids: Tensor, target_ids: Optional[Tensor] = None) -> CausalLMOutput:
        """
        mels: log-mel spectrum from audio array
        input_ids: input token ids for decoder (inclu. sot sequence)
        target_ids: target token ids to compute cross_entropy loss
        """

        # NOTE: in Whisper Audio Encoder, the audio encoding is already normalized by ln_post()
        aud_enc, aud_mask = self.encoder.forward(mels)
        tok_mask = input_ids != self.config.pad_id
        tok_embed = self.get_token_embedding(input_ids)
        if target_ids is None:
            logits = self.logits(aud_enc, tok_embed, aud_mask, tok_mask)  # (b, L, pad_vocab_size)
            # do not compute loss
            return CausalLMOutput(logits=logits)
        else:
            # predict sequence length
            # len_logits is bounded on [-1, 1] via tanh
            len_logits = self.len_predictor(aud_enc.detach(), input_ids[:, [2]], aud_mask)
            # normalize target length to [-1, 1], so that Huber loss has comparable scale to cross_entropy and lp_factor can be set to 1.0
            norm_tgt_len = torch.sum(tok_mask, dim=-1, dtype=torch.float32)
            norm_tgt_len = (norm_tgt_len / (0.5 * self.config.n_text_ctx)) - 1
            loss = F.smooth_l1_loss(len_logits.float(), norm_tgt_len.float(), beta=1.0)

            logits = self.logits(aud_enc, tok_embed, aud_mask, tok_mask)  # (b, L, pad_vocab_size)
            loss += F.cross_entropy(logits.flatten(end_dim=1).float(), target_ids.flatten(), ignore_index=-100, reduction='mean')

            return CausalLMOutput(logits=logits, loss=loss)

    def compute_cos_sin(
            self,
            seq_len: float,
            device: torch.device,
            actual_seq_len: Optional[Tensor] = None
        ) -> tuple[Tensor, Tensor]:
        # seq_len may include pad tokens, actual_seq_len does not
        pos = torch.arange(int(seq_len), device=device, dtype=torch.float32)
        if actual_seq_len is None:
            # cos_sin for a single sequence
            pos = self.gamma * pos / seq_len
            freqs = torch.outer(pos, self.inv_freq).unsqueeze(0).unsqueeze(0)
        else:
            # cos_sin for batched sequences
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
    def detect_language(self, mel: Tensor, tokenizer, kv_cache):
        """
        Predicting the language of audio
        NOTE: I could have called `forward` but there is a careful attention required for cos, sin
        """
        mels = mel.unsqueeze(0)
        aud_enc, aud_mask = self.encoder.forward(mels)

        idx = torch.as_tensor([[tokenizer.sot]], dtype=torch.long, device=self.device)
        tok_embed = self.get_token_embedding(idx)
        logits = self.logits(aud_enc, tok_embed, aud_mask=aud_mask, pred_seq_len=1.0)

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
    def decode(self, mel: Tensor, max_tokens: int, temperature=1.0, top_k=None, seed=168):
        """
        mel: (n_mels,)
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

        mels = mel.unsqueeze(0)
        aud_enc, aud_mask = self.encoder.forward(mels)
        idx = torch.as_tensor([self.prefix], dtype=torch.long, device=mels.device)
        tok_embed = self.get_token_embedding(idx)
        n = idx.shape[1]

        # predict seq_len
        len_logits = self.len_predictor(aud_enc, idx[:, [2]].clone(), aud_mask)
        pred_seq_len = 0.5 * self.config.n_text_ctx * (len_logits + 1)

        next_embed = tok_embed
        for i in range(n, max_tokens):
            logits = self.logits(aud_enc, next_embed, aud_mask=aud_mask, pred_seq_len=pred_seq_len, kv_cache=kv_cache)
            logits[:, -1, self.vocab_size:] = -float('inf')     # suppress the pad tokens

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
            if id_next.item() == self.config.eot_id:
                break
            next_embed = self.get_token_embedding(id_next)
        return idx

    @torch.no_grad()
    def init_weights(self):
        nn.init.normal_(self.encoder.conv1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.encoder.conv2.weight, mean=0.0, std=0.02)

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

        nn.init.kaiming_normal_(self.len_predictor.conv3.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.len_predictor.task_embed.weight, mean=0.0, std=0.8)
        nn.init.normal_(self.len_predictor.head.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.len_predictor.beta_proj.weight)
        nn.init.zeros_(self.len_predictor.beta_proj.bias)
        nn.init.zeros_(self.len_predictor.alpha_proj.weight)
        # nn.init.ones_(self.len_predictor.alpha_proj.bias)

        # Embedding
        nn.init.normal_(self.tok_embed.weight, mean=0.0, std=0.001)
        # tie token embedding
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embed.weight

        nn.init.trunc_normal_(self.pos_basis, std=1.0)

        # Rotary embeddings
        cos, sin = precompute_rotary_emb(self.config.n_text_ctx, self.head_dim, device=self.device)
        self.cos, self.sin = cos, sin
        self.inv_freq = 1.0 / (
            1e4 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )

        self.mask = torch.empty(self.config.n_text_ctx, self.config.n_text_ctx).fill_(-float('inf')).triu_(1)

        if self.tok_embed.weight.device.type == 'cuda':
            self.tok_embed.weight = self.tok_embed.weight.to(torch.bfloat16)
            self.pos_basis = self.pos_basis.to(torch.bfloat16)
