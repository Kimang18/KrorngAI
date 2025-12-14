# Author: KrorngAI Org.
# Date: December, 2025


from typing import Iterable, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from whisper.model import (
        ModelDimensions,
        MultiHeadAttention,
        Whisper
    )
    from whisper.decoding import detect_language as detect_language_function
except (ImportError, ModuleNotFoundError):
    print("You need to install openai-whisper package: pip install git+https://github.com/openai/whisper.git")
    raise

from .decoding import decode as decode_function
from .nn_utils import (
    norm,
    LinearWrapper,
    LayerNormWrapper,
    Conv1D,
    KVCache,
    CausalSelfAttention
)
from .common import print_banner


@dataclass
class NeoModelDimensions(ModelDimensions):
    n_text_kv_head: int


class MLP(nn.Module):
    def __init__(self, n_state: int):
        super().__init__()
        self.c_fc = Conv1D(n_state, 4 * n_state)
        self.c_proj = Conv1D(4 * n_state, n_state)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class ResidualAttentionBlock(nn.Module):
    """
    Attention block for text decoder
    Text decoder has cross attention to align audio with text
    Since the n_audio_ctx=1500 != n_text_ctx, we need additional modification to RoPE
    To avoid complication, I fallback to original MultiHeadAttention of whisper package for cross attention
    """

    def __init__(self, layer_idx: int, n_state: int, n_head: int, n_kv_head: int):
        super().__init__()

        self.attn = CausalSelfAttention(layer_idx, n_state, n_head, n_kv_head)
        self.cross_attn = MultiHeadAttention(n_state, n_head)
        self.cross_attn_ln = LayerNormWrapper(n_state)
        self.mlp = MLP(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        cos_sin=None,
        kv_cache: Optional[dict] = None,
    ):
        attn_kv_cache: KVCache = kv_cache['neo'] if kv_cache else None
        x = x + self.attn(norm(x), cos_sin=cos_sin, kv_cache=attn_kv_cache)
        x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(norm(x))
        return x


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.n_state = n_state
        self.n_head = n_head

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(layer_idx, n_state, n_head, n_head)
                for layer_idx in range(n_layer)
            ]
        )
        self.lm_head = LinearWrapper(n_state, n_vocab, bias=False)

        self.rotary_seq_len = n_ctx * 10
        head_dim = n_state // n_head
        cos, sin = self._precompute_rotary_emb(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        nn.init.zeros_(self.lm_head.weight)

        for block in self.blocks:
            nn.init.zeros_(block.mlp.c_proj.weight)
            nn.init.zeros_(block.attn.out.weight)
            nn.init.zeros_(block.cross_attn.out.weight)

        head_dim = self.n_state // self.n_head
        cos, sin = self._precompute_rotary_emb(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.token_embedding.weight.device.type == "cuda":
            self.token_embedding.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * \
                min(1.0, math.sqrt(fan_out / fan_in))
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _precompute_rotary_emb(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.token_embedding.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_text_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """

        n_batch, n_ctx = x.size()

        assert n_ctx <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {n_ctx} > {self.cos.size(1)}"
        assert x.device == self.cos.device, f"Rotary embeddings and x are on different devices: {x.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"

        T0 = kv_cache['neo'].get_pos() if kv_cache else 0
        cos_sin = self.cos[:, T0:T0+n_ctx], self.sin[:, T0:T0+n_ctx]

        x = self.token_embedding(x)
        x = norm(x)
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, cos_sin=cos_sin, kv_cache=kv_cache)
        x = norm(x)

        logits = self.lm_head(x)
        softcap = 15
        logits = softcap * torch.tanh(logits / softcap)
        logits = logits.float()

        return logits


class NeoWhisper(Whisper):
    def __init__(self, dims: NeoModelDimensions, verbose=False):
        super().__init__(dims)
        del self.decoder
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        self.decoder.init_weights()
        if verbose:
            print_banner()

    @property
    def num_languages(self):
        return 99

    @property
    def is_multilingual(self):
        return True

    @torch.inference_mode()
    def generate(self, mels, sot_tokens, max_tokens=100, temperature=1.0, top_k=None, seed=42):
        # naif implementation for speech decoding
        assert mels.shape[0] == 1, "Does not support batch audio yet"
        assert isinstance(sot_tokens, list)
        assert len(sot_tokens) == 4, "Only support no_timestamps and cannot infer language given speech"
        device = self.device
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([sot_tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(mels, ids)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size) only consider the last prediction
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token

    detect_language = detect_language_function
    decode = decode_function
