# Author: KrorngAI Org.
# Date: December, 2025


from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor


def norm(x):
    return F.rms_norm(x, (x.size(-1), ))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last time into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)  # re-assemble
    out = out.to(x.dtype)  # ensure input/output dtypes match
    return out


class LinearWrapper(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype)
        )


class LayerNormWrapper(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Conv1dWrapper(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


class Conv1D(nn.Module):
    """
    Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py#L97

    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nx (`int`): The number of input features.
        nf (`int`): The number of output features.
    """

    def __init__(self, nx, nf, bias=True):
        super().__init__()
        self.nx = nx
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.hasBias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def __repr__(self) -> str:
        return "Conv1D(nf={nf}, nx={nx})".format(**self.__dict__)

    def forward(self, x: Tensor) -> Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        if self.hasBias:
            x = torch.addmm(self.bias.to(x.dtype), x.view(-1, x.size(-1)), self.weight.to(x.dtype))
        else:
            x = torch.mm(x.view(-1, x.size(-1)), self.weight.to(x.dtype))
        x = x.view(size_out)
        return x


class KVCache:
    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        self.kv_shape = (num_layers, 2, batch_size,
                         num_heads, seq_len, head_dim)
        self.kv_cache = None
        self.pos = 0

    def reset(self):
        self.pos = 0

    def get_pos(self):
        return self.pos

    def rearrange(self, pos):
        self.pos = pos
        # self.kv_cache[:, :, :, :, :pos, :] =

    def prefill(self, other):
        assert self.kv_cache is None, "Cannot prefill a non-empty KV cache"
        assert other.kv_cache is not None, "Cannot prefill with None KV cache"

        # Extract dimensions
        self_layers, self_kv, self_batch, self_heads, self_seq, self_head_dim = self.kv_shape
        other_layers, other_kv, other_batch, other_heads, other_seq, other_head_dim = other.kv_shape

        # validate dimensions
        assert self_layers == other_layers, f"Layer count mismatch: {self_layers} != {other_layers}"
        assert self_kv == other_kv, f"K/V dimension mismatch: {self_kv} != {other_kv}"
        assert self_heads == other_heads, f"Head count mismatch: {self_heads} != {other_heads}"
        assert self_head_dim == other_head_dim, f"Head dim mismatch: {self_head_dim} != {other_head_dim}"

        # batch size can be expanded (other can be 1, self can be larger)
        assert self_batch == other_batch or other_batch == 1, f"Batch size mismatch: {self_batch} vs {other_batch}"

        # sequence length: self must be longer than other
        assert self_seq >= other_seq, f"Sequence length mismatch: {self_seq} < {other_seq}"

        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        self.kv_cache[:, :, :, :, :other.pos, :] = other.kv_cache
        self.pos = other.pos

    def insert_kv(self, layer_idx, k, v):
        # Lazy initialize the cache here because we need to know the dtype/device
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)

        B, H, T_add, D = k.size()
        t0, t1 = self.pos, self.pos + T_add
        # Dynamically grow the cache if needed
        if t1 > self.kv_cache.size(4):
            t_needed = t1 + 1024  # as much as we need plus buffer of 1024
            # then round up to the nearest multiple of 1024
            t_needed = (t_needed + 1023) & ~1023
            additional_shape = list(self.kv_cache.shape)
            additional_shape[4] = t_needed - self.kv_cache.size(4)
            additional_cache = torch.empty(additional_shape, dtype=k.dtype, device=k.device)
            self.kv_cache = torch.cat([self.kv_cache, additional_cache], dim=4).contiguous()
            self.kv_shape = self.kv_cache.shape

        # Insert k, v into the cache
        self.kv_cache[layer_idx, 0, :, :, t0:t1, :] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1, :] = v
        # Return the full cached keys/values up to current position (as a view)
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1, :]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1, :]

        # Increment pos after the last layer of the Transformer processes
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        return key_view, value_view


class CausalSelfAttention(nn.Module):
    def __init__(self, layer_idx: int, n_state: int, n_head: int, n_kv_head: int):
        super().__init__()
        assert n_state % n_head == 0
        assert n_kv_head <= n_head and n_head % n_kv_head == 0

        self.layer_idx = layer_idx
        self.n_state = n_state
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_state // n_head
        self.query = Conv1D(self.n_state, self.n_head * self.head_dim)
        self.key = Conv1D(self.n_state, self.n_kv_head * self.head_dim, bias=False)
        self.value = Conv1D(self.n_state, self.n_kv_head * self.head_dim)
        self.out = Conv1D(self.n_state, self.n_state)

    def forward(
        self,
        x: Tensor,
        cos_sin=None,
        kv_cache: Optional[KVCache] = None,
    ):
        B, T, C = x.size()

        q = self.query(x).view(B, T, self.n_head, self.head_dim)
        k = self.key(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.value(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)

        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2)
        Tk = k.size(2)

        # Group Query Attention (GQA): duplicate key/value heads to match query heads if desired
        enable_gqa = self.n_head != self.n_kv_head
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the key/values in the cache
            a = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/value (i.e. full prefix)
            attn_mask = torch.zeros(
                (Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len = Tk - Tq
            attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Re-assemble the heads side by side and project back to residual stream
        out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.out(out)
