"""
Copied from https://github.com/openai/whisper/blob/main/whisper/model.py
and remove ModelDimensions, TextDecoder and Whisper classes
"""
from typing import Optional, Iterable, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .nn_utils import (
    LinearWrapper,
    LayerNormWrapper,
    Conv1dWrapper
)


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = LinearWrapper(n_embed, n_embed)
        self.key = LinearWrapper(n_embed, n_embed, bias=False)
        self.value = LinearWrapper(n_embed, n_embed)
        self.out = LinearWrapper(n_embed, n_embed)

    def forward(
            self,
            x: Tensor,
            key_mask: Optional[Tensor] = None,
        ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        # q, k = norm(q), norm(k)
        attn_mask = None
        if key_mask is not None:
            b, L = key_mask.shape
            attn_mask = key_mask.view(b, 1, 1, L).expand(-1, self.n_head, L, -1)

        qkv = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        qkv = qkv.permute(0, 2, 1, 3).flatten(start_dim=2)

        return self.out(qkv), None


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_embed: int, n_head: int):
        super().__init__()
        self.attn = MultiHeadAttention(n_embed, n_head)
        self.attn_ln = LayerNormWrapper(n_embed)
        n_mlp = n_embed * 4
        self.mlp = nn.Sequential(
                LinearWrapper(n_embed, n_mlp), nn.GELU(), LinearWrapper(n_mlp, n_embed)
        )
        self.mlp_ln = LayerNormWrapper(n_embed)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        x = x + self.attn(self.attn_ln(x), mask)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(self, n_mels:int, n_ctx: int, n_embed: int, n_head: int, n_layer: int):
        super().__init__()
        self.conv1 = Conv1dWrapper(n_mels, n_embed, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv1dWrapper(n_embed, n_embed, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_embed), persistent=False)

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_embed, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNormWrapper(n_embed)

    def encode(self, x: Tensor):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1).contiguous()

        # assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding[:x.shape[1]]).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x

    def forward(self, x: Tensor):
        mask = torch.zeros((x.shape[0], 1, x.shape[2]), device=x.device, dtype=torch.float32)
        mask.mask_fill_(torch.sum(x - x[:, :, [-1]], dim=1, keepdim=True) > 1e-6, 1.0)
        x = mask * x
        x = F.gelu(mask * self.conv1(x))  # suppress bias before act_fn
        # reduce mask using the same setting of conv2
        mask = F.max_pool1d(mask, kernel_size=3, stride=2, padding=1)
        x = F.gelu(mask * self.conv2(x))  # suppress bias before act_fn
        x = x.permute(0, 2, 1).contiguous()

        # assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding[:x.shape[1]]).to(x.dtype)

        mask = mask.transpose(1, 2).contiguous().to(torch.bool)  # (b, L/2, 1)
        x = mask * x  # suppress pos_embed
        for block in self.blocks:
            x = block(x, mask.squeeze(-1))
            x = mask * x  # suppress bias

        x = self.ln_post(x)
        return x, mask.squeeze(-1)

    def load_weights_from_whisper(self, state_dict: Dict[str, Tensor]):
        new_state_dict = {}
        for old_layer, tensor in state_dict.items():
            if "positional_embedding" in old_layer:
                continue
            new_state_dict[old_layer] = tensor
        self.load_state_dict(new_state_dict, strict=False)
