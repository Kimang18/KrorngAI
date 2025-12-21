# Author: KrorngAI Org.
# Date: December, 2025


from typing import TYPE_CHECKING
import torch
import torch.nn.functional as F

from .nn_utils import KVCache

if TYPE_CHECKING:
    from .slm import SLM


@torch.inference_mode()
def sample_next_token_id(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size)
    Return (B, 1)
    """
    assert temperature >= 0.0, 'temperature must be non-negative'

    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('inf')
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=rng)


@torch.inference_mode()
def generate(
    model: 'SLM',
    tokenizer,
    prompt,
    max_tokens=128,
    temperature=1.0,
    top_k=None,
    stream=False,
    seed=168,
):
    """
    Naif implementation for streaming inference
    initial_tokens: list of tokens
    """
    assert isinstance(prompt, str), 'prompt must be string'
    initial_tokens = tokenizer.encode(prompt)
    kv_model_kwargs = {
        "num_heads": model.config.n_kv_head,
        "head_dim": model.config.n_state // model.config.n_head,
        "num_layers": model.config.n_layer
    }
    kv_cache = KVCache(
        batch_size=1,
        seq_len=len(initial_tokens),
        **kv_model_kwargs
    )
    token_ids = list(initial_tokens)
    if stream:
        def stream_generate():
            for token in model.stream_generate(initial_tokens, max_tokens, temperature, top_k, kv_cache, seed):
                token_ids.append(token)
                yield tokenizer.decode(token_ids)
        return stream_generate()
    else:
        for token in model.stream_generate(initial_tokens, max_tokens, temperature, top_k, kv_cache, seed):
            token_ids.append(token)
        return tokenizer.decode(token_ids)
