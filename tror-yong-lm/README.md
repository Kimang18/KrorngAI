# TrorYong Language Model

`TrorYongGPT`, Small Language Model with Rotary Positional Embeddings, is a re-implementation of GPT2 by OpenAI.

# Installation

You can easily install `tror-yong-lm` using `pip` command as the following:

```bash
pip install tror-yong-lm
```

# Usage

## Loading tokenizer

`TrorYongGPT` is a small language model that you can train from scratch.
With this goal, you can use your own tokenizer to pair with `TrorYongGPT`.
Just make sure that the __tokenizer used for training__ and the __tokenizer used for inference__ is __the same__.

For example, we can use a tokenizer from `tiktoken` of OpenAI as the following:

```python
import tiktoken

tokenizer = tiktoken.get_encoding('gpt2')
print(tokenizer.n_vocab)
```

When preparing a dataset to train `TrorYongGPT`, you just need to transform the text into token ids using the tokenizer
```python
sentence = 'Cambodia needs peace.'
token_ids = tokenizer.encode(sentence)
```

## Loading TrorYongGPT model

```python
import torch
from tror_yong_lm import TrorYongGPT, TrorYongConfig
config = TrorYongConfig(
    n_vocab=tokenizer.n_vocab, # use the tokenizer's vocab size
    n_ctx=64,
    n_layer=4,
    n_head=6,
    n_kv_head=6,
    n_state=384,
)
model = TrorYongGPT(config)
token_ids = [100, 103, 104] # suppose we have this tokens
torch_arr = torch.tensor([token_ids], dtype=torch.long) # (B, T) = (1, 3)
logits = model(torch_arr) # (B, T, n_vocab) = (1, 3, n_vocab)
```

## Train TrorYongGPT

(To be done)

## Inference

We also provide `generate` function to do text completion.
```python
import tiktoken
import torch
from tror_yong_lm import TrorYongConfig, TrorYongGPT, generate

tokenizer = tiktoken.get_encoding('tokenizer/used/to/train/your/model')

config = TrorYongConfig(
    n_vocab=tokenizer.n_vocab,
    ...
)
model = TrorYongGPT(config)
best_model_params_path = "path/to/your/weights.pt"
model.load_state_dict(torch.load(best_model_params_path))

sentence = 'Once upon a time,'
# streaming
for text in generate(model, tokenizer, sentence, stream=True):
    print(text, end='', flush=True)

# or no stream
result_text = generate(model, tokenizer, sentence)
print(result_text)
```

## TODO:
- [X] implement model with KV cache `TrorYongGPT`
- [ ] notebook colab for training `TrorYongGPT`
- [ ] benchmarking
