# TrorYong Language Model

`TrorYongGPT`, Small Language Model with Rotary Positional Embeddings, is a re-implementation of GPT2 of OpenAI.

`TrorYong` (ត្រយ៉ង) is Khmer word for giant ibis, the bird that symbolises __Cambodia__.

## Support My Work

While this work comes truly from the heart, each project represents a significant investment of time -- from deep-dive research and code preparation to the final narrative and editing process.
I am incredibly passionate about sharing this knowledge, but maintaining this level of quality is a major undertaking.
If you find my work helpful and are in a position to do so, please consider supporting my work with a donation.
You can click <a href="https://pay.ababank.com/oRF8/8yp6hy53">here</a> to donate or scan the QR code below.
Your generosity acts as a huge encouragement and helps ensure that I can continue creating in-depth, valuable content for you.

<figure>
  <div style="text-align: center;"><a name='slotMachine' ><img src="https://kimang18.github.io/assets/fig/aba_qr_kimang.JPG" width="500" /></a></div>
  <figcaption> Using Cambodian bank account, you can donate by scanning my ABA QR code here. (or click <a href="https://pay.ababank.com/oRF8/8yp6hy53">here</a>. Make sure that receiver's name is 'Khun Kim Ang'.) </figcaption>
</figure>

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

You can check out the notebook below to train your own Small Language Model.
I would like to highlight that you can __use your own tokenizer__ to train `TrorYongGPT` and I recommend to do so __for Khmer language__.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kimang18/rag-demo-with-mlx/blob/main/TrorYong_Small_Language_Model_from_Scratch.ipynb)

I also have a video about training TrorYongGPT below

[![Watch the video](https://i9.ytimg.com/vi/e7wEAVeIo0Y/mqdefault_custom_1.jpg?v=69569cdc&sqp=CNjorssG&rs=AOn4CLC7rnplNZJUDmBvKJrGz3tKW_W_Yw)](https://youtu.be/e7wEAVeIo0Y)

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
- [X] notebook colab for training `TrorYongGPT`
- [ ] benchmarking
