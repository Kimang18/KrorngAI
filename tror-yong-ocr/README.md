# TrorYong OCR Model

`TrorYongOCR`, is an Optical Character Recognition Model implemented by KrorngAI.

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

You can easily install `tror-yong-ocr` using `pip` command as the following:

```bash
pip install tror-yong-ocr
```

# Usage

## Loading tokenizer

`TrorYongOCR` is a small optical character recognition model that you can train from scratch.
With this goal, you can use your own tokenizer to pair with `TrorYongOCR`.
Just make sure that the __tokenizer used for training__ and the __tokenizer used for inference__ is __the same__.

Your tokenizer must contain begin of sequence (`bos`), end of sequence (`eos`) and padding (`pad`) tokens.
`bos` token id and `eos` token id are used in decoding function.
`pad` token id is used during training.

I also provide a tokenizer that supports Khmer and English.

```python
from tror_yong_ocr import get_tokenizer

tokenizer = get_tokenizer(charset=None)
print(len(tokenizer)) # you should receive 185
text = 'Amazon បង្កើនការវិនិយោគជិត១'
print(tokenizer.decode(tokenizer.encode(data[0]['text'], add_special_tokens=True), ignore_special_tokens=False))
# this should print <s>Amazon បង្កើនការវិនិយោគជិត១</s>
```

When preparing a dataset to train `TrorYongOCR`, you just need to transform the text into token ids using the tokenizer
```python
sentence = 'Cambodia needs peace.'
token_ids = tokenizer.encode(sentence, add_special_tokens=True)
```

__NOTE:__ I want to highlight that my tokenizer works at character level.

## Loading TrorYongOCR model

Inspired by [`PARSeq`](https://github.com/baudm/parseq/tree/main) and [`DTrOCR`](https://github.com/arvindrajan92/DTrOCR), I design `TrorYongOCR` as the following: given `n_layer` transformer layers
- `n_layer-1` are encoding layers for encoding a given image
- the final layer is a decoding layer without cross-attention mechanism
- for the decoding layer,
  - the __latent state__ of an image (the output of encoding layers) is concatenated with the __input character embedding__ (token embedding including `bos` token plus position embedding) to create __context vector__, _i.e._ __key and value vectors__ (think of it like a prompt prefill)
  - and the __input character embedding__ (token embedding plus position embedding) is used as __query vector__.

The architecture of TrorYongOCR can be found in Figure 1 below.

<figure>
  <div style="text-align: center;"><a name='slotMachine' ><img src="https://raw.githubusercontent.com/Kimang18/KrorngAI/refs/heads/main/tror-yong-ocr/TrorYongOCR.drawio.png" width="500" /></a></div>
  <figcaption> Figure 1: TrorYongOCR architecture overview. The input image is transformed into patch embedding. Image embedding is obtained by additioning patch embedding and position embedding. The image embedding is passed through L-1 encoder blocks to generate image encoding (latent state). The image encoding is concatenated with character embedding (i.e. token embedding plus position embedding) before undergoing causal self-attention mechanism in the single decoder block to generate next token.</figcaption>
</figure>

New technologies in Attention mechanism such as Rotary Positional Embedding (RoPE), and Sigmoid Linear Unit (SiLU) and Gated Linear Unit (GLU) in MLP of Transformer block are implemented in TrorYongOCR.

### Compared to PARSeq

For `PARSeq` model which is an encoder-decoder architecture, text decoder uses position embedding as __query vector__, character embedding (token embedding plus position embedding) as __context vector__, and the __latent state__ from image encoder as __memory__ for the cross-attention mechanism (see Figure 3 of their paper).

### Compared to DTrOCR

For DTrOCR which is a decoder-only architecture, the image embedding (patch embedding plus position embedding) is concatenated with input character embedding (a `[SEP]` token is added at the beginning of input character embedding to indicate sequence separation. `[SEP]` token is equivalent to `bos` token in `TrorYongOCR`), and causal self-attention mechanism is applied to the concatenation from layer to layer to generate text autoregressively (see Figure 2 of their paper).

```python
from tror_yong_ocr import TrorYongOCR, TrorYongConfig
from tror_yong_ocr import get_tokenizer

tokenizer = get_tokenizer()

config = TrorYongConfig(
    img_size=(32, 128),
    patch_size=(4, 8),
    n_channel=3,
    vocab_size=len(tokenizer),
    block_size=192,
    n_layer=4,
    n_head=6,
    n_embed=384,
    dropout=0.1,
    bias=True,
)
model = TrorYongOCR(config, tokenizer)
```

## Train TrorYongOCR

You can check out the notebook below to train your own Small OCR Model.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kimang18/rag-demo-with-mlx/blob/main/https://colab.research.google.com/github/Kimang18/rag-demo-with-mlx/blob/main/FinetuneTrorYongOCR.ipynb)

I also have a video about training TrorYongOCR below

[![Watch the video](https://i9.ytimg.com/vi/3W8P0mByFBY/mqdefault.jpg?v=6995e008&sqp=CMSg4cwG&rs=AOn4CLBmVopfxv_RJGQPJE5qU9eQP4_XBw)](https://youtu.be/3W8P0mByFBY)

## Inference

I also provide `decode` function to decode image in `TrorYongOCR` class.
Note that it can process only one image at a time.
```python
from tror_yong_ocr import TrorYongOCR, TrorYongConfig
from tror_yong_ocr import get_tokenizer


tokenizer = get_tokenizer()

config = TrorYongConfig(
    img_size=(32, 128),
    patch_size=(4, 8),
    n_channel=3,
    vocab_size=len(tokenizer), # exclude pad and unk tokens
    block_size=192,
    n_layer=4,
    n_head=6,
    n_embed=384,
    dropout=0.1,
    bias=True,
)
model = TrorYongOCR(config, tokenizer)
model.load_state_dict(torch.load('path/to/your/weights.pt', map_location='cpu'))

pred = model.decode(batch['img_tensor'][0], max_tokens=192, temperature=0.001, top_k=None)
print(tokenizer.decode(pred[0].tolist(), ignore_special_tokens=True))
```

## TODO:
- [X] implement model with KV cache `TrorYongOCR`
- [X] notebook colab for training `TrorYongOCR`
- [ ] benchmarking
