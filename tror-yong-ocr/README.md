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

When preparing a dataset to train `TrorYongOCRModel`, you just need to transform the text into token ids using the tokenizer
```python
sentence = 'Cambodia needs peace.'
token_ids = tokenizer.encode(sentence, add_special_tokens=True)
```

__NOTE:__ I want to highlight that my tokenizer works at character level.

## Loading TrorYongOCRModel

Get started with the code below

```python
import torch
from torchvision.transforms import v2 as transforms
from PIL import Image # pip install pillow
from tror_yong_ocr import get_tokenizer, TrorYongOCRModel

img = Image.open("your/file/image").convert('RGB')

processor = transforms.Compose(
    [
        transforms.Resize((32, 128)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

img_tensor = processor(img)

tokenizer = get_tokenizer()
model = TrorYongOCRModel.from_pretrained('KrorngAI/TrorYongOCR')
model.eval()

# suppose that you have an image array in numpy
pred_ids = model.decode(img_tensor, 192, temperature=0.01, top_k=25)
print(tokenizer.decode(pred_ids[0].tolist(), ignore_special_tokens=True))
```

TrorYongOCR is designed as the following: given $L$ transformer blocks

- $L-1$ are encoding blocks that encode a given image
- the last block is a single decoding block without cross-attention mechanism
- each transformer is implemented with exclusive self-attention [@zhai2026exclusive] style and SwiGLU in MLP

For the single decoding block,

- the latent state of an image (the output of encoding blocks) is concatenated with the input character embedding (token embedding including bos token) to create context vector, _i.e._ key and value vectors (think of it like a prefill prompt)

The architecture of TrorYongOCR can be found in Figure 1 below.

<figure>
  <div style="text-align: center;"><a name='architecture' ><img src="https://huggingface.co/KrorngAI/TrorYongOCR/resolve/main/figures/architecture.png" width="500" /></a></div>
  <figcaption> Figure 1: TrorYongOCR architecture overview. The input image is transformed into patch embedding. Image embedding is obtained by additioning patch embedding and position embedding. The image embedding is passed through L-1 encoder blocks to generate image encoding (latent state). The image encoding is concatenated with character embedding (i.e. token embedding plus position embedding) before undergoing causal self-attention mechanism in the single decoder block to generate next token.</figcaption>
</figure>


### Compared to PARSeq

For `PARSeq` model which is an encoder-decoder architecture, text decoder uses position embedding as __query vector__, character embedding (token embedding plus position embedding) as __context vector__, and the __latent state__ from image encoder as __memory__ for the cross-attention mechanism (see Figure 3 of their paper).

### Compared to DTrOCR

For DTrOCR which is a decoder-only architecture, the image embedding (patch embedding plus position embedding) is concatenated with input character embedding (a `[SEP]` token is added at the beginning of input character embedding to indicate sequence separation. `[SEP]` token is equivalent to `bos` token in `TrorYongOCR`), and causal self-attention mechanism is applied to the concatenation from layer to layer to generate text autoregressively (see Figure 2 of their paper).


## Fine-tuning TrorYongOCR

You can check out the notebook below to train your own Small OCR Model.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kimang18/SourceCode-KrorngAI-YT/blob/main/FinetuneTrorYongOCR.ipynb)

I also have a video about training TrorYongOCR below

<a href="http://www.youtube.com/watch?feature=player_embedded&v=3W8P0mByFBY" target="_blank">
 <img src="http://img.youtube.com/vi/3W8P0mByFBY/mqdefault.jpg" alt="Watch the video" height="240" border="1" />
</a>


## TODO:
- [X] implement model with KV cache `TrorYongOCRModel`
- [X] notebook colab for fine-tuning `TrorYongOCRModel`
- [ ] benchmarking
