# NeoWhisper
Improve `Whisper` of OpenAI by integrating Rotary Positional Embeddings (RoPE) and adding more options for tokenizers available in pypi package `tiktoken`.

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
```bash
pip install neo-whisper
```

## Requirement
```bash
pip install git+https://github.com/openai/whisper.git
```

# Usage

## Loading tokenizer
```python
from neo_whisper import get_tokenizer
tokenizer_name = 'cl100k_base'
tokenizer = get_tokenizer(multilingual=True, language='km', task='transcribe', encoder_name=tokenizer_name)
print(tokenizer.eot)
```

## Loading NeoWhisper model
```python
from neo_whisper import NeoWhisper, NeoModelDimensions
dims = NeoModelDimensions(
    n_vocab=tokenizer.encoding.n_vocab, # use the tokenizer's vocab size
    n_mels=80,       # or whatever context size you're training with
    n_audio_ctx=1500,
    n_audio_state=384,
    n_audio_head=6,
    n_audio_layer=4,
    n_text_ctx=448,
    n_text_state=384,
    n_text_head=6,
    n_text_kv_head=6,
    n_text_layer=4
)
model = NeoWhisper(dims)
```

This `model` works like the original model of OpenAI whisper (`NeoWhisper` inherits from `Whisper` of openai-whisper. TextDecoder of `NeoWhisper` is different from the one of `Whisper` in the sense that `RoPE` is integrated in `NeoWhisper`.).

## Loading Original Whisper model
It is possible to load the model implemented in openai-whisper but with new tokenizer (such as `cl100k_base`).
```python
from neo_whisper import Whisper, ModelDimensions
dims = ModelDimensions(
    n_vocab=tokenizer.encoding.n_vocab, # use the tokenizer's vocab size
    n_mels=80,       # or whatever context size you're training with
    n_audio_ctx=1500,
    n_audio_state=384,
    n_audio_head=6,
    n_audio_layer=4,
    n_text_ctx=448,
    n_text_state=384,
    n_text_head=6,
    n_text_layer=4
)
model = Whisper(dims)
```
__NOTE:__ When using __new__ tokenizer, you need to train the Text Decoder of your model.

## Train TextDecoder

You can check out the notebook below to train your own NeoWhisper.
I would like to highlight that you can __use your own tokenizer__ as long as it is available in `tiktoken` pypi package to train `NeoWhisper` and I recommend to do so __for Khmer language__.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kimang18/rag-demo-with-mlx/blob/main/TrorYong_Small_Language_Model_from_Scratch.ipynb)

I also have a video about training Text Decoder of NeoWhisper below

[![Watch the video](https://i9.ytimg.com/vi/e7wEAVeIo0Y/mqdefault_custom_1.jpg?v=69569cdc&sqp=CKDR48oG&rs=AOn4CLCdBBobdPzfCVmrLvt4GHjR4w7EKA)](https://youtu.be/e7wEAVeIo0Y)

__Remark__

When the config of `AudioEncoder` is the same as the original whisper audio encoder trained by OpenAI, we can load pre-trained weight for the encoder and just train the text decoder.
To load model with `AudioEncoder` of OpenAI whisper, simply provide `neo_encoder=False` when initialize `NeoWhisper` (by default, `neo_encoder=True`).

```python
from neo_whisper import NeoWhisper, NeoModelDimensions
import whisper

dims = NeoModelDimensions(
    n_vocab=tokenizer.encoding.n_vocab, # use the tokenizer's vocab size
    n_mels=80,       # or whatever context size you're training with
    n_audio_ctx=1500,
    n_audio_state=384,
    n_audio_head=6,
    n_audio_layer=4,
    n_text_ctx=448,
    n_text_state=384,
    n_text_head=6,
    n_text_kv_head=6,
    n_text_layer=4
)
model = NeoWhisper(dims, neo_encoder=False)
# load pre-trained weight of audio encoder
model.encoder.load_state_dict(whisper.load_model("tiny").encoder.state_dict())
# freeze the pre-trained weight
for p in model.encoder.parameters():
    p.requires_grad = False
```

## Transcription
We can use trained model for transcription in the same way as `openai-whisper` pypi.
The only difference is that you must specify `tokenizer_name` properly.
Concretely, tokenizer used in the transcription task must be the tokenizer used to train the model.
So, `tokenizer_name` __must be provided__ in the arguments of `transcribe`.

```python
from neo_whisper import (
    get_tokenizer,
    NeoWhisper,
    NeoModelDimensions,
    transcribe
)
tokenizer_name = 'cl100k_base'
tokenizer = get_tokenizer(multilingual=True, task='transcribe', encoder_name=tokenizer_name)
dims = NeoModelDimensions(
    n_vocab=tokenizer.encoding.n_vocab, # use the tokenizer's vocab size
    n_mels=80,       # or whatever context size you're training with
    n_audio_ctx=1500,
    n_audio_state=384,
    n_audio_head=6,
    n_audio_layer=4,
    n_text_ctx=448,
    n_text_state=384,
    n_text_head=6,
    n_text_kv_head=6,
    n_text_layer=4
)
model = NeoWhisper(dims, neo_encoder=False) # if you use neo_encoder, specify accordingly
best_model_params_path = "path/to/your/weights.pt"
model.load_state_dict(torch.load(best_model_params_path))

result = transcribe(wmodel, audio_array, verbose=True, tokenizer_name=tokenizer_name)
print(result['text'])
```

## TODO:
- [X] implement decoding function for `NeoWhisper` and `Whisper`
- [X] implement transcription for `NeoWhisper` and `Whisper`
- [X] notebook colab for training `NeoWhisper`
- [ ] benchmarking
