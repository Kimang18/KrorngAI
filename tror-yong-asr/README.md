# TrorYong ASR Model

`TrorYongASR`, is an Automatic Speech Recognition Model implemented by KrorngAI.

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

You can easily install `tror-yong-asr` using `pip` command as the following:

```bash
pip install tror-yong-asr
```

To use TrorYongASR, there are few dependencies: `transformers`, `safetensors`, and `torchaudio`.

# Usage

Get started with the code below
```python
from transformers import AutoProcessor
from tror_yong_asr import TrorYongASRModel, transcribe, translate, detect_language


model_id = "KrorngAI/TrorYongASR-tiny"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = TrorYongASRModel.from_pretrained(model_id)

result1 = detect_language('/path/to/audio_file.mp3', model, processor)
print(result1)

result2 = transcribe('/path/to/audio_file.mp3', model, processor, max_tokens=64)
print(result2)

result3 = translate('/path/to/audio_file.mp3', model, processor, max_tokens=64)
print(result3)
```

TrorYongASR has 2 pre-trained weights that support Khmer and English:

- Tiny version with `model_id=KrorngAI/TrorYongASR-tiny`
- Small version with `model_id=KrorngAI/TrorYongASR-small`

<figure>
  <div style="text-align: center;"><a name='TrorYongASR' ><img src="https://huggingface.co/KrorngAI/TrorYongASR-tiny/resolve/main/figures/architecture.png" width="500" /></a></div>
  <figcaption> Figure 1: TrorYongASR architecture. Dropout layers are omitted due to space constraints, [B], [L], [T], [E], and [P] are begin-of-sequence, language, task, end-of-sequence, and padding tokens, respectively. This figures presents the case having 16 distinct target prediction positions. The QKV-projection is explicitly shown here because particularly for TrorYongASR, the single position basis p is used for each position to directly form query projection The last linear layer outputs logits over the vocabulary set. These logits are then used to compute cross-entropy loss.</figcaption>
</figure>

# Evaluation

TrorYongASR was evaluated on `test-split` of `google/fleurs` with code `km-kh` for Khmer and `librispeech.clean` for English.

**WER Comparison with Whisper:**

<div align="center">

| Tiny        | Parameters | Khmer (`fleurs`)            | English (`librispeech.clean`) |
| -------     | --------   | --------------------------- | ---                           |
| TrorYongASR | 29M        | 75.88%                      | 54.33%                        |
| Whisper     | 39M        | 100.6%                      | 7.6%                          |

| Small       | Parameters | Khmer (`fleurs`)            | English (`librispeech.clean`) |
| -------     | --------   | --------------------------- | ---                           |
| TrorYongASR | 135M       | 50.46%                      | 21.75%                        |
| Whisper     | 244M       | 104.4%                      | 3.4%                          |
</div>

# Fine-tune TrorYongASR

Below is the notebook of fine-tuning tutorial.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kimang18/rag-demo-with-mlx/blob/main/Finetune_TrorYongASR.ipynb)

If you speak Khmer, you can watch my YouTube video explaining each step of the fine-tuning below.

<a href="http://www.youtube.com/watch?feature=player_embedded&v=hohJ_ZVkYjg" target="_blank">
 <img src="http://img.youtube.com/vi/hohJ_ZVkYjg/mqdefault.jpg" alt="Watch the video" height="240" border="1" />
</a>

**Note**: from version __v.1.1__ onward, you can use functions `push_to_hub`, `save_pretrained`, and `from_pretrained` like any models of `transformers`.

```python
from transformers import AutoProcessor
from tror_yong_asr import TrorYongASRModel

original_model_id="KrorngAI/TrorYongASR-tiny"
processor = AutoProcessor.from_pretrained(original_model_id, trust_remote_code=True)
model = TrorYongASRModel.from_pretrained(original_model_id)

new_model_id="your_hf_repo"
processor.push_to_hub(new_model_id)
model.push_to_hub(new_model_id)
```
