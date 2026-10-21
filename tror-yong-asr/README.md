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
from tror_yong_asr import TrorYongASRModel, transcribe, translate, identify_language


model_id = "KrorngAI/TrorYongASR2-small"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = TrorYongASRModel.from_pretrained(model_id)

result1 = identify_language('/path/to/audio_file.mp3', model, processor)
print(result1)

result2 = transcribe('/path/to/audio_file.mp3', model, processor, max_tokens=64)
print(result2)

result3 = translate('/path/to/audio_file.mp3', model, processor, max_tokens=64)
print(result3)
```

TrorYongASR has 2 pre-trained weights that support Khmer and English:

- Tiny version with `model_id=KrorngAI/TrorYongASR2-tiny`
- Small version with `model_id=KrorngAI/TrorYongASR2-small`

<figure>
  <div style="text-align: center;"><a name='TrorYongASR' ><img src="https://kimang18.github.io/krorngai-blog/TrorYongASR2/figures/architecture6.png" width="500" /></a></div>
  <figcaption> Figure 1: TrorYongASR2 architecture. Dropout layers are omitted due to space constraints, [B], [L], [T], [E], and [P] are begin-of-sequence, language, task, end-of-sequence, and pad tokens respectively.
    This figure presents the case having 16 distinct target prediction positions.
    The auditory-lingual decoder is just a single normalization-free Transformer where the standard normalization layers are replaced by point-wise Dynamic rescaled Gaussian cumulative distribution function (Derf).
    All projections in attention layers are explicitly shown here because specific to TrorYongASR2, the single position basis \\( \boldsymbol{p} \\) is used for each position to directly form query vectors (differently from the standard decoder, the linear layer for query projection is not required) while context vectors are the token embeddings multiplied by the position basis and are projected into key and value vectors.
    With gated mechanism, masked multi-head attention (Masked MHA) generates gate score from context vectors while MHA for speech-text alignment computes gate score from the hidden state, the output of Masked MHA.
    Inspired by QK-normalization, a Derf activation is applied to the query and key vectors of Masked MHA right before \\( p \\)-RoPE transformation.
    For the cross-attention layer, the query and key vectors are transformed by first Derf and then \\( p \\)-LARoPE.
    Attention masks are generated from randomly chosen permutations and are used only for the context-position attention.
    The last linear layer outputs logits over the vocabulary set.
    These logits are then used to compute cross-entropy loss.
    The audio encoder is simply Whisper's Audio Encoder, but it can process audio of arbitrary length that is shorter than 30 seconds.
  </figcaption>
</figure>

## Evaluation

### Dataset
The evaluation assesses transcription capability on two datasets: 1) [`google/fleurs`](https://huggingface.co/datasets/Kimang18/google-fleurs-km-kh) for Khmer and 2) [`openslr/librispeech_asr`](https://huggingface.co/datasets/openslr/librispeech_asr) for English.
All results are from the **test split** of each dataset.

| Dataset               | Language   | Testing samples  | Description                                       |
|-----------------------|------------|------------------|---------------------------------------------------|
| **google/fleurs**     | Khmer      | 765              | Multi-lingual dataset with Khmer language samples |
| **librispeech_asr.clean** | English    | 2611             | Clean speech dataset for English transcription    |

Audio longer than 30 seconds are excluded from the evaluation (that is why `google/fleurs` has 765 samples instead of 771 and `librispeech_asr.clean` has 2611 instead of 2620).


### Metrics

For transcription task, 3 metrics below are used

- **Token Error Rate (TER)**: Proportion of incorrectly predicted tokens
- **Character Error Rate (CER)**: Proportion of characters that are incorrect
- **Word Error Rate (WER)**: Proportion of words that are incorrect

TER measures model's capability in predicting the next token given the audio input and the current sequence of tokens.
This metric is weaker than WER and CER because it doesn't account for insertions, deletions, substitutions, and autoregression as comprehensively.


### Results

| Model     | Metric | Khmer (`fleurs`) | English (`librispeech_asr.clean`) |
|-----------|--------|------------------|-------------------------------|
| **Tiny**  | WER    | 57.07%           | 20.35%                        |
|           | CER    | 43.95%           | 12.42%                        |
|           | TER    | 49%              | 11%                           |
| **Small** | WER    | 36.70%           | 10.12%                        |
|           | CER    | 29.26%           | 6.05%                         |
|           | TER    | 38%              | 5%                            |


Both configurations performance well on English language.
However, Khmer language remains a challenge.

**Note:** To compute `CER` and `WER`, whitespaces are added between words in Khmer text (Khmer text does not have word boundaries like English text). To do so, `khmercut` PyPI package is used to tokenize Khmer text into words, and then the words are joined back together with whitespaces.
It is due to no-boundary Khmer words that Token Error Rate is utilized as one of the metrics.


**Comparison with TrorYongASR1 and Whisper:**


| Size     | Model        | Parameters | Khmer (`fleurs`)             | English (`librispeech_asr.clean`) |
| -------- | ------------ | ---------- | ---------------------------  | ----------------------------- |
| **Tiny** | **TrorYongASR2** | 29M        | **57.07%**                      | 20.35%                         |
|          | TrorYongASR1 | 29M        | 75.88%                      | 54.33%                         |
|          | Whisper      | 39M        | 100.6%                      | **7.6%**                           |
|          |              |            |                              |                               |
| **Small**| **TrorYongASR2** | 136M       | **36.70%**                      | 10.12%                         |
|          | TrorYongASR1 | 135M       | 50.46%                      | 21.75%                         |
|          | Whisper      | 244M       | 104.4%                      | **3.4%**                           |


**Key Observations:**

- As mentioned above, TrorYongASR2 clearly outperforms TrorYongASR1 and requires just a small increase in the number of parameters (the increase comes from additional length predictor and gate projection layers)
- Whisper has the highest number of parameters for comparable configuration (39M vs 29M for `tiny`, and 244M vs 136M for `small`)
- Whisper shows significantly lower word error rates on English (7.6% for `tiny`, 3.4% for `small`)
- TrorYongASR2 performs best on Khmer dataset (57.07% for `tiny`, and 36.70% for `small`)

**Note:** `WER` data of Whisper is taken directly from their [paper](https://arxiv.org/abs/2212.04356).

**Note on Translation Task:** According to [KrorngAI](https://huggingface.co/KrorngAI), the 2 translation datasets above are built using the API of [TranslateKH](https://translatekh.mptc.gov.kh/) which, in turn, uses an AI model for the translation (Khmer to English and vice-versa).
Without any means to check the correctness of the translation over 209 hours of audio, TrorYongASR2 is not evaluated on any translation datasets.
It can be assumed that the translation performance of TrorYongASR2 is upper bounded by the AI model used in the backend of [TranslateKH](https://translatekh.mptc.gov.kh/).

## Fine-tuning

You can follow this notebook below to learn to fine-tune TrorYongASR2.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/#fileId=https://huggingface.co/KrorngAI/TrorYongASR2-small/blob/main/fine_tune_TrorYongASR2.ipynb)

## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```bibtex
@online{khun2026,
  author = {Khun, Kimang},
  title = {TrorYongASR2: {Permuted} {Language} {Modeling} with
    {Monotonic} {Alignment,} {Gated} {Attention,} and
    {Normalization-free} {Design}},
  date = {2026-10-19},
  url = {https://kimang18.github.io/krorngai-blog/TrorYongASR2/},
  langid = {en}
}
```