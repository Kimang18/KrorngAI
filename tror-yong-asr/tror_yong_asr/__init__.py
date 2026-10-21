from typing import Union
import torch
from torch import Tensor
from collections import namedtuple
from functools import partial
import numpy as np
from .model import TrorYongASRModel, TrorYongASRConfig
from ._version import __version__


TaskOutput = namedtuple("TaskOutput", "pred_ids text")


def load_mel_spectrogram(audio: Union[str, np.ndarray], processor) -> Tensor:
    SAMPLE_RATE = 16_000  # TrorYongASR is trained with this rate
    if isinstance(audio, str):  # file_name
        import torchaudio

        # load audio directly to a tensor
        waveform, sample_rate = torchaudio.load(audio)
        # convert to mono channel
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # resample if necessary (my model accepts 16kHz)
        if sample_rate != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
            waveform = resampler(waveform)
        waveform = waveform.squeeze()  # 1D-array
    elif isinstance(audio, np.ndarray):
        waveform = torch.tensor(audio)
    return processor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding="longest").input_features[0]  # (80, n_frames)


def identify_language(audio: Union[str, np.ndarray], model: TrorYongASRModel, processor):
    """
    Args:
        audio: file path to audio (tested with mp3, but m4a or wav are not tested yet) or audio array
    """

    mel = load_mel_spectrogram(audio, processor)
    return model.identify_language(mel, processor.tokenizer)


def _task_handling(audio: Union[str, np.ndarray], model: TrorYongASRModel, processor, task: str, max_tokens: int, temperature=1.0, top_k=None, seed=168, verbose=False) -> TaskOutput:
    """
    Args:
        audio: file path to audio (tested with mp3, but m4a or wav are not tested yet) or audio array
        max_tokens: maximum number of tokens to be generated
        temperature: calibrate randomness (e.g., 0.0 means greedy decoding)
        verbose: if true and processor.language is None, then print the predicted language
    """

    if processor.tokenizer.language is None:
        raise ValueError(f"Please set language attribute of processor.tokenizer. Call identify_language for language identification.")

    lang_token_id = processor.tokenizer.language_token

    task_token_id = getattr(processor.tokenizer, task, None)
    if task_token_id is None:
        raise ValueError(f"Not supported for task {task}. Either 'transcribe' or 'translate' is supported.")

    model.set_prefix([processor.tokenizer.sot, lang_token_id, task_token_id])

    mel = load_mel_spectrogram(audio, processor)

    pred_ids = model.decode(mel, max_tokens, temperature, top_k, seed)
    text = processor.decode(pred_ids, skip_special_tokens=True)
    return TaskOutput(text=text, pred_ids=pred_ids)


transcribe = partial(_task_handling, task='transcribe')
translate = partial(_task_handling, task='translate')
