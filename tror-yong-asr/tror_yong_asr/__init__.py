from torch import Tensor
from collections import namedtuple
from functools import partial
import torchaudio
from .model import TrorYongASRModel, TrorYongASRConfig
from ._version import __version__


TaskOutput = namedtuple("TaskOutput", "output_ids text")
# hard-coded audio hyperparameters
SAMPLE_RATE = 16_000


def load_mel_from_file(file_name: str, processor) -> Tensor:
    # load audio directly to a tensor
    waveform, sample_rate = torchaudio.load(file_name)
    # resample if necessary (my model accepts 16kHz)
    if sample_rate != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
        waveform = resampler(waveform)
    waveform = waveform.squeeze()  # 1D-array
    return processor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding="longest").input_features[0]  # (80, n_frames)


def detect_language(file_name: str, model: TrorYongASRModel, processor):
    """
    Args:
    file_name: file path to audio (tested with mp3, but m4a or wav are not tested yet)
    """

    mel = load_mel_from_file(file_name, processor)
    return model.detect_language(mel, processor.tokenizer)


def _task_handling(file_name: str, model: TrorYongASRModel, processor, task: str, max_tokens: int, temperature=1.0, top_k=None, seed=168, verbose=False) -> TaskOutput:
    """
    Args:
    file_name: file path to audio (tested with mp3, but m4a or wav are not tested yet)
    max_tokens: maximum number of tokens to be generated
    temperature: calibrate randomness (e.g., 0.0 means greedy decoding)
    verbose: if true and processor.language is None, then print the predicted language
    """
    task_token_id = getattr(processor.tokenizer, task, None)
    if task_token_id is None:
        raise ValueError(f"No support for task {task}. Either 'transcribe' or 'translate' is supported.")

    mel = load_mel_from_file(file_name, processor)

    if processor.tokenizer.language is None:
        lang_token_id, lang_probs = model.detect_language(mel, processor.tokenizer)
        if verbose:
            print("Language Prediction", lang_probs)
        lang_token_id = lang_token_id.item()
    else:
        lang_token_id = processor.tokenizer.language_token

    if lang_token_id == processor.tokenizer.no_speech:
        return TaskOutput(text='', output_ids=[processor.tokenizer.sot, lang_token_id, processor.tokenizer.eot])

    model.set_prefix([processor.tokenizer.sot, lang_token_id, task_token_id])

    output_ids = model.decode(mel, max_tokens, temperature, top_k, seed)
    text = processor.decode(output_ids, skip_special_tokens=True)
    return TaskOutput(text=text, output_ids=output_ids)


transcribe = partial(_task_handling, task='transcribe')
translate = partial(_task_handling, task='translate')
