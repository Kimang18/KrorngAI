from collections import namedtuple
from functools import partial
import torchaudio
from .model import TrorYongASRModel, TrorYongASRConfig
from ._version import __version__


TaskOutput = namedtuple("TaskOutput", "output_ids text")


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

    # load audio directly to a tensor
    waveform, sample_rate = torchaudio.load(file_name)
    # resample if necessary (my model excepts 16kHz)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    audio_array = waveform.squeeze().numpy()  # 1D-array

    mels = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features[0]  # (80, 3000)

    if processor.tokenizer.language is None:
        lang_token_id, lang_probs = model.detect_language(mels, processor.tokenizer)
        if verbose:
            print("Language Prediction", lang_probs)
        lang_token_id = lang_token_id.item()
    else:
        lang_token_id = processor.tokenizer.language_token

    model.set_prefix([processor.tokenizer.sot, lang_token_id, task_token_id])

    output_ids = model.decode(mels, max_tokens, temperature, top_k, seed)
    text = processor.decode(output_ids[0], skip_special_tokens=True)
    return TaskOutput(text=text, output_ids=output_ids[0])


transcribe = partial(_task_handling, task='transcribe')
translate = partial(_task_handling, task='translate')
