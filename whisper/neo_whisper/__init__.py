try:
    from whisper.decoding import DecodingResult
    from whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim
except (ImportError, ModuleNotFoundError):
    print("You need to install openai-whisper package: pip install git+https://github.com/openai/whisper.git")
    raise

from .tokenizer import get_tokenizer
from .whisper import NeoWhisper
from .whisper import NeoModelDimensions
from .model import Whisper, ModelDimensions
from .decoding import NeoDecodingOptions as DecodingOptions
from .decoding import decode, detect_language
from .transcribe import transcribe
from ._version import __version__
