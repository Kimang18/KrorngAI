try:
    from whisper.decoding import DecodingResult, decode, detect_language
except (ImportError, ModuleNotFoundError):
    print("You need to install openai-whisper package: pip install git+https://github.com/openai/whisper.git")
    raise

from .tokenizer import get_tokenizer
from .decoding import NeoDecodingOptions as DecodingOptions
from .whisper import NeoWhisper
from .whisper import NeoModelDimensions
from .model import Whisper, ModelDimensions
