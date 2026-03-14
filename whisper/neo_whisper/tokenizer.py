# Author: KrorngAI Org.
# Date: December, 2025


from typing import List, Optional
import regex as re

try:
    from whisper.tokenizer import (
        lru_cache,
        Tokenizer,
        tiktoken,
        LANGUAGES,
        TO_LANGUAGE_CODE
    )
except (ImportError, ModuleNotFoundError):
    print("You need to install openai-whisper package: pip install git+https://github.com/openai/whisper.git")
    raise


class NeoTokenizer(Tokenizer):
    def decode(self, token_ids: List[int], skip_special_tokens: bool = False, **kwargs) -> str:
        """
        Re-implement by adding skip_special_tokens option
        """
        token_ids = [t for t in token_ids if t < self.timestamp_begin]
        if skip_special_tokens:
            token_ids = [
                t for t in token_ids if t not in self.special_tokens.values()]
        return self.encoding.decode(token_ids, **kwargs)


def is_english_or_khmer(token_bytes: bytes, allowed_pattern) -> bool:
    try:
        # Step 1: Decode bytes to UTF-8 string
        text = token_bytes.decode('utf-8')

        # Step 2: Use fullmatch to ensure EVERY character in the token
        # belongs to the allowed ranges
        return bool(allowed_pattern.fullmatch(text))

    except UnicodeDecodeError:
        # If it can't be decoded as UTF-8, they cannot be checked by regex.
        # We return False here; the main loop will still keep these
        # if they are single bytes (len == 1) for safety.
        return False


@lru_cache(maxsize=None)
def get_encoding(name: str = "gpt2", num_languages: int = 99):
    """
    Inspired by https://github.com/openai/tiktoken/tree/main, Section Extending tiktoken
    And tokenizer.py of OpenAI/whisper
    """
    if name == "km-en":
        encoder_decoder = tiktoken.get_encoding("o200k_base")
        original_ranks: dict[bytes, int] = encoder_decoder._mergeable_ranks
        # filter for Khmer and English tokens
        # around 130000 tokens remain

        # Unicode ranges:
        # English/Basic Latin: 0x0000-0x007F
        # Khmer: 0x1780-0x17FF
        # Khmer Symbols: 0x19E0-0x19FF
        allowed = re.compile(
            r'^[\u0000-\u007F'
            r'\u1780-\u17FF'
            r'\u19E0-\u19FF]+$'
        )
        subset_ranks = {}
        for token, rank in original_ranks.items():
            if len(token) == 1:   # Always keep single bytes (the 256 'atoms')
                subset_ranks[token] = rank
                continue

            if is_english_or_khmer(token, allowed):
                subset_ranks[token] = rank

        # Re-index: Create a new mapping from token bytes to a new contiguous ID
        # We sort by the original rank to maintain the BPE merge priority order
        sorted_ranks = sorted(subset_ranks, key=lambda t: original_ranks[t])
        mergeable_ranks = {token: i for i, token in enumerate(sorted_ranks)}
    else:
        encoder_decoder = tiktoken.get_encoding(name)
        mergeable_ranks = encoder_decoder._mergeable_ranks

    n_vocab = len(mergeable_ranks)
    special_tokens = {}

    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())[:num_languages]],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]

    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1

    return tiktoken.Encoding(
        name=f"{name}_im",
        explicit_n_vocab=n_vocab,
        pat_str=encoder_decoder._pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )


@lru_cache(maxsize=None)
def get_tokenizer(
    multilingual: bool,
    *,
    num_languages: int = 99,
    language: Optional[str] = None,
    task: Optional[str] = None,  # Literal["transcribe", "translate", None]
    encoder_name: Optional[str] = None
) -> Tokenizer:
    if language is not None:
        language = language.lower()
        if language not in LANGUAGES:
            if language in TO_LANGUAGE_CODE:
                language = TO_LANGUAGE_CODE[language]
            else:
                raise ValueError(f"Unsupported language: {language}")

    if multilingual:
        encoding_name = "multilingual"
        language = language or "en"
        task = task or "transcribe"
    else:
        encoding_name = "gpt2"
        language = None
        task = None
    if encoder_name is not None:
        encoding_name = encoder_name

    encoding = get_encoding(name=encoding_name, num_languages=num_languages)

    return NeoTokenizer(
        encoding=encoding, num_languages=num_languages, language=language, task=task
    )
