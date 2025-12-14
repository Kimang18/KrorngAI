# Author: KrorngAI Org.
# Date: December, 2025


from typing import List, Optional

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


@lru_cache(maxsize=None)
def get_encoding(name: str = "gpt2", num_languages: int = 99):
    """
    Inspired by https://github.com/openai/tiktoken/tree/main, Section Extending tiktoken
    And tokenizer.py of OpenAI/whisper
    """
    encoder_decoder = tiktoken.get_encoding(name)

    n_vocab = len(encoder_decoder._mergeable_ranks)
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
        mergeable_ranks=encoder_decoder._mergeable_ranks,
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
