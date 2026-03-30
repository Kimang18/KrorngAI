# Author: KrorngAI org.
# Date: March 2026
# Inspired by https://github.com/openai/whisper/blob/main/whisper/tokenizer.py

from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from functools import cached_property
from enum import Enum
from transformers import AutoTokenizer


LANGUAGES = {
    "km": "khmer",
    "en": "english"
}
TO_LANGUAGE_CODE = {
    **{lang: code for code, lang in LANGUAGES.items()},
}


class ASRSpecialTokens(str, Enum):
    km_token = "<|km|>" # language token must be added to lm_head of Decoder Model
    en_token = "<|en|>" # language token must be added to lm_head of Decoder Model
    transcribe = "<|transcribe|>"
    translate = "<|translate|>"
    no_speech = "<|nospeech|>"
    @classmethod
    def list(cls):
        return [c.value for c in cls]


@dataclass
class ASRTokenizer:
    """
    Tokenizer for the ASR task.
    It supports only two languages: Khmer and English.
    It does not support timestamps.
    """
    encoding: AutoTokenizer
    num_languages: int = 2 # only khmer and english
    language: Optional[str] = None
    task: Optional[str] = None
    sot_sequence: Tuple[int] = ()
    special_tokens: dict[str, int] = field(default_factory=dict)
    def __post_init__(self):
        self.encoding.add_special_tokens({
            "additional_special_tokens": ASRSpecialTokens.list()
        })

        for special in self.encoding.all_special_tokens:
            special_id = self.encoding.encode(special, add_special_tokens=False)[0]
            self.special_tokens[special] = special_id

        sot: int = self.special_tokens["<s>"]
        translate: int = self.special_tokens["<|translate|>"]
        transcribe: int = self.special_tokens["<|transcribe|>"]

        sot_sequence = [sot]
        if self.language is not None:
            language = self.language.lower()
            if language not in LANGUAGES:
                if language in TO_LANGUAGE_CODE:
                    language = TO_LANGUAGE_CODE[language]
                else:
                    raise ValueError(f"Unsupported language: {language}")

            self.language = f"<|{language}|>"
            lang_id = self.encoding.encode(self.language, add_special_tokens=False)[0]
            sot_sequence.append(lang_id)
        if self.task is not None:
            task_token: int = transcribe if self.task == "transcribe" else translate
            sot_sequence.append(task_token)

        self.sot_sequence = tuple(sot_sequence)

    def encode(self, text, **kwargs):
        encoding = self.encoding.encode(text, add_special_tokens=False, **kwargs)
        return encoding if encoding[0] != 29871 else encoding[1:] # 29871 is whitespace for TinyKhmerTokenizer

    def decode(self, token_ids: List[int], **kwargs) -> str:
        return self.encoding.decode(token_ids, **kwargs)

    @cached_property
    def eot(self) -> int:
        return self.special_tokens["</s>"]

    @cached_property
    def transcribe(self) -> int:
        return self.special_tokens["<|transcribe|>"]

    @cached_property
    def translate(self) -> int:
        return self.special_tokens["<|translate|>"]

    @cached_property
    def sot(self) -> int:
        return self.special_tokens["<s>"]

    @cached_property
    def pad_id(self) -> int:
        return self.special_tokens["<unk>"]

    @cached_property
    def no_speech(self) -> int:
        return self.special_tokens["<|nospeech|>"]

    @cached_property
    def language_token(self) -> int:
        """Returns the token id corresponding to the value of the `language` field"""
        if self.language is None:
            raise ValueError("This tokenizer does not have language token configured")

        return self.to_language_token(self.language)

    def to_language_token(self, language):
        if token := self.special_tokens.get(f"<|{language}|>", None):
            return token

        raise KeyError(f"Language {language} not found in tokenizer.")

    @cached_property
    def all_language_tokens(self) -> Tuple[int]:
        result = []
        for token, token_id in self.special_tokens.items():
            if token.strip("<|>") in LANGUAGES:
                result.append(token_id)
        return tuple(result)

    @cached_property
    def all_language_codes(self) -> Tuple[str]:
        return tuple(self.decode([_l]).strip("<|>") for _l in self.all_language_tokens)

    @cached_property
    def non_speech_tokens(self) -> Tuple[int]:
        """
        Returns the list of tokens to suppress in order to avoid any speaker tags or non-speech
        annotations, to prevent sampling texts that are not actually spoken in the audio, e.g.

        - ♪♪♪
        - ( SPEAKING FOREIGN LANGUAGE )
        - [DAVID] Hey there,

        keeping basic punctuations like commas, periods, question marks, exclamation points, etc.
        """
        symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』')
        symbols += (
            "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()
        )

        # symbols that may be a single token or multiple tokens depending on the tokenizer.
        # In case they're multiple tokens, suppress the first token, which is safe because:
        # These are between U+2640 and U+267F miscellaneous symbols that are okay to suppress
        # in generations, and in the 3-byte UTF-8 representation they share the first two bytes.
        miscellaneous = set("♩♪♫♬♭♮♯")
        assert all(0x2640 <= ord(c) <= 0x267F for c in miscellaneous)

        # allow hyphens "-" and single quotes "'" between words, but not at the beginning of a word
        result = {self.encoding.encode(" -", add_special_tokens=False)[0], self.encoding.encode(" '", add_special_tokens=False)[0]}
        for symbol in symbols + list(miscellaneous):
            for tokens in [
                self.encoding.encode(symbol, add_special_tokens=False),
                self.encoding.encode(" " + symbol, add_special_tokens=False),
            ]:
                if len(tokens) == 1 or symbol in miscellaneous:
                    result.add(tokens[0])

        return tuple(sorted(result))
