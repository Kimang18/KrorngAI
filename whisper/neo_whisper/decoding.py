# Author: KrorngAI Org.
# Date: December, 2025


from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, List, Tuple, Union

import torch
from torch import Tensor

try:
    from whisper.audio import CHUNK_LENGTH
    from whisper.decoding import (
        DecodingOptions,
        DecodingResult,
        PyTorchInference,
        MaximumLikelihoodRanker,
        GreedyDecoder,
        BeamSearchDecoder,
        SuppressBlank,
        SuppressTokens,
        ApplyTimestampRules,
        DecodingTask,
    )
except (ImportError, ModuleNotFoundError):
    print("You need to install openai-whisper package: pip install git+https://github.com/openai/whisper.git")
    raise

if TYPE_CHECKING:
    from .model import Whisper
    from .whisper import NeoWhisper

from .tokenizer import get_tokenizer, NeoTokenizer
from .nn_utils import KVCache


@dataclass(frozen=True)
class NeoDecodingOptions(DecodingOptions):
    # WARN: add new option: tokenizer_name
    tokenizer_name: str = "gpt2"


class NeoPyTorchInference(PyTorchInference):
    def __init__(self, model: "NeoWhisper", initial_token_length: int):
        super().__init__(model, initial_token_length)
        self.kv_model_kwargs = {
            "num_heads": self.model.dims.n_text_head,
            "head_dim": self.model.dims.n_text_state // self.model.dims.n_text_head,
            "num_layers": self.model.dims.n_text_layer,
        }

    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        if not self.kv_cache:
            self.kv_cache, self.hooks = self.model.install_kv_cache_hooks()

        if self.kv_cache.get('neo', None) is None:
            batch_size = audio_features.shape[0]
            self.kv_cache['neo'] = KVCache(
                batch_size=batch_size,
                seq_len=self.initial_token_length,
                **self.kv_model_kwargs,
            )

        if tokens.shape[-1] > self.initial_token_length:
            # only need to use the last token except in the first forward pass
            tokens = tokens[:, -1:]

        return self.model.decoder(tokens, audio_features, kv_cache=self.kv_cache)


class NeoDecodingTask(DecodingTask):
    def __init__(self, model: "NeoWhisper", options: DecodingOptions):
        self.model = model

        tokenizer_name = options.tokenizer_name or "gpt2"
        language = options.language or "en"
        tokenizer = get_tokenizer(
            model.is_multilingual,
            num_languages=model.num_languages,
            language=language,
            task=options.task,
            encoder_name=tokenizer_name
        )
        self.tokenizer: NeoTokenizer = tokenizer
        self.options: DecodingOptions = self._verify_options(options)

        self.n_group: int = options.beam_size or options.best_of or 1
        self.n_ctx: int = model.dims.n_text_ctx
        self.sample_len: int = options.sample_len or model.dims.n_text_ctx // 2

        self.sot_sequence: Tuple[int] = tokenizer.sot_sequence
        if self.options.without_timestamps:
            self.sot_sequence = tokenizer.sot_sequence_including_notimestamps

        self.initial_tokens: Tuple[int] = self._get_initial_tokens()
        self.sample_begin: int = len(self.initial_tokens)
        self.sot_index: int = self.initial_tokens.index(tokenizer.sot)

        # inference: implements the forward pass through the decoder, including kv caching
        #TODO : this does not work for Whisper yet
        self.inference = NeoPyTorchInference(model, len(self.initial_tokens))

        # sequence ranker: implements how to rank a group of sampled sequences
        self.sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)

        # decoder: implements how to select the next tokens, given the autoregressive distribution
        if options.beam_size is not None:
            self.decoder = BeamSearchDecoder(
                options.beam_size, tokenizer.eot, self.inference, options.patience
            )
        else:
            self.decoder = GreedyDecoder(options.temperature, tokenizer.eot)

        # logit filters: applies various rules to suppress or penalize certain tokens
        self.logit_filters = []
        if self.options.suppress_blank:
            self.logit_filters.append(SuppressBlank(
                self.tokenizer, self.sample_begin))
        if self.options.suppress_tokens:
            self.logit_filters.append(
                SuppressTokens(self._get_suppress_tokens()))
        if not options.without_timestamps:
            precision = CHUNK_LENGTH / model.dims.n_audio_ctx  # usually 0.02 seconds
            max_initial_timestamp_index = None
            if options.max_initial_timestamp:
                max_initial_timestamp_index = round(
                    self.options.max_initial_timestamp / precision
                )
            self.logit_filters.append(
                ApplyTimestampRules(
                    tokenizer, self.sample_begin, max_initial_timestamp_index
                )
            )


@torch.no_grad()
def decode(
    model: "Whisper",
    mel: Tensor,
    options: DecodingOptions = DecodingOptions(),
    **kwargs,
) -> Union[DecodingResult, List[DecodingResult]]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model: Whisper
        the Whisper model instance

    mel: torch.Tensor, shape = (80, 3000) or (*, 80, 3000)
        A tensor containing the Mel spectrogram(s)

    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments

    Returns
    -------
    result: Union[DecodingResult, List[DecodingResult]]
        The result(s) of decoding contained in `DecodingResult` dataclass instance(s)
    """
    if single := mel.ndim == 2:
        mel = mel.unsqueeze(0)

    if kwargs:
        options = replace(options, **kwargs)

    result = NeoDecodingTask(model, options).run(mel)

    return result[0] if single else result
