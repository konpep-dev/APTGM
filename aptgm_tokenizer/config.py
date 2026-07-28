from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Union


class PreTokenizer(str, Enum):
    BYTE_LEVEL = "byte_level"
    CHARACTER = "character"
    WHITESPACE = "whitespace"
    METASPACE_BPE = "metaspace_bpe"
    PUNCTUATION = "punctuation"


def _resolve_pt(pt: Union[str, PreTokenizer]) -> PreTokenizer:
    if isinstance(pt, str):
        return PreTokenizer(pt)
    return pt


@dataclass
class TokenizerConfig:
    vocab_size: int = 16384
    pre_tokenizer: Union[str, PreTokenizer] = PreTokenizer.BYTE_LEVEL
    unk_token: str = "<|UNK|>"
    bos_token: str = "<|BOS|>"
    eos_token: str = "<|EOS|>"
    pad_token: str = "<|PAD|>"
    mask_token: str = "<|MASK|>"
    add_prefix_space: bool = True
    trim_offsets: bool = True
    lowercase: bool = False
    max_token_length: Optional[int] = None
    additional_special_tokens: list[str] = field(default_factory=list)

    @property
    def special_tokens(self) -> list[str]:
        return [
            self.unk_token,
            self.bos_token,
            self.eos_token,
            self.pad_token,
            self.mask_token,
        ] + (self.additional_special_tokens or [])

    @property
    def special_tokens_list(self) -> list[tuple[str, int]]:
        return [(t, i) for i, t in enumerate(self.special_tokens)]


@dataclass
class TrainerConfig:
    initial_alphabet: list[str] = field(default_factory=lambda: [chr(i) for i in range(32, 127)])
    limit_alphabet: int = 1000
    min_frequency: int = 2
    show_progress: bool = True
    special_tokens_first: bool = True
    max_token_length: Optional[int] = None
    num_threads: Optional[int] = None


DEFAULT_CONFIG = TokenizerConfig()
SMALL_CONFIG = TokenizerConfig(vocab_size=8192)
LARGE_CONFIG = TokenizerConfig(vocab_size=32768)
MQAR_CONFIG = TokenizerConfig(vocab_size=256, pre_tokenizer=PreTokenizer.CHARACTER)
