import os
from pathlib import Path
from typing import Optional, Union

from .core import APTGMTokenizer
from .config import TokenizerConfig, TrainerConfig


def train_tokenizer(
    input_files: Union[str, list[str]],
    output_dir: Union[str, Path],
    *,
    vocab_size: int = 16384,
    pre_tokenizer: str = "byte_level",
    lowercase: bool = False,
    min_frequency: int = 2,
    limit_alphabet: int = 1000,
    additional_special_tokens: Optional[list[str]] = None,
    verbose: bool = True,
) -> APTGMTokenizer:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(input_files, str):
        input_files = [input_files]

    cfg = TokenizerConfig(
        vocab_size=vocab_size,
        pre_tokenizer=pre_tokenizer,
        lowercase=lowercase,
        additional_special_tokens=additional_special_tokens or [],
    )

    tok = APTGMTokenizer(cfg)
    tok.train(input_files, vocab_size=vocab_size)
    tok.save(output_dir)

    if verbose:
        print(f"Tokenizer trained: vocab_size={tok.vocab_size}")
        print(f"Saved to: {output_dir}")
        special = tok.special_token_ids
        for name, tid in special.items():
            print(f"  {name}: {tid}")

    return tok


def train_tokenizer_from_texts(
    texts: list[str],
    output_dir: Union[str, Path],
    *,
    vocab_size: int = 16384,
    pre_tokenizer: str = "byte_level",
    verbose: bool = True,
) -> APTGMTokenizer:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = TokenizerConfig(
        vocab_size=vocab_size,
        pre_tokenizer=pre_tokenizer,
    )

    tok = APTGMTokenizer(cfg)
    tok.train_from_iterator(texts, vocab_size=vocab_size)
    tok.save(output_dir)

    if verbose:
        print(f"Tokenizer trained from iterator: vocab_size={tok.vocab_size}")
        print(f"Saved to: {output_dir}")

    return tok
