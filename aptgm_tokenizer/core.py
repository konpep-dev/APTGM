import json
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from tokenizers import (
    Tokenizer as HFTokenizer,
    models,
    normalizers,
    pre_tokenizers,
    decoders,
    processors,
    trainers,
)

from .config import TokenizerConfig, PreTokenizer, _resolve_pt


class APTGMTokenizer:
    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self._tokenizer: Optional[HFTokenizer] = None
        self._special_token_ids: dict[str, int] = {}

    @property
    def tokenizer(self) -> HFTokenizer:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call .train() or .load() first.")
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def special_token_ids(self) -> dict[str, int]:
        if not self._special_token_ids:
            self._build_special_map()
        return self._special_token_ids

    def _build_special_map(self):
        self._special_token_ids = {}
        for token in self.config.special_tokens:
            tid = self.tokenizer.token_to_id(token)
            if tid is not None:
                self._special_token_ids[token] = tid

    def _init_empty(self):
        self._tokenizer = HFTokenizer(models.BPE(unk_token=self.config.unk_token))
        self._tokenizer.add_special_tokens(self.config.special_tokens)
        self._apply_normalizer()
        self._apply_pre_tokenizer()
        self._apply_decoder()
        self._build_special_map()

    def _apply_normalizer(self):
        if self.config.lowercase:
            self._tokenizer.normalizer = normalizers.Sequence([
                normalizers.NFKC(),
                normalizers.Lowercase(),
            ])
        else:
            self._tokenizer.normalizer = normalizers.NFKC()

    def _apply_pre_tokenizer(self):
        cfg = self.config
        pt = _resolve_pt(cfg.pre_tokenizer)

        if pt == PreTokenizer.BYTE_LEVEL:
            self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
                add_prefix_space=cfg.add_prefix_space
            )
        elif pt == PreTokenizer.CHARACTER:
            self._tokenizer.pre_tokenizer = pre_tokenizers.CharDelimiterSplit(" ")
        elif pt == PreTokenizer.WHITESPACE:
            self._tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        elif pt == PreTokenizer.METASPACE_BPE:
            self._tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
                add_prefix_space=cfg.add_prefix_space
            )
        elif pt == PreTokenizer.PUNCTUATION:
            self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.Whitespace(),
                pre_tokenizers.Punctuation(),
            ])
        else:
            raise ValueError(f"Unknown pre_tokenizer: {pt}")

    def _apply_decoder(self):
        cfg = self.config
        pt = _resolve_pt(cfg.pre_tokenizer)

        if pt == PreTokenizer.BYTE_LEVEL:
            self._tokenizer.decoder = decoders.ByteLevel()
        elif pt == PreTokenizer.METASPACE_BPE:
            self._tokenizer.decoder = decoders.Metaspace()
        else:
            self._tokenizer.decoder = decoders.BPEDecoder()

    def _get_trainer(self) -> trainers.BpeTrainer:
        tcfg = self.config
        return trainers.BpeTrainer(
            vocab_size=tcfg.vocab_size,
            min_frequency=2,
            special_tokens=tcfg.special_tokens,
            show_progress=True,
            initial_alphabet=[chr(i) for i in range(32, 127)],
            limit_alphabet=1000,
        )

    def train(self, files: Union[str, list[str]], *, vocab_size: Optional[int] = None, **kwargs):
        if vocab_size is not None:
            self.config.vocab_size = vocab_size
        self._init_empty()
        self.tokenizer.train(files, self._get_trainer())
        self._apply_postprocessor()
        self._build_special_map()

    def train_from_iterator(self, texts: list[str], *, vocab_size: Optional[int] = None):
        if vocab_size is not None:
            self.config.vocab_size = vocab_size
        self._init_empty()
        trainer = self._get_trainer()
        self.tokenizer.train_from_iterator(texts, trainer)
        self._apply_postprocessor()
        self._build_special_map()

    def _apply_postprocessor(self):
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{self.config.bos_token} $A {self.config.eos_token}",
            pair=f"{self.config.bos_token} $A {self.config.eos_token} $B:1 {self.config.eos_token}:1",
            special_tokens=[
                (self.config.bos_token, self.special_token_ids[self.config.bos_token]),
                (self.config.eos_token, self.special_token_ids[self.config.eos_token]),
            ],
        )

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if add_special_tokens:
            return self.tokenizer.encode(text).ids
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: list[str], add_special_tokens: bool = True) -> list[list[int]]:
        out = self.tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
        return [o.ids for o in out]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, batch: list[list[int]], skip_special_tokens: bool = True) -> list[str]:
        return self.tokenizer.decode_batch(batch, skip_special_tokens=skip_special_tokens)

    def id_to_token(self, token_id: int) -> Optional[str]:
        return self.tokenizer.id_to_token(token_id)

    def token_to_id(self, token: str) -> Optional[int]:
        return self.tokenizer.token_to_id(token)

    def pad(
        self,
        sequences: list[list[int]],
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> dict[str, Union[list, np.ndarray]]:
        pad_id = self.special_token_ids.get(self.config.pad_token, 0)
        if max_length is None:
            max_length = max(len(s) for s in sequences) if sequences else 0

        padded = []
        attention_mask = []
        for seq in sequences:
            if len(seq) >= max_length:
                padded.append(seq[:max_length])
                attention_mask.append([1] * max_length)
            else:
                pad_len = max_length - len(seq)
                padded.append(seq + [pad_id] * pad_len)
                attention_mask.append([1] * len(seq) + [0] * pad_len)

        if return_tensors == "pt":
            import torch
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        return {
            "input_ids": padded,
            "attention_mask": attention_mask,
        }

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path / "tokenizer.json"))
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size": self.config.vocab_size,
                "pre_tokenizer": _resolve_pt(self.config.pre_tokenizer).value,
                "unk_token": self.config.unk_token,
                "bos_token": self.config.bos_token,
                "eos_token": self.config.eos_token,
                "pad_token": self.config.pad_token,
                "mask_token": self.config.mask_token,
                "add_prefix_space": self.config.add_prefix_space,
                "lowercase": self.config.lowercase,
                "additional_special_tokens": self.config.additional_special_tokens or [],
            }, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "APTGMTokenizer":
        path = Path(path)
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                data = json.load(f)
            cfg = TokenizerConfig(
                vocab_size=data.get("vocab_size", 16384),
                pre_tokenizer=PreTokenizer(data.get("pre_tokenizer", "byte_level")),
                unk_token=data.get("unk_token", "<|UNK|>"),
                bos_token=data.get("bos_token", "<|BOS|>"),
                eos_token=data.get("eos_token", "<|EOS|>"),
                pad_token=data.get("pad_token", "<|PAD|>"),
                mask_token=data.get("mask_token", "<|MASK|>"),
                add_prefix_space=data.get("add_prefix_space", True),
                lowercase=data.get("lowercase", False),
                additional_special_tokens=data.get("additional_special_tokens", []),
            )
        else:
            cfg = TokenizerConfig()

        tok = cls(cfg)
        tok._tokenizer = HFTokenizer.from_file(str(path / "tokenizer.json"))
        tok._build_special_map()
        return tok
