from typing import Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader

from .core import APTGMTokenizer


class TextDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        tokenizer: APTGMTokenizer,
        max_length: int = 512,
        stride: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length

        all_ids = tokenizer.encode(" ".join(texts))
        self.examples = []
        for i in range(0, len(all_ids) - max_length + 1, self.stride):
            chunk = all_ids[i : i + max_length]
            self.examples.append(chunk)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk = self.examples[idx]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


def create_dataloader(
    texts: list[str],
    tokenizer: APTGMTokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TextDataset(texts, tokenizer, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def collate_for_aptgm(
    batch: list[dict[str, torch.Tensor]],
    pad_token_id: int = 0,
) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids = []
    labels = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len
        input_ids.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        )
        labels.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack([
            torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
            for seq_len in [item["input_ids"].shape[0] for item in batch]
        ]),
    }
