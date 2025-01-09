# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional, Union

from torch.utils.data import DataLoader

from litgpt.tokenizer import Tokenizer
from litgpt.data import DataModule


@dataclass
class OpenWebMath(DataModule):
    """The OpenWebMath data module for pretraining."""

    data_path: Union[str, Path] = Path("/n/netscratch/glassman_lab/Lab/zhentingqi/my_projects/overtraining/data/continued_pretrain")
    """The path to the data directory, containing two folders 'train' and 'val'
    which are the output of the preprocessing step. The path can also be a remote path (e.g., s3://)."""
    val_split_fraction: float = 0.0005
    """The fraction of data that should be put aside for validation."""
    seed: int = 42
    """The seed to use for shuffling the training data."""
    num_workers: int = 8
    """The number of workers to use for the dataloaders."""

    tokenizer: Optional[Tokenizer] = field(default=None, repr=False, init=False)
    batch_size: int = field(default=1, repr=False, init=False)
    seq_length: int = field(default=2048, repr=False, init=False)

    def __post_init__(self) -> None:
        super().__init__()
        # Could be a remote path (s3://) or a local path
        self.data_path_train = str(self.data_path).rstrip("/") + "/train"
        self.required_paths = [self.data_path_train]

    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = 2048
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = max_seq_length + 1  # Increase by one because we need the next token as well

    def prepare_data(self) -> None:
        for path in self.required_paths:
            if not path.startswith("s3://") and not Path(path).is_dir():
                raise FileNotFoundError(
                    "The data path for OpenWebMath is expected to be the directory containing these subdirectories:"
                    f" `train`, `val`. The directory {path} does not exist."
                    " Set it via `--data.data_path=...`"
                )

    def train_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader

        train_dataset = StreamingDataset(
            input_dir=self.data_path_train,
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
        )
        train_dataloader = StreamingDataLoader(
            train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        return None
