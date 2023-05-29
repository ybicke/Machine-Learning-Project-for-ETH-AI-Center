"""Module for fine-tuning a reward model using preference data."""
import math
import os
import pickle
from itertools import chain
from os import path
from pathlib import Path
from typing import Union

import numpy
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, Dataset, random_split

from ..types import FloatNDArray, Trajectory
from .network import LightningNetwork, LightningRNNNetwork

# Utilize Tensor Cores of NVIDIA GPUs
torch.set_float32_matmul_precision("high")

# File paths
script_path = Path(__file__).parent.resolve()
file_path = path.join(script_path, "preference_dataset.pkl")


class PreferenceDataset(Dataset):
    """PyTorch Dataset for loading preference data."""

    def __init__(self, dataset_path: str):
        """Initialize dataset."""
        with open(dataset_path, "rb") as handle:
            trajectory_pairs: list[tuple[Trajectory, Trajectory]] = pickle.load(handle)

        steps = [
            (
                (
                    trajectory1[index]["obs"].astype("float32"),
                    trajectory2[index]["obs"].astype("float32"),
                )
                for index in range(len(trajectory1))
            )
            for (trajectory1, trajectory2) in trajectory_pairs
        ]

        self.steps = list(chain.from_iterable(steps))

    def __len__(self):
        """Return size of dataset."""
        return len(self.steps)

    def __getitem__(self, index: int):
        """Return item with given index."""
        return self.steps[index][0], self.steps[index][1]


# pylint: disable=too-few-public-methods
class MultiStepPreferenceDataset(Dataset):
    """PyTorch Dataset for loading preference data for multiple steps."""

    def __init__(self, dataset_path: str, sequence_length: int):
        """Initialize dataset."""
        with open(dataset_path, "rb") as handle:
            trajectory_pairs: list[tuple[Trajectory, Trajectory]] = pickle.load(handle)

        sequence_pairs: list[tuple[FloatNDArray, FloatNDArray]] = []

        for trajectory1, trajectory2 in trajectory_pairs:
            for index in range(len(trajectory1) - sequence_length + 1):
                # find the end of this pattern
                end_index = index + sequence_length

                # gather input and output parts of the pattern
                sequence_pair = (
                    numpy.array(
                        [step["obs"] for step in trajectory1[index:end_index]],
                        dtype=numpy.float32,
                    ),
                    numpy.array(
                        [step["obs"] for step in trajectory2[index:end_index]],
                        dtype=numpy.float32,
                    ),
                )
                sequence_pairs.append(sequence_pair)

        self.sequence_pairs = sequence_pairs
        self.sequence_length = sequence_length

    def __len__(self):
        """Return size of dataset."""
        return len(self.sequence_pairs)

    def __getitem__(self, index: int):
        """Return items with given index."""
        return self.sequence_pairs[index]


def train_reward_model(
    reward_model: LightningModule,
    dataset: Union[PreferenceDataset, MultiStepPreferenceDataset],
    epochs: int,
    batch_size: int,
    split_ratio: float = 0.8,
):
    """Train a reward model given preference data."""
    train_size = math.floor(split_ratio * len(dataset))
    train_set, val_set = random_split(
        dataset, lengths=[train_size, len(dataset) - train_size]
    )

    cpu_count = os.cpu_count()
    cpu_count = cpu_count if cpu_count is not None else 8

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cpu_count,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=cpu_count,
    )

    trainer = Trainer(
        max_epochs=epochs,
        log_every_n_steps=5,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
    )

    trainer.fit(reward_model, train_loader, val_loader)

    return reward_model


def main():
    """Run reward model fine-tuning."""
    # Train MLP
    dataset = PreferenceDataset(file_path)

    reward_model = LightningNetwork(
        input_dim=17, hidden_dim=256, layer_num=3, output_dim=1
    )

    train_reward_model(reward_model, dataset, epochs=100, batch_size=4)

    # Train RNN
    dataset = MultiStepPreferenceDataset(file_path, 32)

    reward_model = LightningRNNNetwork(
        input_size=17, hidden_size=128, num_layers=3, dropout=0.2
    )

    train_reward_model(reward_model, dataset, epochs=100, batch_size=4)


if __name__ == "__main__":
    main()
