"""Module for fine-tuning a reward model using preference data."""
import math
import os
import pickle
from itertools import chain
from os import path
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, Dataset, random_split

from .network import LightningNetwork

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
            self.pairs_of_trajectories = pickle.load(handle)

        steps = [
            (
                (
                    step1.astype("float32"),
                    trajectory2[index].astype("float32"),
                )
                for [index, step1] in enumerate(trajectory1)
            )
            for (trajectory1, trajectory2) in self.pairs_of_trajectories
        ]

        self.steps = list(chain.from_iterable(steps))

    def __len__(self):
        """Return size of dataset."""
        return len(self.steps)

    def __getitem__(self, idx: int):
        """Return item with given index."""
        return self.steps[idx][0], self.steps[idx][1]


def train_reward_model(
    reward_model: LightningNetwork,
    dataset: PreferenceDataset,
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
    # Load data
    dataset = PreferenceDataset(file_path)

    # Initialize network
    reward_model = LightningNetwork(
        layer_num=3, input_dim=17, hidden_dim=256, output_dim=1
    )

    # Train model
    train_reward_model(reward_model, dataset, epochs=100, batch_size=32)


if __name__ == "__main__":
    main()
