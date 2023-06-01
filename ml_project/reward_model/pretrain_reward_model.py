"""Module for pre-training a reward model from trajectory data."""
import math
import os
import pickle
from os import path
from pathlib import Path

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import Tensor
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, Dataset, random_split

from .networks_old import LightningNetwork

ALGORITHM = "sac"  # "ppo" or "sac"
ENVIRONMENT_NAME = "HalfCheetah-v3"
USE_REWARD_MODEL = False
USE_SDE = True

model_id = f"{ALGORITHM}_{ENVIRONMENT_NAME}"
model_id += "_sde" if USE_SDE else ""
model_id += "_finetuned" if USE_REWARD_MODEL else ""

# File paths
script_path = Path(__file__).parent.resolve()
file_path = path.join(
    script_path, "..", "rl", "reward_data_final", f"{model_id}_reward_dataset.pkl"
)
models_path = path.join(script_path, "models_final")

cpu_count = os.cpu_count()
cpu_count = cpu_count if cpu_count is not None else 8

# Utilize Tensor Cores of NVIDIA GPUs
torch.set_float32_matmul_precision("high")


def calculate_mse_loss(network: LightningModule, batch: Tensor):
    """Calculate the mean squared erro loss for the reward."""
    return mse_loss(network(batch[0]), batch[1].unsqueeze(1), reduction="sum")


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for loading trajectories data."""

    def __init__(self, dataset_path: str):
        """Initialize dataset."""
        with open(dataset_path, "rb") as handle:
            self.trajectories = pickle.load(handle)
        self.data = [t["obs"].astype("float32") for t in self.trajectories]
        self.target = [t["reward"].astype("float32") for t in self.trajectories]

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return item with given index."""
        return self.data[idx], self.target[idx]


def train_reward_model(
    reward_model: LightningModule,
    dataset: TrajectoryDataset,
    epochs: int,
    batch_size: int,
    split_ratio: float = 0.8,
):
    """Train a reward model given trajectories data."""
    train_size = math.floor(split_ratio * len(dataset))
    train_set, val_set = random_split(
        dataset, lengths=[train_size, len(dataset) - train_size]
    )

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

    checkpoint_callback = ModelCheckpoint(
        dirpath=models_path,
        filename=f"{model_id}",
        monitor="val_loss",
        save_weights_only=True,
    )

    trainer = Trainer(
        max_epochs=epochs,
        log_every_n_steps=5,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min"), checkpoint_callback],
    )

    trainer.fit(reward_model, train_loader, val_loader)

    return reward_model


def main():
    """Run reward model pre-training."""
    # Load data
    dataset = TrajectoryDataset(file_path)

    reward_model = LightningNetwork(
        input_dim=17,
        hidden_dim=256,
        layer_num=12,
        output_dim=1,
        calculate_loss=calculate_mse_loss,
    )

    train_reward_model(reward_model, dataset, epochs=100, batch_size=32)


if __name__ == "__main__":
    main()
