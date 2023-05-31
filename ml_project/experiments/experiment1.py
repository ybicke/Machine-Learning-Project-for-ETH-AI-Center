"""Module for conducting an experiment."""
import math
import os
from os import path
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, Dataset, random_split

from ml_project.reward_model.networks import (
    LightningTrajectoryNetworkExperiment,
    LightningTrajectoryNetworkPreTrain,
)
from ml_project.types import FloatNDArray

script_path = Path(__file__).parent.resolve()

pretrained_model_path = path.join(
    script_path,
    "..",
    "..",
    "lightning_logs",
    "experiment1_pretrained_linear",
    "checkpoints",
    "epoch=22-step=5750.ckpt",
)


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for loading trajectories data."""

    def __init__(self, trajectories: list[tuple[int, int]]):
        """Initialize dataset."""
        self.data = [obs for (obs, _) in trajectories]
        self.target = [reward for (_, reward) in trajectories]

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return item with given index."""
        return self.data[idx], self.target[idx]


class MultiStepPreferenceDataset(Dataset):
    """PyTorch Dataset for loading preference data for multiple steps."""

    # Specify -1 for `sequence_length` to use the full trajectories
    def __init__(self, data: list[tuple[list[int], list[int]]], sequence_length: int):
        """Initialize dataset."""
        trajectory_pairs: list[tuple[list[int], list[int]]] = data

        sequence_pairs: list[tuple[FloatNDArray, FloatNDArray]] = []

        for trajectory1, trajectory2 in trajectory_pairs:
            trajectory_length = (
                len(trajectory1) if sequence_length == -1 else sequence_length
            )

            for index in range(len(trajectory1) - trajectory_length + 1):
                # find the end of this pattern
                end_index = index + trajectory_length

                # gather input and output parts of the pattern
                sequence_pair = (
                    np.array(
                        trajectory1[index:end_index],
                        dtype=np.float32,
                    ),
                    np.array(
                        trajectory2[index:end_index],
                        dtype=np.float32,
                    ),
                )
                sequence_pairs.append(sequence_pair)

        self.sequence_pairs = sequence_pairs

    def __len__(self):
        """Return size of dataset."""
        return len(self.sequence_pairs)

    def __getitem__(self, index: int):
        """Return items with given index."""
        return self.sequence_pairs[index]


def train_reward_model(
    reward_model: LightningModule,
    dataset: Union[TrajectoryDataset, MultiStepPreferenceDataset],
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


def true_reward(state: int):
    """Return true reward."""
    return 0.8 * state


def engineered_reward(state: int):
    """Return engineered reward."""
    return state


def generate_reward_data(amount: int = 10000):
    """Genereate synthetic reward data."""
    states = 200 * np.random.rand(amount) - 100
    rewards = [engineered_reward(state) for state in states]
    return list(zip(states, rewards))


def generate_preference_data(amount: int = 100):
    """Genereate synthetic preference data."""
    pairs = 200 * np.random.rand(amount, 2) - 100

    preference_data = []

    for first, second in pairs:
        if true_reward(first) < true_reward(second):
            preference_data.append(([first], [second]))
        elif true_reward(first) > true_reward(second):
            preference_data.append(([second], [first]))
        else:
            preference_data.append(([first], [second]))
            preference_data.append(([second], [first]))

    return preference_data


def main():
    """Run experiment."""
    reward_data = generate_reward_data()
    preference_data = generate_preference_data()

    TrajectoryDataset(reward_data)
    finetune_dataset = MultiStepPreferenceDataset(preference_data, sequence_length=-1)

    reward_model = LightningTrajectoryNetworkPreTrain(
        layer_num=3, input_dim=1, hidden_dim=64, output_dim=1
    )
    # train_reward_model(reward_model, pretrain_dataset, epochs=100, batch_size=32)
    checkpoint = torch.load(pretrained_model_path)
    reward_model.load_state_dict(checkpoint["state_dict"])

    f_reward_model = LightningTrajectoryNetworkExperiment(
        layer_num=3, input_dim=1, hidden_dim=64, output_dim=1
    )

    # load pre-trained model
    checkpoint = torch.load(pretrained_model_path)
    f_reward_model.load_state_dict(checkpoint["state_dict"])

    train_reward_model(f_reward_model, finetune_dataset, epochs=100, batch_size=1)

    # load fine-tuned model
    # ...

    x_axis = np.arange(-100, 100, 1)

    true_r = [true_reward(x) for x in x_axis]
    engineered_r = [engineered_reward(x) for x in x_axis]
    with torch.no_grad():
        pretrained_r = [reward_model(torch.Tensor([x])).numpy()[0] for x in x_axis]
        finetuned_r = [f_reward_model(torch.Tensor([[x]])).numpy()[0] for x in x_axis]

    ys_values = [true_r, engineered_r, pretrained_r, finetuned_r]

    labels = [
        "true reward",
        "engineered reward",
        "pre-trained model",
        "fine-tuned model",
    ]
    colors = [
        "tab:blue",
        "tab:red",
        "tab:green",
        "tab:purple",
    ]

    for index, label in enumerate(labels):
        plt.plot(x_axis, ys_values[index], color=colors[index], label=label)
    plt.xlabel("State", fontsize="xx-large")
    plt.ylabel("Reward", fontsize="xx-large")
    plt.grid(True, color="gray", alpha=0.5)
    plt.legend(fontsize="x-large", borderpad=0.1, labelspacing=0.1)
    plt.savefig(
        "ml_project/experiments/experiment1_with_pretraining.png",
        dpi=100,
        bbox_inches="tight",
        pad_inches=0,
    )


if __name__ == "__main__":
    main()
