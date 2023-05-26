"""Module for instantiating a neural network."""
# pylint: disable=arguments-differ
from typing import Type, Union

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn


class LightningNetwork(LightningModule):
    """Neural network to model the RL agent's reward using Pytorch Lightning."""

    def __init__(
        self,
        layer_num: int,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        activation_function: Type[nn.Module] = nn.ReLU,
        last_activation: Union[Type[nn.Module], None] = None,
    ):
        super().__init__()

        # Initialize the network
        layers_unit = [input_dim] + [hidden_dim] * (layer_num - 1)

        layers: list[nn.Module] = []

        for idx in range(len(layers_unit) - 1):
            layers.append(nn.Linear(layers_unit[idx], layers_unit[idx + 1]))
            layers.append(activation_function())

        layers.append(nn.Linear(layers_unit[-1], output_dim))

        if last_activation is not None:
            layers.append(last_activation())

        self.network = nn.Sequential(*layers)

        # Initialize the weights
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)

                layer.bias.data.zero_()
                layer.bias.data.zero_()

    def forward(self, batch: Tensor, _batch_idx: int):
        """Do a forward pass through the neural network (inference)."""
        batch = self.network(batch)
        return batch

    def training_step(self, batch: Tensor, _batch_idx: int):
        """Compute the loss for training."""
        return self._calculate_loss(batch)

    def validation_step(self, batch: Tensor, _batch_idx: int):
        """Compute the loss for validation."""
        loss = self._calculate_loss(batch)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        """Configure optimizer to optimize the neural network."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _calculate_loss(self, batch: Tensor):
        rewards0 = self.network(batch[0]).sum(dim=1)
        rewards1 = self.network(batch[1]).sum(dim=1)

        probs_softmax = torch.exp(rewards0) / (
            torch.exp(rewards0) + torch.exp(rewards1)
        )

        loss = -torch.sum(torch.log(probs_softmax))

        return loss


class Network(nn.Module):
    """Neural network to model the RL agent's reward."""

    def __init__(
        self,
        layer_num,
        input_dim,
        output_dim,
        hidden_dim,
        activation_function=torch.relu,
        last_activation=None,
    ):
        super().__init__()
        self.activation = activation_function
        self.last_activation = last_activation
        layers_unit = [input_dim] + [hidden_dim] * (layer_num - 1)
        layers = [
            nn.Linear(layers_unit[idx], layers_unit[idx + 1])
            for idx in range(len(layers_unit) - 1)
        ]
        self.layers = nn.ModuleList(layers)
        self.last_layer = nn.Linear(layers_unit[-1], output_dim)
        self.network_init()

    def forward(self, data: Tensor):
        """Do a forward pass through the neural network (inference)."""
        return self._forward(data)

    def _forward(self, data: Tensor):
        for layer in self.layers:
            data = self.activation(layer(data))
        data = self.last_layer(data)
        if self.last_activation is not None:
            data = self.last_activation(data)
        return data

    def network_init(self):
        """Initialize the neural network."""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()
