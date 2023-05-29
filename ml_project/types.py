"""Common types for all scripts in the project."""

from typing import TypedDict

import numpy
from numpy.typing import NDArray
from torch import Tensor

FloatNDArray = NDArray[numpy.float_]


class Step(TypedDict):
    """Trajectory dictionary type."""

    obs: FloatNDArray
    reward: float


Trajectory = list[Step]
Trajectories = dict[int, Trajectory]
Batch = list[tuple[list[FloatNDArray], list[FloatNDArray]]]
TensorBatch = list[tuple[Tensor, Tensor]]
