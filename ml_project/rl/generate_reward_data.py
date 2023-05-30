"""Module for saving data of an RL agent's trajectories after training."""
import os
import pickle
from os import path
from pathlib import Path
from typing import Type, Union

import gym
import numpy as np
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.sac.sac import SAC

from ..types import Trajectory

ALGORITHM = "ppo"
ENVIRONMENT_NAME = "HalfCheetah-v3"

STEPS_PER_CHECKPOINT = 10000

script_path = Path(__file__).parent.resolve()
models_path = path.join(script_path, "models")


def generate_data(algorithm: Union[Type[PPO], Type[SAC]], environment: gym.Env):
    """Generate agent's observations and rewards in the training environment."""
    data: Trajectory = []

    for file in os.listdir(models_path):
        if file.startswith(ALGORITHM + "_" + ENVIRONMENT_NAME):
            model = algorithm.load(path.join(models_path, file[:-4]))

            n_step = 0
            state = environment.reset()
            while n_step < STEPS_PER_CHECKPOINT:
                action, _states = model.predict(state, deterministic=True)
                next_state, reward, terminated, _info = environment.step(action)

                observation = np.concatenate((state, action, next_state))
                data.append({"obs": observation, "reward": reward})

                if terminated:
                    next_state = environment.reset()

                state = next_state
                n_step += 1

    return data


def main():
    """Run data generation."""
    env = gym.make(ENVIRONMENT_NAME)

    if ALGORITHM == "sac":
        ALGO = SAC
    elif ALGORITHM == "ppo":
        ALGO = PPO
    else:
        raise NotImplementedError(f"{ALGORITHM} not implemented")

    obs_reward_data = generate_data(ALGO, env)

    with open(
        path.join(
            script_path,
            "reward_data",
            f"{ALGORITHM}_{ENVIRONMENT_NAME}_obs_reward_dataset.pkl",
        ),
        "wb",
    ) as handle:
        pickle.dump(obs_reward_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
