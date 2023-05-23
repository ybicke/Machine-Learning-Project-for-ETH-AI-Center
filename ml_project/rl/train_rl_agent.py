"""Module for training an RL agent."""
# ruff: noqa: E402
# pylint: disable=wrong-import-position

import os
from os import path
from pathlib import Path

os.add_dll_directory(path.join(Path.home(), ".mujoco", "mjpro150", "bin"))

import gym
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.sac.sac import SAC

ALGORITHM = "ppo"
ENVIRONMENT_NAME = "HalfCheetah-v3"

script_path = Path(__file__).parent.resolve()
models_path = path.join(script_path, "models")

env = gym.make(ENVIRONMENT_NAME)


def main():
    """Run RL agent training."""
    if ALGORITHM == "sac":
        model = SAC("MlpPolicy", env, verbose=1)
    elif ALGORITHM == "ppo":
        model = PPO("MlpPolicy", env, verbose=1)
    else:
        raise NotImplementedError(f"{ALGORITHM} not implemented")

    iterations = 5

    # STEPS_PER_ITER = 60000
    steps_per_iteration = 5000

    for i in range(iterations):
        model.learn(
            total_timesteps=steps_per_iteration,
            reset_num_timesteps=False,
            log_interval=4,
        )

        model.save(
            path.join(
                models_path,
                f"{ALGORITHM}_{ENVIRONMENT_NAME}_{steps_per_iteration*i}",
            )
        )


if __name__ == "__main__":
    main()
