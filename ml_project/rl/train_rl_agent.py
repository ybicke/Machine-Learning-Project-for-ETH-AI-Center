"""Module for training an RL agent."""
# ruff: noqa: E402
# pylint: disable=wrong-import-position
import os
from os import path
from pathlib import Path

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.sac.sac import SAC

ALGORITHM = "ppo"  # "ppo" or "sac"
ENVIRONMENT_NAME = "HalfCheetah-v3"

script_path = Path(__file__).parent.resolve()
models_path = path.join(script_path, "models")
tensorboard_path = path.join(script_path, "..", "..", "rl_logs")


def main():
    """Run RL agent training."""
    cpu_count = os.cpu_count()
    cpu_count = cpu_count if cpu_count is not None else 8

    env = make_vec_env(ENVIRONMENT_NAME, n_envs=cpu_count)

    if ALGORITHM == "sac":
        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_path)
    elif ALGORITHM == "ppo":
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_path)
    else:
        raise NotImplementedError(f"{ALGORITHM} not implemented")

    iterations = 10

    steps_per_iteration = 250000

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
