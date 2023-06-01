"""Module for saving data of an RL agent's trajectories after training."""
import os
import pickle
import re
from os import path
from pathlib import Path
from typing import Type, Union

import gym
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.sac.sac import SAC

ALGORITHM = "sac"  # "ppo" or "sac"
ENVIRONMENT_NAME = "HalfCheetah-v3"
USE_REWARD_MODEL = False
USE_SDE = True

model_id = f"{ALGORITHM}_{ENVIRONMENT_NAME}"
model_id += "_sde" if USE_SDE else ""
model_id += "_finetuned" if USE_REWARD_MODEL else ""

STEPS_PER_CHECKPOINT = 10000

script_path = Path(__file__).parent.resolve()
models_path = path.join(script_path, "models_final")


def generate_data(algorithm: Union[Type[PPO], Type[SAC]], environment: gym.Env):
    """Generate agent's observations and rewards in the training environment."""
    data = []
    model_count = 0

    for file in os.listdir(models_path):
        if re.search(f"{model_id}_[0-9]", file):
            model = algorithm.load(path.join(models_path, file[:-4]))

            obs = environment.reset()
            for _ in range(STEPS_PER_CHECKPOINT):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, _info = environment.step(action)

                data.append({"obs": obs, "reward": reward})

                if terminated:
                    obs = environment.reset()

            model_count += 1
            print(f"Model #{model_count}")

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
            "reward_data_final",
            f"{model_id}_reward_dataset.pkl",
        ),
        "wb",
    ) as handle:
        pickle.dump(obs_reward_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
