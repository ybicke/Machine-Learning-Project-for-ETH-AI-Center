"""Module for saving videos and data of an RL agent's trajectories."""
import os
import pickle
import re
from os import path
from pathlib import Path
from typing import Type, Union

import gym
import numpy as np
from gym.wrappers import RecordVideo
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.sac.sac import SAC

from ..types import Obs, RewardlessTrajectories

ALGORITHM = "sac"
ENVIRONMENT_NAME = "HalfCheetah-v3"
USE_REWARD_MODEL = False
USE_SDE = False

model_id = f"{ALGORITHM}_{ENVIRONMENT_NAME}"
model_id += "_sde" if USE_SDE else ""
model_id += "_finetuned" if USE_REWARD_MODEL else ""

RECORD_INTERVAL = 500
RECORD_LENGTH = 100
VIDEOS_PER_CHECKPOINT = 2

script_path = Path(__file__).parent.resolve()
models_path = path.join(script_path, "models_final")


def record_videos(
    algorithm: Union[Type[PPO], Type[SAC]], environment: gym.wrappers.RecordVideo
):
    """Record videos of the training environment."""
    obs_dataset: RewardlessTrajectories = {}
    observations: list[Obs] = []

    n_step = 0

    for _, file in enumerate(os.listdir(models_path)):
        if re.search(f"{model_id}_[0-9]", file):
            model = algorithm.load(path.join(models_path, file[:-4]))

            state = environment.reset()
            while n_step < VIDEOS_PER_CHECKPOINT * RECORD_INTERVAL:
                action, _states = model.predict(state, deterministic=True)
                next_state, _reward, terminated, _info = environment.step(action)

                environment.render(mode="rgb_array")

                i = n_step % RECORD_INTERVAL
                if i < RECORD_LENGTH:
                    if i == 0:
                        observations = []

                    observation = np.concatenate((state, action, next_state))
                    observations.append(observation)

                    if i == RECORD_LENGTH - 1:
                        obs_dataset[n_step - RECORD_LENGTH + 1] = observations

                    if terminated:
                        next_state = environment.reset()

                state = next_state
                n_step += 1

    return obs_dataset


def main():
    """Run video generation."""
    env = gym.make(ENVIRONMENT_NAME)
    env = RecordVideo(
        env,
        video_folder=path.join(script_path, "..", "static", "videos"),
        step_trigger=lambda n: n % RECORD_INTERVAL == 0,
        video_length=RECORD_LENGTH,
        name_prefix=f"{model_id}",
    )

    if ALGORITHM == "sac":
        ALGO = SAC
    elif ALGORITHM == "ppo":
        ALGO = PPO
    else:
        raise NotImplementedError(f"{ALGORITHM} not implemented")

    obs_dataset = record_videos(ALGO, env)

    with open(
        path.join(
            script_path,
            "observation_data_corresponding_to_videos",
            f"{model_id}_obs_dataset.pkl",
        ),
        "wb",
    ) as handle:
        pickle.dump(obs_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
