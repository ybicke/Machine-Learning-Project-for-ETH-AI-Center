"""Module for training an RL agent and saving videos and data of its trajectories."""
import os
import pickle

import gymnasium as gym
from stable_baselines3 import PPO, SAC

algo = "ppo"
env_name = "HalfCheetah-v4"

record_video_interval = 500
video_length = 100
n_videos_per_model_checkpoint = 5

env = gym.make(env_name, render_mode="rgb_array")
env = gym.wrappers.RecordVideo(
    env,
    video_folder="videos",
    step_trigger=lambda n: n % record_video_interval == 0,
    video_length=video_length,
    name_prefix=f"{algo}-{env_name}",
    disable_logger=True,
)

if algo == "sac":
    ALGO = SAC
elif algo == "ppo":
    ALGO = PPO
else:
    raise NotImplementedError(f"{algo} not implemented")

trajectory_dataset = {}
trajectory = []

n_step = 0
for n_checkpoint, file in enumerate(os.listdir("models")):
    if file.startswith(algo + "_" + env_name):
        file_name = f"models/{file[:-4]}"
        model = ALGO.load(file_name)

        obs, info = env.reset()
        while (
            n_step
            < (n_checkpoint + 1) * n_videos_per_model_checkpoint * record_video_interval
        ):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            i = n_step % record_video_interval
            if i < video_length:
                if i == 0:
                    trajectory = []

                transition = {"obs": obs, "reward": reward}
                trajectory.append(transition)

                if i == video_length - 1:
                    trajectory_dataset[n_step - video_length + 1] = trajectory

            if terminated or truncated:
                obs, info = env.reset()
            n_step += 1

with open(f"reward_data/{algo}_{env_name}_obs_reward_dataset.pkl", "wb") as handle:
    pickle.dump(trajectory_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
