import gymnasium as gym
from stable_baselines3 import PPO, SAC

algo = "ppo"
env_name = "HalfCheetah-v4"

env = gym.make(env_name)

if algo == "sac":
    model = SAC("MlpPolicy", env, verbose=1)
elif algo == "ppo":
    model = PPO("MlpPolicy", env, verbose=1)
else:
    raise NotImplementedError(f"{algo} not implemented")

ITER = 5
# STEPS_PER_ITER = 60000
STEPS_PER_ITER = 5000

for i in range(ITER):
    model.learn(
        total_timesteps=STEPS_PER_ITER, reset_num_timesteps=False, log_interval=4
    )
    model.save(f"models/{algo}_{env_name}_{STEPS_PER_ITER*i}")
