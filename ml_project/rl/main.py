"""Module for training an RL agent on a gymnasium environment"""
import os
from argparse import ArgumentParser
from configparser import ConfigParser
from os import path
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from agents.ddpg import DDPG
from agents.ppo import PPO
from agents.sac import SAC
from utils.utils import Dict, RunningMeanStd, make_transition

current_path = Path(__file__).parent.resolve()
model_path = path.join(current_path, "model_weights")
config_path = path.join(current_path, "config.ini")

os.makedirs(model_path, exist_ok=True)

parser = ArgumentParser("parameters")

parser.add_argument(
    "--env_name",
    type=str,
    default="HalfCheetah-v4",
    help="'Ant-v4','HalfCheetah-v4','Hopper-v4','Humanoid-v4','HumanoidStandup-v4',\
          'InvertedDoublePendulum-v4', 'InvertedPendulum-v4' (default : Hopper-v4)",
)
parser.add_argument(
    "--algo", type=str, default="ppo", help="algorithm to adjust (default : ppo)"
)
parser.add_argument("--train", type=bool, default=True, help="(default: True)")
parser.add_argument("--render", type=bool, default=False, help="(default: False)")
parser.add_argument(
    "--epochs", type=int, default=1000, help="number of epochs, (default: 1000)"
)
parser.add_argument(
    "--tensorboard", type=bool, default=False, help="use_tensorboard, (default: False)"
)
parser.add_argument(
    "--load", type=str, default="no", help="load network name in ./model_weights"
)
parser.add_argument(
    "--save_interval", type=int, default=100, help="save interval(default: 100)"
)
parser.add_argument(
    "--print_interval", type=int, default=1, help="print interval(default : 20)"
)
parser.add_argument(
    "--use_cuda", type=bool, default=True, help="cuda usage(default : True)"
)
parser.add_argument(
    "--reward_scaling", type=float, default=0.1, help="reward scaling(default : 0.1)"
)
args = parser.parse_args()
parser = ConfigParser()
parser.read(config_path)
agent_args = Dict(parser, args.algo)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if not args.use_cuda:
    DEVICE = "cpu"

if args.tensorboard:
    from torch.utils.tensorboard.writer import SummaryWriter

    WRITER = SummaryWriter()
else:
    WRITER = None

if args.render:
    env = gym.make(args.env_name, render_mode="human")
else:
    env = gym.make(args.env_name)

action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
state_rms = RunningMeanStd(state_dim)


if args.algo == "ppo":
    agent = PPO(WRITER, DEVICE, state_dim, action_dim, agent_args)
elif args.algo == "sac":
    agent = SAC(WRITER, DEVICE, state_dim, action_dim, agent_args)
elif args.algo == "ddpg":
    from utils.noise import OUNoise

    noise = OUNoise(action_dim, 0)
    agent = DDPG(WRITER, DEVICE, state_dim, action_dim, agent_args, noise)


if (torch.cuda.is_available()) and (args.use_cuda):
    agent = agent.cuda()

if args.load != "no":
    agent.load_state_dict(torch.load("./model_weights/" + args.load))

score_lst = []
state_lst = []

if agent_args.on_policy:
    score = 0.0
    state_ = env.reset()[0]
    state = np.clip((state_ - state_rms.mean) / (state_rms.var**0.5 + 1e-8), -5, 5)
    for n_epi in range(args.epochs):
        for t in range(agent_args.traj_length):
            if args.render:
                env.render()
            state_lst.append(state_)
            mu, sigma = agent.get_action(torch.from_numpy(state).float().to(DEVICE))
            dist = torch.distributions.Normal(mu, sigma[0])
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            next_state_, reward, terminated, truncated, info = env.step(
                action.cpu().numpy()
            )
            done = terminated or truncated
            next_state = np.clip(
                (next_state_ - state_rms.mean) / (state_rms.var**0.5 + 1e-8), -5, 5
            )
            transition = make_transition(
                state,
                action.cpu().numpy(),
                np.array([reward * args.reward_scaling]),
                next_state,
                np.array([done]),
                log_prob.detach().cpu().numpy(),
            )
            agent.put_data(transition)
            score += reward
            if done:
                state_ = env.reset()[0]
                state = np.clip(
                    (state_ - state_rms.mean) / (state_rms.var**0.5 + 1e-8), -5, 5
                )
                score_lst.append(score)
                if args.tensorboard:
                    WRITER.add_scalar("score/score", score, n_epi)
                score = 0
            else:
                state = next_state
                state_ = next_state_

        agent.train_net(n_epi)
        state_rms.update(np.vstack(state_lst))
        if n_epi % args.print_interval == 0 and n_epi != 0:
            print(
                "# of episode :{}, avg score : {:.1f}".format(
                    n_epi, sum(score_lst) / len(score_lst)
                )
            )
            score_lst = []
        if n_epi % args.save_interval == 0 and n_epi != 0:
            torch.save(agent.state_dict(), "./model_weights/agent_" + str(n_epi))

else:  # off policy
    for n_epi in range(args.epochs):
        score = 0.0
        state = env.reset()[0]
        done = False
        while not done:
            if args.render:
                env.render()
            action, _ = agent.get_action(torch.from_numpy(state).float().to(DEVICE))
            action = action.cpu().detach().numpy().flatten()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            transition = make_transition(
                state,
                action,
                np.array([reward * args.reward_scaling]),
                next_state,
                np.array([done]),
            )
            agent.put_data(transition)

            state = next_state

            score += reward
            if agent.data.data_idx > agent_args.learn_start_size:
                agent.train_net(agent_args.batch_size, n_epi)
        score_lst.append(score)
        if args.tensorboard:
            WRITER.add_scalar("score/score", score, n_epi)
        if n_epi % args.print_interval == 0 and n_epi != 0:
            print(
                "# of episode :{}, avg score : {:.1f}".format(
                    n_epi, sum(score_lst) / len(score_lst)
                )
            )
            score_lst = []
        if n_epi % args.save_interval == 0 and n_epi != 0:
            torch.save(agent.state_dict(), "./model_weights/agent_" + str(n_epi))
            torch.save(agent.state_dict(), "./model_weights/agent_" + str(n_epi))
