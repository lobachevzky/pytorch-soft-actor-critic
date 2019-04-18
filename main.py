# -*- coding: utf-8 -*-

import argparse
import itertools
import random

import gym
import ipdb
import numpy as np
from tensorboardX import SummaryWriter
import torch

from normalized_actions import NormalizedActions
from replay_memory import ReplayMemory
from sac import SAC
from util import hard_update
from utils import space_to_size

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--env-name',
    default="HalfCheetah-v2",
    help='name of the environment to run')
parser.add_argument(
    '--policy',
    default="Gaussian",
    help='algorithm to use: Gaussian | Deterministic')
parser.add_argument(
    '--eval',
    type=bool,
    default=True,
    help='Evaluates a policy a policy every 10 episode (default:True)')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='G',
    help='discount factor for reward')
parser.add_argument(
    '--smoothing',
    type=float,
    default=0.01,
    metavar='G',
    help='target smoothing coefficient(τ)')
parser.add_argument('--clip', type=float, help='gradient clipping')
parser.add_argument(
    '--critic-lr', type=float, default=5e-4, help='critic learning rate')
parser.add_argument(
    '--actor-lr', type=float, default=5e-4, help='actor learning rate')
parser.add_argument(
    '--alpha-lr', type=float, default=0.0003, help='alpha learning rate')
parser.add_argument(
    '--alpha',
    type=float,
    default=0.01,
    metavar='G',
    help=
    'Temperature parameter α determines the relative importance of the entropy term against the reward'
)
parser.add_argument(
    '--automatic_entropy_tuning',
    type=bool,
    default=False,
    metavar='G',
    help='Temperature parameter α automaically adjusted.')
parser.add_argument(
    '--seed', type=int, default=0, metavar='N', help='random seed')
parser.add_argument(
    '--batch_size', type=int, default=256, metavar='N', help='batch size')
parser.add_argument('--no-cuda', dest='cuda', action='store_false')
parser.add_argument(
    '--num_steps',
    type=int,
    default=1e10,
    metavar='N',
    help='maximum number of steps')
parser.add_argument(
    '--hidden_size', type=int, default=300, metavar='N', help='hidden size')
parser.add_argument(
    '--updates_per_step',
    type=int,
    default=1,
    metavar='N',
    help='model updates per simulator step')
parser.add_argument(
    '--start_steps',
    type=int,
    default=10000,
    metavar='N',
    help='Steps sampling random actions')
parser.add_argument(
    '--target_update_interval',
    type=int,
    default=1,
    metavar='N',
    help='Value target update per no. of updates per step')
parser.add_argument(
    '--replay_size',
    type=int,
    default=1000000,
    metavar='N',
    help='size of replay buffer')
parser.add_argument('--logdir')
parser.add_argument('--algo', default='sac')
parser.add_argument('--tau', type=float)
parser.add_argument('--tau_', type=float)
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
env.action_space.np_random.seed(args.seed)
env.unwrapped.np_random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

writer = SummaryWriter(args.logdir)

# Agent
agent = SAC(
    space_to_size(env.observation_space),
    env.action_space,
    args,
    writer=writer)

# Memory
memory = ReplayMemory(args.replay_size)

# Training Loop
rewards = []
test_rewards = []
total_numsteps = 0
updates = 0

for i_episode in itertools.count():
    state = env.reset()

    episode_reward = 0
    while True:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)  # Sample action from policy
        next_state, reward, done, _ = env.step(action)  # Step

        mask = not done  # 1 for not done and 0 for done
        memory.push(state, action, reward, next_state,
                    mask)  # Append transition to memory

        if len(memory) > args.batch_size:
            if agent.algo == 'pmac':
                hard_update(agent.reference_policy, agent.policy)
            for i in range(args.updates_per_step
                           ):  # Number of updates per step in environment
                # Sample a batch from memory
                state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(
                    args.batch_size)
                # Update parameters of all the networks
                agent.update_parameters(state_batch, action_batch,
                                        reward_batch, next_state_batch,
                                        mask_batch, updates)

                updates += 1

        state = next_state
        total_numsteps += 1
        episode_reward += reward

        if done:
            break

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('train reward', episode_reward, i_episode)
    rewards.append(episode_reward)
    print("Episode: {}, total numsteps: {}, reward: {}, average reward: {}".
          format(i_episode, total_numsteps, np.round(rewards[-1], 2),
                 np.round(np.mean(rewards[-100:]), 2)))

    if i_episode % 10 == 0 and args.eval == True:
        state = env.reset()
        episode_reward = 0
        while True:
            action = agent.select_action(state, eval=True)

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            state = next_state
            if done:
                break

        writer.add_scalar('test reward', episode_reward, i_episode)

        test_rewards.append(episode_reward)
        print("----------------------------------------")
        print("Test Episode: {}, reward: {}".format(i_episode,
                                                    test_rewards[-1]))
        print("----------------------------------------")

env.close()
