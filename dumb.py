#! /usr/bin/env python
import gym

env = gym.make('Pendulum-v0')
env.seed(0)
print(env.action_space.sample())
