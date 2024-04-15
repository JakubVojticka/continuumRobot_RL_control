import sys
import time
sys.path.append('../Pytorch')

import numpy as np
import random
import copy
from collections import namedtuple, deque

import tensorflow as tf
import torch
import torch.nn.functional as F
import torch.optim as optim

from ddpg_agent import Agent

# Create a DDPG instance
agent = Agent(state_dim, action_dim)

# Train the agent for max_episodes
for i in range(max_episode):
    total_reward = 0
    step =0
    state = env.reset()
    for  t in range(max_time_steps):
        action = agent.select_action(state)
        # Add Gaussian noise to actions for exploration
        action = (action + np.random.normal(0, 1, size=action_dim)).clip(-max_action, max_action)
        #action += ou_noise.sample()
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        if render and i >= render_interval : env.render()
        agent.replay_buffer.push((state, next_state, action, reward, np.float(done)))
        state = next_state
        if done:
            break
        step += 1
        
    score_hist.append(total_reward)
    total_step += step+1
    print("Episode: \t{}  Total Reward: \t{:0.2f}".format( i, total_reward))
    agent.update()
    if i % 10 == 0:
        agent.save()
env.close()