import numpy as np
import random
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf
from collections import namedtuple, deque
from td3_model import Actor, Critic

BUFFER_SIZE = int(5e4)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-2        # learning rate of the critic
WEIGHT_DECAY = 1e-4     # L2 weight decay / 0.0001

# BUFFER_SIZE = int(5e5)  # replay buffer size
# BATCH_SIZE = 128        # minibatch size
# GAMMA = 0.99            # discount factor
# TAU = 5e-3              # for soft update of target parameters
# LR_ACTOR = 1e-4         # learning rate of the actor 
# LR_CRITIC = 1e-3        # learning rate of the critic
# WEIGHT_DECAY = 0.0001   # L2 weight decay
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == 'cpu':
    print('GPU is not available')
print('Using device:', device)
print()
# Additional Info when using cuda
if device.type == 'cuda:0':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

devices = torch.cuda.device_count()
print(f'{devices} Number of Devices Exists')

class Agent:
    def __init__(self, state_dim, action_dim, seed):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seed = random.seed(seed)

        self.actor_local = Actor(state_dim, action_dim, seed).to(device)
        self.actor_target = Actor(state_dim, action_dim, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        self.critic1_local = Critic(state_dim, action_dim, seed).to(device)
        self.critic1_target = Critic(state_dim, action_dim, seed).to(device)
        self.critic1_optimizer = optim.Adam(self.critic1_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.critic2_local = Critic(state_dim, action_dim, seed).to(device)
        self.critic2_target = Critic(state_dim, action_dim, seed).to(device)
        self.critic2_optimizer = optim.Adam(self.critic2_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.noise = OUNoise(action_dim, seed)
        self.memory = ReplayBuffer(action_dim, BUFFER_SIZE, BATCH_SIZE, seed)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

def choose_action(self, observation):
    if self.time_step < self.warmup:
        mu = np.random.normal(scale=self.noise, size=(self.n_actions,))
    else:
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        # returns a batch size of 1, want a scalar array
        mu = self.actor(state)[0]
    mu_prime = mu + np.random.normal(scale=self.noise)
    mu_prime = tf.clip_by_value(mu_prime, self.min_action, self.max_action)
    self.time_step += 1  # inicializácia a zvýšenie časového kroku
    return mu_prime


def learn(self, experiences, discount):
    states, actions, rewards, next_states, dones = experiences

    # Sample a mini-batch from the replay buffer
    batch_size = len(states)
    indices = np.random.choice(len(self.memory), size=batch_size, replace=False)
    states = states[indices]
    actions = actions[indices]
    rewards = rewards[indices]
    next_states = next_states[indices]
    dones = dones[indices]

    # Compute critic loss
    next_actions = self.actor_target(next_states)
    noise = torch.randn_like(next_actions) * 0.2
    noise = torch.clamp(noise, -0.5, 0.5)
    next_actions = (next_actions + noise).clamp(-1, 1)

    Q_targets_next1 = self.critic1_target(next_states, next_actions)
    Q_targets_next2 = self.critic2_target(next_states, next_actions)
    Q_targets_next = torch.min(Q_targets_next1, Q_targets_next2)
    Q_targets = rewards + (discount * Q_targets_next * (1 - dones))

    Q_expected1 = self.critic1_local(states, actions)
    Q_expected2 = self.critic2_local(states, actions)

    critic_loss1 = F.mse_loss(Q_expected1, Q_targets.detach())
    critic_loss2 = F.mse_loss(Q_expected2, Q_targets.detach())

    self.critic1_optimizer.zero_grad()
    self.critic2_optimizer.zero_grad()
    critic_loss1.backward()
    critic_loss2.backward()
    self.critic1_optimizer.step()
    self.critic2_optimizer.step()

    # Delayed policy updates
    if len(self.memory) % 2 == 0:
        actor_loss = -self.critic1_local(states, self.actor_local(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic1_local, self.critic1_target, TAU)
        self.soft_update(self.critic2_local, self.critic2_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)


def soft_update(self, local_model, target_model, tau):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class OUNoise:
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)