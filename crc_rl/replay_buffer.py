# Simple Replay Buffer for CRC-RL
import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, obs_shape, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.obs_buf = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.action_buf = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.done_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr, self.size = 0, 0
    def add(self, obs, action, reward, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=torch.tensor(self.obs_buf[idxs], device=self.device),
            action=torch.tensor(self.action_buf[idxs], device=self.device),
            reward=torch.tensor(self.reward_buf[idxs], device=self.device),
            next_obs=torch.tensor(self.next_obs_buf[idxs], device=self.device),
            done=torch.tensor(self.done_buf[idxs], device=self.device)
        )
        return batch
