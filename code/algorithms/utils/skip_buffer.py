from .vanilla_buffer import ReplayBuffer

import torch
import numpy as np

class SkipReplayBuffer(ReplayBuffer):
    def __init__(self, state_dim, action_dim, buffer_size, debug_mode, max_repetition, device):
        super(SkipReplayBuffer, self).__init__(state_dim, action_dim, buffer_size, device)
        self.debug_mode: bool = debug_mode
        self.max_repetition = max_repetition

        self.rep = np.zeros((self.max_size, 1))
        self.future_states = np.zeros((self.max_size, self.max_repetition, state_dim))
        self.discounted_rewards = np.zeros((self.max_size, self.max_repetition, 1))
        self.future_not_dones = np.zeros((self.max_size, self.max_repetition, 1))
        
    def add(
        self, 
        state, 
        action, 
        rep, 
        next_state, 
        reward, 
        done, 
        future_states = None,
        discounted_rewards = None,
        future_dones = None
    ):
        
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.rep[self.ptr] = rep
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        
        if self.debug_mode:
            self.future_states[self.ptr] = future_states
            self.discounted_rewards[self.ptr] = discounted_rewards
            self.future_not_dones[self.ptr] = 1 - future_dones

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        if self.debug_mode:
            return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.rep[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device),
                torch.FloatTensor(self.future_states[ind]).to(self.device),
                torch.FloatTensor(self.discounted_rewards[ind]).to(self.device),
                torch.FloatTensor(self.future_not_dones[ind]).to(self.device),
            )
        else:
            return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.rep[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device),
            )
