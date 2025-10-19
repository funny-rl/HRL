from .ddpg import DDPG

import torch
import torch.nn as nn
import torch.nn.functional as F

class SKIP_Q(nn.Module):
    """
    Simple fully connected Q function. Also used for skip-Q when concatenating behaviour action and state together.
    Used for simpler environments such as mountain-car or lunar-lander.
    """

    def __init__(self, state_dim, action_dim, skip_dim):
        super(SKIP_Q, self).__init__()
        # We follow the architecture of the Actor and Critic networks in terms of depth and hidden units
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, skip_dim)
        self._non_linearity = F.relu

    def forward(self, x):
        x = self._non_linearity(self.fc1(x))
        x = self._non_linearity(self.fc2(x))
        return self.fc3(x)


class TempoRLDDPG(DDPG):
    def __init__(self, model_dict, device):
        super(TempoRLDDPG, self).__init__(model_dict, device)
        self.repetition = model_dict["model"]["max_repetition"]
        self.skip_q_lr = model_dict["model"]["skip_q_lr"]
        
        self.skip_Q = SKIP_Q(self.state_dim, self.action_dim, self.repetition).to(self.device)
        self.skip_optimizer = torch.optim.Adam(self.skip_Q.parameters(), lr= self.skip_q_lr)
        
    def select_skip(self, state, action):
        """
        Select the skip action.
        Has to be called after select_action
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = torch.FloatTensor(action.reshape(1, -1)).to(self.device)
        return self.skip_Q(torch.cat([state, action], 1)).cpu().data.numpy().flatten()
    
    def train_skip(self, replay_buffer, batch_size):
        state, action, skip, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        # Compute the target Q value
        target_Q = self.target_critic(next_state, self.target_actor(next_state))
        target_Q = reward + (not_done * torch.pow(self.discount, skip + 1) * target_Q).detach()

        # Get current Q estimate
        current_Q = self.skip_Q(torch.cat([state, action], 1)).gather(1, skip.long())
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        
        # Optimize the critic
        self.skip_optimizer.zero_grad()
        critic_loss.backward()
        self.skip_optimizer.step()
