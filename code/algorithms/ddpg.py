import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.max_action * torch.tanh(self.l3(a))
        return a
    
class Critic_Q(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic_Q, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x):
        q = F.relu(self.l1(x))
        q = F.relu(self.l2(q))
        return self.l3(q)

class Critic_V(nn.Module):
    def __init__(self, state_dim):
        super(Critic_V, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x):
        q = F.relu(self.l1(x))
        q = F.relu(self.l2(q))
        return self.l3(q)

class DDPG(object):
    def __init__(self, model_dict, device):
        self.state_dim = model_dict["state_dim"]
        self.action_dim = model_dict["action_dim"]
        self.max_action = model_dict["max_action"]
        
        self.discount = model_dict["discount_factor"]
        self.tau = model_dict["tau"]
        
        self.actor_lr = model_dict["actor_lr"]
        self.critic_lr = model_dict["critic_lr"]
        self.device = device

        self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
        self.target_actor = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        
        self.critic_q = Critic_Q(self.state_dim, self.action_dim).to(self.device)
        self.target_critic_q = deepcopy(self.critic_q)
        self.critic_Q_optimizer = torch.optim.Adam(self.critic_q.parameters(), lr=self.critic_lr)
        
        self.critic_V = Critic_V(self.state_dim).to(self.device)
        self.target_critic_v = deepcopy(self.critic_V)
        self.critic_V_optimizer = torch.optim.Adam(self.critic_V.parameters(), lr=self.critic_lr)
    
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def train(self, replay_buffer, batch_size):
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        # Compute the target Q value
        target_Q = self.target_critic_q(torch.cat([next_state, self.target_actor(next_state)], dim=-1))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic_q(torch.cat([state, action], dim=-1))        

        # Compute critic loss
        critic_Q_loss = F.mse_loss(current_Q, target_Q)
        
        # Optimize the critic
        self.critic_Q_optimizer.zero_grad()
        critic_Q_loss.backward()
        self.critic_Q_optimizer.step()
        
        # get the V value
        target_V = self.target_critic_v(next_state)
        target_V = reward + (not_done * self.discount * target_V).detach()
        current_V = self.critic_V(state)
        critic_V_loss = F.mse_loss(current_V, target_V)
        
        # Optimize the V critic
        self.critic_V_optimizer.zero_grad()
        critic_V_loss.backward()
        self.critic_V_optimizer.step()
        
        
        # Compute actor loss
        actor_loss = -self.critic_q(torch.cat([state, self.actor(state)], dim=-1)).mean()
        
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # soft parameter update about target networks
        for param, target_param in zip(self.critic_q.parameters(), self.target_critic_q.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.critic_V.parameters(), self.target_critic_v.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        log_dict = {
            "actor_loss": actor_loss.item(),
            "critic_Q_loss": critic_Q_loss.item(),
            "critic_V_loss": critic_V_loss.item(),
        }

        return log_dict