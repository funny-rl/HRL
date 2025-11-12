from .ddpg import DDPG

import torch
import wandb 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.interpolate import CubicSpline

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
    def __init__(self, debug_mode, model_dict, device):
        super(TempoRLDDPG, self).__init__(model_dict, device)
        self.debug_mode: bool = debug_mode
        self.max_repetition = model_dict["model"]["max_repetition"]
        self.skip_q_lr = model_dict["model"]["skip_q_lr"]
        self.use_interpolation = model_dict["model"]["use_interpolation"]
        self.alpha = model_dict["model"]["alpha"]

        self.skip_Q = SKIP_Q(self.state_dim, self.action_dim, self.max_repetition).to(self.device)
        self.skip_optimizer = torch.optim.Adam(self.skip_Q.parameters(), lr= self.skip_q_lr)
        self.target_q_diffs = []

        if self.use_interpolation:
            self.step1_slopes = 0.0
            self.ewma_alpha = 0.1

    def select_skip(self, state, action):
        """
        Select the skip action.
        Has to be called after select_action
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = torch.FloatTensor(action.reshape(1, -1)).to(self.device) # 1x10x17 / 1x10x4 1x10x1

        output = self.skip_Q(torch.cat([state, action], dim=-1)).cpu().flatten()
        
        if self.use_interpolation:
            interpolation = -1 * self.alpha * self.step1_slopes * (torch.arange(0, self.max_repetition))
            output = output + torch.FloatTensor(interpolation) 
            
        return int(torch.argmax(output).item()) + 1


    def build_debug_log(self, step):
        log_dict = {}
        for idx, idx_step in enumerate(self.target_Q_estimation):
            log_dict[f"N-step Target Q estimation/{idx+1}-step"] = idx_step
        wandb.log(log_dict, step=step)
        
        log_dict = {}
        for idx, (mean, std) in enumerate(zip(self.q_bias_mean, self.q_bias_std)):
            log_dict[f"N-step Q Value Difference/{idx+1}-step"] = mean
            log_dict[f"N-step Q Value Std/{idx+1}-step"] = std
        wandb.log(log_dict, step=step)
        
    def train_skip(self, replay_buffer, batch_size):
        if self.debug_mode:
            (
                state, 
                action, 
                skip, 
                next_state, 
                reward, 
                not_done, 
                future_states, # batch x max_rep x state_dim
                discounted_rewards, # batch x max_rep x 1
                future_not_dones # batch x max_rep x 1
            ) = replay_buffer.sample(batch_size)
            
            with torch.no_grad():
                future_states = future_states.view(-1, future_states.shape[-1]) # (batch*max_rep) x state_dim
                # target_critic_Q = self.target_critic_q(
                #     torch.cat(
                #         [future_states, self.target_actor(future_states)],
                #         dim=-1
                #     )
                # )
                target_critic_Q = self.target_critic_v(
                    future_states
                )
                num_discount = torch.arange(1, self.max_repetition + 1).view(1, self.max_repetition, 1).repeat(batch_size, 1, 1).to(self.device) # batch x max_rep x 1
                target_critic_Q = target_critic_Q.view(batch_size, self.max_repetition, 1) # batch x max_rep x 1
                target_critic_Q = discounted_rewards + (future_not_dones * torch.pow(self.discount, num_discount) * target_critic_Q).detach()
                target_critic_Q = target_critic_Q.view(batch_size, self.max_repetition) # batch x max_rep

        else:
            (
                state, 
                action, 
                skip, 
                next_state, 
                reward, 
                not_done, 
            ) = replay_buffer.sample(batch_size)

        # Compute the target Q value
        # target_Q = self.target_critic_q(
        #     torch.cat(
        #         [next_state, self.target_actor(next_state)],
        #         dim=-1
        #     )
        # )
        target_Q = self.target_critic_v(
            next_state
        )
        target_Q = reward + (not_done * torch.pow(self.discount, skip+1) * target_Q).detach()
        n_stepR_Q = self.skip_Q(torch.cat([state, action], dim=-1))
        current_Q = n_stepR_Q.gather(1, skip.long())

        if self.debug_mode:
            with torch.no_grad():
                # real_target_critic_Q = target_critic_Q.gather(1, skip.long())
                # if not torch.allclose(target_Q, real_target_critic_Q):
                #     print(target_Q[-10:], real_target_critic_Q[-10:])
                # assert torch.allclose(target_Q, real_target_critic_Q), "Target Q values do not match!"
                
                self.target_Q_estimation = target_critic_Q.mean(dim=0).flatten().cpu().numpy().tolist() 
                #print(self.target_Q_estimation)
                target_q_diff = []
                for idx in range(self.max_repetition -1):
                    target_q_diff.append(self.target_Q_estimation[idx +1] - self.target_Q_estimation[idx])
                self.target_q_diffs.append(target_q_diff)

                # print(np.mean(self.target_q_diffs, axis=0))

                q_bias = (n_stepR_Q - target_critic_Q).detach().cpu() 
                self.q_bias_mean = q_bias.mean(dim=0).flatten().numpy().tolist()
                self.q_bias_std = q_bias.std(dim=0).flatten().numpy().tolist()
                                
                if self.use_interpolation:
                    x_data = np.arange(10) + 1
                    mean_n_stepR_Q = target_critic_Q.mean(dim=0).flatten().cpu().numpy()
                    spline_function = CubicSpline(x_data, mean_n_stepR_Q)
                    self.step1_slopes = self.ewma_alpha * self.step1_slopes + (1 - self.ewma_alpha) * spline_function(1, 1)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        
        # Optimize the critic
        self.skip_optimizer.zero_grad()
        critic_loss.backward()
        self.skip_optimizer.step()

        log_dict = {
            "rep_critic_loss": critic_loss.cpu().item()
        }

        # with torch.no_grad():
        #     skip_step_rate_flat = skip_step_rate.cpu().flatten()
        #     logging_step = [0.50, 0.60, 0.70, 0.80, 0.90] # time step rates to log
        #     target_Q_values = {}
        #     for _step in logging_step:
        #         if _step in skip_step_rate_flat:
        #             mask = (skip_step_rate_flat == _step)
        #             target_Q_values[f"stepR{_step}_undisountedQ"] = target_Q_func[mask].mean().item()
        #     log_dict = log_dict | target_Q_values
            
        # with torch.no_grad():
        #     skip_step = (skip.cpu().flatten()).int() + 1
        #     skip_step_rates = skip_step_rate.cpu().flatten()
        #     logging_skip = [1, 3, 5] # skips to log
        #     logging_steps = [0.1, 0.5, 0.8]
        #     unique_skip_step = skip_step.unique()
        #     target_values = {}
        #     for _skip in logging_skip:
        #         if _skip in unique_skip_step:
        #             mask = (skip_step == _skip)
        #             for log_step in logging_steps:
        #                 step_mask = (skip_step_rates == log_step)
        #                 if torch.sum(mask & step_mask) > 0: # avoid empty mask
        #                     target_values[f"stepR{log_step}_skip{_skip}_target_Q"] = target_Q[mask & step_mask].mean().item()
        #     log_dict = log_dict | target_values
        
        # with torch.no_grad():
        #     for rep in range(self.max_repetition):
        #         log_dict[f"rep_{rep}_n_stepR_Q"] = mean_n_stepR_Q[rep]
            
        return log_dict