from .ddpg import DDPG

import torch
import torch.nn as nn
import torch.nn.functional as F

class SKIP_Q(nn.Module):

    def __init__(self, state_dim, action_dim, rep_dim=1):
        super(SKIP_Q, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim + rep_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)
        self._non_linearity = F.relu

    def forward(self, x):
        x = self._non_linearity(self.fc1(x))
        x = self._non_linearity(self.fc2(x))
        return self.fc3(x)

class SKIP_actor(nn.Module):
    def __init__(self, state_dim, action_dim, output_dim):
        super(SKIP_actor, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, output_dim)
        self._non_linearity = F.relu
        
    def forward(self, x):
        x = self._non_linearity(self.fc1(x))
        x = self._non_linearity(self.fc2(x))
        return self.fc3(x)

class NewDDPG(DDPG):
    def __init__(self, model_dict, device):
        super(NewDDPG, self).__init__(model_dict, device)
        self.max_repetition = model_dict["model"]["max_repetition"]
        self.rep_dist = model_dict["model"]["rep_dist"]
        self.softmax_tau = model_dict["model"]["softmax_tau"]
        self.scale_coef = model_dict["model"]["scale_coef"]

        if self.rep_dist == "poisson":
            self.skip_Actor = SKIP_actor(self.state_dim, self.action_dim, output_dim = 1).to(self.device)
        self.skip_actor_optimizer = torch.optim.Adam(self.skip_Actor.parameters(), lr= model_dict["model"]["skip_actor_lr"])

        self.skip_Q = SKIP_Q(self.state_dim, self.action_dim).to(self.device)
        self.skip_q_optimizer = torch.optim.Adam(self.skip_Q.parameters(), lr= model_dict["model"]["skip_q_lr"])
        
    def  truncated_poisson_pmf(self, lam, N):
        """
        ks: Tensor of shape (batch_size, N) containing integers from 0 to N-1
        """
        ks = torch.arange(0, N, device=lam.device, dtype=lam.dtype)
        log_p = ks * torch.log(lam) - lam - torch.lgamma(ks + 1)
        p = torch.exp(log_p / self.softmax_tau)
        Z = p.sum(dim=-1, keepdim=True)
        p_trunc = p / Z
        return p_trunc
        
    def select_rep_k(self, state, action, deterministic=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = torch.FloatTensor(action.reshape(1, -1)).to(self.device)
        if self.rep_dist == "poisson":
            lam = self.skip_Actor(
                torch.cat(
                    [state, action]
                    , 1
                )
            )
            lam = self.scale_coef  * F.softplus(lam) + 1e-5
            p_trunc = self.truncated_poisson_pmf(lam = lam, N = self.max_repetition)
            if deterministic:
                return int(torch.argmax(p_trunc).cpu().item()) + 1
            else:
                sampled_idx = int(torch.multinomial(p_trunc, num_samples=1).item())
                return sampled_idx + 1
            
    def train_skip_Q(self, replay_buffer, batch_size):
        state, action, skip, next_state, reward, not_done = replay_buffer.sample(batch_size)
        with torch.no_grad():
            target_Q = self.target_critic(next_state, self.target_actor(next_state))
            target_Q = reward + (not_done * torch.pow(self.discount, skip + 1) * target_Q).detach()
        behavior_Q = self.skip_Q(torch.cat([state, action, skip / self.max_repetition], 1))
        critic_loss = F.mse_loss(behavior_Q, target_Q)
        self.skip_q_optimizer.zero_grad()
        critic_loss.backward()
        self.skip_q_optimizer.step()

        log_dict = {
            "rep_critic_loss": critic_loss.cpu().item()
        }
        
        with torch.no_grad():
            skip_flat = skip.flatten() + 1.0 # +1 for repetition
            loss_flat = target_Q # pos: overestimate / neg: underestimate
            skip_loss_mean = {}
            for rep in skip_flat:
                mask = (skip_flat == rep)
                skip_loss_mean[f"{int(rep)}_skip_Q_loss"] = loss_flat[mask].mean().item()
            log_dict = log_dict | skip_loss_mean
        
        return log_dict
    
    def train_skip_actor(self, replay_buffer, batch_size):
        # 1. Sample batch
        state, action, skip, next_state, reward, not_done = replay_buffer.sample(batch_size)

        if self.rep_dist == "poisson":
            # 2. Actor forward
            lam = self.skip_Actor(torch.cat([state, action], dim=1))
            lam = self.scale_coef * F.softplus(lam) + 1e-5

            # 3. Truncated Poisson probability
            p_trunc = self.truncated_poisson_pmf(lam=lam, N=self.max_repetition)

            # 4. Gather probabilities of executed skips
            skip_p = p_trunc.gather(dim=-1, index=skip.long())
            log_skip_p = torch.log(skip_p + 1e-10)

            rep_adv_list: list[float] = []

            # 5. Compute behavior_Q and behavior_V safely with no gradient
            
            with torch.no_grad():
                behavior_Q = self.skip_Q(torch.cat([state, action, skip / self.max_repetition], dim=1))
                behavior_V = torch.zeros_like(behavior_Q)
                batch_size = state.size(0)
                # Precompute all repetitions
                for num_rep in range(1, self.max_repetition + 1):
                    num_rep_tensor = torch.full((batch_size, 1), num_rep, dtype=torch.float32, device=self.device)
                    Q_per_rep = self.skip_Q(torch.cat([state, action, num_rep_tensor / self.max_repetition], dim=1))
                    # detach p_trunc to prevent gradient flow
                    rep_idx = num_rep - 1
                    prob_rep = p_trunc[:, rep_idx].unsqueeze(-1).detach()
                    behavior_V += prob_rep * Q_per_rep
                    rep_adv_list.append((Q_per_rep - behavior_V).mean(dim=0, keepdim=False).item())

            # 6. Advantage and actor loss
            behavior_A = behavior_Q - behavior_V

            actor_loss = -(log_skip_p * behavior_A.detach()).mean()

            # 7. Backprop only for actor
            self.skip_actor_optimizer.zero_grad()
            actor_loss.backward()
            self.skip_actor_optimizer.step()
            
            log_dict = {
                "rep_actor_loss" : actor_loss.cpu().item(),
                "max_rep_Adv" : rep_adv_list[-1]
            }
            
            return log_dict
                
                