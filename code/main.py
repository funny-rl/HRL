import pdb
import hydra
import torch
import numpy as np

from pprint import pprint

import gymnasium as gym
import gymnasium_robotics
from gymnasium import spaces
gym.register_envs(gymnasium_robotics)

from omegaconf import OmegaConf

from utils import set_seed, state_transform, store_future_states
from algorithms import ALGO_REGISTRY
from algorithms.utils import BUFFER_REGISTRY    



    

def eval(policy, env_name, seed, eval_episodes, model_name, use_dict_state, use_step_rate):
    if "LunarLander" in env_name:
        eval_env = gym.make(env_name, continuous=True)
    else:
        eval_env = gym.make(env_name)
    avg_reward = 0.
    avg_steps = 0.
    avg_decision = 0.
    avg_mean_rep: list[float] = []
    avg_std_rep: list[float] = []
    is_successes: list[float] = []
    
    for _ in range(eval_episodes):
        state, _ = eval_env.reset(seed = seed + 100)
        repetition = 1
        done = False
        mean_rep: list[int] = []
        is_success: list[float] = []

        while not done:
            state = state_transform(
                state, 
                use_dict_state, 
                use_step_rate, 
                eval_env._elapsed_steps / eval_env._max_episode_steps
            )
            action = policy.select_action(state)
            if model_name == TEMPORL:
                repetition = policy.select_skip(state, action)
            avg_decision += 1
            mean_rep.append(repetition)
            for _ in range(repetition):
                next_state, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                avg_reward += reward
                avg_steps += 1
                state = next_state
                if "is_success" in info.keys():
                    is_success.append(info["is_success"])

                if done:
                    is_successes.append(np.mean(is_success))
                    break
        
        avg_mean_rep.append(np.mean(mean_rep))
        avg_std_rep.append(np.std(mean_rep))
        
    eval_env.close()

    avg_reward /= eval_episodes
    avg_steps /= eval_episodes
    avg_decision /= eval_episodes
    avg_mean_rep = np.mean(avg_mean_rep) 
    avg_std_rep = np.mean(avg_std_rep) 
    
   
    log_dict = {
            "eval_epi_reward_mean": avg_reward,
            "eval_epi_length_mean": avg_steps,
            "eval_avg_decision": avg_decision,
            "eval_mean_repetition": avg_mean_rep,
            "eval_std_repetition": avg_std_rep,
    }
    
    if len(is_successes) > 0:
       log_dict["eval_success_rate"] = np.mean(is_successes)

    return log_dict


@hydra.main(config_path="configs/", config_name="config", version_base=None)
def main(args):
    print(OmegaConf.to_yaml(args))
    
    env_name = args.env_name
    algo_args = args.algos
    model_args = algo_args.model
    debug_mode  = args.debug

    model_name = model_args.name
    total_training_steps = args.total_training_steps
    warmup_steps = args.warmup_steps
    eval_n_episodes = args.eval_n_episodes
    seed = args.seed
    use_wandb = args.use_wandb
    use_step_rate = algo_args.use_step_rate

    env = gym.make(env_name)

    set_seed(seed)
    state, _ =  env.reset(seed = seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not env_name in gym.envs.registry.keys():
        raise NotImplementedError(f"{env_name} is not supported yet.")
    
    if isinstance(env.observation_space, spaces.Box):
        use_dict_state = False
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        
    elif isinstance(env.observation_space, gym.spaces.Dict):
        use_dict_state = True
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
    else:
        raise NotImplementedError(
            f"Unsupported observation space type: {type(env.observation_space)}"
        )
    
    state = state_transform(
        state, 
        use_dict_state, 
        use_step_rate, 
        0.0
    )
    state_dim = len(state)
    
    env_dict = {
        "env": env,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
    }
    
    pprint(env_dict)

    model_dict = {
        **OmegaConf.to_container(algo_args, resolve=True),
        **env_dict
    }

    policy = ALGO_REGISTRY[model_name](debug_mode, model_dict, device)
    
    replay_buffer = BUFFER_REGISTRY[DDPG](state_dim, action_dim, algo_args.buffer_size, device)
    
    if model_name == TEMPORL:
        max_repetition = model_args.max_repetition
        skip_e_greedy = model_args.skip_e_greedy
        skip_replay_buffer = BUFFER_REGISTRY[model_name](
            state_dim, 
            action_dim, 
            algo_args.buffer_size, 
            debug_mode, 
            max_repetition,
            device
        )

    if use_wandb:
        import wandb 
        wandb.init(
            project=env_name,       
            name=f"{env_name}_{model_name}_seed{seed}",
            group=args.group_name,
            config=OmegaConf.to_container(args, resolve=True)
        )

    episode_reward = 0
    training_steps = 0
    num_decisions = 0
    mean_rep: list[int] = []

    while training_steps < total_training_steps:
        if training_steps < warmup_steps:
            action = env.action_space.sample()

            if model_name == TEMPORL or model_name == NEW_Model:
                repetition = int(np.random.randint(1, max_repetition + 1))
            elif model_name == DDPG:
                repetition = 1

        else:
            action = (
                policy.select_action(state) + \
                    np.random.normal(0, max_action * algo_args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)
            
            if model_name == TEMPORL:
                if np.random.random() < skip_e_greedy:
                    repetition = int(np.random.randint(1, max_repetition + 1))
                else:
                    repetition = policy.select_skip(
                        state, 
                        action, 
                    )
            elif model_name == DDPG:
                repetition = 1
            else:
                raise NotImplementedError
                
        num_decisions += 1 
        mean_rep.append(repetition)

        if debug_mode:
            future_states, future_rewards, future_dones = store_future_states(
                env = env,
                action = action, 
                repetition = repetition,
                max_repetition = max_repetition,
                use_step_rate = use_step_rate,
                use_dict_state = use_dict_state
            )
            
        skip_states, skip_rewards = [], []
        
        for repeat_step in range(repetition):
            step_rate = env._elapsed_steps / env._max_episode_steps
            training_steps += 1
        
            next_state, reward, terminated, truncated, _ = env.step(action)

            next_state = state_transform(
                next_state, 
                use_dict_state, 
                use_step_rate, 
                env._elapsed_steps / env._max_episode_steps
            )
            done = terminated or truncated
            
            skip_states.append(state)
            skip_rewards.append(reward)
            
            replay_buffer.add(
                state, action, next_state, reward, done
            )

            if model_name == TEMPORL or model_name == NEW_Model:
                skip_id = 0
                for idx, start_state in enumerate(skip_states):
                    skip_reward = 0
                    for exp, r in enumerate(skip_rewards[skip_id:]):
                        skip_reward += np.power(policy.discount, exp) * r
                    skip_step = repeat_step - skip_id

                    if debug_mode:
                        """
                        After get fr_list[0] -> get next_state fs_list[0] and done signal dn_list[0]
                        """
                        fs_list = future_states[idx: idx + max_repetition]
                        fr_list = future_rewards[idx : idx + max_repetition] # future rewards
                        dr_list = future_dones[idx : idx + max_repetition] # discounted rewards
                        for r_idx, r in enumerate(fr_list):
                            discounted_r = 0
                            for num_exp in range(r_idx + 1):
                                discounted_r += np.power(policy.discount, num_exp) * fr_list[num_exp]
                            dr_list[r_idx] = discounted_r
                        dn_list = future_dones[idx : idx + max_repetition] # done flags
                        skip_replay_buffer.add(
                            start_state, 
                            action, 
                            skip_step, 
                            next_state, 
                            skip_reward, 
                            done,
                            np.array(fs_list),
                            np.expand_dims(dr_list, axis=-1), 
                            np.expand_dims(dn_list, axis=-1)
                        )
                    
                    else:
                        skip_replay_buffer.add(
                            start_state, 
                            action, 
                            skip_step, 
                            next_state, 
                            skip_reward, 
                            done,
                        )
                    skip_id += 1
                
            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if training_steps >= warmup_steps:
                policy_log_dict = policy.train(replay_buffer, algo_args.batch_size)
                if model_name == TEMPORL:
                    rep_critic_loss = policy.train_skip(skip_replay_buffer, algo_args.batch_size)
                    log_dict = policy_log_dict | rep_critic_loss
                elif model_name == DDPG:
                    log_dict = policy_log_dict
                else:
                    raise ValueError(f"Unsupported model name for training. [{model_name}]")
                
                if use_wandb:
                    for k, v in log_dict.items():
                        wandb.log({k:v}, step = training_steps)
                        
                    if debug_mode and model_name == TEMPORL:
                        policy.build_debug_log(step = training_steps)
                        
            # Evaluate episode
            if (training_steps + 1) % args.eval_interval == 0:
                eval_log_dict = eval(
                    policy, 
                    env_name, 
                    seed, 
                    eval_episodes=eval_n_episodes,
                    model_name=model_name, 
                    use_dict_state=use_dict_state,
                    use_step_rate=use_step_rate,
                )
                if use_wandb:
                    for k, v in eval_log_dict.items():
                        wandb.log({k:v}, step = training_steps)

            if done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(
                    f"Training steps: {training_steps + 1} Reward: {episode_reward:.3f}")
                if model_name == TEMPORL or model_name == NEW_Model:
                    print("Repetition list:", mean_rep)
                if use_wandb:
                    wandb.log(
                        {
                            "episode_reward": episode_reward,
                            "episode_length": env._elapsed_steps,
                            "num_decisions": num_decisions,
                            "mean_repetition": np.mean(mean_rep),
                            "std_repetition": np.std(mean_rep)
                        }, step=training_steps
                    )

                # Reset environment
                state, _ = env.reset()
                
                state = state_transform(
                    state, 
                    use_dict_state, 
                    use_step_rate, 
                    0.0
                )
                episode_reward = 0
                num_decisions = 0
                mean_rep.clear()
                break
        

if __name__ == "__main__":
    TEMPORL = "temporl"
    DDPG = "vanilla_ddpg"
    NEW_Model = "new"
    main()
