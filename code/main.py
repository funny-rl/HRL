import hydra
import torch
import numpy as np
from pprint import pprint
from omegaconf import OmegaConf
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing

from utils import set_seed
from algorithms import ALGO_REGISTRY
from algorithms.utils import BUFFER_REGISTRY    

def eval(policy, env_name, seed, eval_episodes, model_name):
    eval_env = gym.make(env_name)
    eval_env.reset(seed = seed + 100)
    avg_reward = 0.
    avg_steps = 0.
    
    for _ in range(eval_episodes):
        state, _ = eval_env.reset()
        repetition = 1
        done = False
        
        while not done:
            action = policy.select_action(np.array(state))
            if model_name == "temporl":
                repetition = np.argmax(policy.select_skip(np.array(state), action)) + 1
            
            for _ in range(repetition):
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                avg_reward += reward
                avg_steps += 1
                state = next_state
                if done:
                    break
        eval_env.close()

    avg_reward /= eval_episodes
    avg_steps /= eval_episodes
    
    return avg_reward, avg_steps


@hydra.main(config_path="configs/", config_name="config", version_base=None)
def main(args):
    print(OmegaConf.to_yaml(args))
    
    env_name = args.env_name
    algo_args = args.algos
    model_args = algo_args.model
    model_name = model_args.name
    total_training_steps = args.total_training_steps
    warmup_steps = args.warmup_steps
    eval_n_episodes = args.eval_n_episodes
    seed = args.seed
    use_wandb = args.use_wandb
    
    env = gym.make(env_name)
    env.reset(seed = seed)
    set_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
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

    policy = ALGO_REGISTRY[model_name](model_dict, device)
    
    replay_buffer = BUFFER_REGISTRY["vanilla_ddpg"](state_dim, action_dim, algo_args.buffer_size, device)
    if model_name == "temporl":
        max_repetition = model_args.max_repetition
        skip_e_greedy = model_args.skip_e_greedy
        skip_replay_buffer = BUFFER_REGISTRY[model_name](state_dim, action_dim, algo_args.buffer_size, device)

    if use_wandb:
        import wandb 
        wandb.init(
            project=env_name,       
            name=f"{env_name}_{model_name}_seed{seed}",
            config=OmegaConf.to_container(args, resolve=True)
        )

    state, _ = env.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    avg_reward, avg_steps = eval(
        policy, 
        env_name, 
        seed, 
        eval_episodes=eval_n_episodes,
        model_name=model_name
    )

    training_steps = 0
    
    while training_steps < total_training_steps:
        

        if training_steps < warmup_steps:
            action = env.action_space.sample() 
            if model_name == "temporl":
                repetition = np.random.randint(1, max_repetition + 1)
            elif model_name == "vanilla_ddpg":
                repetition = 1
        else:
            action = (
                policy.select_action(np.array(state)) + \
                    np.random.normal(0, max_action * algo_args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)
            if model_name == "temporl":
                if np.random.random() < skip_e_greedy:
                    repetition = np.random.randint(1, max_repetition + 1)
                else:
                    repetition = policy.select_skip(state, action)
                    repetition = np.argmax(repetition) + 1
                    
            elif model_name == "vanilla_ddpg":
                repetition = 1
                    

        # number of repetitions for action repetition
        skip_states, skip_rewards = [], []
        for repeat_step in range(repetition):
            
            episode_timesteps += 1
            training_steps += 1
        
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            skip_states.append(state)
            skip_rewards.append(reward)
            
            replay_buffer.add(
                state, action, next_state, reward, done
            )
            
            if model_name == "temporl":
                skip_id = 0
                for start_state in skip_states:
                    skip_reward = 0
                    for exp, r in enumerate(skip_rewards[skip_id:]):
                        skip_reward += np.power(policy.discount, exp) * r
                    skip_replay_buffer.add(
                        start_state, 
                        action, 
                        repeat_step - skip_id, 
                        next_state, 
                        skip_reward, 
                        done
                    )
                    skip_id += 1
                
            state = next_state
            episode_reward += reward
            
            # Train agent after collecting sufficient data
            if training_steps >= warmup_steps:
                policy.train(replay_buffer, algo_args.batch_size)
                if model_name == "temporl":
                    policy.train_skip(skip_replay_buffer, algo_args.batch_size)
                
            if done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(
                    f"Training steps: {training_steps + 1} Episode Num: {episode_num + 1} Episode_Length: {episode_timesteps} Reward: {episode_reward:.3f}")
                if use_wandb:
                    wandb.log(
                        {
                            "episode_reward": episode_reward,
                            "episode_length": episode_timesteps,
                        }, step=episode_num
                    )
                # Reset environment
                state, _ = env.reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
                break
        
            # Evaluate episode
            if (training_steps + 1) % args.eval_interval == 0:
                avg_reward, avg_steps = eval(
                    policy, 
                    env_name, 
                    seed, 
                    eval_episodes=eval_n_episodes,
                    model_name=model_name
                )
                if use_wandb:
                    wandb.log(
                        {
                            "eval_epi_reward_mean": avg_reward,
                            "eval_epi_length_mean": avg_steps,
                        }, step=episode_num
                    )

if __name__ == "__main__":
    
    main()
