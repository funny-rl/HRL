import mujoco
import random 
import numpy as np
import torch

def state_transform(_s, use_dict_state, use_step_rate, step_rate):
    if use_dict_state:
        _s = np.concatenate(
            [
                _s["observation"], 
                _s["desired_goal"], 
                _s["achieved_goal"]
            ], axis=-1
        )

    if use_step_rate:
        _s = np.concatenate([_s, [step_rate]])
    return _s


def set_seed(seed: int):
    """Seed the program."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def dict_equal_exact(d1, d2):
    if d1.keys() != d2.keys():
        return False
    
    for k in d1:
        if not np.array_equal(d1[k], d2[k]):
            return False
    return True

def save_state(env):
    unwrapped_env = env.unwrapped
    state_info = {
        "qpos": np.copy(unwrapped_env.data.qpos),
        "qvel": np.copy(unwrapped_env.data.qvel),
        "time": unwrapped_env.data.time,
        "elapsed_steps": getattr(env, "_elapsed_steps", 0),
    }
    return state_info

def restore_state(env, saved_info):
    unwrapped_env = env.unwrapped
    unwrapped_env.data.qpos[:] = saved_info["qpos"]
    unwrapped_env.data.qvel[:] = saved_info["qvel"]
    unwrapped_env.data.time = saved_info["time"]
    mujoco.mj_forward(
        unwrapped_env.model, 
        unwrapped_env.data
    )
    env._elapsed_steps = saved_info["elapsed_steps"]
    
    assert dict_equal_exact(unwrapped_env._get_obs(), unwrapped_env._get_obs()), "State restoration failed!"

def store_future_states(
    env, 
    action,
    repetition,
    max_repetition,
    use_step_rate,
    use_dict_state
):
    saved_info = save_state(env)

    s_list = []
    r_list = []
    d_list = []
    
    total_rep = repetition + max_repetition
    for _ in range(total_rep):
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = state_transform(
            next_state,
            use_dict_state,
            use_step_rate, 
            env._elapsed_steps / env._max_episode_steps
        )
        s_list.append(next_state)
        r_list.append(reward)
        d_list.append(done)
    restore_state(env, saved_info)

    assert len(s_list) == total_rep and \
              len(r_list) == total_rep and \
                len(d_list) == total_rep, "Storing future states failed!"

    return s_list, r_list, d_list
    
