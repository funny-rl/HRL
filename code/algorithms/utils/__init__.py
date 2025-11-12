from algorithms.utils.vanilla_buffer import ReplayBuffer
from algorithms.utils.skip_buffer import SkipReplayBuffer
from algorithms.utils.skip_buffer import SkipReplayBuffer

BUFFER_REGISTRY = {
    "vanilla_ddpg": ReplayBuffer,
    "temporl": SkipReplayBuffer,
    "new": SkipReplayBuffer,
} 