from algorithms.ddpg import DDPG
from algorithms.temporl_ddpg import TempoRLDDPG
ALGO_REGISTRY = {
    "vanilla_ddpg": DDPG,
    "temporl": TempoRLDDPG,
}

