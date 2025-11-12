from algorithms.ddpg import DDPG
from algorithms.temporl_ddpg import TempoRLDDPG
from algorithms.new_ddpg import NewDDPG

ALGO_REGISTRY = {
    "vanilla_ddpg": DDPG,
    "temporl": TempoRLDDPG,
    "new": NewDDPG,
}

