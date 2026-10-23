"""
Example taken from https://github.com/hsahovic/poke-env/blob/master/examples/self_play.py
"""
from gymnasium.spaces import Box
from poke_env.environment import SinglesEnv
from poke_env.battle import AbstractBattle
from agents.EXAMPLE_policy_SB3 import PolicyPlayer
import numpy as np
N_FEATURES = 12


class SelfPlayEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: Box(-1, 4, shape=(N_FEATURES,), dtype=np.float32)
            for agent in self.possible_agents
        }

    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle,
            fainted_value=2.0,
            hp_value=1.0,
            status_value=0.5,
            victory_value=30.0,
        )

    def embed_battle(self, battle: AbstractBattle):
        return PolicyPlayer.embed_battle(battle)