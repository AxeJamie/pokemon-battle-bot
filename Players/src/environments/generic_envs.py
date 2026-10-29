from environments.observation_tokens import ObservableExtractorV1
from environments.rewards_calculator import DeterministicRewardCalculator
from poke_env.battle import AbstractBattle
from poke_env.environment import SinglesEnv
from gymnasium.spaces import Box
import numpy as np

class CustomisableEnv(SinglesEnv):
    """ Environment subclass with changeable embedding and reward functions"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedding = ObservableExtractorV1()
        self.rewards = DeterministicRewardCalculator()
        #self.observation_spaces = {agent: Box(-1, 4, shape=(N_FEATURES,), dtype=np.float32)}

    def calc_reward(self, battle : AbstractBattle) -> float:
        return self.rewards.calc_reward(battle)

    def embed_battle(self, battle: AbstractBattle):
        return self.embedding.battle_to_tokens(battle)