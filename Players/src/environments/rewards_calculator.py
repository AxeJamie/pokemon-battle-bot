from poke_env.environment.env import PokeEnv
from poke_env.battle import AbstractBattle

class DeterministicRewardCalculator:
    def __init__(self):
        self.MULTIPLIERS = {"fainted_value":2.0,"hp_value":1.0,"status_value":0.5,"victory_value":30.0} # EXAMPLE values from the docs
        self.helper = PokeEnv

    def calc_reward(self, battle : AbstractBattle) -> float:
        return self.helper.reward_computing_helper(battle=battle,**self.MULTIPLIERS)