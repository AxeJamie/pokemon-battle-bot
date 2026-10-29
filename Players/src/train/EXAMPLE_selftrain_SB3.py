"""
Example taken from https://github.com/hsahovic/poke-env/blob/master/examples/self_play.py
"""


import asyncio
from typing import Any, Awaitable

import numpy as np
import supersuit as ss
import torch
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from features.EXAMPLE_featureExtractor_SB3 import  EXAMPLEFeaturesExtractor as FeaturesExtractor

from poke_env.battle import AbstractBattle
from poke_env.data import GenData
from poke_env.environment import SinglesEnv
from poke_env.player import (
    BattleOrder,
    DefaultBattleOrder,
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from agents.EXAMPLE_policy_SB3 import MaskedActorCriticPolicy
from environments.EXAMPLE_basic_singles_env import SelfPlayEnv

BATTLE_FORMAT = "gen9randombattle"
NUM_LEARN_STEPS = 1000 #98_304


class EXAMPLE_selftrain():
    def __init__(self):
        self.ppo = self.train()

    def train(self):
        num_envs = 2
        env = SelfPlayEnv(battle_format=BATTLE_FORMAT, log_level=40, open_timeout=None)
        vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
        vec_env = ss.concat_vec_envs_v1(
            vec_env,
            num_vec_envs=num_envs,
            num_cpus=num_envs,
            base_class="stable_baselines3",
        )
        ppo = PPO(
            MaskedActorCriticPolicy,
            vec_env,
            learning_rate=3e-4,
            n_steps=3072 // (2 * num_envs),
            batch_size=128,
            gamma=0.99,
            ent_coef=0.01,
            device="cpu",
        )

        # train
        ppo.learn(NUM_LEARN_STEPS)
        vec_env.close()

        return ppo

    def get_ppo(self):
        return self.ppo