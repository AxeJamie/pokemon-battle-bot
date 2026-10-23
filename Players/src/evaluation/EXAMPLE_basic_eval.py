"""
Example taken from https://github.com/hsahovic/poke-env/blob/master/examples/self_play.py
"""
from train.EXAMPLE_selftrain_SB3 import EXAMPLE_selftrain
import asyncio
from agents.EXAMPLE_policy_SB3 import PolicyPlayer

from poke_env.player import (
    BattleOrder,
    DefaultBattleOrder,
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)

BATTLE_FORMAT = "gen9randombattle"


def run_evaluate():
    policy_trainer = EXAMPLE_selftrain()
    ppo = policy_trainer.get_ppo()
    # evaluate
    agent = PolicyPlayer(
        policy=ppo.policy, battle_format=BATTLE_FORMAT, max_concurrent_battles=10
    )
    opponents: list[Player] = [
        c(battle_format=BATTLE_FORMAT, max_concurrent_battles=10)
        for c in [RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer]
    ]
    asyncio.run(agent.battle_against(*opponents, n_battles=100))
    print("--- Win rates vs bots ---")
    for opp in opponents:
        win_rate = round(100 * opp.n_lost_battles / opp.n_finished_battles)
        print(f"{opp.username}: {win_rate}%")


if __name__ == "__main__":
    run_evaluate()
