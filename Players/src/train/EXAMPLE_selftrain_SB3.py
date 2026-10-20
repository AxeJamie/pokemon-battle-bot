"""
Example taken from https://github.com/hsahovic/poke-env/blob/master/examples/self_play.py
"""


class EXAMPLE_selftrain():
    def __init__(self):
        ppo = self.train()

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
        ppo.learn(98_304)
        vec_env.close()

        return ppo
