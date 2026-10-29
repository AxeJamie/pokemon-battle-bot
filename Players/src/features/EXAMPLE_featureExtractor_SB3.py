"""
Example taken from https://github.com/hsahovic/poke-env/blob/master/examples/self_play.py
"""

N_FEATURES = 12
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
class EXAMPLEFeaturesExtractor(BaseFeaturesExtractor):
    """Extracts the observation tensor from the dict obs and declares features_dim
    so SB3 builds the MLP with the right input size."""

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=N_FEATURES)

    def forward(self, obs):
        return obs["observation"]