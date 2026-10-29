from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """Custom Extractor implementing a transformer to turn observations
    into exciting representations"""

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=N_FEATURES)

    def forward(self, obs):
        return obs["observation"]