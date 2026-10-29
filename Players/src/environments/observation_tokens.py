"""
We designate a class here tasked with taking a battle state, and returning the features that a player may consider when training.
More precicely, we wish to define the input tokens, and any useful aditional information
"""
from poke_env.battle import AbstractBattle
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.field import Field
from poke_env.battle.weather import Weather
from poke_env.battle.pokemon_type import PokemonType

import numpy as np

class ObservableExtractorV1:
    # Used for the embed_battle method in the SB3 agents

    def __init__(self):
        # Define the enumerations for one-hot encoding
        self.WEATHER_IDX = {weather: i for i, weather in enumerate(Weather)}
        self.FIELD_IDX = {field: i for i, field in enumerate(Field)}
        self.SIDE_IDX = {condition: i for i, condition in enumerate(SideCondition)}
        self.TYPE_IDX = {condition: i for i, condition in enumerate(PokemonType)}



        self.battlefield_token_length = len(self.WEATHER_IDX) + len(self.FIELD_IDX) + 2 * len(self.SIDE_IDX)
        self.player_token_length = 10

    def encode_field(self,battle:AbstractBattle):

        tokens = np.zeros(self.battlefield_token_length,dtype=np.float32,)

        offset = 0

        # Weather: one-hot
        for weather in battle.weather:
            if weather in self.WEATHER_IDX:
                tokens[offset + self.WEATHER_IDX[weather]] = 1.0
            else:
                raise EncodingWarning
        offset += len(self.WEATHER_IDX)

        # Fields: one-hot
        for field in battle.fields:
            if field in self.FIELD_IDX:
                tokens[offset + self.FIELD_IDX[field]] = 1.0
            else:
                raise EncodingWarning
        offset += len(self.FIELD_IDX)

        # Side conditions: encode remaining turns (or presence if unavailable)
        for condition,count in battle.side_conditions.items():
            if condition in self.SIDE_IDX:
                tokens[offset + self.SIDE_IDX[condition]] = count #Could explicitly cast to float but no need
        offset += len(self.SIDE_IDX)

        for condition, count in battle.opponent_side_conditions.items():
            if condition in self.SIDE_IDX:
                tokens[offset + self.SIDE_IDX[condition]] = count

        return tokens

    def encode_mons(self,battle:AbstractBattle):

        tokens = np.zeros(self.player_token_length,dtype=np.float32,)

        return tokens

    def encode_active_mon(self,battle:AbstractBattle):

        tokens = np.zeros(self.player_token_length,dtype=np.float32,)

        # One hot for which mon is the active mon
        # Encoding for battle.pokemon.effects

        return tokens

    def encode_matchups(self,battle:AbstractBattle):

        tokens = np.zeros(self.player_token_length,dtype=np.float32,)

        return tokens

    def battle_to_tokens(self,battle: AbstractBattle):

        #Field State Tokens - One hot - [Weather,Terrain,Speed Effects..]
        field_tokens = self.encode_field(battle)

        # Player Choices
        # It is worth noting switching is always an option that can be inferred from knowledge of your party - and of arena conditions



        return np.concatenate(
            [
                field_tokens
            ],
            dtype=np.float32,
        )



