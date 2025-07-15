import logging
import os
import json

from clemcore.clemgame import GameInstanceGenerator
from clemcore.clemgame.resources import GameResourceLocator
from textarena.envs.registration import ENV_REGISTRY

# Seed for reproducibility, to be passed to the TextArena environments
SEED = 525119131

N_GAMES = 2  # Number of games to generate for each game type

class TextArenaInstanceGenerator(GameInstanceGenerator):
    """
    TextArenaInstanceGenerator is a custom instance generator for the TextArena environment.
    It just holds the configuration for the games, and passes the random seed to the TextArena environment,
    which handle the actual instance generation.
    """

    def __init__(self):
        super().__init__(os.path.dirname(__file__))
    
    def on_generate(self, seed: int, **kwargs):
        # load all games from the ENV_REGISTRY that use the specified environment
        ta_name = kwargs.get("ta_name")
        ta_env = kwargs.get("ta_env")
        player_specs = kwargs.get("player_specs")
        for ta_game in ENV_REGISTRY:
            if ta_game.startswith(ta_name) and ENV_REGISTRY[ta_game].entry_point == ta_env:
                experiment = self.add_experiment(ta_game)
                for i in range(N_GAMES):
                    game_instance = self.add_game_instance(experiment, game_id=i)
                    game_instance["ta_env"] = ta_env
                    game_instance["seed"] = seed + i # Ensure different seeds for each game instance
                    game_instance["player_specs"] = player_specs

if __name__ == "__main__":
    # load clemgame.json to get the games
    clemgame_registry = json.load(open(os.path.join(os.path.dirname(__file__), "clemgame.json"), "r"))

    for game in clemgame_registry:
        TextArenaInstanceGenerator().generate(ta_name=game["ta_name"], ta_env=game["ta_env"], filename=game["instances"] + '.json', player_specs=game["player_specs"], seed=SEED)
