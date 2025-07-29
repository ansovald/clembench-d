import logging
import os
import json

from clemcore.clemgame import GameInstanceGenerator
from clemcore.clemgame.resources import GameResourceLocator
from textarena.envs.registration import ENV_REGISTRY

# Seed for reproducibility, to be passed to the TextArena environments
SEED = 525119131

# N_GAMES = 1  # Number of games to generate for each game type

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
        for ta_game in ENV_REGISTRY:
            if ta_game.startswith(ta_name):
                # We take the "-raw" version of the environment, since it does not have any wrappers applied.
                if ta_game.endswith("-raw"):
                    experiment = self.add_experiment(ta_game[:-4])
                    for i in range(kwargs.get("n_instances")):
                        game_instance = self.add_game_instance(experiment, game_id=i)
                        game_instance["entry_point"] = ENV_REGISTRY[ta_game].entry_point
                        game_instance["env_specs"] = ENV_REGISTRY[ta_game].kwargs
                        game_instance["seed"] = seed + i # Ensure different seeds for each game instance
                        # game_instance["player_specs"] = kwargs.get("player_specs")
                        if "override_variables" in kwargs:
                            game_instance["override_variables"] = kwargs["override_variables"]

if __name__ == "__main__":
    # load clemgame.json to get the games
    clemgame_registry = json.load(open(os.path.join(os.path.dirname(__file__), "clemgame.json"), "r"))

    for game in clemgame_registry:
        TextArenaInstanceGenerator().generate(**game, filename=game["instances"] + '.json', seed=SEED)
