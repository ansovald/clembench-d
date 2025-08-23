import os
import json
import regex as re

from clemcore.clemgame import GameInstanceGenerator
# from clemcore.clemgame.resources import GameResourceLocator
from textarena.envs.registration import ENV_REGISTRY

# Seed for reproducibility, to be passed to the TextArena environments
SEED = 525119131

class TextArenaInstanceGenerator(GameInstanceGenerator):
    """
    TextArenaInstanceGenerator is a custom instance generator for the TextArena environment.
    It just holds the configuration for the games, and passes the random seed to the TextArena environment,
    which handle the actual instance generation.
    """

    def __init__(self):
        super().__init__(os.path.dirname(__file__))
    
    def on_generate(self, seed: int, **kwargs):
        entry_point = kwargs.get("entry_point")
        assert entry_point is not None, "entry_point must be specified in the game config!"
        game_name = kwargs.get("game_name")
        experiments = kwargs.get("experiments")
        n_instances = kwargs.get("n_instances", 1)
        if not experiments or isinstance(experiments, list):
            # load all games from the ENV_REGISTRY that use the specified entry point
            for ta_game in ENV_REGISTRY:
                if ENV_REGISTRY[ta_game].entry_point == entry_point and ta_game.endswith("-raw"):
                    difficulty = ta_game.split("-")[-2]
                    if re.match(r"v[0-9]+", difficulty):
                        difficulty = "default"
                    if isinstance(experiments, list) and difficulty not in experiments:
                        continue
                    config = {
                        "env_id": ta_game,
                        "entry_point": entry_point,
                        "env_specs": ENV_REGISTRY[ta_game].kwargs
                    }
                    experiment = self.add_experiment(difficulty)
                    self.generate_instances(experiment, n_instances, config, seed)
        elif isinstance(experiments, dict):
            for difficulty in experiments:
                experiment = self.add_experiment(difficulty)
                config = {
                    "env_id": f"{game_name}-{difficulty}",
                    "register_env": True,
                    "entry_point": entry_point,
                    "env_specs": experiments[difficulty]
                }
                self.generate_instances(experiment, n_instances, config, seed)
        else:
            raise ValueError("experiments must be either None, a list of difficulty levels, or a dict of experiment configs.")
        
    def generate_instances(self, experiment, n_instances: int, config: dict, seed: int=SEED):
        for i in range(n_instances):
            game_instance = self.add_game_instance(experiment, game_id=i)
            for key, value in config.items():
                game_instance[key] = value
            game_instance["seed"] = seed + i

if __name__ == "__main__":
    # load clemgame.json to get the games
    clemgame_registry = json.load(open(os.path.join(os.path.dirname(__file__), "clemgame.json"), "r"))

    for game in clemgame_registry:
        TextArenaInstanceGenerator().generate(**game, filename=game["instances"] + '.json', seed=SEED)
