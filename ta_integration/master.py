from clemcore.clemgame import GameBenchmark, GameScorer
from clemcore.clemgame.registry import GameSpec
from clemcore.clemgame.metrics import BENCH_SCORE

from ta_master import TextArenaGameMaster
from submasters import *
from metrics import *

from typing import List, Dict, Tuple
import logging

# Local type aliases to avoid import issues with the framework. Copied from textarena/core.py
Message = Tuple[int, str]
Observations = Dict[int, List[Message]]

logger = logging.getLogger(__name__)

class TextArenaBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        logger.info(f"Initializing TextArenaBenchmark with game_spec: {game_spec}")
        super().__init__(game_spec)
        assert 'master' in game_spec and 'scorer' in game_spec, "Both 'master' and 'scorer' must be specified in the game spec."
        self.master_class = globals()[game_spec['master']]
        self.scorer_class = globals()[game_spec['scorer']]

    def create_game_master(self, experiment, player_models):
        return self.master_class(self.game_spec, experiment, player_models)
    
    def create_game_scorer(self, experiment, game_instance):
        return self.scorer_class(self.game_spec['game_name'], experiment, game_instance)
    