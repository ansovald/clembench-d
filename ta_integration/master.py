from clemcore.clemgame import GameBenchmark, GameScorer
from clemcore.clemgame.registry import GameSpec
from clemcore.clemgame.metrics import BENCH_SCORE

from ta_master import TextArenaGameMaster
from submasters import *

from typing import List, Dict, Tuple
import logging

# Local type aliases to avoid import issues with the framework. Copied from textarena/core.py
Message = Tuple[int, str]
Observations = Dict[int, List[Message]]

logger = logging.getLogger(__name__)

class TextArenaScorer(GameScorer):
    """
    Default scorer for the TextArena environment.
    """
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_episode_scores(self, episode_interactions: Dict):
        self.log_episode_score(BENCH_SCORE, episode_interactions[BENCH_SCORE])

class TextArenaBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        logger.info(f"Initializing TextArenaBenchmark with game_spec: {game_spec}")
        super().__init__(game_spec)
        if 'master' in game_spec:
            self.master_class = globals()[game_spec['master']]
        else:
            self.master_class = TextArenaGameMaster
        if 'scorer' in game_spec:
            # Not used as of yet, but might be handy at some point
            self.scorer_class = globals()[game_spec['scorer']]
        else:
            self.scorer_class = TextArenaScorer

    def create_game_master(self, experiment, player_models):
        return self.master_class(self.game_spec, experiment, player_models)
    
    def create_game_scorer(self, experiment, game_instance):
        return self.scorer_class(self.game_spec['game_name'], experiment, game_instance)
    