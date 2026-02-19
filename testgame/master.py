from clemcore import backends
from clemcore.backends import Model
from clemcore.clemgame import Player, GameBenchmark, GameMaster
from clemcore.clemgame.legacy.scorer import GameScorer
from clemcore.clemgame.legacy.master import DialogueGameMaster
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_LOSE, METRIC_SUCCESS, METRIC_REQUEST_COUNT, \
    METRIC_REQUEST_COUNT_PARSED, METRIC_REQUEST_COUNT_VIOLATED, BENCH_SCORE
import random
from typing import Dict, List
import numpy as np


"""
THIS IS A GAME MEANT FOR TESTING THE INTEGRATION OF MODELS.
The player is asked exactly one question, then the game ends.
We can then check the transcripts and requests to see if everything is set up correctly.
"""

class TestPlayer(Player):
    def __init__(self, model: Model):
        super().__init__(model)

    def _custom_response(self, context: Dict) -> str:
        return "I don't know"


class TestMaster(DialogueGameMaster):
    def _on_setup(self, **instance):
        # Setup game state
        initial_prompt = f"Answer the following question truthfully in one sentence: {instance['input']}"

        # Setup player
        self.test_player = TestPlayer(self.player_models[0])
        self.add_player(self.test_player, initial_context=initial_prompt)

    def _does_game_proceed(self):
        return False  # End game after one response
    
    def _on_valid_player_response(self):
        pass
    
    def _validate_player_response(self, player: Player, response: str) -> bool:
        pass

    def _on_after_game(self):
        self.log_key(METRIC_ABORTED, np.nan)
        self.log_key(METRIC_SUCCESS, np.nan)
        self.log_key(METRIC_LOSE, np.nan)

class TestScorer(GameScorer):
    def score_turns(self, episode_interactions: Dict) -> None:
        pass  # single-turn

    def log_main_score(self, episode_interactions: Dict):
        accuracy = 100 if episode_interactions[METRIC_SUCCESS] else 0
        self.log_episode_score(BENCH_SCORE, accuracy)

class TestGameBenchmark(GameBenchmark):
    def create_game_master(self, experiment: Dict, player_models: List[backends.Model]) -> GameMaster:
        return TestMaster(self.game_spec, experiment, player_models)