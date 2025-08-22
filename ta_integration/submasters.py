"""
Subclasses of TextArena's GameMaster to handle scoring
and functionalities to ensure deterministic behavior for specific games
"""

from operator import inv
from sys import stdout

from regex import B
from ta_master import TextArenaGameMaster
from clemcore.clemgame import GameMaster, GameScorer, Player
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, BENCH_SCORE
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def reward_for_player(rewards: list[Dict], player_id: int=0) -> float:
    """
    Args:
        rewards: list of dicts as passed by ta_env.close()
        player_id: ID of the player for which to extract the reward
    Returns: 
        numeric_reward for the player
    """
    assert player_id in rewards[0], f"Player ID {player_id} not found in rewards! {rewards[0].keys()}"
    # assert player_id in rewards[1], f"Player ID {player_id} not found in other_rewards! {rewards[1].keys()}"
    return rewards[0][player_id] #, rewards[1][player_id]['invalid_move']

class WordChainsMaster(TextArenaGameMaster):
    def setup(self, **kwargs):
        super().setup(**kwargs)
        self.start_word = self.env.state.game_state['current_word']

    def _override_variables(self):
        # Sort word_list of self.env to ensure deterministic behavior
        self.env.word_list.sort()
    
    def _log_metrics(self, rewards: Dict):
        self.end_word = self.env.state.game_state['current_word']
        word_length_diff = len(self.end_word) - len(self.start_word)
        word_stats = {
            'start_word': self.start_word,
            'start_word_length': len(self.start_word),
            'end_word': self.end_word,
            'end_word_length': len(self.end_word),
            'word_length_diff': word_length_diff
        }
        self.log_key("word_stats", word_stats)
        if word_length_diff == 0:
            self.log_key(METRIC_ABORTED, 1)
            self.log_key(METRIC_SUCCESS, 0)
            self.log_key(METRIC_LOSE, 1)
            self.log_key(BENCH_SCORE, np.nan)
        else:
            self.log_key(METRIC_ABORTED, 0)
            self.log_key(METRIC_SUCCESS, 1)
            self.log_key(METRIC_LOSE, 0)
            # Bench score: We assume a perfect game reaches a word with 20 letters.
            main_score = min(1, (len(self.end_word) / 20)) * 100
            self.log_key(BENCH_SCORE, main_score)

class SinglePlayerMaster(TextArenaGameMaster):
    def _log_metrics(self, rewards):
        """
        Reward structures:
        Tower of Hanoi ( virtually identical for Minesweeper etc.):
        Can end three ways:
            1. Player wins by moving all disks to tower C:           +1
            2. Player loses by running out of turns: float within    [0,1] 
                (percentage completion, can be used as bench score)
            3. Game ends because of invalid moves:                   -1, according to https://www.textarena.ai/environments/towerofhanoi
                Currently, a game ended with invalid move still returns a float within [0,1]
        """
        num_reward = reward_for_player(rewards, player_id = 0)
        # if invalid_move: num_reward = -1
        metrics = self._init_metrics()
        if num_reward == 1:
            metrics[METRIC_SUCCESS] = 1
            metrics[BENCH_SCORE] = 100
        elif num_reward == -1:
            metrics[METRIC_ABORTED] = 1
            metrics[METRIC_LOSE] = 1
        else:
            # Neither success nor loss
            metrics[BENCH_SCORE] = num_reward * 100
        
        self._log_metric_keys(metrics)
