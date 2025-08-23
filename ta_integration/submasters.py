"""
Subclasses of TextArena's GameMaster to handle scoring
and functionalities to ensure deterministic behavior for specific games
"""

from re import M
from sys import stdout

from py import log
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
    Reward structures:
    Tower of Hanoi (largely identical with most single player games) can end in three ways:
        1. Player wins by moving all disks to tower C:           +1
        2. Player loses by running out of turns: float within    [0,1] 
            (as calculated by `ta_env._get_percentage_completion()`, can be used as basic bench score)
        3. Game ends because of invalid moves:                   -1, according to https://www.textarena.ai/environments/towerofhanoi
            Currently, a game ended with invalid move still returns a float within [0,1]
            (bypassed by `reward_for_player` function)
    Args:
        rewards: list of dicts as passed by ta_env.close()
        player_id: ID of the player for which to extract the reward
    Returns: 
        numeric_reward for the player
    """
    assert player_id in rewards[0], f"Player ID {player_id} not found in rewards! {rewards[0].keys()}"
    assert player_id in rewards[1], f"Player ID {player_id} not found in other_rewards! {rewards[1].keys()}"
    numeric_reward = rewards[0][player_id]
    # invalid_move = rewards[1][player_id]['invalid_move']
    # if invalid_move:
    #     numeric_reward = -1  # Bypass the float reward and sets it to -1, as described on TA website
    return numeric_reward

class WordChainsMaster(TextArenaGameMaster):
    def setup(self, **kwargs):
        super().setup(**kwargs)
        self.final_keys['start_word'] = self.env.state.game_state['current_word']
        self.final_keys['start_word_length'] = len(self.final_keys['start_word'])

    def _on_before_reset(self):
        # Sort word_list of self.env to ensure deterministic sampling of start word
        self.env.word_list.sort()
    
    def log_success(self, rewards: Dict = None, last_move_invalid: bool = False):
        self.final_keys['end_word'] = self.env.state.game_state['current_word']
        self.final_keys['end_word_length'] = len(self.final_keys['end_word'])
        word_length_diff = self.final_keys['end_word_length'] - self.final_keys['start_word_length']
        self.final_keys['word_length_diff'] = word_length_diff
        
        if word_length_diff == 0:
            self.log_key(METRIC_ABORTED, 1)
            self.log_key(METRIC_SUCCESS, 0)
            self.log_key(METRIC_LOSE, 1)
        else:
            self.log_key(METRIC_ABORTED, 0)
            self.log_key(METRIC_SUCCESS, 1)
            self.log_key(METRIC_LOSE, 0)

class SinglePlayerMaster(TextArenaGameMaster):
    """
    Master class for single-player games in TextArena.
    It handles basic scoring and logging functionalities.
    """
    def log_success(self, rewards, last_move_invalid: bool = False):
        numeric_reward = reward_for_player(rewards, player_id=0)
        self.final_keys['numeric_reward'] = numeric_reward
        metrics = self.prepare_metrics(numeric_reward)
        self.log_keys(metrics)

    def prepare_metrics(self, numeric_reward=None) -> Dict[str, float]:
        """
        Returns default values for the metrics.
        """
        metrics = super().prepare_metrics()
        if numeric_reward:
            if numeric_reward == -1:
                metrics[METRIC_ABORTED] = 1
                metrics[METRIC_LOSE] = 1
            elif numeric_reward == 1:
                metrics[METRIC_SUCCESS] = 1
        return metrics
    
class MinesweeperMaster(SinglePlayerMaster):
    """
    BENCH_SCORE is here the percentage of cells cleared by 
    the player, excluding the ones revealed in the first turn.
    """

    def _on_before_game(self):
        """
        Mines are only distributed with the first move, so we first reveal the 
        center cell to ensure the game is always initialized deterministically.
        """
        rows = self.env.rows
        cols = self.env.cols
        x = rows // 2
        y = cols // 2
        faux_response = f"[{x} {y}]"
        player, context = self.observe()
        player.perceive_context(context)
        player.perceive_response(faux_response)
        self.step(faux_response)

class HangmanMaster(SinglePlayerMaster):
    def _on_before_game(self):
        self.final_keys['target_word'] = self.env.state.game_state['target_word']
        self.final_keys['lives'] = self.env.state.game_state['tries_left']

    def collect_ta_env_info(self):
        self.final_keys['lives_left'] = self.env.state.game_state['tries_left']
        self.final_keys['guessed_letters'] = self.env.state.game_state['guessed_letters']

class TwoPlayerMaster(TextArenaGameMaster):
    """
    Master class for competitive two-player games in TextArena.
    It handles basic scoring and logging functionalities.
    """
    def log_success(self, rewards: Dict = None, last_move_invalid: bool = False):
        self.final_keys['rewards'] = rewards

        if last_move_invalid:
            self.log_key(METRIC_ABORTED, 1)
            self.log_key(METRIC_SUCCESS, 0)
            self.log_key(METRIC_LOSE, 1)
        else:
            self.log_key(METRIC_ABORTED, 0)
            self.log_key(METRIC_SUCCESS, 1)
            self.log_key(METRIC_LOSE, 0)
    
class BattleshipMaster(TwoPlayerMaster):
    """
    Master class for the Battleship game in TextArena.
    It handles basic scoring and logging functionalities.
    """
    def collect_ta_env_info(self):
        boards = {}
        for player_id in self.env.state.game_state['board']:
            board = self.env.state.game_state['board'][player_id]
            board = "\n".join("".join(f"{cell}" for cell in board[i]) for i in range(len(board)))
            boards[player_id] = board
        self.final_keys['cell_counts'] = { 
                'total_cells': self.env.grid_size ** 2,
                'total_ship_cells': sum(value for value in self.env.ships.values())
            }
        for player_id in boards:
            # count the number of 'X' and 'O' on the board
            hits = boards[player_id].count('X')
            misses = boards[player_id].count('O')
            water = boards[player_id].count('~')
            remaining_ships = self.env.grid_size ** 2 - (hits + misses + water)
            self.final_keys['cell_counts'][player_id] = {
                'hits': hits,
                'misses': misses,
                'water': water,
                'remaining_ship_cells': remaining_ships
            }