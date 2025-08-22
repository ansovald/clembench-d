"""
This module contains the default TextArena game master, scorer and player classes.
"""

from typing import Dict, Optional, Tuple, List
import random
import numpy as np
import textarena as ta
from clemcore.backends import Model
from clemcore.clemgame import Player, GameMaster, GameBenchmark, GameScorer
from clemcore.clemgame.registry import GameSpec
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, BENCH_SCORE
import logging
from typing import List, Dict, Optional, Tuple, Union

from clem_observation_wrapper import ClemObservationWrapper

logger = logging.getLogger(__name__)

class TextArenaPlayer(Player):
    """
    TextArenaPlayer is a custom player for the TextArena environment.
    It is subclassed for each player role defined in the game specification.
    """
    def __init__(self, model: Model, ta_player_id: int, master: GameMaster, custom_response: list = None):
        super().__init__(model=model)
        self.custom_response = custom_response if custom_response else ["Whatever"]
        self.ta_id = ta_player_id  # This is the player ID used by TextArena, which may differ from the GameMaster's player ID
        self.game_master = master  # Reference to the GameMaster for accessing game state and methods
    
    def _custom_response(self, context):
        return random.choice(self.custom_response)
    
    def perceive_context(self, context, *, log_event=True, memorize=True):
        self.game_master.played_in_turn.add(self.ta_id)  # Mark this player as having played in the current turn
        self.game_master.last_player = self.ta_id  # Update the last player who played
        return super().perceive_context(context, log_event=log_event, memorize=memorize)

class TextArenaGameMaster(GameMaster):
    """
    TextArenaGameMaster is a custom game master for the TextArena environment.
    It inherits from the GameMaster class and implements the required methods.
    """
    
    def __init__(self, game_spec: GameSpec, experiment: Dict, player_models: List[Model]):
        logger.info(f"Initializing {self.__class__.__name__} with game_spec: {game_spec}")
        super().__init__(game_spec=game_spec, experiment=experiment, player_models=player_models)
        self.game_name = game_spec['game_name']
        self.ta_env_id = experiment['name']
        self.players_by_id = { -1: "GM" }     # This corresponds to the player IDs used by TextArena
        self.logged_observation_count = {}
        self.request_violation = False  # Flag to indicate if a request violation occurred in last turn
        self.played_in_turn = set()  # Keep track of players who have played in the current turn
        self.last_player = None # Needed to detect possible request violation in last turn in _on_after_game()
        self.started = False  # Flag to indicate if the game has started
        self.done = False  # Flag to indicate if the game is done
        self.checked_observations = {}

    def setup(self, **kwargs):
        self.started = True
        self.env = ta.make(env_id=self.ta_env_id + "-raw")        
        self.env = ClemObservationWrapper(env=self.env, game_master=self, num_players=len(self.player_models))
        self._override_variables()
        # This should be redundant, since the environment is reset with the seed, but it doesn't hurt
        random.seed(kwargs['seed'])
        np.random.seed(kwargs['seed'])
        self.env.reset(num_players=len(self.player_models), seed=kwargs['seed'])     # reset sets the initial prompts for each player
        for player_id, (player_spec, player_model) in enumerate(zip(self.game_spec.player_specs, self.player_models)):
            # make a custom Player subclass for each player_spec, named after the role
            role = player_spec.get('role', 'Player')
            custom_response = player_spec.get('custom_response', None)
            player_class = type(role, (TextArenaPlayer,), {})
            # Create an instance of the player class with the model
            player_instance = player_class(model=player_model, master=self, custom_response=custom_response, ta_player_id=player_id)
            self.add_player(player_instance, player_id=player_id)

    def _override_variables(self):
        """
        Placeholder for functionalities in subclasses to override variables
        """
        pass

    def has_started(self):
        return self.started
    
    def is_done(self):
        return self.done

    def add_player(self, player: Player, player_id: int):
        """
        Add a player to the game master.
        """
        player.register_many(self._loggers)
        self.players_by_id[player_id] = player
        player.name = f"Player {player_id + 1}"        # TA player ID 0 is Player 1, etc.
        self.log_player(player.name, game_role=player.game_role, model_name=player.model.name)
        self.checked_observations[player_id] = 0

    def get_current_player(self) -> Optional[Player]:
        """
        Playpen needs this.
        Get the current player based on the environment's state.
        """
        return self.players_by_id[self.env.state.current_player_id]
    
    def get_context_for(self, player: Player) -> Dict:
        """
        Playpen needs this.
        Directly call the ClemObservationWrapper's observation method and bypass all other logic.
        """
        context = self.env.observation(player_id=player.ta_id)
        return context
    
    def observe(self) -> Tuple[Player, Dict]:
        player = self.get_current_player()
        context = self.get_context_for(player)
        return player, context
    
    def play(self):
        """
        Main play loop method. This method is called to run the game for benchmarking.
        """
        done = False
        while not done:
            player, context = self.observe()
            response = player(context)
            done, info = self.step(response=response)

    def step(self, response: str) -> Tuple[Union[bool, List], Union[Dict, List]]:
        self.request_violation = False  # Reset request violation for this turn
        done, info = self.env.step(action=response)
        self.request_violation = self._check_move_validity()  # Check if the last player made an invalid move
        self.done = done
        if info:
            logger.info(f"Game info passed by TextArena's env.step(): {info}")
        if not done:
            if self._start_next_round():
                self._prepare_next_round()
        else:
            self._on_after_game()
        return done, info
    
    def _start_next_round(self):
        """
        check if all players have played in the current turn and no request violation occurred.
        """
        return len(self.played_in_turn) == len(self.players_by_id) - 1 and not self.request_violation

    def _prepare_next_round(self):
        self.played_in_turn.clear()  # Clear the set for the next round
        self.log_next_round()

    def _check_move_validity(self):
        """
        Check if the last player made an invalid move.
        """
        player_id = self.env.state.current_player_id
        checked_observations = self.checked_observations[player_id]
        last_observations = self.env.state.observations[player_id][checked_observations:]
        self.checked_observations[player_id] = len(self.env.state.observations[player_id])  # Update the checked observations count
        for observation in last_observations:
            if observation and "attempted an invalid move" in observation[1]:
                self.count_request_violation()
                return True
        return False
    
    def _on_after_game(self):
        """
        This method is called after the game ends.
        It retrieves the rewards generated by env.close()
        """
        rewards = self.env.close()
        self.log_to_self(type_='numeric rewards', value=rewards[0])
        self.log_to_self(type_='other rewards', value=rewards[1])
        self.log_key("ta_rewards", rewards)
        self._log_metrics(rewards)
        self.log_game_end()
        

    def _log_metrics(self, rewards: Dict):
        # """
        # Log the metrics for the game.
        # Done in GameMaster, because the TA env still exists
        # """
        # # Rewards appears to be a tuple:
        # # First, success/loss for each player (1 for success, 0 for loss),
        # # Then: last message to each player, e.g. `'invalid_move': False, 'turn_count': 7, 'reason': 'Congratulations! You solved the Tower of Hanoi puzzle.'`
        # player_success = rewards[0]
        # # success is mean of player_success values. Not sure if this works for all games.
        # success = np.mean(list(player_success.values()))
        # last_messages = rewards[1]
        # self.log_to_self(type_='game_end', value=str(last_messages))
        # aborted = 0
        # for player_id in self.players_by_id:
        #     if player_id == -1:
        #         continue
        #     if last_messages[player_id]['invalid_move'] and self.last_player == player_id:
        #         self.count_request_violation()
        #         aborted = 1
        # self.log_key(METRIC_ABORTED, aborted)
        # self.log_key(METRIC_SUCCESS, success)
        # if success == 1:
        #     self.log_key(METRIC_LOSE, 0)
        # else:
        #     self.log_key(METRIC_LOSE, 1)
        # self.log_key(BENCH_SCORE, sum(player_success.values()) / len(player_success))
        # self.log_game_end()
        # TODO: I suppose I need subclasses for each game to log the metrics correctly.
        pass

    def _init_metrics(self):
        """
        returns default values for the metrics.
        """
        return {
            METRIC_ABORTED: 0,
            METRIC_SUCCESS: 0,
            METRIC_LOSE: 0,
            BENCH_SCORE: np.nan
        }

    def _log_metric_keys(self, metrics: Dict[str, int]):
        """
        Log the metric keys for the game.
        """
        self.log_key(METRIC_ABORTED, metrics[METRIC_ABORTED])
        self.log_key(METRIC_SUCCESS, metrics[METRIC_SUCCESS])
        self.log_key(METRIC_LOSE, metrics[METRIC_LOSE])
        self.log_key(BENCH_SCORE, metrics[BENCH_SCORE])
