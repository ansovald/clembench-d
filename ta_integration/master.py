from math import exp
from re import M
import ast

from py import log
from clemcore import backends
from clemcore.backends import Model
from clemcore.clemgame import Player, GameMaster, GameBenchmark, GameScorer
from clemcore.clemgame.registry import GameSpec
from clemcore.clemgame.recorder import NoopGameRecorder
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, BENCH_SCORE

import textarena as ta
from textarena.agents.basic_agents import STANDARD_GAME_PROMPT
from textarena.core import ObservationWrapper, Env, ObservationType

from typing import List, Callable, Dict, Text, Optional, Tuple, Union
import logging
import random
import numpy as np

# Local type aliases to avoid import issues with the framework. Copied from textarena/core.py
Message = Tuple[int, str]
Observations = Dict[int, List[Message]]

# TODO:
#    Problem with messages containing restricted words: Player 0 is reprompted, but eventually, 
#    the message containing restricted content is still passed on to Player 1. 
#    This appears to be a bug of the TabooEnv


logger = logging.getLogger(__name__)

class TextArenaPlayer(Player):
    """
    TextArenaPlayer is a custom player for the TextArena environment.
    It will be subclassed for each player role defined in the game specification.
    """
    
    def __init__(self, model: Model, ta_player_id: int, custom_response: list = None):
        super().__init__(model=model)
        self.custom_response = custom_response if custom_response else ["Whatever"]
        self.ta_id = ta_player_id  # This is the player ID used by TextArena, which may differ from the GameMaster's player ID
    
    def _custom_response(self, context):
        return random.choice(self.custom_response)
    
class ClemObservationWrapper(ObservationWrapper):
    """
    ClemObservationWrapper is a custom observation wrapper for the TextArena environment.
    """
    
    def __init__(self, env: Env, game_master: GameMaster):
        super().__init__(env)
        # self.temp_observations: Dict[int, List[Tuple[int, str]]] = {}
        self.temp_observations = {}
        self.game_master = game_master
    
    def _convert_obs_to_context(self, player_id):
        """
        Takes the observations the player hasn't seen yet, and prepares them as content in a context dict.
        """
        # logger.info(f"Converting observations for player {player_id} to context.")
        # logger.info(f"Current temp observations: {self.temp_observations[player_id]}")
        # logger.info(f"All observations: {self.env.state.observations}")
        content = ""
        if self.game_master.players_by_id[player_id]._is_initial_call:
            content += STANDARD_GAME_PROMPT + "\n\n"
        
        if player_id in self.temp_observations:
            for sender_id, message, observation_type in self.temp_observations[player_id]:
                if sender_id != player_id:
                    # logger.info(f"Adding observation from sender {sender_id}:\n{message}\n(type: {observation_type})")
                    sender_name = self.env.state.role_mapping.get(sender_id, f"Player {sender_id}")
                    content += f"\n[{sender_name}] {message}"
                # else:
                #     logger.info(f"Skipping observation from self ({sender_id}) for player {player_id}: {message}")
        # logger.info(f"Final content for player {player_id}: {content}")
        self.temp_observations[player_id] = []
        # logger.info(f"Cleared temp observations for player {player_id}.")

        return {'role': 'user', 'content': content}

    def observation(self, player_id: int, observation: Optional[Observations]):
        # Every time a player receives an observation, reset the request violation flag.
        # If it is true, turn counter will not be incremented in play()
        self.game_master.request_violation = False
        logger.info(f"Resetting request violation flag.")
        # logger.info(f"\n\nClemObservationWrapper received observation for player {player_id}: {observation}")
        if observation is None:
            logger.info(f"ClemObservationWrapper received None observation for player {player_id}.")
            return self._convert_obs_to_context(player_id=player_id)

        # Extend the full observations with the current observations without duplicates
        if player_id not in self.temp_observations:
            self.temp_observations[player_id] = []

        # Append new observations in sequence
        if type(observation) is str:
            # logger.info(f"\t\tObservation for player {player_id} is a string")
            self.temp_observations[player_id].append(tuple([-1, observation, ObservationType.PROMPT]))
        else:
            # logger.info(f"\t\tObservation for player {player_id} is of type {type(observation)}")
            print(observation[-1])

            # TODO: Something is off with WordChain. 
            # Players are mixed up in the messages, and penalized for words from the other player, or something like that.
            if "attempted an invalid move" in observation[-1][1]:
                logger.info(f"Player {player_id} attempted an invalid move, counting as a request violation.")
                self.game_master.request_violation = True
                logger.info(f"Setting request violation flag to True")
                self.game_master.game_recorder.count_request_violation()
            self.temp_observations[player_id].extend(observation)

        return self._convert_obs_to_context(player_id=player_id)


class TextArenaGameMaster(GameMaster):
    """
    TextArenaGameMaster is a custom game master for the TextArena environment.
    It inherits from the GameMaster class and implements the required methods.
    """
    
    def __init__(self, game_spec: GameSpec, experiment: Dict, player_models: List[backends.Model]):
        super().__init__(game_spec=game_spec, experiment=experiment, player_models=player_models)
        self.ta_env_id = experiment['name']
        self.players_by_id = { -1: "GM" }     # This corresponds to the player IDs used by TextArena
        self.context_for_player: Dict[str, Dict] = dict()  # context entries look like {"role": "user", "content": ...}
        self.game_recorder = NoopGameRecorder()
        self.request_violation = False
        self.last_player = None # Needed to detect possible request violation in last turn

    def setup(self, **kwargs):
        random.seed(kwargs['seed'])
        np.random.seed(kwargs['seed'])
        self.env = ta.make(env_id=self.ta_env_id + "-raw")
        
        # Apply variable overrides from the game specification.
        # This is to ensure deterministic behavior of the game, e.g. for sorting the word list in WordChains,
        # to make sampling from the word list deterministic.
        if 'override_variables' in kwargs:
            self._apply_variable_overrides(kwargs['override_variables'])
        # TODO: check if this might lead to problems, since some games have more than one default wrapper
        #       I am currently using the `-raw` version of the environment, which does not have any wrappers applied.
        #       Instead, just use ClemObservationWrapper
        self.env = ClemObservationWrapper(env=self.env, game_master=self)
        self.env.reset(num_players=len(self.player_models), seed=kwargs['seed'])     # reset sets the initial prompts for each player
        for player_id, (player_spec, player_model) in enumerate(zip(self.game_spec.player_specs, self.player_models)):
            # make a custom Player subclass for each player_spec, named after the role
            role = player_spec.get('role', 'Player')
            custom_response = player_spec.get('custom_response', None)
            player_class = type(role, (TextArenaPlayer,), {})
            # Create an instance of the player class with the model
            player_instance = player_class(model=player_model, custom_response=custom_response, ta_player_id=player_id)

            self.add_player(player_instance, player_id=player_id)

    def _apply_variable_overrides(self, override_variables: Dict[str, str]):
        """
        Apply variable overrides from the game specification to the environment.
        The override_variables dict contains variable paths as keys and expressions as values.
        """
        print(f"Applying variable overrides: {override_variables}")
        
        for variable_path, expression in override_variables.items():
            try:
                logger.info(f"Applying override for {variable_path} with expression: {expression}")
                # Parse the variable path (e.g., "self.env.word_list")
                path_parts = variable_path.split('.')
                # Start from self and navigate to the parent object
                current_obj = self
                for part in path_parts[:-1]:
                    current_obj = getattr(current_obj, part)
                # Get the attribute name to set
                attr_name = path_parts[-1]
                # Get the current value of the attribute
                current_value = getattr(current_obj, attr_name)
                # Evaluate the expression
                if expression.startswith('lambda'):
                    # Handle lambda expressions
                    func = eval(expression)
                    new_value = func(current_value)
                else:
                    # Raise an error if the expression is not a lambda
                    raise ValueError(f"Unsupported expression format: {expression}. Only lambda expressions are supported.")
                # Set the new value
                setattr(current_obj, attr_name, new_value)
                logger.info(f"Successfully overrode self.{variable_path}")
                
            except Exception as e:
                logger.error(f"Failed to apply override for {variable_path}: {e}")
                # Continue with other overrides even if one fails

    def add_player(self, player: Player, player_id: int):
        """
        Add a player to the game master.
        """
        player.game_recorder = self.game_recorder  # Ensure the player uses the same game recorder
        self.players_by_id[player_id] = player
        # self.players_by_id[player_id + 1] = player
        player.game_recorder = self.game_recorder  # player should record to the same interaction log
        player.name = f"Player {player_id + 1}"        # TA player ID 0 is Player 1, etc.
        self.log_player(player)

    def get_current_player(self) -> Optional[Player]:
        # Playpen needs this
        return self.players_by_id[self.env.state.current_player_id]

    def play(self):
        """
        Main play loop method. This method is called to run the game for benchmarking.
        """
        done = False
        played_in_turn = set()  # Keep track of players who have played in the current turn
        player_id, context = self.env.get_observation()
        while not done:
            logger.info(f"\n\nStarting new turn. Current player ID: {self.env.state.current_player_id}")
            played_in_turn.add(player_id)
            self.last_player = player_id  # Update the last player who played
            logger.info(f"In play loop.\nCurrent player ID: {player_id}\nContext: {context}")
            response = self.players_by_id[player_id](context)
            logger.info(f"Player {player_id} responded with: {response}")
            done, info = self.step(response)
            if info:
                logger.info(f"Game info: {info}")
            if not done:
                player_id, context = self.env.get_observation()
                if len(played_in_turn) == len(self.players_by_id) - 1 and not self.request_violation:  # All players have played in this turn, last move was valid
                    logger.info(f"All players have played in this turn, last move was valid: {played_in_turn}, {self.request_violation}")
                    played_in_turn.clear()  # Clear the set for the next turn
                    self.game_recorder.log_next_round()
            else:
                self._on_after_game()

    def step(self, response: str) -> Tuple[Union[bool, List], Union[Dict, List]]:
        return self.env.step(action=response)
    
    def _on_after_game(self):
        """
        TODO: Get the latest observations and scoring information from the environment.
              self.env.state.rewards is only for after game.
              The ToH env does submit a 'reward' when calling `set_invalid_move`, 
              but this is (currently?) not used in SinglePlayerState.
        """
        logger.info("In _on_after_game")
        rewards = self.env.close()
        self.log_key("rewards", rewards)
        # Rewards appears to be a tuple:
        # First, success/loss for each player (1 for success, 0 for loss),
        # Then: last message to each player, e.g. `'invalid_move': False, 'turn_count': 7, 'reason': 'Congratulations! You solved the Tower of Hanoi puzzle.'`
        player_success = rewards[0]
        # success is mean of player_success values. Not sure if this works for all games.
        success = np.mean(list(player_success.values()))
        last_messages = rewards[1]
        self.log_to_self(type_='game_end', value=str(last_messages))
        aborted = 0
        for player_id in self.players_by_id:
            if player_id == -1:
                continue
            if last_messages[player_id]['invalid_move'] and self.last_player == player_id:
                self.game_recorder.count_request_violation()
                aborted = 1
        self.log_key(METRIC_ABORTED, aborted)
        self.log_key(METRIC_SUCCESS, success)
        if success == 1:
            self.log_key(METRIC_LOSE, 0)
        else:
            self.log_key(METRIC_LOSE, 1)
        self.log_key(BENCH_SCORE, sum(player_success.values()) / len(player_success))

class TextArenaScorer(GameScorer):
    """
    TextArenaScorer is a custom game scorer for the TextArena environment.
    TODO: Add scoring logic to game_instance?
    """
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_episode_scores(self, episode_interactions: Dict):
        bench_score = episode_interactions.get(BENCH_SCORE, 0)
        success = episode_interactions.get(METRIC_SUCCESS, 0)
        aborted = episode_interactions.get(METRIC_ABORTED, 0)
        lose = episode_interactions.get(METRIC_LOSE, 0)
        if aborted:
            self.log_episode_score(BENCH_SCORE, np.nan)
        else:
            self.log_episode_score(BENCH_SCORE, bench_score)
        self.log_episode_score(METRIC_SUCCESS, success)
        self.log_episode_score(METRIC_ABORTED, aborted)
        self.log_episode_score(METRIC_LOSE, lose)

class TextArenaBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        logger.info(f"Initializing TextArenaBenchmark with game_spec: {game_spec}")
        super().__init__(game_spec)
        self.game_spec = game_spec

    def create_game_master(self, experiment, player_models):
        return TextArenaGameMaster(self.game_spec, experiment, player_models)
    
    def create_game_scorer(self, experiment, game_instance):
        return TextArenaScorer(self.game_spec['game_name'], experiment, game_instance)