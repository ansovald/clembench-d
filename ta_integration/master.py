from math import exp

from py import log
from clemcore import backends
from clemcore.backends import Model
from clemcore.clemgame import Player, GameMaster, GameBenchmark
from clemcore.clemgame.registry import GameSpec
from clemcore.clemgame.recorder import NoopGameRecorder

import textarena as ta
from textarena.agents.basic_agents import STANDARD_GAME_PROMPT
from textarena.core import ObservationWrapper, Env, ObservationType
from textarena.envs.registration import ENV_REGISTRY

from typing import List, Callable, Dict, Text, Optional, Tuple, Union
import logging
import random
import numpy as np

# Local type aliases to avoid import issues with the framework. Copied from textarena/core.py
Message = Tuple[int, str]
Observations = Dict[int, List[Message]]

# TODO:
# 1. Set up proper logging (half done...?)
# 2. Find a way to tap into invalid messages and log them properly.
#    Right now, we have no GM to GM logs, but GM "broadcasts" show up in both player's messages.
# 3. Problem with messages containing restricted words: Player 0 is reprompted, but eventually, 
#    the message containing restricted content is still passed on to Player 1. 
#    Is this a bug in the TabooEnv, or in my Wrapper?


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

        # TODO: penalty handling! e.g.  (-1, 'You attempted an invalid move. Reason: You tried to move a disk from an empty tower. Please resubmit a valid move and remember to follow the game rules to avoid penalties.', <ObservationType.GAME_ADMIN: 6>)
        # TODO: Also: now the whole observation historey shows up in every message again.
        logger.info(f"Converting observations for player {player_id} to context.")
        logger.info(f"Current temp observations: {self.temp_observations[player_id]}")
        logger.info(f"All observations: {self.env.state.observations}")
        content = ""
        if self.game_master.players_by_id[player_id]._is_initial_call:
            content += STANDARD_GAME_PROMPT + "\n\n"
        
        if player_id in self.temp_observations:
            for sender_id, message, observation_type in self.temp_observations[player_id]:
                logger.info(f"Adding observation from sender {sender_id}:\n{message}\n(type: {observation_type})")
                if sender_id != player_id:
                    sender_name = self.env.state.role_mapping.get(sender_id, f"Player {sender_id}")
                content += f"\n[{sender_name}] {message}"
            # print(f'type of self.temp_observations[player_id]: {type(self.temp_observations[player_id])}')
            # content = ''.join(self.temp_observations[player_id])
        logger.info(f"Final content for player {player_id}: {content}")
        self.temp_observations[player_id] = []
        logger.info(f"Cleared temp observations for player {player_id}.")

        return {'role': 'user', 'content': content}

    def observation(self, player_id: int, observation: Optional[Observations]):
        logger.info(f"\n\nClemObservationWrapper received observation for player {player_id}: {observation}")
        if observation is None:
            logger.info(f"ClemObservationWrapper received None observation for player {player_id}.")
            return self._convert_obs_to_context(player_id=player_id)

        # Extend the full observations with the current observations without duplicates
        if player_id not in self.temp_observations:
            self.temp_observations[player_id] = []

        # Append new observations in sequence
        if type(observation) is str:
            logger.info(f"\t\tObservation for player {player_id} is a string")
            # TODO: Should probably add ObservationType
            self.temp_observations[player_id].append(tuple([-1, observation, ObservationType.PROMPT]))
        else:
            logger.info(f"\t\tObservation for player {player_id} is of type {type(observation)}")
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
        # self.players_by_id = {}                  # ta_id `0` is Player 1, `1` is Player 2, etc.
        self.context_for_player: Dict[str, Dict] = dict()  # context entries look like {"role": "user", "content": ...}
        self.game_recorder = NoopGameRecorder()

    def setup(self, **kwargs):
        random.seed(kwargs['seed'])
        np.random.seed(kwargs['seed'])
        self.env = ta.make(env_id=self.ta_env_id + "-raw")
        # TODO: check if this might lead to problems, since some games have more than one default wrapper
        #       I am currently using the `-raw` version of the environment, which does not have any wrappers applied.
        #       Instead, just use ClemObservationWrapper
        self.env = ClemObservationWrapper(env=self.env, game_master=self)
        self.env.reset(num_players=len(self.player_models))     # reset sets the initial prompts for each player
        for player_id, (player_spec, player_model) in enumerate(zip(self.game_spec.player_specs, self.player_models)):
            # make a custom Player subclass for each player_spec, named after the role
            role = player_spec.get('role', 'Player')
            custom_response = player_spec.get('custom_response', None)
            player_class = type(role, (TextArenaPlayer,), {})
            # Create an instance of the player class with the model
            player_instance = player_class(model=player_model, custom_response=custom_response, ta_player_id=player_id)

            self.add_player(player_instance, player_id=player_id)

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
        while not done:
            logger.info(f"\n\nStarting new turn. Current player ID: {self.env.state.current_player_id}")
            player_id, context = self.env.get_observation()
            logger.info(f"In play loop.\nCurrent player ID: {player_id}\nContext: {context}")
            response = self.players_by_id[player_id](context)
            logger.info(f"Player {player_id} responded with: {response}")
            done, info = self.step(response)
            if info:
                logger.info(f"Game info: {info}")
            if done:
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
        logger.info(f"self.env.state.observations: {self.env.state.observations}")
        rewards = self.env.close()
        logger.info(f"Rewards: {rewards}")

class TextArenaBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        logger.info(f"Initializing TextArenaBenchmark with game_spec: {game_spec}")
        super().__init__(game_spec)
        self.game_spec = game_spec

    def create_game_master(self, experiment, player_models):
        return TextArenaGameMaster(self.game_spec, experiment, player_models)