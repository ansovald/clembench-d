from math import exp
from clemcore import backends
from clemcore.backends import Model
from clemcore.clemgame import Player, GameMaster, GameBenchmark
from clemcore.clemgame.registry import GameSpec
from clemcore.clemgame.recorder import NoopGameRecorder

import textarena as ta
from textarena.agents.basic_agents import STANDARD_GAME_PROMPT
from textarena.core import ObservationWrapper, Env
from textarena.envs.registration import ENV_REGISTRY
from textarena.wrappers import LLMObservationWrapper

from typing import List, Callable, Dict, Text, Optional, Tuple, Union
import logging
import random
import numpy as np


# TODO:
# 1. Set up proper logging
# 2. Find a way to tap into invalid messages and log them properly. They do show up in the observations.


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

    def __call__(self, context: Dict, memorize: bool = True) -> str:
        """
        Let the player respond (act verbally) to a given context.

        Args:
            context: The context to which the player should respond.
            memorize: Whether the context and response are to be added to the player's message history.
        Returns:
            The textual response.
        """
        assert context["role"] == "user", f"The context must be given by the user role, but is {context['role']}"
        return super().__call__(context=context, memorize=False)
    
    def _custom_response(self, context):
        return random.choice(self.custom_response)
    
class ClemWrapper(ObservationWrapper):
    """
    ClemWrapper is a custom observation wrapper for the TextArena environment.
    """
    
    def __init__(self, env: Env, game_master: GameMaster):
        super().__init__(env)
        self.full_observations = {}
        self.full_contexts = {}

    # TODO: mirror LLMObservationWrapper's observation method
    #       and tap into it to build the logs

    def observation(self, player_id: int, observation: str) -> str:
        """
        Process the observation for a specific player and return a formatted string.
        
        Args:
            player_id (int): The ID of the player.
            observation (str): The observation to process.
        
        Returns:
            str: The formatted observation string.
        """
        print(f"ClemWrapper.observation(player_id={player_id}, observation={observation})")
        if player_id not in self.full_observations:
            self.full_observations[player_id] = []
        
        self.full_observations[player_id].append((player_id, observation))
        return "\n".join([f"[Player {pid}] {obs}" for pid, obs in self.full_observations[player_id]])

class TextArenaGameMaster(GameMaster):
    """
    TextArenaGameMaster is a custom game master for the TextArena environment.
    It inherits from the GameMaster class and implements the required methods.
    """
    
    def __init__(self, game_spec: GameSpec, experiment: Dict, player_models: List[backends.Model]):
        super().__init__(game_spec=game_spec, experiment=experiment, player_models=player_models)
        self.ta_env_id = experiment['name']
        self.players_by_id = { -1: "GM" }     # This corresponds to the player IDs used by TextArena
        self.context_for_player: Dict[str, Dict] = dict()  # context entries look like {"role":"user", "content": ...}
        self.game_recorder = NoopGameRecorder()  # NoopGameRecorder is used to avoid recording interactions in this benchmark

    def setup(self, **kwargs):
        random.seed(kwargs['seed'])
        np.random.seed(kwargs['seed'])
        self.env = ta.make(env_id=self.ta_env_id)
        self.env = LLMObservationWrapper(env=self.env)
        self.env.reset(num_players=len(self.player_models))
        observations = self.env.state.observations
        for player_id, (player_spec, player_model) in enumerate(zip(self.game_spec.player_specs, self.player_models)):
            # make a custom Player subclass for each player_spec, named after the role
            role = player_spec.get('role', 'Player')
            custom_response = player_spec.get('custom_response', None)
            player_class = type(role, (TextArenaPlayer,), {})
            # Create an instance of the player class with the model
            player_instance = player_class(model=player_model, custom_response=custom_response, ta_player_id=player_id)

            initial_prompt = observations[player_id][0][1]  # Get the initial observation for the player
            self.add_player(player_instance, player_id=player_id, initial_context=initial_prompt)

    def add_player(self, player: Player, player_id: int, initial_context: str):
        """
        Add a player to the game master.
        """
        self.players_by_id[player_id] = player
        player.game_recorder = self.game_recorder  # player should record to the same interaction log
        player.name = f"Player {player_id}"        # Mirrors TextArena's player ID
        self.set_context_for(player, initial_context)
        # print(f"Initial context for {player.name}:\n{initial_context}\n")
    
    def set_context_for(self, player: Player, content: str, **extras):
        """
        Set the context for the specified Player. The player will be prompted with the context on its next turn.

        The context always has a 'role' and 'content' entry where the 'role' is always set to 'user'.
        Args:
            player: The player to set the context for.
            content: The text content to be added to the context.
            extras: Additional content to be merged into the context e.g. information about images. 
                    NOTE: TA does not currently support images, so this is just a placeholder.
        """
        if player is None:
            return
        message = {"role": "user", "content": STANDARD_GAME_PROMPT + "\n\n" + content}
        context = {**extras, **message}
        self.context_for_player[player.name] = context
        # print(f"Setting context for player {player.name}:\n\trole:{context['role']}\n\tcontent:{context['content']}\n\n")
    
    def get_current_player(self) -> Optional[Player]:
        # Playpen needs this
        return self.players_by_id[self.env.state.current_player_id]
    
    def get_context_for(self, player) -> Dict:
        assert player is not None, "Cannot get player context for 'None'"
        assert player.name in self.context_for_player, f"No context set for {player.name}"
        context = self.context_for_player[player.name]
        assert "role" in context, f"Player context must have a 'role' entry"
        assert context["role"] == "user", f"Role of player context must be 'user'"
        assert "content" in context, f"Player context must have a 'content' entry"
        # print(f"Getting context for player {player.name}:\n\trole:{context['role']}\n\tcontent:{context['content']}\n\n")
        return context

    def play(self):
        """
        Main play loop method. This method is called to run the game for benchmarking.
        """
        done = False
        while not done:
            current_player = self.get_current_player()
            context = self.get_context_for(current_player)
            response = current_player(context)
            print(f"Player {current_player.name} response: '{response}'")
            done, info = self.step(response)
            if info:
                logger.info(f"Game info: {info}")
            _, observation = self.env.get_observation()
            self.set_context_for(self.get_current_player(), observation)

    def step(self, response: str) -> Tuple[Union[bool, List], Union[Dict, List]]:
        return self.env.step(action=response)

class TextArenaBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        logger.info(f"Initializing TextArenaBenchmark with game_spec: {game_spec}")
        super().__init__(game_spec)
        self.game_spec = game_spec

    def create_game_master(self, experiment, player_models):
        return TextArenaGameMaster(self.game_spec, experiment, player_models)