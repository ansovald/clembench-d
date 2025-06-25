import random
from typing import List, Dict
import numpy as np
import re
import logging

from clemcore.backends import Model
from clemcore.clemgame import DialogueGameMaster, Player, GameBenchmark, GameScorer, GameSpec
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, BENCH_SCORE, METRIC_LOSE, METRIC_REQUEST_COUNT, \
    METRIC_REQUEST_COUNT_PARSED, METRIC_REQUEST_COUNT_VIOLATED

logger = logging.getLogger(__name__)


class InstructionFollower(Player):

    def _custom_response(self, context):
        answer = random.choice(["first", "second", "third"])
        return f"Answer: {answer}"
    
    def __call__(self, context: Dict) -> str:
        response_text = super().__call__(context)
        return response_text


class InstructionGiver(Player):

    def _custom_response(self, context):
        answer = random.choice(["Expression: The one that looks like the target."])
        return "Expression: The one that looks like the target."
    
    def __call__(self, context: Dict) -> str:
        response_text = super().__call__(context)
        return response_text


class MultimodalReferenceGameMaster(DialogueGameMaster):

    def __init__(self, game_name: str, game_path: str, experiment: Dict, player_models: List[Model]):
        super().__init__(game_name, game_path, experiment, player_models)
        self.experiment = experiment
        self.game = None

    def _on_setup(self, **game_instance):
        self.game_instance = game_instance
        self.game_id = game_instance['game_id']
        self.player_1_prompt_header = game_instance['player_1_prompt_header']
        self.player_2_prompt_header = game_instance['player_2_prompt_header']
        self.target_image_number = game_instance['target_image_number']

        self.player_1_response_pattern = r'{}'.format(game_instance['player_1_response_pattern'])
        self.player_2_response_pattern = r'{}'.format(game_instance['player_2_response_pattern'])

        self.player_1_images = game_instance['player_1_images']
        self.player_2_images = game_instance['player_2_images']
        self.accepted_answers = game_instance['accepted_answers']
        self.all_accepted_answers = [item for sublist in self.accepted_answers for item in sublist]


        self.instruction_giver = InstructionGiver(self.player_models[0],
                                                  name="Player 1",
                                                  game_recorder=self.game_recorder)
        self.instruction_follower = InstructionFollower(self.player_models[1],
                                                        name="Player 2",
                                                        game_recorder=self.game_recorder)
        p1_initial_context = dict(role="user",
                       content=self.player_1_prompt_header,
                       image=self.player_1_images)
        self.add_player(self.instruction_giver, initial_context=p1_initial_context)
        self.add_player(self.instruction_follower)
        self.terminate = False
        self.aborted = False
        self.success = False
        self.aborted_at_player = None

    def _validate_player_response(self, player, response):
        """
        Decide if a player response matches the valid response patterns.
        An invalid response breaks the game rules and ends the game.

        Args:
            player: The player that gave the response.
            response: The response of the current player.
        Returns:
            True, if the response is fine. Otherwise, False.
        """
        if player == self.instruction_giver:
            # Player 1 response validation
            p1_match = re.compile(self.player_1_response_pattern, re.IGNORECASE).match(response)
            if p1_match and p1_match.group('remainder') == "":
                return True
            self.terminate = True
            # if the Player 1 message don't match the rule => start with "Expression: " and contains only one paragraph
            # log the message and abort the game
            action = {'type': 'invalid format', 'content': 'Invalid generated expression',
                      'original_content': response}
            self.log_event(from_="GM", to="GM", action=action)
            self.aborted = True
            self.aborted_at_player = '1'
            return False
        elif player == self.instruction_follower:
            # Game only has one round, so we terminate regardless of the response
            self.terminate = True
            # Player 2 response validation
            p2_match = re.compile(self.player_2_response_pattern, re.IGNORECASE).match(response)
            if p2_match and p2_match.group('remainder') == "":
                return True
            # abort the game if the output doesn't match the rule
            action = {'type': 'invalid format', 'content': 'Invalid generated choice',
                      'original_content': response}
            self.log_event(from_="GM", to="GM", action=action)
            self.aborted = True
            self.aborted_at_player = '2'
            return False
        
    def _parse_response(self, player, response):
        """ Takes a valid player response and parses it.

        Args:
            player: The Player instance that produced the response.
            response: The response of the current player.
        Returns:
            The parsed response
        """
        if player == self.instruction_giver:
            # Player 1 response parsing
            p1_match = re.compile(self.player_1_response_pattern, re.IGNORECASE).match(response)
            if p1_match:
                action = {'type': 'parse', 'content': response,
                          'expression': p1_match.group('content')}
                self.log_event(from_="GM", to="GM", action=action)
                return response
        elif player == self.instruction_follower:
            # Player 2 response parsing
            p2_match = re.compile(self.player_2_response_pattern, re.IGNORECASE).match(response)
            if p2_match:
                action = {'type': 'parse', 'content': response,
                      'answer': p2_match.group('content')}
                self.log_event(from_="GM", to="GM", action=action)
                return p2_match.group('content').lower()  # return the label only
        return None
    
    def _on_valid_player_response(self, player: Player, parsed_response: str) -> None:
        """Method executed after a player response has been parsed and validated.
        This method is used to set the context for the other player.

        Args:
            player: The Player instance that produced the response (or has been modified by the GM).
            parsed_response: The parsed and valid response of the current player.
        """
        if player == self.instruction_giver:
            # Game only has one round, so we can use the initial prompt header here
            self.set_context_for(self.instruction_follower, 
                                 content = self.player_2_prompt_header.replace('TARGET_EXPRESSION', parsed_response), 
                                 image = self.player_2_images)
        else:
            if parsed_response in self.accepted_answers[int(self.target_image_number) - 1]:
                self.log_to_self('guess_correct', parsed_response)
                self.success = True
            else:
                self.log_to_self('guess_wrong', f'expected answer: {self.target_image_number}')
    
    def _does_game_proceed(self):
        if self.terminate:
            return False
        return True
    
    def _on_after_game(self):
        self.log_key(METRIC_ABORTED, int(self.aborted))
        self.log_key(METRIC_SUCCESS, int(self.success))
        if self.aborted:
            self.log_key('Aborted at Player', self.aborted_at_player)


class MultimodalReferenceGameScorer(GameScorer):

    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)
        self.target_grid_name = game_instance["target_image_number"]
        self.player_2_response_pattern = game_instance["player_2_response_pattern"]

    def compute_scores(self, episode_interactions: Dict) -> None:
        '''
        Compute and log scores for one episode of referencegame.
        :param episode_interactions: the game episode interactions log
        '''

        self.log_episode_score(METRIC_ABORTED, episode_interactions[METRIC_ABORTED])
        aborted_at_player_1 = 0
        aborted_at_player_2 = 0
        if episode_interactions[METRIC_ABORTED] == 1:
            if episode_interactions['Aborted at Player'] == '1':
                aborted_at_player_1 = 1
            elif episode_interactions['Aborted at Player'] == '2':
                aborted_at_player_2 = 1
        self.log_episode_score('Aborted at Player 1', aborted_at_player_1)
        self.log_episode_score('Aborted at Player 2', aborted_at_player_2)

        request_count = 1
        parsed_request_count = 0
        turn = episode_interactions["turns"][0]
        # log generated expression length
        if aborted_at_player_1 == 0:
            parsed_request_count += 1
            p1_expression = turn[2]['action']['expression']
            expression_length = len(p1_expression)
            self.log_turn_score(0, 'Generated Expression Length', expression_length)
            self.log_episode_score('Generated Expression Length', expression_length)
            number_of_tokens = len(p1_expression.split(' '))
            self.log_turn_score(0, 'Generated Expression Number of Tokens', number_of_tokens)
            self.log_episode_score('Generated Expression Number of Tokens', number_of_tokens)
            request_count += 1
            if aborted_at_player_2 == 0:
                # Player 2 response is valid
                parsed_request_count += 1
        else:
            self.log_turn_score(0, 'Generated Expression Length', np.nan)
            self.log_episode_score('Generated Expression Length', np.nan)
            self.log_turn_score(0, 'Generated Expression Number of Tokens', np.nan)
            self.log_episode_score('Generated Expression Number of Tokens', np.nan)

        violated_request_count = request_count - parsed_request_count
        self.log_turn_score(0, METRIC_REQUEST_COUNT, request_count)
        self.log_turn_score(0, METRIC_REQUEST_COUNT_PARSED, parsed_request_count)
        self.log_turn_score(0, METRIC_REQUEST_COUNT_VIOLATED, violated_request_count)
        self.log_episode_score(METRIC_REQUEST_COUNT, request_count)
        self.log_episode_score(METRIC_REQUEST_COUNT_PARSED, parsed_request_count)
        self.log_episode_score(METRIC_REQUEST_COUNT_VIOLATED, violated_request_count)

        self.log_episode_score(METRIC_LOSE, 1 - episode_interactions[METRIC_SUCCESS])

        bench_score = np.nan
        if episode_interactions[METRIC_ABORTED] == 0:
            bench_score = episode_interactions[METRIC_SUCCESS] * 100
            
        self.log_episode_score(BENCH_SCORE, bench_score)


class MultimodalReferenceGameBenchmark(GameBenchmark):

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> DialogueGameMaster:
        return MultimodalReferenceGameMaster(self.game_name, self.game_path, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return MultimodalReferenceGameScorer(self.game_name, experiment, game_instance)
