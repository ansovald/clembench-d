import os
import json

from clemcore.clemgame import GameInstanceGenerator

import sys
import os
import random

sys.path.append(os.path.abspath('../clemgames/textmapworld'))
from textmapworld_utils import load_check_graph, generate_filename, create_graphs_file


"Enter the parameters for the game instance generator"
"-------------------------------------------------------------------------------------------------------------"
"°°°°°°°changeable parameters°°°°°°°"
SEED = 42

strict = True
create_new_graphs = False # True or False   !if True, the graphs will be created again, threfore pay attention!
n = 4
m = 4
instance_number = 10
game_type = "named_graph" #"named_graph" or "unnamed_graph"
cycle_type="cycle_false" #"cycle_true" or "cycle_false"
ambiguity= None #(repetition_rooms, repetition_times) or None
if strict:
    RESPONSE_REGEX = '^\{\s*"action":\s*"([^{}]*?)"\s*,\s*"graph":\s*(\{\s*"nodes"\s*:\s*\[.*?\]\s*,\s*"edges"\s*:\s*\{.*?\}\s*\})\s*\}$'
    DONE_REGEX = '^DONE$'
    MOVE_REGEX = '^GO:\s*(north|east|west|south)$'
else:
    RESPONSE_REGEX = "^\{[\s]*\"action\":\s*\"([^\{]*?)\"\s*,\s*\"graph\":\s*(\{\s*\"nodes\"\s*:\s*\[.*\]\s*,\s*\"edges\"\s*:\s*\{.*\})\s*\}"
    DONE_REGEX = 'DONE'
    MOVE_REGEX = 'GO:\s*(north|east|west|south)'
loop_reminder = False
max_turns_reminder = False
experiments = {"small": (4,"cycle_false"), "medium": (6, "cycle_false"), "large": (8, "cycle_false")}

"°°°°°°°imported parameters°°°°°°°"
prompt_file_name = 'PromptNamedGame.template'
prompt_file_name = os.path.join('resources', 'initial_prompts', prompt_file_name)
game_name = "textmapworld_graphreasoning"

with open(os.path.join("..", "clemgames", "textmapworld", game_name, 'resources', 'initial_prompts', "answers.json")) as json_file:
    answers_file = json.load(json_file)
with open(os.path.join("..", "clemgames", "textmapworld", game_name, 'resources', 'initial_prompts', "reminders.json")) as json_file:
    reminders_file = json.load(json_file)
"-------------------------------------------------------------------------------------------------------------"

class GraphGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self,  ):
        super().__init__(os.path.dirname(os.path.abspath(__file__)))

    def on_generate(self):
        player_a_prompt_header =  self.load_template(prompt_file_name)
        Player2_positive_answer = answers_file["PositiveAnswerNamedGame"] 
        Player2_negative_answer = answers_file["NegativeAnswerNamedGame"]
        game_id = 0
        for key, value in experiments.items():
            experiment = self.add_experiment(key)
            size, cycle_type = value
            created_name= generate_filename(game_type, size, cycle_type, ambiguity)
            file_graphs = os.path.join("..", "clemgames", "textmapworld", game_name, 'files', created_name)
            if not create_new_graphs:
                if not os.path.exists(file_graphs):
                    raise ValueError("New graphs are not created, but the file does not exist. Please set create_new_graphs to True.")
            else:
                if os.path.exists(file_graphs):
                    raise ValueError("The file already exists, please set create_new_graphs to False.")
                create_graphs_file(file_graphs, instance_number, game_type, n, m, size, cycle_type, ambiguity, game_name)
                
            if os.path.exists(file_graphs):
                grids = load_check_graph(file_graphs, instance_number, game_type)
                for grid in grids:
                    game_instance = self.add_game_instance(experiment, game_id)
                    game_instance["Prompt"] = player_a_prompt_header
                    game_instance["Player2_positive_answer"] = Player2_positive_answer
                    game_instance["Player2_negative_answer"] = Player2_negative_answer
                    game_instance["Move_Construction"] = MOVE_REGEX
                    game_instance["Stop_Construction"] = DONE_REGEX 
                    game_instance["Response_Construction"] = RESPONSE_REGEX
                    game_instance["Grid_Dimension"] = str(grid["Grid_Dimension"])
                    game_instance['Graph_Nodes'] = str(grid['Graph_Nodes'])
                    game_instance['Graph_Edges'] = str(grid['Graph_Edges'])
                    game_instance['Current_Position'] = str(grid['Initial_Position'])
                    game_instance['Picture_Name'] = grid['Picture_Name']
                    game_instance["Directions"] = str(grid["Directions"])
                    game_instance["Moves"] = str(grid["Moves"])
                    game_instance['Cycle'] = grid['Cycle']
                    game_instance['Ambiguity'] = grid['Ambiguity']
                    game_instance['Game_Type'] = game_type
                    game_instance["Loop_Reminder"] = loop_reminder
                    game_instance["Loop_Reminder_Text"] = reminders_file["loop_reminder"]
                    game_instance["Max_Turns_Reminder"] = max_turns_reminder
                    game_instance["Max_Turns_Reminder_Text"] = reminders_file["max_turns_reminder"]
                    game_instance["Mapping"] = str(grid["Mapping"])
                    game_instance["Strict"] = strict
                    game_id += 1


                        

if __name__ == '__main__':
    random.seed(SEED)

    # always call this, which will actually generate and save the JSON file
    GraphGameInstanceGenerator().generate()