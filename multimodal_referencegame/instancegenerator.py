"""
Generate instances for the referencegame
Version 1.6 (strict regex parsing)

Reads grids_v1.5.json from resources/ (grids don't change in this version)
Creates instances.json in instances/
"""
import os
import random
from clemcore.clemgame import GameInstanceGenerator
import shutil
import matplotlib.pyplot as plt
import json
import logging
from collections import OrderedDict

random.seed(123)

logger = logging.getLogger(__name__)

MAX_NUMBER_INSTANCES = 30
LANGUAGES = ['en']

class ReferenceGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self):
        super().__init__(os.path.dirname(os.path.abspath(__file__)))

    def general_instance(self, language):
        """
        returns a dictionary with the initial prompts and keywords for the game,
        to be added to each game instance
        """
        instance = {}
        instance['player_1_prompt_header'] = self.load_template(os.path.join("resources", "initial_prompts", language, "player_1_prompt_images.template"))
        instance['player_2_prompt_header'] = self.load_template(os.path.join("resources", "initial_prompts", language, "player_2_prompt_images.template"))
        keywords = self.load_json("resources/keywords.json")[language]
        instance['player_1_response_pattern'] = f'^{keywords["expression"]}:\s(?P<content>.+)\n*(?P<remainder>.*)'
        content = '|'.join([item for sublist in keywords['accepted_answers'] for item in sublist])
        instance['player_2_response_pattern'] = f'^{keywords["answer"]}:\s(?P<content>{content})\n*(?P<remainder>.*)'
        # 'content' can directly be compared to gold answer
        # 'remainder' should be empty (if models followed the instructions)

        # the following two fields are no longer required, but kept for backwards compatibility with previous instance versions
        instance["player_1_response_tag"] = f"{keywords['expression']}:"
        instance["player_2_response_tag"] = f"{keywords['answer']}:"
        instance["target"] = keywords['target']
        instance["distractor"] = keywords['distractor']
        instance["accepted_answers"] = keywords['accepted_answers']
        return instance
    
    def prepare_player_1_prompt(self, prompt: str, target_img: int):
        """
        Prepares the player A prompt by replacing FIRST/SECOND/THIRD_IMAGE placeholders with "the target"/"a distractor
        """
        if target_img == 1:
            return prompt.replace("FIRST_IMAGE", "the target").replace("SECOND_IMAGE", "a distractor").replace("THIRD_IMAGE", "a distractor")
        elif target_img == 2:
            return prompt.replace("FIRST_IMAGE", "a distractor").replace("SECOND_IMAGE", "the target").replace("THIRD_IMAGE", "a distractor")
        elif target_img == 3:
            return prompt.replace("FIRST_IMAGE", "a distractor").replace("SECOND_IMAGE", "a distractor").replace("THIRD_IMAGE", "the target")
        else:
            raise ValueError("target_img must be 1, 2 or 3")

    def get_ade_dataset(self):
        sequences = self.load_csv(os.path.join("resources", "sequences.csv"))

        ade_dataset = dict()
        for s in sequences:
            line = s[0].split("\t")

            if line[0] == '':
                continue

            image_path = os.path.join("resources", "ade_images", line[3].split("/")[-1])
            image_category = line[4]

            if image_category not in ade_dataset:
                ade_dataset[image_category] = [image_path]
            else:
                ade_dataset[image_category].append(image_path)
        return ade_dataset

    def get_docci_dataset(self):

        file = open('resources/docci_dataset/docci_metadata.jsonlines', 'r')

        dataset = dict()
        for s in file.readlines():
            line = json.loads(s)

            image_path = os.path.join("resources", "docci_dataset", "images", line["example_id"], ".jpg")

            for ann in line["cloud_vision_api_responses"]["labelAnnotations"]:
                if ann["score"] >= 0.8:
                    image_category = ann["description"].lower()

                    if image_category not in dataset:
                        dataset[image_category] = [image_path]
                    else:
                        if image_path not in dataset[image_category]:
                            dataset[image_category].append(image_path)
        return dataset

    def get_clevr_dataset(self):

        files_to_process = ['CLEVR_train_scenes.json', 'CLEVR_val_scenes.json']
        category2image = dict()
        image2category = dict()

        for file in files_to_process:
            data = self.load_json(os.path.join('resources', 'CLEVR_v1.0', 'scenes', file))

            for scene in data['scenes']:
                if file == 'CLEVR_train_scenes.json':
                    image_path = os.path.join('resources', 'CLEVR_v1.0', 'images', 'train', scene['image_filename'])
                else:
                    image_path = os.path.join('resources', 'CLEVR_v1.0', 'images', 'val', scene['image_filename'])

                # TODO: for just creating image category dict, we don't need this check
                # if os.path.exists(image_path) == False:
                #     continue

                for object in scene['objects']:

                    image_category = object['size']+ ' '+object['color']+' '+object['shape'] +' '+object['material']

                    if image_category not in category2image:
                        category2image[image_category] = [image_path]
                    else:
                        if image_path not in category2image[image_category]:
                            category2image[image_category].append(image_path)

                    if image_path not in image2category:
                        image2category[image_path] = [image_category]
                    else:
                        if image_category not in image2category[image_path]:
                            image2category[image_path].append(image_category)

        return category2image, image2category

    def select_random_item(self, images:list):
        random_index = random.randint(0, len(images)-1)
        return images[random_index]

    def plot_grid(self, grid, file_path):
        fig, ax = plt.subplots()
        ax.set_xticks([])
        ax.set_yticks([])

        for y in range(len(grid)):
            for x in range(len(grid[y])):
                if grid[y][x] == "▢":
                    ax.add_patch(plt.Rectangle((x, -y - 1), 1, 1, facecolor="white", edgecolor="black"))
                else:
                    ax.add_patch(plt.Rectangle((x, -y - 1), 1, 1, facecolor="white", edgecolor="black"))
                    ax.text(x + 0.5, -y - 0.5, grid[y][x], ha='center', va='center', color='black', fontsize=12)

        ax.autoscale_view()
        # plt.show()
        plt.savefig(file_path)

    def process_grid(self, saved_grids, grid):

        if grid not in saved_grids:
            ascii_grid = []
            lines = grid.split('\n')
            for l in lines:
                line = l.split(' ')
                ascii_grid.append(line)

            file_path = os.path.join("resources", "grid_images", f"{len(saved_grids)}.png")
            self.plot_grid(ascii_grid, file_path)

            saved_grids[grid] = os.path.join("games", "multimodal_referencegame", file_path)
        return saved_grids[grid]

    def generate_grid_instances(self, general_instance=None, language=None):
        # GRID EXPERIMENT
        instances = {}
        saved_grids = {}

        with open(os.path.join("resources", "ascii_game_instances.json")) as json_file:
            instances = json.load(json_file)

        for exp in instances['experiments']:

            game_counter = 0
            experiment = self.add_experiment(exp['name'])

            for instance in exp['game_instances']:

                player_1_target_grid = self.process_grid(saved_grids, instance['player_1_target_grid'])
                player_1_second_grid = self.process_grid(saved_grids, instance['player_1_second_grid'])
                player_1_third_grid = self.process_grid(saved_grids, instance['player_1_third_grid'])

                player_2_first_grid = self.process_grid(saved_grids, instance['player_2_first_grid'])
                player_2_second_grid = self.process_grid(saved_grids, instance['player_2_second_grid'])
                player_2_third_grid = self.process_grid(saved_grids, instance['player_2_third_grid'])

                game_instance = self.add_game_instance(experiment, game_counter)

                # import all keywords from the general instance
                game_instance.update(general_instance)
                game_instance['player_1_prompt_header'] = self.prepare_player_1_prompt(game_instance['player_1_prompt_header'], 1)
                game_instance['player_1_images'] = [player_1_target_grid, player_1_second_grid, player_1_third_grid]
                game_instance['player_2_images'] = [player_2_first_grid, player_2_second_grid, player_2_third_grid]
                game_instance['target_image_number'] = instance['target_grid_name'][2]

                game_counter += 1


    def add_shuffled_instances(self, general_instance: dict, image_paths: list[str], experiment: OrderedDict, game_counter: int):
        for target_p2 in [1, 2, 3]:
            game_instance = self.add_game_instance(experiment, game_counter)
            game_instance.update(general_instance)
            game_instance.update(self.prepare_instance(game_instance['player_1_prompt_header'], image_paths, target_p2))

            game_counter += 1

            if game_counter >= MAX_NUMBER_INSTANCES:
                break

        return game_counter


    def prepare_instance(self, player_1_prompt_header: str, image_paths: list[str], target_p2: int):
        instance = {}
        # this assumes that image_paths[0] is the target image, the others are distractors
        if target_p2 == 1:
            instance["player_1_images"] = [image_paths[2], image_paths[1], image_paths[0]]
            instance["player_2_images"] = image_paths
            target_p1 = 3

        elif target_p2 == 2:
            instance["player_1_images"] = image_paths
            instance["player_2_images"] = [image_paths[1], image_paths[0], image_paths[2]]
            target_p1 = 1

        elif target_p2 == 3:
            instance["player_1_images"] = [image_paths[2], image_paths[0], image_paths[1]]
            instance["player_2_images"] = [image_paths[2], image_paths[1], image_paths[0]]
            target_p1 = 2
        
        instance["player_1_prompt_header"] = self.prepare_player_1_prompt(player_1_prompt_header, target_p1)
        instance["target_image_number"] = target_p2
        
        return instance

    def generate_scene_instances(self, general_instance=None, language=None, sample_images=False):
        ade_dataset = self.get_ade_dataset()

        game_counter = 0
        image_counter = 1
        experiment = self.add_experiment('ADE_images')
        for target_category in ade_dataset:

            image_paths = []

            if sample_images:
                target_category_images = ade_dataset[target_category]
                target_image = self.select_random_item(target_category_images)
                shutil.copyfile(target_image, os.path.join("resources", "scene_images", f"{str(image_counter)}.jpg"))
            image_paths.append(os.path.join("games", "multimodal_referencegame", "resources", "scene_images", f"{str(image_counter)}.jpg"))
            image_counter += 1

            if sample_images:
                # remove the target image from the list, select another image from the same category
                target_category_images.remove(target_image)
                distractor1 = self.select_random_item(target_category_images)
                shutil.copyfile(distractor1, os.path.join("resources", "scene_images", f"{str(image_counter)}.jpg"))
            image_paths.append(os.path.join("games", "multimodal_referencegame", "resources", "scene_images", f"{str(image_counter)}.jpg"))
            image_counter += 1

            if sample_images:
                # remove the target image from the list, select another image from the same category
                target_category_images.remove(distractor1)
                distractor2 = self.select_random_item(target_category_images)
                shutil.copyfile(distractor2, os.path.join("resources", "scene_images", f"{str(image_counter)}.jpg"))
            image_paths.append(os.path.join("games", "multimodal_referencegame", "resources", "scene_images", f"{str(image_counter)}.jpg"))
            image_counter += 1

            game_counter = self.add_shuffled_instances(general_instance=general_instance, image_paths=image_paths, experiment=experiment, game_counter=game_counter)

            if game_counter >= MAX_NUMBER_INSTANCES:
                break

    def generate_scene_static_target_instances(self, general_instance=None, language=None, sample_images=False):
        ade_dataset = self.get_ade_dataset()

        game_counter = 0
        image_counter = 1

        if os.path.exists(os.path.join("resources", "scene_images")):
            # image count used for dynamic instances:
            # next multiple of 3, starting from MAX_NUMBER_INSTANCES
            if MAX_NUMBER_INSTANCES % 3 == 0:
                image_counter = MAX_NUMBER_INSTANCES
            else:
                image_counter = MAX_NUMBER_INSTANCES + (3 - (MAX_NUMBER_INSTANCES % 3))
            # + 2, since the original sampling process skipped one image
            image_counter += 2

        experiment = self.add_experiment('ADE_static_target_images')

        target_image = None
        for target_category in ade_dataset:
            if not target_image:
                if sample_images:
                    target_category_images = ade_dataset[target_category]
                    target_image = self.select_random_item(target_category_images)
                    shutil.copyfile(target_image, os.path.join("resources", "scene_images", f"{str(image_counter)}.jpg"))
                    target_category_images.remove(target_image)
                target_image = os.path.join("games", "multimodal_referencegame", "resources", "scene_images", f"{str(image_counter)}.jpg")
                image_counter += 1

            image_paths = [target_image]

            if sample_images:
                distractor1 = self.select_random_item(target_category_images)
                shutil.copyfile(distractor1, os.path.join("resources", "scene_images", f"{str(image_counter)}.jpg"))
            image_paths.append(os.path.join("games", "multimodal_referencegame", "resources", "scene_images", f"{str(image_counter)}.jpg"))
            image_counter += 1

            if sample_images:
                # remove the target image from the list, select another image from the same category
                target_category_images.remove(distractor1)
                distractor2 = self.select_random_item(target_category_images)
                shutil.copyfile(distractor2, os.path.join("resources", "scene_images", f"{str(image_counter)}.jpg"))
            image_paths.append(os.path.join("games", "multimodal_referencegame", "resources", "scene_images", f"{str(image_counter)}.jpg"))
            image_counter += 1

            game_counter = self.add_shuffled_instances(general_instance=general_instance, image_paths=image_paths, experiment=experiment, game_counter=game_counter)

            if game_counter >= MAX_NUMBER_INSTANCES:
                break

    def generate_docci_instances(self, general_instance=None, language=None, sample_images=False):
        docci_dataset = self.get_docci_dataset()

        game_counter = 0
        image_counter = 1
        experiment = self.add_experiment('DOCCI_images')
        for target_category in docci_dataset:
            image_paths = []
            if sample_images:
                target_category_images = docci_dataset[target_category]
                target_image = self.select_random_item(target_category_images)
                shutil.copyfile(target_image, os.path.join("resources", "docci_images", f"{str(image_counter)}.jpg"))
            image_paths.append(os.path.join("games", "multimodal_referencegame", "resources", "docci_images", f"{str(image_counter)}.jpg"))
            image_counter += 1

            if sample_images:
                # remove the target image from the list, select another image from the same category
                target_category_images.remove(target_image)
                distractor1 = self.select_random_item(target_category_images)
                shutil.copyfile(distractor1, os.path.join("resources", "docci_images", f"{str(image_counter)}.jpg"))
            image_paths.append(os.path.join("games", "multimodal_referencegame", "resources", "docci_images", f"{str(image_counter)}.jpg"))
            image_counter += 1

            if sample_images:
                # remove the distractor1 image from the list, select another image from the same category
                target_category_images.remove(distractor1)
                distractor2 = self.select_random_item(target_category_images)
                shutil.copyfile(distractor2, os.path.join("resources", "docci_images", f"{str(image_counter)}.jpg"))
            image_paths.append(os.path.join("games", "multimodal_referencegame", "resources", "docci_images", f"{str(image_counter)}.jpg"))
            image_counter += 1

            game_counter = self.add_shuffled_instances(general_instance=general_instance, image_paths=image_paths, experiment=experiment, game_counter=game_counter)

            if game_counter >= MAX_NUMBER_INSTANCES:
                break

    def generate_docci_static_target_instances(self, general_instance=None, language=None, sample_images=False):
        if sample_images:
            docci_dataset = self.get_docci_dataset()

        game_counter = 0
        image_counter = 1
        experiment = self.add_experiment('DOCCI_static_target_images')


        if os.path.exists(os.path.join("resources", "docci_images")):
            # image count used for dynamic instances:
            # next multiple of 3, starting from MAX_NUMBER_INSTANCES
            if MAX_NUMBER_INSTANCES % 3 == 0:
                image_counter = MAX_NUMBER_INSTANCES
            else:
                image_counter = MAX_NUMBER_INSTANCES + (3 - (MAX_NUMBER_INSTANCES % 3))
            # + 2, since the original sampling process skipped one image
            image_counter += 2

        # target_image_path = ''
        if sample_images:
            target_category = 'dog breed'
        # select the key from docci_dataset that has the most images
        # for category in docci_dataset:
        #     if target_category == '' or len(docci_dataset[category]) > len(docci_dataset[target_category]):
        #         target_category = category

        if sample_images:
            target_category_images = docci_dataset[target_category]
            target_image = self.select_random_item(target_category_images)
            shutil.copyfile(target_image, os.path.join("resources", "docci_images", f"{str(image_counter)}.jpg"))
            target_category_images.remove(target_image)
        target_image_path = os.path.join("games", "multimodal_referencegame", "resources", "docci_images", f"{str(image_counter)}.jpg")
        image_counter += 1

        while True:
            target_images = [target_image_path]

            if sample_images:
                distractor1 = self.select_random_item(target_category_images)
                shutil.copyfile(distractor1, os.path.join("resources", "docci_images", f"{str(image_counter)}.jpg"))
            target_images.append(os.path.join("games", "multimodal_referencegame", "resources", "docci_images", f"{str(image_counter)}.jpg"))
            image_counter += 1

            if sample_images:
                # remove the distractor1 image from the list, select another image from the same category
                target_category_images.remove(distractor1)
                distractor2 = self.select_random_item(target_category_images)
                shutil.copyfile(distractor2, os.path.join("resources", "docci_images", f"{str(image_counter)}.jpg"))
            target_images.append(os.path.join("games", "multimodal_referencegame", "resources", "docci_images", f"{str(image_counter)}.jpg"))
            image_counter += 1

            game_counter = self.add_shuffled_instances(general_instance=general_instance, image_paths=target_images, experiment=experiment, game_counter=game_counter)

            if game_counter >= MAX_NUMBER_INSTANCES:
                break

    def select_distractor_for_clevr(self, target_categories, category2image, image2category):
        # get the categories of the selected image as target
        # target_categories = image2category[target_image]

        # loop over each category of the target image and find the image that has the most common categories with the target image
        distractor1 = ""
        max_common_categories = 0

        for category in target_categories:
            for image in category2image[category]:

                if image not in image2category:
                    continue

                common_categories = set(target_categories).intersection(image2category[image])
                num_common_categories = len(common_categories)
                if num_common_categories > max_common_categories:
                    distractor1 = image
                    max_common_categories = num_common_categories

        return distractor1

    def generate_clevr_instances(self, general_instance=None, language=None, sample_images=False):

        category2image, image2category = self.get_clevr_dataset()

        game_counter = 0
        image_counter = 1
        experiment = self.add_experiment('CLEVR_images')
        for target_category in category2image:
            image_paths = []
            if sample_images:
                # TODO: the following code only sampled the target image, the distractors were later overwritten

                target_category_images = category2image[target_category]
                target_image = self.select_random_item(target_category_images)

                target_categories = image2category[target_image]
                image2category.pop(target_image)



                distractor1 = self.select_distractor_for_clevr(target_categories, category2image, image2category)

                image2category.pop(distractor1)


                distractor2 = self.select_distractor_for_clevr(target_categories, category2image, image2category)
                image2category.pop(distractor2)


                shutil.copyfile(target_image, os.path.join("resources", "clevr_images", f"{str(image_counter)}.jpg"))
            image_paths.append(os.path.join("games", "multimodal_referencegame", "resources", "clevr_images", f"{str(image_counter)}.jpg"))
            image_counter += 1

            if sample_images:
                # remove the target image from the list, select another image from the same category
                target_category_images.remove(target_image)
                distractor1 = self.select_random_item(target_category_images)
                shutil.copyfile(distractor1, os.path.join("resources", "clevr_images", f"{str(image_counter)}.jpg"))
            image_paths.append(os.path.join("games", "multimodal_referencegame", "resources", "clevr_images", f"{str(image_counter)}.jpg"))
            image_counter += 1

            if sample_images:
                # remove the distractor1 image from the list, select another image from the same category
                target_category_images.remove(distractor1)
                distractor2 = self.select_random_item(target_category_images)
                shutil.copyfile(distractor2, os.path.join("resources", "clevr_images", f"{str(image_counter)}.jpg"))
            image_paths.append(os.path.join("games", "multimodal_referencegame", "resources", "clevr_images", f"{str(image_counter)}.jpg"))
            image_counter += 1

            game_counter = self.add_shuffled_instances(general_instance=general_instance, image_paths=image_paths, experiment=experiment, game_counter=game_counter)

            if game_counter >= MAX_NUMBER_INSTANCES:
                break

    def generate_clevr_static_target_instances(self, general_instance=None, language=None, sample_images=False):
        game_counter = 0
        image_counter = 1
        image_directory = os.path.join("resources", "clevr_images")
        
        # find out if there are any images in the image_directory and if there is any, get the last image number by looking at the suffix before the .jpg file extension
        if os.path.exists(image_directory):
            # files = os.listdir(image_directory)
            # if len(files) > 0:
            #     image_counter = len(files) + 1
            # image count used for dynamic instances:
            # next multiple of 3, starting from MAX_NUMBER_INSTANCES
            if MAX_NUMBER_INSTANCES % 3 == 0:
                image_counter = MAX_NUMBER_INSTANCES
            else:
                image_counter = MAX_NUMBER_INSTANCES + (3 - (MAX_NUMBER_INSTANCES % 3))
            # + 2, since the original sampling process skipped one image
            image_counter += 2

        category2image, image2category = self.get_clevr_dataset()

        experiment = self.add_experiment('CLEVR_static_target_images')

        target_image_path = ''
        target_categories = []

        for target_category in category2image:

            if target_image_path == '':
                if sample_images:
                    target_category_images = category2image[target_category]
                    target_image = self.select_random_item(target_category_images)
                    target_categories = image2category[target_image]
                    image2category.pop(target_image)

                    shutil.copyfile(target_image, os.path.join("resources", "clevr_images", f"{str(image_counter)}.jpg"))
                target_image_path = os.path.join("resources", "clevr_images", f"{str(image_counter)}.jpg")
                image_counter += 1
            
            image_paths = [target_image_path]

            if sample_images:
                distractor1 = self.select_distractor_for_clevr(target_categories, category2image, image2category)
                image2category.pop(distractor1)

                distractor2 = self.select_distractor_for_clevr(target_categories, category2image, image2category)
                image2category.pop(distractor2)

                shutil.copyfile(distractor1, os.path.join("resources", "clevr_images", f"{str(image_counter)}.jpg"))
            image_paths.append(os.path.join("resources", "clevr_images", f"{str(image_counter)}.jpg"))
            image_counter += 1

            if sample_images:
                shutil.copyfile(distractor2, os.path.join("resources", "clevr_images", f"{str(image_counter)}.jpg"))
            image_paths.append(os.path.join("resources", "clevr_images", f"{str(image_counter)}.jpg"))
            image_counter += 1

            game_counter = self.add_shuffled_instances(general_instance=general_instance, image_paths=image_paths, experiment=experiment, game_counter=game_counter)

            if game_counter >= MAX_NUMBER_INSTANCES:
                break

    def generate_pentomino_instances(self, general_instance=None, language=None):
        game_counter = 0
        image_directory = os.path.join("resources", "pentomino_images")

        image_files = []
        if os.path.exists(image_directory):
            image_files = os.listdir(image_directory)

        experiment = self.add_experiment('pentomino_images')

        # loop over image image_files and take the first 7 sets of images: 1st image is the target image, the next 2 are distractors, and create 3 sets of tuples  where one is the target image and the other two are the distractors
        for i in range(1, len(image_files)+1, 7):

            image_paths = [os.path.join(image_directory, f"{i}.jpg")]

            for j in range(i + 1, i + 7, 2):
                image_paths.append(os.path.join(image_directory, f"{j}.jpg"))
                image_paths.append(os.path.join(image_directory, f"{j+1}.jpg"))
                
                game_counter = self.add_shuffled_instances(general_instance=general_instance, image_paths=image_paths, experiment=experiment, game_counter=game_counter)

            if game_counter >= MAX_NUMBER_INSTANCES:
                break

    def on_generate(self):
        for language in LANGUAGES:
            general_instance = self.general_instance(language)
            self.generate_grid_instances(general_instance, language)
            self.generate_scene_instances(general_instance, language)
            self.generate_docci_instances(general_instance, language)
            self.generate_clevr_instances(general_instance, language)
            self.generate_clevr_static_target_instances(general_instance, language)
            self.generate_docci_static_target_instances(general_instance, language)
            self.generate_scene_static_target_instances(general_instance, language)
            self.generate_pentomino_instances(general_instance, language)

if __name__ == '__main__':
    ReferenceGameInstanceGenerator().generate(filename="instances.json")
