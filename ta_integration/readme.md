# TextArena integration

This folder contains the instance generator, game master and game specs necessary to integrate games from [TextArena](https://github.com/LeonGuertler/TextArena/tree/main) into the clem framework.

## Adding games

### TextArena structure

In the TextArena repo, the game environments containing all game logic are stored in `textarena/envs`. Each env inherits from `ta.Env`, and has a game `ta.State` depending on the number of players. They can be wrapped in different ObservationWrappers for different frontends.

### `clemgame.json`

Check `TextArena/textarena/envs/__init__.py` [(link)](https://github.com/LeonGuertler/TextArena/blob/main/textarena/envs/__init__.py) for the full game registry.
Once you have chosen a game, you can make a new entry in `clemgamge.json` of the structure:

```
{
        "game_name": "ta_word_chains",
        "main_game": "ta_word_chains",
        "ta_name": "WordChains-v0",
        "n_instances": 10,
        "instances": "instances_word_chains",
        "players": 2,
        "image": null,
        "languages": [
            "en"
        ],
        "player_specs": [
            {
                "role": "Player 0",
                "custom_response": ["lighter", "elephant", "tiger", "rabbit"]
            },
            {
                "role": "Player 1",
                "custom_response": ["elephant", "tiger", "rabbit", "banana"]
            }
        ],
        "override_variables": {
            "env.word_list": "lambda x: sorted(x)"
        }
    }
```

* You can choose `game_name` freely, but it should begin with `ta_`. 
* `main_game`: copy `game_name`, is not currently used
* `ta_name` is the `id` of TA's game registry, without any suffixes. If there are several difficulty levels (such as `Minesweeper-v0-small`, `-medium`, `-hard`), they will be automatically added as experiments.
* `n_instances`is the number of instances per experiment, but be aware that some games always have the same starting configuration (such as Tower of Hanoi, where all disks are always on the starting peg).
* `instances` is the name of the resulting json file, should always start with `instances_`.
* `n_players` must match the number of players TA requires, cf. [envs Readme](https://github.com/LeonGuertler/TextArena/blob/main/textarena/envs/README.md)
* always add `"image": null` and `"languages": ["en"]`; as of July 2025, TA neither supports multimodal input nor other languages.
* `player specs`: Must match `n_players`. For role, you can use descriptive names such us `Guesser` etc. `custom_response` is for programmatic answers that are sampled from when playing with `-m mock`.
* `override_variables`: Usually not needed (can be deleted). See section [Reproducibility](#reproducibility).

Note on registry entries:
We always use the `-raw` registry entries, since these don't use default wrappers, and we can solely use `ClemObservationWrapper`. This is handled internally in `instancegenerator.py`.

### Instance Generation

Generated instances contain the correct entry point, i.e., the Env subclass that is to be instantiated. For reference, it also copies the `env_specs` from the TA registry, but internally, the values from TA are used during game play.

Each instance also contains a seed that is passed to the environment to ensure reproducibility.

### Reproducibility

Despite passing the seed, there might occur some non-deterministic behavior, which might be fixed by `override_variables` in `clemgame.json`.

For example, in WordChains, the start word is sampled from a word list generated from a set, i.e., `list(set(word.lower() for word in nltk.corpus.words.words()))`. Sets are nondeterministic, and sampling accordingly doesn't work. As a workaround, `override_variables` is added to the `clemgame.json` to sort the list:

```
"override_variables": {
    "env.word_list": "lambda x: sorted(x)"
}
```

After creating the TA environment, but before initiating the game state, any variable in this entry is replaced according to the lambda function given as value. For security reasons, this functionality should never be extended to execute any code other than replacing a specific variable!

Despite this, some games will still behave somewhat indeterministic. For example, in Minesweeper, the placement of mines depends on the first move. Thus, a different first move will also lead to a different placement of mines.

# Scoring

Scoring is somewhat difficult and depends on the game. TextArena usually assigns different scores for each player, often just win/lose.

## Word Chains
For Word Chains, the player that provides the last valid word wins, the other one loses. However, to benchmark a model, it is better to base scoring on the length of the final word.
We assume that a perfect game lasts 8 rounds, reaching a length 16 letters longer than the original word.