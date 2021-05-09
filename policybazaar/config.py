import os
from pathlib import Path

POLICY_BAZAAR_DIR = os.getenv('POLICY_BAZAAR_DIR', default=os.path.join(str(Path.home()), '.policybazaar'))
MIN_PRE_TRAINED_LEVEL = 1
MAX_PRE_TRAINED_LEVEL = 4

MAZE_BASE_PROJECT_URL = 'koulanurag/pytorch-drl/'
MUJOCO_BASE_PROJECT_URL = 'koulanurag/pytorch-drl/'

MAZE_ENV_IDS = {
    # sparse
    'd4rl:maze2d-open-v0': {'wandb_run_path': MAZE_BASE_PROJECT_URL + '3rd5zlnh',
                            # pre_trained_id : wand_model_save_id
                            # 1: best, 4: worst
                            'models': {1: 50, 2: 20, 3: 10, 4: 1}},
    'd4rl:maze2d-medium-v1': {'wandb_run_path': MAZE_BASE_PROJECT_URL + 'gplzxs8a',
                              'models': {1: 25, 2: 47, 3: 2, 4: 1}},
    'd4rl:maze2d-umaze-v1': {'wandb_run_path': MAZE_BASE_PROJECT_URL + 'e15ildhv',
                             'models': {1: 50, 2: 12, 3: 5, 4: 1}},
    'd4rl:maze2d-large-v1': {'wandb_run_path': MAZE_BASE_PROJECT_URL + '1vi4gkf8',
                             'models': {1: 35, 2: 30, 3: 15, 4: 1}},

    # dense
    'd4rl:maze2d-open-dense-v0': {'wandb_run_path': MAZE_BASE_PROJECT_URL + '111lwjn6',
                                  'models': {1: 50, 2: 15, 3: 10, 4: 1}},
    'd4rl:maze2d-medium-dense-v1': {'wandb_run_path': MAZE_BASE_PROJECT_URL + '1huhpfht',
                                    'models': {1: 15, 2: 40, 3: 2, 4: 1}},
    'd4rl:maze2d-umaze-dense-v1': {'wandb_run_path': MAZE_BASE_PROJECT_URL + '1rsf3y79',
                                   'models': {1: 50, 2: 8, 3: 4, 4: 1}},
    'd4rl:maze2d-large-dense-v1': {'wandb_run_path': MAZE_BASE_PROJECT_URL + '20cxp3qr',
                                   'models': {1: 25, 2: 14, 3: 10, 4: 1}},

    # antmaze
    'd4rl:antmaze-umaze-v0': {'wandb_run_path': MAZE_BASE_PROJECT_URL + '1yv3t59a',
                              'models': {1: 25, 2: 15, 3: 10, 4: 1}},
    'd4rl:antmaze-medium-diverse-v0': {'wandb_run_path': MAZE_BASE_PROJECT_URL + '20l3hp3w',
                                       'models': {1: 25, 2: 15, 3: 10, 4: 1}},
    'd4rl:antmaze-large-diverse-v0': {'wandb_run_path': MAZE_BASE_PROJECT_URL + '1c6b1a2e',
                                      'models': {1: 25, 2: 15, 3: 10, 4: 1}},
}

# mujoco(gym)
MUJOCO_ENV_IDS = {'HalfCheetah-v2': {'wandb_run_path': MUJOCO_BASE_PROJECT_URL + 't669pz0z',
                                     'models': {1: 25, 2: 15, 3: 10, 4: 1}},
                  'Hopper-v2': {'wandb_run_path': MUJOCO_BASE_PROJECT_URL + '21k6p0fq',
                                'models': {1: 50, 2: 35, 3: 8, 4: 1}},
                  'Walker2d-v2': {'wandb_run_path': MUJOCO_BASE_PROJECT_URL + '17mv3xec',
                                  'models': {1: 35, 2: 15, 3: 8, 4: 1}}}

# Cassie
CASSIE_BASE_PROJECT_URL = 'offline-drl-team/cassie/'
CASSIE_ENV_IDS = {
    'cassie:CassieWalkSlow-v0': {'wandb_run_path': CASSIE_BASE_PROJECT_URL + '3cxlmmfu', 'models': {1: 0}},
    'cassie:CassieStand-v0': {'wandb_run_path': CASSIE_BASE_PROJECT_URL + '18par2uc', 'models': {1: 0}},
    'cassie:CassieWalkFast-v0': {'wandb_run_path': CASSIE_BASE_PROJECT_URL + '2q816ycl', 'models': {1: 0}}}

ENV_IDS = {**MAZE_ENV_IDS, **MUJOCO_ENV_IDS, **CASSIE_ENV_IDS}

PARENT_CHILD_ENVS = {'d4rl:antmaze-umaze-v0': ['d4rl:antmaze-umaze-diverse-v0'],
                     'd4rl:antmaze-medium-diverse-v0': ['d4rl:antmaze-medium-play-v0'],
                     'd4rl:antmaze-large-diverse-v0': ['d4rl:antmaze-large-play-v0'],
                     'Walker2d-v2': ['d4rl:walker2d-random-v0', 'd4rl:walker2d-medium-v0', 'd4rl:walker2d-expert-v0',
                                     'd4rl:walker2d-medium-replay-v0', 'd4rl:walker2d-medium-expert-v0'],
                     'Hopper-v2': ['d4rl:hopper-random-v0', 'd4rl:hopper-medium-v0', 'd4rl:hopper-expert-v0',
                                   'd4rl:hopper-medium-replay-v0', 'd4rl:hopper-medium-expert-v0'],
                     'HalfCheetah-v2': ['d4rl:halfcheetah-random-v0', 'd4rl:halfcheetah-medium-v0',
                                        'd4rl:halfcheetah-expert-v0', 'd4rl:halfcheetah-medium-replay-v0',
                                        'd4rl:halfcheetah-medium-expert-v0']}

CHILD_PARENT_ENVS = {}
for parent_env in PARENT_CHILD_ENVS:
    for child_env in PARENT_CHILD_ENVS[parent_env]:
        CHILD_PARENT_ENVS[child_env] = parent_env

ENV_PERFORMANCE_STATS = {'d4rl:maze2d-open-v0': {1: {'score_mean': 122.2, 'score_std': 10.61},
                                                 2: {'score_mean': 104.9, 'score_std': 22.19},
                                                 3: {'score_mean': 18.05, 'score_std': 14.85},
                                                 4: {'score_mean': 4.85, 'score_std': 8.62}},
                         'd4rl:maze2d-medium-v1': {1: {'score_mean': 245.55, 'score_std': 272.75},
                                                   2: {'score_mean': 203.75, 'score_std': 252.61},
                                                   3: {'score_mean': 256.65, 'score_std': 260.16},
                                                   4: {'score_mean': 258.55, 'score_std': 262.81}},
                         'd4rl:maze2d-umaze-v1': {1: {'score_mean': 235.5, 'score_std': 35.45},
                                                  2: {'score_mean': 197.75, 'score_std': 58.21},
                                                  3: {'score_mean': 23.4, 'score_std': 73.24},
                                                  4: {'score_mean': 3.2, 'score_std': 9.65}},
                         'd4rl:maze2d-large-v1': {1: {'score_mean': 231.35, 'score_std': 268.37},
                                                  2: {'score_mean': 160.8, 'score_std': 201.97},
                                                  3: {'score_mean': 50.65, 'score_std': 76.94},
                                                  4: {'score_mean': 9.95, 'score_std': 9.95}},
                         'd4rl:maze2d-open-dense-v0': {1: {'score_mean': 127.18, 'score_std': 9.17},
                                                       2: {'score_mean': 117.53, 'score_std': 10.21},
                                                       3: {'score_mean': 63.96, 'score_std': 16.03},
                                                       4: {'score_mean': 26.82, 'score_std': 9.19}},
                         'd4rl:maze2d-medium-dense-v1': {1: {'score_mean': 209.25, 'score_std': 190.66},
                                                         2: {'score_mean': 192.36, 'score_std': 193.29},
                                                         3: {'score_mean': 225.54, 'score_std': 183.33},
                                                         4: {'score_mean': 232.94, 'score_std': 184.62}},
                         'd4rl:maze2d-umaze-dense-v1': {1: {'score_mean': 240.22, 'score_std': 25.1},
                                                        2: {'score_mean': 201.12, 'score_std': 21.35},
                                                        3: {'score_mean': 121.94, 'score_std': 10.71},
                                                        4: {'score_mean': 45.5, 'score_std': 44.53}},
                         'd4rl:maze2d-large-dense-v1': {1: {'score_mean': 168.83, 'score_std': 225.78},
                                                        2: {'score_mean': 239.1, 'score_std': 208.43},
                                                        3: {'score_mean': 204.39, 'score_std': 197.96},
                                                        4: {'score_mean': 90.89, 'score_std': 70.61}},
                         'd4rl:antmaze-umaze-v0': {1: {'score_mean': 0.0, 'score_std': 0.0},
                                                   2: {'score_mean': 0.0, 'score_std': 0.0},
                                                   3: {'score_mean': 0.0, 'score_std': 0.0},
                                                   4: {'score_mean': 0.0, 'score_std': 0.0}},
                         'd4rl:antmaze-medium-diverse-v0': {1: {'score_mean': 0.0, 'score_std': 0.0},
                                                            2: {'score_mean': 0.0, 'score_std': 0.0},
                                                            3: {'score_mean': 0.0, 'score_std': 0.0},
                                                            4: {'score_mean': 0.0, 'score_std': 0.0}},
                         'd4rl:antmaze-large-diverse-v0': {1: {'score_mean': 0.0, 'score_std': 0.0},
                                                           2: {'score_mean': 0.0, 'score_std': 0.0},
                                                           3: {'score_mean': 0.0, 'score_std': 0.0},
                                                           4: {'score_mean': 0.0, 'score_std': 0.0}},
                         'HalfCheetah-v2': {1: {'score_mean': 1169.13, 'score_std': 80.45},
                                            2: {'score_mean': 1044.39, 'score_std': 112.61},
                                            3: {'score_mean': 785.88, 'score_std': 303.59},
                                            4: {'score_mean': 94.79, 'score_std': 40.88}},
                         'Hopper-v2': {1: {'score_mean': 1995.84, 'score_std': 794.71},
                                       2: {'score_mean': 1466.71, 'score_std': 497.1},
                                       3: {'score_mean': 1832.43, 'score_std': 560.86},
                                       4: {'score_mean': 236.51, 'score_std': 1.09}},
                         'Walker2d-v2': {1: {'score_mean': 2506.9, 'score_std': 689.45},
                                         2: {'score_mean': 811.28, 'score_std': 321.66},
                                         3: {'score_mean': 387.01, 'score_std': 42.82},
                                         4: {'score_mean': 162.7, 'score_std': 102.14}}}

for env_name in ENV_IDS:
    ENV_IDS[env_name]['model_name'] = 'agent_model'
