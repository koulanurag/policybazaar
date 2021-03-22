import os
from pathlib import Path

POLICY_BAZAAR_DIR = os.getenv('POLICY_BAZAAR_DIR', default=os.path.join(str(Path.home()), '.policybazaar'))
MIN_PRE_TRAINED_LEVEL = 1
MAX_PRE_TRAINED_LEVEL = 4

ENV_INFO = {'d4rl:maze2d-umaze-dense-v1': {'wandb_run_id': 'koulanurag/pytorch-drl/1x308yro',
                                           'info': {'pre_trained=4': {'score_mean': 154.2, 'score_std': 13.8},
                                                    'pre_trained=3': {'score_mean': 231.0, 'score_std': 66.6},
                                                    'pre_trained=2': {'score_mean': 243.0, 'score_std': 27.1},
                                                    'pre_trained=1': {'score_mean': 198.8, 'score_std': 88.5}}},
            'd4rl:maze2d-medium-dense-v1': {'wandb_run_id': 'koulanurag/pytorch-drl/1chrtriz',
                                            'info': {'pre_trained=4': {'score_mean': 286.4, 'score_std': 172.5},
                                                     'pre_trained=3': {'score_mean': 200.9, 'score_std': 194.9},
                                                     'pre_trained=2': {'score_mean': 247.3, 'score_std': 180.7},
                                                     'pre_trained=1': {'score_mean': 274.9, 'score_std': 176.9}}},
            'd4rl:maze2d-umaze-v1': {'wandb_run_id': 'koulanurag/pytorch-drl/3m0f08zs',
                                     'info': {'pre_trained=4': {'score_mean': 39.5, 'score_std': 94.7},
                                              'pre_trained=3': {'score_mean': 223.9, 'score_std': 60.2},
                                              'pre_trained=2': {'score_mean': 253.5, 'score_std': 30.4},
                                              'pre_trained=1': {'score_mean': 243.5, 'score_std': 35.4}}},
            'd4rl:maze2d-open-dense-v0': {'wandb_run_id': 'koulanurag/pytorch-drl/3aupc8y7',
                                          'info': {'pre_trained=4': {'score_mean': 44.3, 'score_std': 12.7},
                                                   'pre_trained=3': {'score_mean': 105.3, 'score_std': 14.6},
                                                   'pre_trained=2': {'score_mean': 120.3, 'score_std': 9.4},
                                                   'pre_trained=1': {'score_mean': 122.1, 'score_std': 9.5}}},
            'd4rl:maze2d-open-v0': {'wandb_run_id': 'koulanurag/pytorch-drl/10u4yfqz',
                                    'info': {'pre_trained=4': {'score_mean': 6.7, 'score_std': 8.1},
                                             'pre_trained=3': {'score_mean': 26.2, 'score_std': 10.8},
                                             'pre_trained=2': {'score_mean': 120.0, 'score_std': 15.5},
                                             'pre_trained=1': {'score_mean': 117.7, 'score_std': 10.2}}},
            'd4rl:maze2d-large-dense-v1': {'wandb_run_id': 'koulanurag/pytorch-drl/im68hs90',
                                           'info': {'pre_trained=4': {'score_mean': 206.5, 'score_std': 159.0},
                                                    'pre_trained=3': {'score_mean': 251.0, 'score_std': 192.7},
                                                    'pre_trained=2': {'score_mean': 297.3, 'score_std': 230.7}}},
            'd4rl:maze2d-large-v1': {'wandb_run_id': 'koulanurag/pytorch-drl/kbwinqwa',
                                     'info': {'pre_trained=4': {'score_mean': 12.3, 'score_std': 9.1},
                                              'pre_trained=3': {'score_mean': 92.8, 'score_std': 164.7},
                                              'pre_trained=2': {'score_mean': 5.3, 'score_std': 12.7},
                                              'pre_trained=1': {'score_mean': 33.0, 'score_std': 45.8}}},
            'd4rl:maze2d-medium-v1': {'wandb_run_id': 'koulanurag/pytorch-drl/3ku1yoo5',
                                      'info': {'pre_trained=4': {'score_mean': 339.2, 'score_std': 227.8},
                                               'pre_trained=3': {'score_mean': 281.2, 'score_std': 256.8},
                                               'pre_trained=2': {'score_mean': 290.5, 'score_std': 263.7},
                                               'pre_trained=1': {'score_mean': 293.4, 'score_std': 267.5}}}}

for env_name in ENV_INFO:
    ENV_INFO[env_name]['model_name'] = 'agent_model'
