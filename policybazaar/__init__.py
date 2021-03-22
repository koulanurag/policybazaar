import os

import torch
import wandb

from .config import ENV_INFO, POLICY_BAZAAR_DIR, MIN_PRE_TRAINED_LEVEL, MAX_PRE_TRAINED_LEVEL
from .model import ActorCriticNetwork


def get_policy(env_name: str, pre_trained: int = 1) -> (ActorCriticNetwork, dict):
    """
    Retrieves policies for the environment with the pre-trained marker quality.

    :param env_name:  name of the environment
    :param pre_trained: pre_trained level . It should be between 1 and 5 ,
                        where 1 indicates best model and 5 indicates worst level.

    Example:
        >>> import policybazaar
        >>> policybazaar.get_policy('d4rl:maze2d-open-v0',pre_trained=1)
    """

    assert MIN_PRE_TRAINED_LEVEL <= pre_trained <= MAX_PRE_TRAINED_LEVEL, \
        'pre_trained marker should be between [{},{}] where {} indicates the best model' \
        ' and {} indicates the worst model'.format(MIN_PRE_TRAINED_LEVEL, MAX_PRE_TRAINED_LEVEL, MIN_PRE_TRAINED_LEVEL,
                                                   MAX_PRE_TRAINED_LEVEL)
    assert env_name in ENV_INFO, '`{}` not found. It should be among following: {}'.format(env_name, ENV_INFO.keys())

    run_id = ENV_INFO[env_name]['wandb_run_id']
    info = ENV_INFO[env_name]['info']['pre_trained={}'.format(pre_trained)]

    # retrieve model
    run = wandb.Api().run(run_id)
    env_root = os.path.join(POLICY_BAZAAR_DIR)
    os.makedirs(env_root, exist_ok=True)
    model_name = '{}_interval_{}.p'.format(ENV_INFO[env_name]['model_name'],
                                           (MAX_PRE_TRAINED_LEVEL + 1) - pre_trained)
    wandb.restore(name=model_name, run_path=run_id, replace=True, root=env_root)
    model = ActorCriticNetwork(run.config['observation_size'],
                               run.config['action_size'],
                               hidden_dim=64,
                               action_std=0.5)
    model.load_state_dict(torch.load(os.path.join(env_root, model_name), map_location=torch.device('cpu')))
    return model, info
