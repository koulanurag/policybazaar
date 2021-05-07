import os

import torch
import wandb

from .config import ENV_IDS, POLICY_BAZAAR_DIR, MIN_PRE_TRAINED_LEVEL, MAX_PRE_TRAINED_LEVEL
from .config import ENV_PERFORMANCE_STATS, CHILD_PARENT_ENVS


def get_policy(env_name: str, pre_trained: int = 1):
    """
    Retrieves policies for the environment with the pre-trained quality marker.

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
    assert env_name in ENV_IDS or env_name in CHILD_PARENT_ENVS, \
        '`{}` not found. It should be among following: {}'.format(env_name,
                                                                  list(ENV_IDS.keys()) + list(CHILD_PARENT_ENVS.keys()))

    if env_name not in ENV_IDS:
        env_name = CHILD_PARENT_ENVS[env_name]
    if env_name in ENV_PERFORMANCE_STATS and pre_trained in ENV_PERFORMANCE_STATS[env_name]:
        info = ENV_PERFORMANCE_STATS[env_name][pre_trained]
    else:
        info = {}

    run_path = ENV_IDS[env_name]['wandb_run_path']
    run = wandb.Api().run(run_path)
    env_root = os.path.join(env_name, POLICY_BAZAAR_DIR, 'models', env_name)
    os.makedirs(env_root, exist_ok=True)

    if 'cassie' in env_name:
        # retrieve model
        model_name = '{}_{}.0.p'.format(ENV_IDS[env_name]['model_name'], ENV_IDS[env_name]['models'][pre_trained])
        from .cassie_model import ActorCriticNetwork
        model = ActorCriticNetwork(47, 10, 256)
    else:
        # retrieve model
        model_name = '{}_{}.0.p'.format(ENV_IDS[env_name]['model_name'], ENV_IDS[env_name]['models'][pre_trained])

        from .model import ActorCriticNetwork
        model = ActorCriticNetwork(run.config['observation_size'],
                                   run.config['action_size'],
                                   hidden_dim=64,
                                   action_std=0.5)

    wandb.restore(name=model_name, run_path=run_path, replace=True, root=env_root)
    model.load_state_dict(torch.load(os.path.join(env_root, model_name), map_location=torch.device('cpu')))
    return model, info
