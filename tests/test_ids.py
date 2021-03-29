import gym
import pytest
import torch

import policybazaar
from policybazaar.config import ENV_IDS, MIN_PRE_TRAINED_LEVEL, MAX_PRE_TRAINED_LEVEL, CHILD_PARENT_ENVS


@pytest.mark.parametrize('env_name,pre_trained', [(env_name, pre_trained)
                                                  for env_name in list(ENV_IDS.keys()) + list(CHILD_PARENT_ENVS.keys())
                                                  for pre_trained in range(MIN_PRE_TRAINED_LEVEL,
                                                                           MAX_PRE_TRAINED_LEVEL + 1)])
def test_wandb_ids(env_name, pre_trained):
    model, model_info = policybazaar.get_policy(env_name, pre_trained)

    env = gym.make(env_name)
    obs = env.reset()
    action = model.actor(torch.tensor(obs).unsqueeze(0).float()).mean.data.numpy()[0]
    env.step(action)
    env.close()
