import gym
import policybazaar
import pytest
import torch
from policybazaar.config import ENV_IDS, CHILD_PARENT_ENVS


@pytest.mark.parametrize('env_name,pre_trained',
                         [(env_name, pre_trained)
                          for env_name in list(ENV_IDS.keys()) + list(CHILD_PARENT_ENVS.keys())
                          for pre_trained in
                          (ENV_IDS[env_name]['models']
                          if env_name in ENV_IDS else ENV_IDS[CHILD_PARENT_ENVS[env_name]]['models'])])
def test_model_exists(env_name, pre_trained):
    model, model_info = policybazaar.get_policy(env_name, pre_trained)

    env = gym.make(env_name)
    obs = env.reset()
    action = model.actor(torch.tensor(obs).unsqueeze(0).float()).mean.data.numpy()[0]
    env.step(action)
    env.close()


@pytest.mark.parametrize('env_name,pre_trained',
                         [(env_name, pre_trained)
                          for env_name in list(ENV_IDS.keys()) + list(CHILD_PARENT_ENVS.keys())
                          for pre_trained in
                          (ENV_IDS[env_name]['models']
                          if env_name in ENV_IDS else ENV_IDS[CHILD_PARENT_ENVS[env_name]]['models'])])
def test_dataset_exists(env_name, pre_trained):
    dataset = policybazaar.get_dataset(env_name, pre_trained)

    env = gym.make(env_name)
    assert len(dataset['actions'][0]) == env.action_space.shape[0]
    assert len(dataset['observations'][0]) == env.observation_space.shape[0]
    assert len(dataset['actions']) == len(dataset['observations']) == \
           len(dataset['rewards']) == len(dataset['terminals']) == len(dataset['timeouts'])
