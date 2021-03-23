# policybazaar
A collection of multi-quality  policies for continuous control tasks.

[![Python package](https://github.com/koulanurag/policybazaar/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/koulanurag/policybazaar/actions/workflows/python-package.yml)

## Installation
It requires:

- [Python 3.6+](https://www.python.org/downloads/)
- [mujoco-py](https://github.com/openai/mujoco-py), [mujoco 200](https://www.roboti.us/index.html) and [mujoco license](https://www.roboti.us/license.html). Please, follow `mujoco-py` installation instructions from [here](https://github.com/openai/mujoco-py).
- [Pytorch >= 1.8.0](https://pytorch.org/)

Python package and dependencies could be installed using:
```bash
pip install git+https://github.com/koulanurag/policybazaar@master#egg=policybazaar
```
Or
```bash
git clone https://github.com/koulanurag/policybazaar.git
cd policybazar
pip install -e .
```

## Usage

```python console
>>> import policybazaar, gym
>>> model, model_info = policybazaar.get_policy('d4rl:maze2d-open-v0',pre_trained=1)
>>> model_info
{'score_mean': 113.0, 'score_std': 18.3}

>>> episode_reward = 0
>>> done = False
>>> env = gym.make('d4rl:maze2d-open-v0')
>>> obs = env.reset()
>>> while not done:
...    action_dist = model.actor(torch.tensor(obs).unsqueeze(0).float())
...    action = action_dist.mean.data.numpy()[0]
...    obs, reward, done, step_info = env.step(action)
...    episode_reward += reward
>>> episode_reward
100.0

```

## Testing:

- Install: ```pip install -e ".[test]" ```
- Run: ```pytest```

## What's New:

- **23rd Mar, 2021:**
    - Initial release(alpha) with pre-trained policies for maze2d in d4rl.
    
## Pre-trained Policy Scores
In the following, we report performance of various pre-trained models. These scores are reported over `20` episode runs.

### :small_blue_diamond: [d4rl:maze2d](https://github.com/rail-berkeley/d4rl/wiki/Tasks#maze2d)
<img width="500" alt="maze2d-environments" src="https://github.com/rail-berkeley/offline_rl/raw/assets/assets/mazes_filmstrip.png">

| Environment Name |`pre_trained=1` (best) |`pre_trained=2`  |`pre_trained=3`  |`pre_trained=4` (worst) |
|:------: | :------: | :------: | :------: | :------: | 
|`d4rl:maze2d-umaze-dense-v1`|198.8±88.5 |243.0±27.1 |231.0±66.6 |154.2±13.8 |
|`d4rl:maze2d-medium-dense-v1`|274.9±176.9 |247.3±180.7 |200.9±194.9 |286.4±172.5 |
|`d4rl:maze2d-umaze-v1`|243.5±35.4 |253.5±30.4 |223.9±60.2 |39.5±94.7 |
|`d4rl:maze2d-open-dense-v0`|122.1±9.5 |120.3±9.4 |105.3±14.6 |44.3±12.7 |
|`d4rl:maze2d-open-v0`|117.7±10.2 |120.0±15.5 |26.2±10.8 |6.7±8.1 |
|`d4rl:maze2d-large-dense-v1`|275.2±237.4 |297.3±230.7 |251.0±192.7 |206.5±159.0 |
|`d4rl:maze2d-large-v1`|33.0±45.8 |5.3±12.7 |92.8±164.7 |12.3±9.1 |
|`d4rl:maze2d-medium-v1`|293.4±267.5 |290.5±263.7 |281.2±256.8 |339.2±227.8 |

***

