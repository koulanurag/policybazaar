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
- **29th Mar, 2021:**
    - Initial release(alpha2) with pre-trained policies for maze2d and some environments in mujoco(gym) in d4rl.
    - policies were hand-picked
- **23rd Mar, 2021:**
    - Initial release(alpha1) with pre-trained policies for maze2d in d4rl. 
    
## Pre-trained Policy Scores
In the following, we report performance of various pre-trained models. These scores are reported over `20` episode runs.

### :small_blue_diamond: [d4rl:maze2d](https://github.com/rail-berkeley/d4rl/wiki/Tasks#maze2d)
<img width="500" alt="maze2d-environments" src="https://github.com/rail-berkeley/offline_rl/raw/assets/assets/mazes_filmstrip.png">

| Environment Name |`pre_trained=1` (best) |`pre_trained=2`  |`pre_trained=3`  |`pre_trained=4` (worst) |
|:------: | :------: | :------: | :------: | :------: | 
|`d4rl:maze2d-open-v0`|122.2±10.61 |104.9±22.19 |18.05±14.85 |4.85±8.62 |
|`d4rl:maze2d-medium-v1`|245.55±272.75 |203.75±252.61 |256.65±260.16 |258.55±262.81 |
|`d4rl:maze2d-umaze-v1`|235.5±35.45 |197.75±58.21 |23.4±73.24 |3.2±9.65 |
|`d4rl:maze2d-large-v1`|231.35±268.37 |160.8±201.97 |50.65±76.94 |9.95±9.95 |
|`d4rl:maze2d-open-dense-v0`|127.18±9.17 |117.53±10.21 |63.96±16.03 |26.82±9.19 |
|`d4rl:maze2d-medium-dense-v1`|209.25±190.66 |192.36±193.29 |225.54±183.33 |232.94±184.62 |
|`d4rl:maze2d-umaze-dense-v1`|240.22±25.1 |201.12±21.35 |121.94±10.71 |45.5±44.53 |
|`d4rl:maze2d-large-dense-v1`|168.83±225.78 |239.1±208.43 |204.39±197.96 |90.89±70.61 |



### :small_blue_diamond: [d4rl:antmaze](https://github.com/rail-berkeley/d4rl/wiki/Tasks#antmaze)
<img width="500" alt="maze2d-environments" src="https://github.com/rail-berkeley/offline_rl/raw/assets/assets/ant_filmstrip.png">

| Environment Name |`pre_trained=1` (best) |`pre_trained=2`  |`pre_trained=3`  |`pre_trained=4` (worst) |
|:------: | :------: | :------: | :------: | :------: | 
|`d4rl:antmaze-umaze-v0`|0.0±0.0 |0.0±0.0 |0.0±0.0 |0.0±0.0 |
|`d4rl:antmaze-medium-diverse-v0`|0.0±0.0 |0.0±0.0 |0.0±0.0 |0.0±0.0 |
|`d4rl:antmaze-large-diverse-v0`|0.0±0.0 |0.0±0.0 |0.0±0.0 |0.0±0.0 |

### :small_blue_diamond: [mujoco(gym)](https://github.com/rail-berkeley/d4rl/wiki/Tasks#antmaze)
<img width="200" alt="mujoco-environments" src="https://gym.openai.com/videos/2019-10-21--mqt8Qj1mwo/HalfCheetah-v2/poster.jpg">
<img width="200" alt="mujoco-environments" src="https://i1.wp.com/www.r-craft.org/wp-content/uploads/2018/09/imitation-learning-in-tensorflow-hopper-from-openai-gym.png?fit=890%2C468&ssl=1">
<img width="200" alt="mujoco-environments" src="data:image/jpeg;base64,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">

| Environment Name |`pre_trained=1` (best) |`pre_trained=2`  |`pre_trained=3`  |`pre_trained=4` (worst) |
|:------: | :------: | :------: | :------: | :------: | 
|`HalfCheetah-v2`|1169.13±80.45 |1044.39±112.61 |785.88±303.59 |94.79±40.88 |
|`Hopper-v2`|1995.84±794.71 |1466.71±497.1 |1832.43±560.86 |236.51±1.09 |
|`Walker2d-v2`|2506.9±689.45 |811.28±321.66 |387.01±42.82 |162.7±102.14 |

***

