import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sqrt

LOG_STD_HI = -1.5
LOG_STD_LO = -20


def normc_fn(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


# The base class for an actor. Includes functions for normalizing state (optional)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.is_recurrent = False

        self.welford_state_mean = torch.zeros(1)
        self.welford_state_mean_diff = torch.ones(1)
        self.welford_state_n = 1

        self.env_name = None

    def forward(self):
        raise NotImplementedError

    def normalize_state(self, state, update=True):
        state = torch.Tensor(state)

        if self.welford_state_n == 1:
            self.welford_state_mean = torch.zeros(state.size(-1))
            self.welford_state_mean_diff = torch.ones(state.size(-1))
        #
        # if update:
        #     if len(state.size()) == 1:  # If we get a single state vector
        #         state_old = self.welford_state_mean
        #         self.welford_state_mean += (state - state_old) / self.welford_state_n
        #         self.welford_state_mean_diff += (state - state_old) * (state - state_old)
        #         self.welford_state_n += 1
        #     elif len(state.size()) == 2:  # If we get a batch
        #         print("NORMALIZING 2D TENSOR (this should not be happening)")
        #         for r_n in r:
        #             state_old = self.welford_state_mean
        #             self.welford_state_mean += (state_n - state_old) / self.welford_state_n
        #             self.welford_state_mean_diff += (state_n - state_old) * (state_n - state_old)
        #             self.welford_state_n += 1
        #     elif len(state.size()) == 3:  # If we get a batch of sequences
        #         print("NORMALIZING 3D TENSOR (this really should not be happening)")
        #         for r_t in r:
        #             for r_n in r_t:
        #                 state_old = self.welford_state_mean
        #                 self.welford_state_mean += (state_n - state_old) / self.welford_state_n
        #                 self.welford_state_mean_diff += (state_n - state_old) * (state_n - state_old)
        #                 self.welford_state_n += 1
        return (state - self.welford_state_mean) / sqrt(self.welford_state_mean_diff / self.welford_state_n)

    def copy_normalizer_stats(self, net):
        self.welford_state_mean = net.self_state_mean
        self.welford_state_mean_diff = net.welford_state_mean_diff
        self.welford_state_n = net.welford_state_n

    def initialize_parameters(self):
        self.apply(normc_fn)


class Actor(Net):
    def __init__(self):
        super(Actor, self).__init__()

    def forward(self):
        raise NotImplementedError

    def get_action(self):
        raise NotImplementedError


class Gaussian_FF_Actor(Actor):  # more consistent with other actor naming conventions
    def __init__(self, state_dim, action_dim, layers=(256, 256), nonlinearity=torch.nn.functional.relu,
                 fixed_std=np.exp(-2), normc_init=True):
        super(Gaussian_FF_Actor, self).__init__()

        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.Linear(state_dim, layers[0])]
        for i in range(len(layers) - 1):
            self.actor_layers += [nn.Linear(layers[i], layers[i + 1])]
        self.means = nn.Linear(layers[-1], action_dim)

        self.action = None
        self.action_dim = action_dim
        self.nonlinearity = nonlinearity

        # Initialized to no input normalization, can be modified later
        self.obs_std = 1.0
        self.obs_mean = 0.0
        self.fixed_std = fixed_std

        # weight initialization scheme used in PPO paper experiments
        self.normc_init = normc_init

        self.init_parameters()

    def init_parameters(self):
        if self.normc_init:
            self.apply(normc_fn)
            self.means.weight.data.mul_(0.01)

    def _get_dist_params(self, state):
        state = (state - self.obs_mean) / self.obs_std

        x = state
        for l in self.actor_layers:
            x = self.nonlinearity(l(x))
        mean = self.means(x)

        return mean, self.fixed_std

    def forward(self, state, deterministic=True):
        mu, sd = self._get_dist_params(state)

        self.action = torch.distributions.Normal(mu, sd)

        return self.action

    def get_action(self):
        return self.action

    def distribution(self, inputs):
        mu, sd = self._get_dist_params(inputs)
        return torch.distributions.Normal(mu, sd)


# The base class for a critic. Includes functions for normalizing reward and state (optional)
class Critic(Net):
    def __init__(self):
        super(Critic, self).__init__()

        self.welford_reward_mean = 0.0
        self.welford_reward_mean_diff = 1.0
        self.welford_reward_n = 1

    def forward(self):
        raise NotImplementedError

    def normalize_reward(self, r, update=True):
        if update:
            if len(r.size()) == 1:
                r_old = self.welford_reward_mean
                self.welford_reward_mean += (r - r_old) / self.welford_reward_n
                self.welford_reward_mean_diff += (r - r_old) * (r - r_old)
                self.welford_reward_n += 1
            elif len(r.size()) == 2:
                for r_n in r:
                    r_old = self.welford_reward_mean
                    self.welford_reward_mean += (r_n - r_old) / self.welford_reward_n
                    self.welford_reward_mean_diff += (r_n - r_old) * (r_n - r_old)
                    self.welford_reward_n += 1
            else:
                raise NotImplementedError

        return (r - self.welford_reward_mean) / torch.sqrt(self.welford_reward_mean_diff / self.welford_reward_n)


class FF_V(Critic):
    def __init__(self, state_dim, layers=(256, 256), env_name='NOT SET', nonlinearity=torch.nn.functional.relu,
                 normc_init=True, obs_std=None, obs_mean=None):
        super(FF_V, self).__init__()

        self.critic_layers = nn.ModuleList()
        self.critic_layers += [nn.Linear(state_dim, layers[0])]
        for i in range(len(layers) - 1):
            self.critic_layers += [nn.Linear(layers[i], layers[i + 1])]
        self.network_out = nn.Linear(layers[-1], 1)

        self.env_name = env_name

        self.nonlinearity = nonlinearity

        self.obs_std = obs_std
        self.obs_mean = obs_mean

        # weight initialization scheme used in PPO paper experiments
        self.normc_init = normc_init

        self.init_parameters()
        self.train()

    def init_parameters(self):
        if self.normc_init:
            print("Doing norm column initialization.")
            self.apply(normc_fn)

    def forward(self, inputs):
        if self.training == False:
            inputs = (inputs - self.obs_mean) / self.obs_std

        x = inputs
        for l in self.critic_layers:
            x = self.nonlinearity(l(x))
        value = self.network_out(x)

        return value

    def act(self, inputs):  # not needed, deprecated
        return self(inputs)


class ActorCriticNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(ActorCriticNetwork, self).__init__()
        self.actor = Gaussian_FF_Actor(num_inputs, num_actions, layers=(hidden_dim, hidden_dim))
        self.critic = FF_V(num_inputs, layers=(hidden_dim, hidden_dim))

    def to(self, device):
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        return super(ActorCriticNetwork, self).to(device)
