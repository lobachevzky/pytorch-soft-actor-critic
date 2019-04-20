import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from util import to_numpy, to_torch
from collections import namedtuple

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

ActValues = namedtuple('ActValues',
                       'action mean log_std log_prob std act_tanh dist')


# Initialize Policy weights
def weights_init_(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


class Critic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(Critic, self).__init__()
        self.q1 = QNetwork(num_inputs, num_actions, hidden_dim)
        self.q2 = QNetwork(num_inputs, num_actions, hidden_dim)

    def forward(self, state, action):
        return self.q1.forward(state, action), self.q2.forward(state, action)


class GaussianPolicy(nn.Module):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
            self,
            hidden_size,
            obs_dim,
            action_dim,
            device,
    ):
        self.device = device
        super().__init__()

        self.linear1 = nn.Linear(obs_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, action_dim)
        self.log_std_linear = nn.Linear(hidden_size, action_dim)

        self.apply(weights_init_)

    def select_action(self, obs_np, eval=False):
        actions = self.get_actions(obs_np[None], deterministic=eval)
        return actions[0, :]

    def get_actions(self, obs_np, deterministic=False):
        out = self.forward(to_torch(obs_np, device=self.device),
                           deterministic=deterministic)
        return to_numpy(out.action)
        # return self.eval_np(obs_np, deterministic=deterministic)[0]

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(
            self,
            obs,
            act_tanh=None,
            deterministic=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)

        if deterministic:
            act = torch.tanh(mean)
            log_prob = None
        else:
            normal = Normal(mean, std)
            if act_tanh is None:
                act_tanh = normal.sample()

            act = torch.tanh(act_tanh)
            log_prob = normal.log_prob(act_tanh) - torch.log(1 - act**2 + 1e-6)
            log_prob = log_prob.sum(dim=1, keepdim=True)

        return ActValues(act, mean, log_std, log_prob, std, act_tanh, normal)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x))
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), torch.tensor(0.), mean, torch.tensor(
            0.)
