import argparse
import math
from typing import List

import numpy as np
import gym
import torch

def hierarchical_parse_args(parser: argparse.ArgumentParser,
                            include_positional=False):
    """
    :return:
    {
        group1: {**kwargs}
        group2: {**kwargs}
        ...
        **kwargs
    }
    """
    args = parser.parse_args()

    def key_value_pairs(group):
        for action in group._group_actions:
            if action.dest != 'help':
                yield action.dest, getattr(args, action.dest, None)

    def get_positionals(groups):
        for group in groups:
            if group.title == 'positional arguments':
                for k, v in key_value_pairs(group):
                    yield v

    def get_nonpositionals(groups: List[argparse._ArgumentGroup]):
        for group in groups:
            if group.title != 'positional arguments':
                children = key_value_pairs(group)
                descendants = get_nonpositionals(group._action_groups)
                yield group.title, {**dict(children), **dict(descendants)}

    positional = list(get_positionals(parser._action_groups))
    nonpositional = dict(get_nonpositionals(parser._action_groups))
    optional = nonpositional.pop('optional arguments')
    nonpositional = {**nonpositional, **optional}
    if include_positional:
        return positional, nonpositional
    return nonpositional

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) +
                                param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def space_to_size(space: gym.Space):
    if isinstance(space, gym.spaces.Discrete):
        return space.n
    elif isinstance(space, (gym.spaces.Dict, gym.spaces.Tuple)):
        if isinstance(space, gym.spaces.Dict):
            _spaces = list(space.spaces.values())
        else:
            _spaces = list(space.spaces)
        return sum(space_to_size(s) for s in _spaces)
    else:
        return space.shape[0]

def to_numpy(x):
    return x.detach().cpu().numpy()

def to_torch(x, device=None):
    if x.dtype == np.bool:
        x = x.astype(int)
    device = device or torch.device('cpu')
    return torch.tensor(x, dtype=torch.float).to(device)