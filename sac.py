import copy
import csv
import os
import subprocess
from io import StringIO

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from model import Critic, DeterministicPolicy, GaussianPolicy, ValueNetwork
from util import hard_update, soft_update
from utils import space_to_size


def get_freer_gpu():
    nvidia_smi = subprocess.check_output(
        'nvidia-smi --format=csv --query-gpu=memory.free'.split(),
        universal_newlines=True)
    free_memory = [
        float(x[0].split()[0])
        for i, x in enumerate(csv.reader(StringIO(nvidia_smi))) if i > 0
    ]
    return np.argmax(free_memory).item()


class SAC(object):
    def __init__(
            self,
            num_inputs,
            action_space,
            args,
            writer,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
    ):

        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.writer = writer
        self.algo = args.algo
        if args.algo == 'pmac':
            assert args.tau is not None
            assert args.tau_ is not None
        self.tau = args.tau
        self.tau_ = args.tau_ or args.alpha
        self.num_inputs = num_inputs
        self.action_space = space_to_size(action_space)
        self.gamma = args.gamma
        self.smoothing = args.smoothing
        self.clip = args.clip

        self.device = torch.device('cpu')
        if args.cuda and torch.cuda.is_available():
            self.device = torch.device('cuda', index=get_freer_gpu())

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.critic = Critic(self.num_inputs, self.action_space,
                             args.hidden_size)
        self.q1_optim = Adam(self.critic.q1.parameters(), lr=args.critic_lr)
        self.q2_optim = Adam(self.critic.q2.parameters(), lr=args.critic_lr)

        if self.policy_type == "Gaussian":
            self.alpha = args.alpha
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning:
                self.target_entropy = -torch.prod(
                    torch.Tensor(action_space.shape)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True)
                self.alpha_optim = Adam([self.log_alpha], lr=args.alpha_lr)
            else:
                pass

            self.policy = GaussianPolicy(self.num_inputs, self.action_space,
                                         args.hidden_size)
            self.reference_policy = None
            if args.algo == 'pmac':
                self.reference_policy = copy.deepcopy(self.policy)
                self.reference_policy.to(self.device)
            self.policy_optim = Adam(
                self.policy.parameters(), lr=args.actor_lr)

            self.value = ValueNetwork(self.num_inputs, args.hidden_size)
            self.value_target = ValueNetwork(self.num_inputs, args.hidden_size)
            self.value_optim = Adam(self.value.parameters(), lr=args.critic_lr)
            hard_update(self.value_target, self.value)
        else:
            self.policy = DeterministicPolicy(
                self.num_inputs, self.action_space, args.hidden_size)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

            self.critic_target = Critic(self.num_inputs, self.action_space,
                                        args.hidden_size)
            hard_update(self.critic_target, self.critic)
            self.critic_target.to(self.device)

        self.policy.to(self.device)
        self.critic.to(self.device)
        self.value.to(self.device)
        self.value_target.to(self.device)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if not eval:
            self.policy.train()
            action, _, _, _, _, _ = self.policy.sample(state)
        else:
            self.policy.eval()
            _, _, _, action, _, _ = self.policy.sample(state)
            if self.policy_type == "Gaussian":
                action = torch.tanh(action)
            else:
                pass
        # action = torch.tanh(action)
        action = action.detach().cpu().numpy()
        return action[0]

    def update_parameters(self, state_batch, action_batch, reward_batch,
                          next_state_batch, mask_batch, updates):
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(
            self.device)
        mask_batch = torch.FloatTensor(np.float32(mask_batch)).unsqueeze(1).to(
            self.device)

        import pickle
        with open('batch.pkl', 'rb') as f:
            batch = pickle.load(f)
        state_batch = batch['observations']
        next_state_batch = batch['next_observations']
        action_batch = batch['actions']
        reward_batch = batch['rewards']
        mask_batch = 1. - batch['terminals']

        with open('networks.pkl', 'rb') as f:
            networks = pickle.load(f)

        with open('optim.pkl', 'rb') as f:
            optims = pickle.load(f)
        optim_mapping = {
            self.policy_optim: 'policy',
            self.value_optim: 'vf',
            self.q1_optim: 'qf1',
            self.q2_optim: 'qf2',
        }

        optim_dict = dict(optims)
        for target, source in optim_mapping.items():
            import ipdb
            ipdb.set_trace()
            target.load_state_dict(optim_dict[source])

        print(optim_dict['qf1'])
        print(self.q1_optim.state_dict())

        critic_mapping = {
            self.critic.q1.linear1: ('qf1', 'fc0'),
            self.critic.q1.linear2: ('qf1', 'fc1'),
            self.critic.q1.linear3: ('qf1', 'last_fc'),
            self.critic.q2.linear1: ('qf2', 'fc0'),
            self.critic.q2.linear2: ('qf2', 'fc1'),
            self.critic.q2.linear3: ('qf2', 'last_fc'),
        }
        policy_mapping = {
            self.policy: 'policy',
            self.reference_policy: 'ref_policy',
        }

        for target_layer, (source_net, source_layer) in critic_mapping.items():
            source_dict = dict(networks[source_net])
            target_layer.load_state_dict(
                dict(
                    weight=source_dict[source_layer + '.weight'],
                    bias=source_dict[source_layer + '.bias'],
                ))

        for target_net, source_net in policy_mapping.items():
            layer_mapping = dict(
                linear1='fc0',
                linear2='fc1',
                mean_linear='last_fc',
                log_std_linear='last_fc_log_std')
            source_dict = dict(networks[source_net])
            for target_layer, source_layer in layer_mapping.items():
                target_net._modules[target_layer].load_state_dict(
                    dict(
                        weight=source_dict[source_layer + '.weight'],
                        bias=source_dict[source_layer + '.bias']))

        value_mapping = {
            self.value: 'vf',
            self.value_target: 'target_vf',
        }

        for target_net, source_net in value_mapping.items():
            source_dict = dict(networks[source_net])
            state_dict = dict(
                **{
                    f'linear{n + 1}.weight': source_dict[f'fc{n}.weight']
                    for n in range(2)
                }, **{
                    f'linear{n + 1}.bias': source_dict[f'fc{n}.bias']
                    for n in range(2)
                }, **{
                    'linear3.weight': source_dict['last_fc.weight'],
                    'linear3.bias': source_dict['last_fc.bias'],
                })
            target_net.load_state_dict(state_dict)

        print(list(self.value_target.parameters())[0][0])
        """
            Use two Q-functions to mitigate positive bias in the policy improvement step that is known
        to degrade performance of value based methods. Two Q-functions also significantly speed
        up training, especially on harder task.
        """
        if self.algo == 'pmac':
            *ref_values, _ = self.reference_policy.sample(state_batch)
            new_action, ref_log_prob, ref_act_tanh, _, _ = [
                x.detach() for x in ref_values
            ]
            _, _, pre_tanh_value, policy_mean, log_std, policy_dist = self.policy.sample(
                state_batch)
            log_prob = policy_dist.log_prob(ref_act_tanh)
        else:
            new_action, log_prob, pre_tanh_value, policy_mean, log_std, policy_dist = self.policy.sample(
                state_batch)
        value = self.value(state_batch)
        target_value = self.value_target(next_state_batch)
        next_q_value = reward_batch + mask_batch * self.gamma * (
            target_value).detach()
        q1_value, q2_value = self.critic(state_batch, action_batch)

        if False:  # self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning:
                """
                Alpha Loss
                """
                alpha_loss = -(self.log_alpha * (
                    log_prob + self.target_entropy).detach()).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                if self.clip:
                    torch.nn.utils.clip_grad_norm([self.log_alpha], self.clip)
                self.alpha_optim.step()
                self.alpha = self.log_alpha.exp()
                alpha_logs = self.alpha.clone()  # For TensorboardX logs
            else:
                alpha_loss = torch.tensor(0.)
                alpha_logs = self.alpha  # For TensorboardX logs
            """
            Including a separate function approximator for the soft value can stabilize training.
            """
        elif False:
            """
            There is no need in principle to include a separate function approximator for the state value.
            We use a target critic network for deterministic policy and eradicate the value value network completely.
            """
            alpha_loss = torch.tensor(0.)
            alpha_logs = self.alpha  # For TensorboardX logs
            next_state_action, _, _, _, _, _ = self.policy.sample(
                next_state_batch)
            target_critic_1, target_critic_2 = self.critic_target(
                next_state_batch, next_state_action)
            target_critic = torch.min(target_critic_1, target_critic_2)
            next_q_value = reward_batch + mask_batch * self.gamma * (
                target_critic).detach()
        """
        Soft Q-function parameters can be trained to minimize the soft Bellman residual
        JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        ∇JQ = ∇Q(st,at)(Q(st,at) - r(st,at) - γV(target)(st+1))
        """
        q1_value_loss = F.mse_loss(q1_value, next_q_value)
        q2_value_loss = F.mse_loss(q2_value, next_q_value)
        q1_new, q2_new = self.critic(state_batch, new_action)
        new_q_value = torch.min(q1_new, q2_new)

        if self.algo == 'sac':
            """
            Including a separate function approximator for the soft value can stabilize training and is convenient to 
            train simultaneously with the other networks
            Update the V towards the min of two Q-functions in order to reduce overestimation bias from function 
            approximation error.
            JV = 𝔼st~D[0.5(V(st) - (𝔼at~π[Qmin(st,at) - α * log π(at|st)]))^2]
            ∇JV = ∇V(st)(V(st) - Q(st,at) + (α * logπ(at|st)))
            """
            next_value = new_q_value - (self.tau_ * log_prob)
            value_loss = F.mse_loss(value, next_value.detach())
            """
            Reparameterization trick is used to get a low variance estimator
            f(εt;st) = action sampled from the policy
            εt is an input noise vector, sampled from some fixed distribution
            Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            ∇Jπ = ∇log π + ([∇at (α * logπ(at|st)) − ∇at Q(st,at)])∇f(εt;st)
            """
            policy_loss = ((self.alpha * log_prob) - new_q_value).mean()

            # Regularization Loss
            mean_loss = 0.001 * policy_mean.pow(2).mean()
            std_loss = 0.001 * log_std.pow(2).mean()

            policy_loss += mean_loss + std_loss
        elif self.algo == 'pmac':
            value_loss = F.mse_loss(value, new_q_value.detach())
            #
            # value = self.value(state_batch).detach()
            # log_prob_ref_actions = policy_dist.log_prob(ref_actions)
            target_policy = torch.exp(
                (new_q_value - self.tau_ * ref_log_prob - value) /
                (self.tau + self.tau_))
            # target_policy = torch.clamp(target_policy, max=0.9).detach()
            policy_loss = (target_policy * (target_policy - log_prob)).mean()
            mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**
                                                           2).mean()
            std_reg_loss = self.policy_std_reg_weight * (log_std**2).mean()
            # pre_tanh_value = policy_outputs[-1]
            pre_activation_reg_loss = self.policy_pre_activation_weight * (
                (pre_tanh_value**2).sum(dim=1).mean())
            policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
            policy_loss = policy_loss + policy_reg_loss
            import ipdb
            ipdb.set_trace()
        else:
            raise RuntimeError

        self.q1_optim.zero_grad()
        q1_value_loss.backward()
        if self.clip:
            torch.nn.utils.clip_grad_norm(self.critic.parameters(), self.clip)
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_value_loss.backward()
        if self.clip:
            torch.nn.utils.clip_grad_norm(self.critic.parameters(), self.clip)
        self.q2_optim.step()

        if self.policy_type == "Gaussian":
            self.value_optim.zero_grad()
            value_loss.backward()
            if self.clip:
                torch.nn.utils.clip_grad_norm(self.value.parameters(),
                                              self.clip)
            self.value_optim.step()
        else:
            value_loss = torch.tensor(0.)

        self.policy_optim.zero_grad()
        if self.clip:
            torch.nn.utils.clip_grad_norm(self.policy.parameters(), self.clip)
        policy_loss.backward()
        self.policy_optim.step()
        """
        We update the target weights to match the current value function weights periodically
        Update target parameter after every n(args.target_update_interval) updates
        """
        if updates % self.target_update_interval == 0 and self.policy_type == "Deterministic":
            soft_update(self.critic_target, self.critic, self.smoothing)

        elif updates % self.target_update_interval == 0 and self.policy_type == "Gaussian":
            soft_update(self.value_target, self.value, self.smoothing)

        self.writer.add_scalar('value loss', value_loss.item(), updates)
        self.writer.add_scalar('critic1 loss', q1_value_loss.item(), updates)
        self.writer.add_scalar('critic2 loss', q2_value_loss.item(), updates)
        self.writer.add_scalar('policy loss', policy_loss.item(), updates)
        self.writer.add_scalar('Q', new_q_value.mean().item(), updates)
        self.writer.add_scalar('V', value.mean().item(), updates)
        self.writer.add_scalar('Q1', q1_value.mean().item(), updates)
        self.writer.add_scalar('Q2', q2_value.mean().item(), updates)
        self.writer.add_scalar('value', q2_value.mean().item(), updates)
        self.writer.add_scalar('std dev', log_std.exp().mean().item(), updates)

    # Save model parameters
    def save_model(self,
                   env_name,
                   suffix="",
                   actor_path=None,
                   critic_path=None,
                   value_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        if value_path is None:
            value_path = "models/sac_value_{}_{}".format(env_name, suffix)
        print('Saving models to {}, {} and {}'.format(actor_path, critic_path,
                                                      value_path))
        torch.save(self.value.state_dict(), value_path)
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path, value_path):
        print('Loading models from {}, {} and {}'.format(
            actor_path, critic_path, value_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
        if value_path is not None:
            self.value.load_state_dict(torch.load(value_path))
