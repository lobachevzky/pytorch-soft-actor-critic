import copy
import csv
import os
import subprocess
from io import StringIO

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from model import DeterministicPolicy, GaussianPolicy, QNetwork, ValueNetwork
from util import hard_update, soft_update


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
            obs_dim,
            action_dim,
            updates_per_write,
            algo,
            tau1,
            tau2,
            alpha,
            gamma,
            smoothing,
            clip,
            cuda,
            policy,
            target_update_interval,
            automatic_entropy_tuning,
            hidden_size,
            critic_lr,
            actor_lr,
            alpha_lr,
            writer,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
    ):

        self.updates_per_write = updates_per_write
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.writer = writer
        self.algo = algo
        if algo == 'pmac':
            assert tau1 is not None
            assert tau2 is not None
        self.tau1 = tau1
        self.tau2 = tau2 or alpha
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.smoothing = smoothing
        self.clip = clip

        self.device = torch.device('cpu')
        if cuda and torch.cuda.is_available():
            self.device = torch.device('cuda', index=get_freer_gpu())

        self.policy_type = policy
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.critic = QNetwork(
            num_inputs=self.obs_dim,
            num_actions=self.action_dim,
            hidden_dim=hidden_size)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)

        if self.policy_type == "Gaussian":
            self.alpha = alpha
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning:
                raise NotImplementedError
                # self.target_entropy = -torch.prod(
                #     torch.Tensor(action_space.shape)).item()
                # self.log_alpha = torch.zeros(1, requires_grad=True)
                # self.alpha_optim = Adam([self.log_alpha], lr=alpha_lr)
            else:
                pass

            self.policy = GaussianPolicy(
                hidden_size=self.obs_dim,
                action_dim=self.action_dim,
                obs_dim=obs_dim,
                device=self.device)
            self.reference_policy = None
            if algo == 'pmac':
                self.reference_policy = copy.deepcopy(self.policy)
                self.reference_policy.to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=actor_lr)

            self.value = ValueNetwork(self.obs_dim, hidden_size)
            self.value_target = ValueNetwork(self.obs_dim, hidden_size)
            self.value_optim = Adam(self.value.parameters(), lr=critic_lr)
            hard_update(self.value_target, self.value)
        else:
            self.policy = DeterministicPolicy(self.obs_dim, self.action_dim,
                                              hidden_size)
            self.policy_optim = Adam(self.policy.parameters(), lr=critic_lr)

            self.critic_target = QNetwork(self.obs_dim, self.action_dim,
                                          hidden_size)
            hard_update(self.critic_target, self.critic)
            self.critic_target.to(self.device)

        self.policy.to(self.device)
        self.critic.to(self.device)
        self.value.to(self.device)
        self.value_target.to(self.device)

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if not deterministic:
            self.policy.train()
            action = self.policy.sample(state).action
        else:
            self.policy.eval()
            action = self.policy.sample(state).mean
            if self.policy_type == "Gaussian":
                action = torch.tanh(action)
            else:
                pass
        # action = torch.tanh(action)
        action = action.detach().cpu().numpy()
        return action[0]

    def update_parameters(self, state_batch, action_batch, reward_batch,
                          next_state_batch, done_batch, updates):
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(
            self.device)
        done_batch = torch.FloatTensor(np.float32(done_batch)).unsqueeze(1).to(
            self.device)
        """
            Use two Q-functions to mitigate positive bias in the policy improvement step that is known
        to degrade performance of value based methods. Two Q-functions also significantly speed
        up training, especially on harder task.
        """
        if self.algo == 'pmac':
            ref_values = self.reference_policy.sample(state_batch)
            new_action = ref_values.action.detach()
            ref_log_prob = ref_values.log_prob.detach()
            ref_act_tanh = ref_values.act_tanh.detach()
            policy_values = self.policy.sample(state_batch)
            pre_tanh_value = policy_values.act_tanh
            policy_mean = policy_values.mean
            log_std = policy_values.log_std
            policy_dist = policy_values.dist
            log_prob = policy_dist.log_prob(ref_act_tanh)
        else:
            new_action, policy_mean, log_std, log_prob, _, pre_tanh_value, policy_dist = self.policy.sample(
                state_batch)
        value = self.value(state_batch)
        target_value = self.value_target(next_state_batch)
        next_q_value = reward_batch + (1. - done_batch) * self.gamma * (
            target_value).detach()
        q1_value, q2_value = self.critic(state_batch, action_batch)

        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning:
                """
                Alpha Loss
                """
                raise NotImplementedError
                # alpha_loss = -(self.log_alpha * (
                #     log_prob + self.target_entropy).detach()).mean()
                # self.alpha_optim.zero_grad()
                # alpha_loss.backward()
                # if self.clip:
                #     torch.nn.utils.clip_grad_norm([self.log_alpha], self.clip)
                # self.alpha_optim.step()
                # self.alpha = self.log_alpha.exp()
                # alpha_logs = self.alpha.clone()  # For TensorboardX logs
            # else:
            #     alpha_loss = torch.tensor(0.)
            #     alpha_logs = self.alpha  # For TensorboardX logs
            """
            Including a separate function approximator for the soft value can stabilize training.
            """
        else:
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
            next_q_value = reward_batch + (1. - done_batch) * self.gamma * (
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
            next_value = new_q_value - (self.tau2 * log_prob)
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
                (new_q_value - self.tau2 * ref_log_prob - value) /
                (self.tau1 + self.tau2))
            target_policy = torch.clamp(target_policy, max=0.9).detach()
            policy_loss = (target_policy * (target_policy - log_prob)).mean()
            mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**
                                                           2).mean()
            std_reg_loss = self.policy_std_reg_weight * (log_std**2).mean()
            # pre_tanh_value = policy_outputs[-1]
            pre_activation_reg_loss = self.policy_pre_activation_weight * (
                (pre_tanh_value**2).sum(dim=1).mean())
            policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
            policy_loss = policy_loss + policy_reg_loss
        else:
            raise RuntimeError

        self.critic_optim.zero_grad()
        q1_value_loss.backward()
        if self.clip:
            torch.nn.utils.clip_grad_norm(self.critic.parameters(), self.clip)
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        q2_value_loss.backward()
        if self.clip:
            torch.nn.utils.clip_grad_norm(self.critic.parameters(), self.clip)
        self.critic_optim.step()

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
        Update target parameter after every n(target_update_interval) updates
        """
        if updates % self.target_update_interval == 0 and self.policy_type == "Deterministic":
            soft_update(self.critic_target, self.critic, self.smoothing)

        elif updates % self.target_update_interval == 0 and self.policy_type == "Gaussian":
            soft_update(self.value_target, self.value, self.smoothing)

        if updates % self.updates_per_write == 0 and self.writer:
            self.writer.add_scalar('value loss', value_loss.item(), updates)
            self.writer.add_scalar('critic1 loss', q1_value_loss.item(),
                                   updates)
            self.writer.add_scalar('critic2 loss', q2_value_loss.item(),
                                   updates)
            self.writer.add_scalar('policy loss', policy_loss.item(), updates)
            self.writer.add_scalar('Q', new_q_value.mean().item(), updates)
            self.writer.add_scalar('V', value.mean().item(), updates)
            self.writer.add_scalar('Q1', q1_value.mean().item(), updates)
            self.writer.add_scalar('Q2', q2_value.mean().item(), updates)
            self.writer.add_scalar('value', q2_value.mean().item(), updates)
            self.writer.add_scalar('std dev',
                                   log_std.exp().mean().item(), updates)

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
