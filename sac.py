import copy
import os

import torch
import torch.nn.functional as F
from torch.optim import Adam

from model import DeterministicPolicy, GaussianPolicy, QNetwork, ValueNetwork
from util import hard_update, soft_update
from utils import space_to_size

import numpy as np
import csv
import subprocess
from io import StringIO


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
    def __init__(self, num_inputs, action_space, args, writer):

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

        self.critic = QNetwork(self.num_inputs, self.action_space,
                               args.hidden_size)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.critic_lr)

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

            self.critic_target = QNetwork(self.num_inputs, self.action_space,
                                          args.hidden_size)
            hard_update(self.critic_target, self.critic)
            self.critic_target.to(self.device)

        self.policy.to(self.device)
        self.critic.to(self.device)
        self.value.to(self.device)
        self.value_target.to(self.device)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if eval == False:
            self.policy.train()
            action, _, _, _, _ = self.policy.sample(state)
        else:
            self.policy.eval()
            _, _, _, action, _ = self.policy.sample(state)
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
        """
        Use two Q-functions to mitigate positive bias in the policy improvement step that is known
        to degrade performance of value based methods. Two Q-functions also significantly speed
        up training, especially on harder task.
        """
        expected_q1_value, expected_q2_value = self.critic(
            state_batch, action_batch)
        new_action, log_prob, _, mean, log_std = self.policy.sample(
            state_batch)
        if self.algo == 'pmac':
            ref_actions, ref_log_prob, _, _, _ = self.reference_policy.sample(
                state_batch)

        if self.policy_type == "Gaussian":
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
            expected_value = self.value(state_batch)
            target_value = self.value_target(next_state_batch)
            next_q_value = reward_batch + mask_batch * self.gamma * (
                target_value).detach()
        else:
            """
            There is no need in principle to include a separate function approximator for the state value.
            We use a target critic network for deterministic policy and eradicate the value value network completely.
            """
            alpha_loss = torch.tensor(0.)
            alpha_logs = self.alpha  # For TensorboardX logs
            next_state_action, _, _, _, _, = self.policy.sample(
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
        q1_value_loss = F.mse_loss(expected_q1_value, next_q_value)
        q2_value_loss = F.mse_loss(expected_q2_value, next_q_value)
        q1_new, q2_new = self.critic(state_batch, new_action)
        expected_new_q_value = torch.min(q1_new, q2_new)

        if self.policy_type == "Gaussian":
            """
            Including a separate function approximator for the soft value can stabilize training and is convenient to 
            train simultaneously with the other networks
            Update the V towards the min of two Q-functions in order to reduce overestimation bias from function 
            approximation error.
            JV = 𝔼st~D[0.5(V(st) - (𝔼at~π[Qmin(st,at) - α * log π(at|st)]))^2]
            ∇JV = ∇V(st)(V(st) - Q(st,at) + (α * logπ(at|st)))
            """
            next_value = expected_new_q_value - (self.tau_ * log_prob)
            value_loss = F.mse_loss(expected_value, next_value.detach())
        else:
            pass
        if self.algo == 'sac':
            """
            Reparameterization trick is used to get a low variance estimator
            f(εt;st) = action sampled from the policy
            εt is an input noise vector, sampled from some fixed distribution
            Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            ∇Jπ = ∇log π + ([∇at (α * logπ(at|st)) − ∇at Q(st,at)])∇f(εt;st)
            """
            policy_loss = (
                (self.alpha * log_prob) - expected_new_q_value).mean()

            # Regularization Loss
            mean_loss = 0.001 * mean.pow(2).mean()
            std_loss = 0.001 * log_std.pow(2).mean()

            policy_loss += mean_loss + std_loss
        elif self.algo == 'pmac':
            ref_q = torch.min(*self.critic(state_batch, ref_actions))
            coefficient = torch.exp(
                (ref_q - self.tau * ref_log_prob -
                 expected_value) / (self.tau + self.tau_))
            policy_loss = coefficient.detach() * log_prob
            policy_loss = policy_loss.mean()
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
        self.writer.add_scalar('Q', expected_new_q_value.mean().item(), updates)
        self.writer.add_scalar('V', expected_value.mean().item(), updates)
        self.writer.add_scalar('Q1', expected_q1_value.mean().item(), updates)
        self.writer.add_scalar('Q2', expected_q2_value.mean().item(), updates)
        self.writer.add_scalar('value', expected_q2_value.mean().item(), updates)
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
