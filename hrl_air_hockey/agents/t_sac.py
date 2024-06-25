import torch
import numpy as np
from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.policy import Policy
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.parameters import to_parameter
from hrl_air_hockey.utils.smdp_replay_memory import SMDPReplayMemory

from copy import deepcopy
from itertools import chain

from hrl_air_hockey.bspline_planner.planner import TrajectoryPlanner


class SACPolicy(Policy):
    """
    Class used to implement the policy used by the Soft Actor-Critic algorithm.
    The policy is a Gaussian policy squashed by a tanh. This class implements the compute_action_and_log_prob and the
    compute_action_and_log_prob_t methods, that are fundamental for the internals calculations of the SAC algorithm.

    """

    def __init__(self, mu_approximator, sigma_approximator, min_a, max_a, log_std_min, log_std_max):
        """
        Constructor.

        Args:
            mu_approximator (Regressor): a regressor computing mean in given a state;
            sigma_approximator (Regressor): a regressor computing the variance in given a state;
            min_a (np.ndarray): a vector specifying the minimum action value for each component;
            max_a (np.ndarray): a vector specifying the maximum action value for each component.
            log_std_min ([float, Parameter]): min value for the policy log std;
            log_std_max ([float, Parameter]): max value for the policy log std.

        """
        self._mu_approximator = mu_approximator
        self._sigma_approximator = sigma_approximator

        self._delta_a = to_float_tensor(.5 * (max_a - min_a), self.use_cuda)
        self._central_a = to_float_tensor(.5 * (max_a + min_a), self.use_cuda)

        self._log_std_min = to_parameter(log_std_min)
        self._log_std_max = to_parameter(log_std_max)

        self._eps_log_prob = 1e-6

        use_cuda = self._mu_approximator.model.use_cuda

        if use_cuda:
            self._delta_a = self._delta_a.cuda()
            self._central_a = self._central_a.cuda()

        self._add_save_attr(
            _mu_approximator='mushroom',
            _sigma_approximator='mushroom',
            _delta_a='torch',
            _central_a='torch',
            _log_std_min='mushroom',
            _log_std_max='mushroom',
            _eps_log_prob='primitive'
        )

    def __call__(self, state, action):
        raise NotImplementedError

    def draw_action(self, state):
        return self.compute_action_and_log_prob_t(
            state, compute_log_prob=False).detach().cpu().numpy()

    def compute_action_and_log_prob(self, state):
        """
        Function that samples actions using the reparametrization trick and the log probability for such actions.

        Args:
            state (np.ndarray): the state in which the action is sampled.

        Returns:
            The actions sampled and the log probability as numpy arrays.

        """
        a, log_prob = self.compute_action_and_log_prob_t(state)
        return a.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    def compute_action_with_grad(self, state):
        """
        Function that samples actions using the reparametrization trick and the log probability for such actions.

        Args:
            state (np.ndarray): the state in which the action is sampled.

        Returns:
            The actions sampled as numpy arrays.

        """
        a, log_prob = self.compute_action_and_log_prob_t(state)
        return a.cpu().numpy()

    def compute_action_and_log_prob_t(self, state, compute_log_prob=True):
        """
        Function that samples actions using the reparametrization trick and, optionally, the log probability for such
        actions.

        Args:
            state (np.ndarray): the state in which the action is sampled;
            compute_log_prob (bool, True): whether to compute the log  probability or not.

        Returns:
            The actions sampled and, optionally, the log probability as torch tensors.

        """
        dist = self.distribution(state)
        a_raw = dist.rsample()
        a = torch.tanh(a_raw)
        a_true = a * self._delta_a + self._central_a

        if compute_log_prob:
            log_prob = dist.log_prob(a_raw).sum(dim=1)
            log_prob -= torch.log(1. - a.pow(2) + self._eps_log_prob).sum(dim=1)
            return a_true, log_prob
        else:
            return a_true

    def distribution(self, state):
        """
        Compute the policy distribution in the given states.

        Args:
            state (np.ndarray): the set of states where the distribution is computed.

        Returns:
            The torch distribution for the provided states.

        """
        mu = self._mu_approximator.predict(state, output_tensor=True)
        log_sigma = self._sigma_approximator.predict(state, output_tensor=True)
        # Bound the log_std
        log_sigma = torch.clamp(log_sigma, self._log_std_min(), self._log_std_max())
        return torch.distributions.Normal(mu, log_sigma.exp())

    def entropy(self, state=None):
        """
        Compute the entropy of the policy.

        Args:
            state (np.ndarray): the set of states to consider.

        Returns:
            The value of the entropy of the policy.

        """
        _, log_pi = self.compute_action_and_log_prob(state)
        return -log_pi.mean()

    def set_weights(self, weights):
        """
        Setter.

        Args:
            weights (np.ndarray): the vector of the new weights to be used by the policy.

        """
        mu_weights = weights[:self._mu_approximator.weights_size]
        sigma_weights = weights[self._mu_approximator.weights_size:]

        self._mu_approximator.set_weights(mu_weights)
        self._sigma_approximator.set_weights(sigma_weights)

    def get_weights(self):
        """
        Getter.

        Returns:
             The current policy weights.

        """
        mu_weights = self._mu_approximator.get_weights()
        sigma_weights = self._sigma_approximator.get_weights()

        return np.concatenate([mu_weights, sigma_weights])

    @property
    def use_cuda(self):
        """
        True if the policy is using cuda_tensors.
        """
        return self._mu_approximator.model.use_cuda

    def parameters(self):
        """
        Returns the trainable policy parameters, as expected by torch optimizers.

        Returns:
            List of parameters to be optimized.

        """
        return chain(self._mu_approximator.model.network.parameters(),
                     self._sigma_approximator.model.network.parameters())


class SACPlusTermination(SAC):
    """
    Soft Actor-Critic algorithm.
    "Soft Actor-Critic Algorithms and Applications".
    Haarnoja T. et al.. 2019.

    """

    def __init__(self, mdp_info, actor_mu_params, actor_sigma_params, actor_optimizer, critic_params,
                 nn_planner_params, termination_params, termination_optimizer, batch_size,
                 initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha, num_adv_sample,
                 use_log_alpha_loss=False, log_std_min=-20, log_std_max=2, target_entropy=None, critic_fit_params=None):

        super().__init__(mdp_info=mdp_info, actor_mu_params=actor_mu_params, actor_sigma_params=actor_sigma_params,
                         actor_optimizer=actor_optimizer, critic_params=critic_params, batch_size=batch_size,
                         initial_replay_size=initial_replay_size, max_replay_size=max_replay_size,
                         warmup_transitions=warmup_transitions, tau=tau, lr_alpha=lr_alpha,
                         use_log_alpha_loss=use_log_alpha_loss,
                         log_std_min=log_std_min, log_std_max=log_std_max, target_entropy=target_entropy,
                         critic_fit_params=critic_fit_params)

        self._smdp_replay_memory = SMDPReplayMemory(initial_replay_size, max_replay_size)
        self.traj_planner = TrajectoryPlanner(**nn_planner_params)

        self.termination_approximator = Regressor(TorchApproximator, **termination_params)
        termination_parameters = self.termination_approximator.model.network.parameters()
        self.termination_optimizer = termination_optimizer['class'](termination_parameters,
                                                                    **termination_optimizer['params'])

        self.num_adv_sample = num_adv_sample

        self._add_save_attr(planner="torch", termination_approximator="mushroom")

    def episode_start(self):
        self.cur_smdp_length = 0
        self.sum_reward = 0
        self.initial_state = None
        self.initial_action = None
        self.initial_smdp_state = None
        self.initial_smdp_action = None
        self.trajectory_buffer = None
        self.last_action = None
        self.policy.reset()

    def draw_action(self, state):
        if self.trajectory_buffer is None or len(self.trajectory_buffer) == 0:
            self.last_action = self.policy.draw_action(state)
            q, dq = self._get_joint_pos(state)
            hit_pos, hit_dir, hit_scale, vel_angle = self._get_target_point(self.last_action)
            self.trajectory_buffer = \
            self.traj_planner.plan_trajectory(q, dq, hit_pos, hit_dir, hit_scale, self.last_action)[0]
            termination = np.array([1])
        else:
            term_prob = self.termination_approximator.predict(state, self.last_action, output_tensor=False)
            # term_prob = 0.02
            termination = np.array([0])
            if np.random.uniform() < term_prob:
                self.last_action = self.policy.draw_action(state)

                q, dq = self._get_joint_pos(state)
                hit_pos, hit_dir, hit_scale, vel_angle = self._get_target_point(self.last_action)
                self.trajectory_buffer = \
                self.traj_planner.plan_trajectory(q, dq, hit_pos, hit_dir, hit_scale, self.last_action)[0]
                termination = np.array([1])
        assert len(self.trajectory_buffer) > 0
        joint_command = self.trajectory_buffer[0, :14]
        self.trajectory_buffer = self.trajectory_buffer[1:]

        if termination == 1:
            last_smdp_length = self.cur_smdp_length
            self.cur_smdp_length = 0
        else:
            last_smdp_length = self.cur_smdp_length
            self.cur_smdp_length += 1

        return np.concatenate([joint_command, self.last_action, termination, np.array([last_smdp_length]),
                               np.array([self.cur_smdp_length])])

    def _get_joint_pos(self, state):
        q = state[..., 6:13]
        dq = state[..., 13:20]
        return q, dq

    def _get_target_point(self, action):
        hit_pos = action[:2]
        angle = action[2]
        scale = action[3]
        hit_vel = np.array([np.cos(angle), np.sin(angle)])
        return hit_pos, hit_vel, scale, angle

    def fit(self, dataset, **info):
        smdp_dataset = self.dataset_preprocessor_1(dataset)
        termination_dataset = self.dataset_preprocessor_2(dataset)
        self._smdp_replay_memory.add(smdp_dataset)
        self._replay_memory.add(termination_dataset)
        for i in range(20):
            if self._smdp_replay_memory.initialized:
                # for updating the critic and actor
                smdp_state, smdp_action, smdp_reward, smdp_next_state, absorbing, _, smdp_length = self._smdp_replay_memory.get(
                    self._batch_size())
                # for updating the termination network
                initial_state, initial_action, _, next_state, _, _ = self._replay_memory.get(self._batch_size())

                if self._smdp_replay_memory.size > self._warmup_transitions():
                    action_new, log_prob = self.policy.compute_action_and_log_prob_t(smdp_state)
                    action_new_prime, _ = self.policy.compute_action_and_log_prob_t(next_state)
                    # update actor
                    actor_loss = self._loss(smdp_state, action_new, log_prob)
                    self._optimize_actor_parameters(actor_loss)
                    self._update_alpha(log_prob.detach())
                    # update beta(termination)
                    beta_loss = self.termination_loss(initial_action, next_state, action_new_prime)
                    self.optimize_termination_parameters(beta_loss)

                q_next = self._next_q(smdp_next_state, absorbing)
                q = smdp_reward + (self.mdp_info.gamma ** smdp_length) * q_next

                self._critic_approximator.fit(smdp_state, smdp_action, q, **self._critic_fit_params)

                self._update_target(self._critic_approximator, self._target_critic_approximator)

    def dataset_preprocessor_1(self, dataset):
        smdp_dataset = list()
        for i, d in enumerate(dataset):
            high_action = d[1][14:18]
            termination = d[1][18]
            last_smdp_length = int(d[1][19])
            cur_smdp_length = int(d[1][20])
            reward = d[2]
            next_state = d[3]
            absorbing = d[4]
            last = d[5]
            self.sum_reward += reward * (self.mdp_info.gamma ** cur_smdp_length)
            if termination == 1 or absorbing or last:
                if self.initial_smdp_state is not None:
                    smdp_dataset.append((self.initial_smdp_state, self.initial_smdp_action, self.sum_reward, next_state,
                                         absorbing, last, last_smdp_length))
                self.sum_reward = 0
                self.initial_smdp_state = d[0]
                self.initial_smdp_action = high_action
        return smdp_dataset

    def dataset_preprocessor_2(self, dataset):
        termination_dataset = list()
        for i, d in enumerate(dataset):
            high_action = d[1][14:18]
            termination = d[1][18]
            last_smdp_length = int(d[1][19])
            reward = d[2]
            next_state = d[3]
            absorbing = d[4]
            last = d[5]
            if termination == 1 or absorbing or last:
                self.initial_state = d[0]
                self.initial_action = high_action
            if self.initial_state is not None:
                termination_dataset.append((self.initial_state, self.initial_action, reward, next_state, absorbing, last))
        return termination_dataset

    def termination_loss(self, action, next_state, action_new_prime):
        # - 1/n * sum_n( beta(s_n', w_n') * A(s_n', w_n') )
        adv_func = self.adv_func(action, next_state, action_new_prime)
        beta = self.termination_approximator.predict(next_state, action_new_prime, output_tensor=True)
        adv_func_tensor = torch.tensor(adv_func, device=beta.device, requires_grad=False)
        return (beta * adv_func_tensor.view_as(beta)).mean()

    def adv_func(self, action, next_state, action_new_prime):
        # A(s', w') = Q(s', w') - V(s') =  Q(s', w') - 1/n * [Q(s', w_0) + Q(s', w_1) + ... + Q(s', w_n)]
        q_0 = self._critic_approximator(next_state, action_new_prime, output_tensor=False, idx=0)
        q_1 = self._critic_approximator(next_state, action_new_prime, output_tensor=False, idx=1)
        q = np.minimum(q_0, q_1)
        _q_sum = np.zeros_like(q)
        # sample w_n
        for _ in range(self.num_adv_sample):
            # sample rule: Prob of w_old: 1-beta(s',w). Prob of w_new = beta(s',w) * policy_dist
            termination_prime = self.termination_approximator.predict(next_state, action, output_tensor=False).flatten()
            terminte_mask = np.random.rand(*termination_prime.shape) < termination_prime
            sampled_action = np.zeros_like(action)
            sampled_action[~terminte_mask, :] = action[~terminte_mask, :]
            policy_actions, _ = self.policy.compute_action_and_log_prob_t(next_state[terminte_mask, :])
            sampled_action[terminte_mask, :] = policy_actions.detach()

            _q_0 = self._critic_approximator(next_state, sampled_action, output_tensor=False, idx=0)
            _q_1 = self._critic_approximator(next_state, sampled_action, output_tensor=False, idx=1)
            _q = np.minimum(_q_0, _q_1)
            _q_sum += _q
        v = _q_sum / self.num_adv_sample
        return q - v

    def optimize_termination_parameters(self, loss):
        self.termination_optimizer.zero_grad()
        loss.backward()
        self.termination_optimizer.step()

    def _loss(self, state, action_new, log_prob):
        q_0 = self._critic_approximator(state, action_new, output_tensor=True, idx=0)
        q_1 = self._critic_approximator(state, action_new, output_tensor=True, idx=1)

        q = torch.min(q_0, q_1)

        return (self._alpha * log_prob - q).mean()

    def _update_alpha(self, log_prob):
        if self._use_log_alpha_loss:
            alpha_loss = - (self._log_alpha * (log_prob + self._target_entropy)).mean()
        else:
            alpha_loss = - (self._alpha * (log_prob + self._target_entropy)).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

    def _next_q(self, next_state, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in ``next_state``.

        Returns:
            Action-values returned by the critic for ``next_state`` and the action returned by the actor.

        """
        a, log_prob_next = self.policy.compute_action_and_log_prob(next_state)
        q = self._target_critic_approximator.predict(
            next_state, a, prediction='min') - self._alpha_np * log_prob_next
        q *= 1 - absorbing

        return q

    def _post_load(self):
        self._update_optimizer_parameters(self.policy.parameters())

    @property
    def _alpha(self):
        return self._log_alpha.exp()

    @property
    def _alpha_np(self):
        return self._alpha.detach().cpu().numpy()
