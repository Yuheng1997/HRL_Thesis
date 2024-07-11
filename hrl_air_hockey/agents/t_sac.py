import torch
import numpy as np
from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from hrl_air_hockey.utils.t_replay_memory import TReplayMemory
from hrl_air_hockey.bspline_planner.planner import TrajectoryPlanner


class SACPlusTermination(SAC):
    """
    Soft Actor-Critic algorithm.
    "Soft Actor-Critic Algorithms and Applications".
    Haarnoja T. et al.. 2019.

    """

    def __init__(self, mdp_info, actor_mu_params, actor_sigma_params, actor_optimizer, critic_params,
                 nn_planner_params, termination_params, termination_optimizer, batch_size,
                 initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha, num_adv_sample, device,
                 use_log_alpha_loss=False, log_std_min=-20, log_std_max=2, target_entropy=None, critic_fit_params=None):

        super().__init__(mdp_info=mdp_info, actor_mu_params=actor_mu_params, actor_sigma_params=actor_sigma_params,
                         actor_optimizer=actor_optimizer, critic_params=critic_params, batch_size=batch_size,
                         initial_replay_size=initial_replay_size, max_replay_size=max_replay_size,
                         warmup_transitions=warmup_transitions, tau=tau, lr_alpha=lr_alpha,
                         use_log_alpha_loss=use_log_alpha_loss,
                         log_std_min=log_std_min, log_std_max=log_std_max, target_entropy=target_entropy,
                         critic_fit_params=critic_fit_params)

        self.replay_memory = TReplayMemory(initial_replay_size, max_replay_size)
        self.traj_planner = TrajectoryPlanner(**nn_planner_params)

        self.termination_approximator = Regressor(TorchApproximator, **termination_params)
        termination_parameters = self.termination_approximator.model.network.parameters()
        self.termination_optimizer = termination_optimizer['class'](termination_parameters,
                                                                    **termination_optimizer['params'])

        self.num_adv_sample = num_adv_sample
        self.device = device

        self._add_save_attr(traj_planner="torch", termination_approximator="mushroom")

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
                self.traj_planner.plan_trajectory(q, dq, hit_pos, hit_dir, hit_scale)[0]
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
                    self.traj_planner.plan_trajectory(q, dq, hit_pos, hit_dir, hit_scale)[0]
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
        t_dataset = self.add_t_dataset(dataset)
        self.replay_memory.add(t_dataset)
        for i in range(10):
            if self.replay_memory.initialized:
                state, option, reward, next_state, absorbing, _, can_terminate = self.replay_memory.get(
                    self._batch_size())

                if self.replay_memory.size > self._warmup_transitions():
                    action_new, log_prob = self.policy.compute_action_and_log_prob_t(state)
                    # update actor
                    actor_loss = self._loss(state, action_new, log_prob)
                    self._optimize_actor_parameters(actor_loss)
                    self._update_alpha(log_prob.detach())
                    # update beta(termination)
                    beta_loss = self.termination_loss(state, option, can_terminate)
                    self.optimize_termination_parameters(beta_loss)

                beta_prime = self.termination_approximator.predict(next_state, option, output_tensor=False).squeeze(-1)
                option_prime = self.policy.draw_action(next_state)
                gt = (reward + self.mdp_info.gamma * (1 - beta_prime) * self._target_critic_approximator.predict(
                    next_state, option, prediction='min') +
                    self.mdp_info.gamma * beta_prime * self._target_critic_approximator.predict(
                    next_state, option_prime, prediction='min'))

                self._critic_approximator.fit(state, option, gt, **self._critic_fit_params)

                self._update_target(self._critic_approximator, self._target_critic_approximator)

    def prepare_dataset(self, dataset):
        smdp_dataset = list()
        termination_dataset = list()
        for i, d in enumerate(dataset):
            high_action = d[1][14:18]
            termination = d[1][18]
            last_smdp_length = d[1][19]
            cur_smdp_length = d[1][20]
            reward = d[2]
            next_state = d[3]
            absorbing = d[4]
            last = d[5]
            self.sum_reward += reward * (self.mdp_info.gamma ** cur_smdp_length)

            if self.initial_smdp_state is None:
                self.initial_smdp_state = d[0]
                self.initial_smdp_action = high_action
            termination_dataset.append(([None], self.initial_smdp_action, [None], next_state, [None], [None]))
            if termination == 1 or absorbing or last:
                smdp_dataset.append((self.initial_smdp_state, self.initial_smdp_action, self.sum_reward, d[0],
                                     absorbing, last, last_smdp_length))
                self.sum_reward = 0
                self.initial_smdp_state = None
                self.initial_smdp_action = None

        return smdp_dataset, termination_dataset

    def add_t_dataset(self, dataset):
        t_dataset = list()
        for i, d in enumerate(dataset):
            state = d[0]
            option = d[1][14:18]
            reward = d[2]
            next_state = d[3]
            absorbing = d[4]
            last = d[5]
            cur_smdp_length = d[1][20]
            if absorbing == 1 or last == 1 or cur_smdp_length == 0:
                can_terminate = 0
            else:
                can_terminate = 1
            t_dataset.append((state, option, reward, next_state, absorbing, last, can_terminate))
        return t_dataset

    def termination_loss(self, state, option, can_terminate):
        # Loss = - 1/n * sum_n(beta(s_n', w_n) * A(s_n', w_n)), n = batch_size
        # A(s', w) = Q(s', w) - V(s') =  Q(s', w) - 1/r * [Q(s', w_0') + Q(s', w_1') + ... + Q(s', w_r')], r=sample_num
        batch_size = state.shape[0]

        # sample rule: Prob of w_old: 1-beta(s',w). Prob of w_new = beta(s',w) * policy_dist
        _beta = self.termination_approximator.predict(state, option, output_tensor=False)
        # sample w_n
        option_term_mask = np.random.rand(batch_size) < _beta.squeeze(-1)
        sampled_option = option
        sampled_option[option_term_mask, :] = self.policy.draw_action(state)[option_term_mask, :]
        sampled_option[~can_terminate, :] = option
        # sample for v(s')
        expand_state = np.repeat(state[:, np.newaxis, :], repeats=self.num_adv_sample, axis=1)
        expand_sampled_option = self.policy.draw_action(expand_state)

        adv = self.adv_func(expand_state, expand_sampled_option, state, sampled_option)
        adv_tensor = torch.tensor(adv, requires_grad=False, device=self.device)

        beta = self.termination_approximator.predict(state, sampled_option, output_tensor=True)

        return - (beta * adv_tensor).mean()

    def adv_func(self, expand_state, expand_sampled_option, state, sampled_option):
        batch_size = expand_state.shape[0]

        _v = self._target_critic_approximator.predict(expand_state.reshape(self.num_adv_sample * batch_size, -1),
                                                      expand_sampled_option.reshape(self.num_adv_sample * batch_size, -1),
                                                      prediction='min')
        v = np.mean(_v.reshape(batch_size, self.num_adv_sample), axis=1)

        q = self._target_critic_approximator.predict(state, sampled_option, prediction='min')
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
        super()._post_load()
        self.traj_planner = TrajectoryPlanner(**nn_planner_params)

    @property
    def _alpha(self):
        return self._log_alpha.exp()

    @property
    def _alpha_np(self):
        return self._alpha.detach().cpu().numpy()
