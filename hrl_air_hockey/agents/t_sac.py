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

        self.state_shape = mdp_info.observation_space.shape
        self.action_shape = mdp_info.action_space.shape
        self._replay_memory = TReplayMemory(initial_replay_size, max_replay_size, state_shape=self.state_shape,
                                            action_shape=self.action_shape, device=device)
        self.nn_planner_params = nn_planner_params
        self.traj_planner = TrajectoryPlanner(**nn_planner_params)

        self.termination_approximator = Regressor(TorchApproximator, **termination_params)
        termination_parameters = self.termination_approximator.model.network.parameters()
        self.termination_optimizer = termination_optimizer['class'](termination_parameters,
                                                                    **termination_optimizer['params'])

        self.num_adv_sample = num_adv_sample
        self.device = device

        self._add_save_attr(
            termination_optimizer='torch',
            nn_planner_params='pickle',
            termination_approximator="mushroom",
            num_adv_sample='primitive',
            device='primitive'
        )

    def episode_start(self):
        self.cur_smdp_length = 0
        self.sum_reward = 0
        self.initial_state = None
        self.initial_action = None
        self.initial_smdp_state = None
        self.initial_smdp_action = None
        self.trajectory_buffer = None
        self.last_action = None
        self.last_log_p = None
        self.policy.reset()

    def draw_action(self, state):
        rest_traj_len = 0
        if self.trajectory_buffer is None or len(self.trajectory_buffer) == 0:
            self.last_action, self.last_log_p = self.policy.compute_action_and_log_prob(state.reshape(1, -1))
            self.last_action = self.last_action.squeeze()
            q, dq = self._get_joint_pos(state)
            hit_pos, hit_dir, hit_scale, vel_angle = self._get_target_point(self.last_action)
            self.trajectory_buffer = \
                self.traj_planner.plan_trajectory(q, dq, hit_pos, hit_dir, hit_scale)[0]
            termination = np.array([1])
            beta_termination = np.array([0])
        else:
            term_prob = self.termination_approximator.predict(state, self.last_action, output_tensor=False)
            # term_prob = 0.02
            termination = np.array([0])
            beta_termination = np.array([0])
            if np.random.uniform() < term_prob:
                self.last_action, self.last_log_p = self.policy.compute_action_and_log_prob(state.reshape(1, -1))
                self.last_action = self.last_action.squeeze()
                q, dq = self._get_joint_pos(state)
                hit_pos, hit_dir, hit_scale, vel_angle = self._get_target_point(self.last_action)

                rest_traj_len = len(self.trajectory_buffer)
                self.trajectory_buffer = \
                    self.traj_planner.plan_trajectory(q, dq, hit_pos, hit_dir, hit_scale)[0]
                termination = np.array([1])
                beta_termination = np.array([1])
        assert len(self.trajectory_buffer) > 0
        joint_command = self.trajectory_buffer[0, :14]
        self.trajectory_buffer = self.trajectory_buffer[1:]

        if termination == 1:
            last_smdp_length = self.cur_smdp_length
            self.cur_smdp_length = 0
        else:
            last_smdp_length = self.cur_smdp_length
            self.cur_smdp_length += 1
        # 14 + 4 + 1 + 1+ 1+ 1 + 1 + 1 = 24
        return np.concatenate([joint_command, self.last_action, termination, np.array([last_smdp_length]),
                               np.array([self.cur_smdp_length]), beta_termination, np.array([rest_traj_len]), self.last_log_p])

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
        self._replay_memory.add(t_dataset)
        for i in range(1):
            if self._replay_memory.initialized:
                state, option, reward, next_state, absorbing, _, log_p = self._replay_memory.get(
                    self._batch_size())

                if self._replay_memory.size > self._warmup_transitions():
                    action_new, log_prob = self.policy.compute_action_and_log_prob_t(state)
                    # update actor
                    actor_loss = self._loss(state, action_new, log_prob)
                    self._optimize_actor_parameters(actor_loss)
                    self._update_alpha(log_prob.detach())
                    # update beta(termination)
                    beta_loss = self.termination_loss(next_state, option)
                    self.optimize_termination_parameters(beta_loss)

                beta_prime = self.termination_approximator.predict(next_state, option, output_tensor=True).squeeze(-1)
                option_prime, log_p_prime = self.policy.compute_action_and_log_prob(next_state)

                gt = (reward + self.mdp_info.gamma * (1 - beta_prime.detach()) * self.q_next(next_state, option, absorbing, log_p=None)
                      + self.mdp_info.gamma * beta_prime.detach() * self.q_next(next_state, option_prime, absorbing, log_p=log_p_prime))

                self._critic_approximator.fit(state, option, gt, **self._critic_fit_params)

                self._update_target(self._critic_approximator, self._target_critic_approximator)

    def q_next(self, next_state, option, absorbing, log_p):
        if log_p is None:
            # log_p = self.inv_log_p(next_state, option).detach().cpu().numpy()
            log_p = 0
        q = self._target_critic_approximator.predict(next_state, option, prediction='min') - self._alpha_np * log_p
        q *= 1 - absorbing.cpu().numpy()
        return torch.tensor(q, device=self.device)

    def _loss(self, state, action_new, log_prob):
        q_0 = self._critic_approximator(state, action_new, output_tensor=True, idx=0)
        q_1 = self._critic_approximator(state, action_new, output_tensor=True, idx=1)
        q = torch.min(q_0, q_1)

        return (self._alpha.detach() * log_prob - q.detach()).mean()

    def inv_log_p(self, state, a_true):
        a = (a_true - self.policy._central_a.clone()) / self.policy._delta_a.clone()
        epsilon = 1e-5
        a = torch.clamp(a, min=-1 + epsilon, max=1 - epsilon)
        a_raw = torch.atanh(a)
        mu = self.policy._mu_approximator.predict(state, output_tensor=True).detach()
        log_sigma = self.policy._sigma_approximator.predict(state, output_tensor=True).detach()
        log_sigma = torch.clamp(log_sigma, self.policy._log_std_min(), self.policy._log_std_max())
        dist = torch.distributions.Normal(mu, log_sigma.exp())
        log_p = dist.log_prob(a_raw).sum(dim=1).detach()
        log_p -= torch.log(1. - a.pow(2) + self.policy._eps_log_prob).sum(dim=1)
        return log_p

    def _update_alpha(self, log_prob):
        # alpha_loss = - (self._log_alpha * (log_prob + self._target_entropy)).mean()
        alpha_loss = - (self._alpha * (log_prob + self._target_entropy)).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

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
            state = torch.tensor(d[0], device=self.device)
            option = torch.tensor(d[1][14:18], device=self.device)
            reward = torch.tensor(d[2], device=self.device)
            next_state = torch.tensor(d[3], device=self.device)
            absorbing = torch.tensor(d[4], device=self.device)
            last = torch.tensor(d[5], device=self.device)
            cur_smdp_length = torch.tensor(d[1][20])
            log_p = torch.tensor(d[1][23], device=self.device)
            t_dataset.append((state, option, reward, next_state, absorbing, last, log_p))

        return t_dataset

    def termination_loss(self, next_state, option):
        # Loss = - 1/n * sum_n(beta(s_n', w_n) * A(s_n', w_n)), n = batch_size
        # A(s', w) = Q(s', w) - V(s') =  Q(s', w) - 1/r * [Q(s', w_0') + Q(s', w_1') + ... + Q(s', w_r')], r=sample_num
        batch_size = next_state.shape[0]

        # sample rule: Prob of w_old: 1-beta(s',w). Prob of w_new = beta(s',w) * policy_dist
        _beta = self.termination_approximator.predict(next_state, option, output_tensor=False)
        # sample w_n
        option_term_mask = np.random.rand(batch_size) < _beta.squeeze(-1)
        sampled_option = option.clone()
        sampled_option[option_term_mask, :] = torch.tensor(self.policy.draw_action(next_state)[option_term_mask, :],
                                                           device=self.device)
        # sample for v(s')
        expand_next_state = next_state.clone().unsqueeze(1).repeat(1, self.num_adv_sample, 1)
        expand_sampled_option = self.policy.draw_action(expand_next_state)

        adv = self.adv_func(expand_next_state, expand_sampled_option, next_state, sampled_option)
        adv_tensor = torch.tensor(adv, requires_grad=False, device=self.device)

        beta = self.termination_approximator.predict(next_state, sampled_option, output_tensor=True)

        return (beta * adv_tensor).mean()

    def adv_func(self, expand_next_state, expand_sampled_option, next_state, sampled_option):
        batch_size = expand_next_state.shape[0]

        _v = self._target_critic_approximator.predict(expand_next_state.reshape(self.num_adv_sample * batch_size, -1),
                                                      expand_sampled_option.reshape(self.num_adv_sample * batch_size,
                                                                                    -1), prediction='min')
        v = np.mean(_v.reshape(batch_size, self.num_adv_sample), axis=1)

        q = self._target_critic_approximator.predict(next_state, sampled_option, prediction='min')
        return q - v

    def optimize_termination_parameters(self, loss):
        self.termination_optimizer.zero_grad()
        loss.backward()
        self.termination_optimizer.step()

    def _post_load(self):
        super()._post_load()
        self.state_shape = self.mdp_info.observation_space.shape
        self.action_shape = self.mdp_info.action_space.shape
        self.nn_planner_params['planner_path'] = '../trained_low_agent/Model_5600.pt'
        self.traj_planner = TrajectoryPlanner(**self.nn_planner_params)
