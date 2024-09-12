import torch
import numpy as np
from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from hrl_air_hockey.utils.t_replay_memory import TReplayMemory
from hrl_air_hockey.bspline_planner.planner import TrajectoryPlanner
from hrl_air_hockey.agents.atacom_agent import AirHockeyController
from hrl_air_hockey.bspline_planner.utils.kinematics import forward_kinematics

class SACPlusTermination(SAC):
    """
    Soft Actor-Critic algorithm.
    "Soft Actor-Critic Algorithms and Applications".
    Haarnoja T. et al.. 2019.

    """

    def __init__(self, mdp_info, actor_mu_params, actor_sigma_params, actor_optimizer, critic_params,
                 atacom_planner_params, termination_params, termination_optimizer, batch_size, termination_warmup,
                 initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha, num_adv_sample, device, adv_bonus,
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

        self.traj_planner = AirHockeyController(**atacom_planner_params)
        self.atacom_planner_params = atacom_planner_params

        self.termination_approximator = Regressor(TorchApproximator, **termination_params)
        termination_parameters = self.termination_approximator.model.network.parameters()
        self.termination_optimizer = termination_optimizer['class'](termination_parameters,
                                                                    **termination_optimizer['params'])
        self.termination_warmup = termination_warmup

        self.num_adv_sample = num_adv_sample
        self.device = device
        self.adv_bonus = adv_bonus
        self.adv_list = []

        self._add_save_attr(
            adv_bonus='primitive',
            termination_optimizer='torch',
            atacom_planner_params='pickle',
            termination_approximator="mushroom",
            num_adv_sample='primitive',
            device='primitive',
            termination_warmup='primitive',
        )

    def epoch_start(self):
        self.adv_list = []

    def episode_start(self):
        self.last_option = None
        self.last_log_p = None
        self.policy.reset()

    def draw_action(self, state):
        if self.last_option is None:
            self.last_option, self.last_log_p = self.policy.compute_action_and_log_prob(state.reshape(1, -1))
            self.last_option = self.last_option.squeeze()
        term_prob = self.termination_approximator.predict(state, self.last_option, output_tensor=False)

        if np.random.uniform() < term_prob:
            self.last_option, self.last_log_p = self.policy.compute_action_and_log_prob(state.reshape(1, -1))
            self.last_option = self.last_option.squeeze()
            target_2d = self._get_target_2d(self.last_option, state)
            low_action = self.traj_planner.compute_control(target_2d, state).flatten()
            termination = np.array([1])
        else:
            target_2d = self._get_target_2d(self.last_option, state)
            low_action = self.traj_planner.compute_control(target_2d, state).flatten()
            termination = np.array([0])

        expand_state = np.repeat(state[np.newaxis, :], 50, axis=0)
        expand_new_option = self.policy.draw_action(expand_state)
        q = self._target_critic_approximator.predict(state[np.newaxis, :], self.last_option[np.newaxis, :], prediction='min')
        v = self._target_critic_approximator.predict(expand_state, expand_new_option, prediction='min').mean()
        adv_value = q - v + self.adv_bonus
        # 14 + 2 + 1 + 1 = 18
        return np.concatenate([low_action, self.last_option, termination, np.array([adv_value])])

    def fit(self, dataset, **info):
        t_dataset = self.add_t_dataset(dataset)
        self._replay_memory.add(t_dataset)

        if self._replay_memory.initialized:
            state, option, reward, next_state, absorbing, _ = self._replay_memory.get(
                self._batch_size())

            if self._replay_memory.size > self._warmup_transitions():
                option_new, log_prob = self.policy.compute_action_and_log_prob_t(state)
                # update actor
                actor_loss = self._loss(state, option_new, log_prob)
                self._optimize_actor_parameters(actor_loss)
                self._update_alpha(log_prob.detach())
                # update beta(termination)
                if self._replay_memory.size > self.termination_warmup:
                    beta_loss = self.termination_loss(next_state, option)
                    self.optimize_termination_parameters(beta_loss)

            beta_prime = self.termination_approximator.predict(next_state, option, output_tensor=True).squeeze(-1)
            option_prime, log_p_prime = self.policy.compute_action_and_log_prob_t(next_state)

            gt = reward + self.mdp_info.gamma * ((1 - beta_prime.detach()) * self.q_next(next_state, option, absorbing)
                 + beta_prime.detach() * self.q_next(next_state, option_prime, absorbing) - self._alpha.detach() * log_p_prime.detach())

            self._critic_approximator.fit(state, option, gt, **self._critic_fit_params)

            self._update_target(self._critic_approximator, self._target_critic_approximator)

    def q_next(self, next_state, option, absorbing):
        q = self._target_critic_approximator.predict(next_state, option, prediction='min')
        q *= 1 - absorbing.cpu().numpy()
        return torch.tensor(q, device=self.device)

    def _loss(self, state, action_new, log_prob):
        q_0 = self._critic_approximator(state, action_new, output_tensor=True, idx=0)
        q_1 = self._critic_approximator(state, action_new, output_tensor=True, idx=1)
        q = torch.min(q_0, q_1)

        return (self._alpha * log_prob - q).mean()

    def _get_target_2d(self, target_pos, state):
        ee_pos = forward_kinematics(self.traj_planner.robot_model, self.traj_planner.robot_data, state[6:13], link="ee")[0][:2]
        action = (target_pos[:2] - ee_pos[:2]) * 30
        action = np.clip(action, -1.5, 1.5)
        return action

    def add_t_dataset(self, dataset):
        t_dataset = list()
        for i, d in enumerate(dataset):
            state = torch.tensor(d[0][:20], device=self.device)
            option = torch.tensor(d[1][14:16], device=self.device)
            reward = torch.tensor(d[2], device=self.device)
            next_state = torch.tensor(d[3][:20], device=self.device)
            absorbing = torch.tensor(d[4], device=self.device)
            last = torch.tensor(d[5], device=self.device)
            t_dataset.append((state, option, reward, next_state, absorbing, last))
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

        adv = self.adv_func(expand_next_state, expand_sampled_option, next_state, sampled_option) + self.adv_bonus
        adv = np.clip(adv, -5 * self.adv_bonus, 5 * self.adv_bonus)
        adv_tensor = torch.tensor(adv, requires_grad=False, device=self.device)

        self.adv_list.append(np.mean(adv))

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
        self.adv_list = []
        self.adv_bonus = 0.01
        self.state_shape = self.mdp_info.observation_space.shape
        self.action_shape = self.mdp_info.action_space.shape
        self.traj_planner = AirHockeyController(**self.atacom_planner_params)