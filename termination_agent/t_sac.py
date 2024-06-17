import numpy as np

import torch
import torch.optim as optim

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.policy import Policy
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.parameters import to_parameter
from utils.replay_memory import PlusReplayMemory

from copy import deepcopy
from itertools import chain


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


class SACPlusTermination(DeepAC):
    """
    Soft Actor-Critic algorithm.
    "Soft Actor-Critic Algorithms and Applications".
    Haarnoja T. et al.. 2019.

    """
    def __init__(self, mdp_info, actor_mu_params, actor_sigma_params, terminate_mu_params, terminate_sigma_params,
                 actor_optimizer, critic_params, batch_size, initial_replay_size, max_replay_size, warmup_transitions,
                 tau, lr_alpha, termination_optimizer, num_adv_sample, use_log_alpha_loss=False, log_std_min=-20,
                 log_std_max=2, target_entropy=None, critic_fit_params=None):
        """
        Constructor.

        Args:
            actor_mu_params (dict): parameters of the actor mean approximator to build;
            actor_sigma_params (dict): parameters of the actor sigma approximator to build;
            actor_optimizer (dict): parameters to specify the actor optimizer algorithm;
            critic_params (dict): parameters of the critic approximator to build;
            batch_size ((int, Parameter)): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before starting the learning;
            max_replay_size (int): the maximum number of samples in the replay memory;
            warmup_transitions ([int, Parameter]): number of samples to accumulate in the replay memory to start the
                policy fitting;
            tau ([float, Parameter]): value of coefficient for soft updates;
            lr_alpha ([float, Parameter]): Learning rate for the entropy coefficient;
            use_log_alpha_loss (bool, False): whether to use the original implementation loss or the one from the
                paper;
            log_std_min ([float, Parameter]): Min value for the policy log std;
            log_std_max ([float, Parameter]): Max value for the policy log std;
            target_entropy (float, None): target entropy for the policy, if None a default value is computed;
            critic_fit_params (dict, None): parameters of the fitting algorithm of the critic approximator.

        """
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        self._batch_size = to_parameter(batch_size)
        self._warmup_transitions = to_parameter(warmup_transitions)
        self._tau = to_parameter(tau)

        self.num_adv_sample = num_adv_sample

        self._use_log_alpha_loss = use_log_alpha_loss

        if target_entropy is None:
            self._target_entropy = -np.prod(mdp_info.action_space.shape).astype(np.float32)
        else:
            self._target_entropy = target_entropy

        self._replay_memory = PlusReplayMemory(initial_replay_size, max_replay_size)

        if 'n_models' in critic_params.keys():
            assert critic_params['n_models'] == 2
        else:
            critic_params['n_models'] = 2

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(TorchApproximator, **critic_params)
        self._target_critic_approximator = Regressor(TorchApproximator, **target_critic_params)

        self._init_target(self._critic_approximator, self._target_critic_approximator)

        # high_level_policy
        actor_mu_approximator = Regressor(TorchApproximator, **actor_mu_params)
        actor_sigma_approximator = Regressor(TorchApproximator, **actor_sigma_params)

        high_policy = SACPolicy(actor_mu_approximator, actor_sigma_approximator,  mdp_info.action_space.low,
                           mdp_info.action_space.high, log_std_min, log_std_max)

        self._log_alpha = torch.tensor(0., dtype=torch.float32)

        if high_policy.use_cuda:
            self._log_alpha = self._log_alpha.cuda().requires_grad_()
        else:
            self._log_alpha.requires_grad_()
        self._alpha_optim = optim.Adam([self._log_alpha], lr=lr_alpha)

        high_policy_parameters = chain(actor_mu_approximator.model.network.parameters(),
                                       actor_sigma_approximator.model.network.parameters())

        # termination_policy
        termination_mu_approximator = Regressor(TorchApproximator, **terminate_mu_params)
        termination_sigma_approximator = Regressor(TorchApproximator, **terminate_sigma_params)

        termination_policy = SACPolicy(termination_mu_approximator, termination_sigma_approximator,  mdp_info.termination_space.low,
                                       mdp_info.termination_space.high, log_std_min, log_std_max)

        termination_policy_parameters = chain(termination_mu_approximator.model.network.parameters(),
                                        termination_sigma_approximator.model.network.parameters())

        self.termination_class = DeepAC(mdp_info, termination_policy, termination_optimizer, termination_policy_parameters)

        self._add_save_attr(
            _critic_fit_params='pickle',
            _batch_size='mushroom',
            _warmup_transitions='mushroom',
            _tau='mushroom',
            _target_entropy='primitive',
            _replay_memory='mushroom',
            _critic_approximator='mushroom',
            _target_critic_approximator='mushroom',
            _use_log_alpha_loss='primitive',
            _log_alpha='torch',
            _alpha_optim='torch'
        )

        super().__init__(mdp_info, high_policy, actor_optimizer, high_policy_parameters)

    def fit(self, dataset, **info):
        self._replay_memory.add(dataset)
        for i in range(20):
            if self._replay_memory.initialized:
                # action_t = [pos_x, pos_y, vel_angle, scale, termination]
                state, action_t, reward, next_state, absorbing, _, smdp_length = self._replay_memory.get(self._batch_size())
                action = action_t[:, :4]
                if self._replay_memory.size > self._warmup_transitions():
                        action_new, log_prob = self.policy.compute_action_and_log_prob_t(state)
                        action_new_prime, _ = self.policy.compute_action_and_log_prob_t(next_state)
                        # update actor
                        actor_loss = self._loss(state, action_new, log_prob)
                        self._optimize_actor_parameters(actor_loss)
                        self._update_alpha(log_prob.detach())
                        # update beta(termination)
                        beta_loss = self.beta_loss(action, next_state, action_new_prime)
                        self.termination_class._optimize_actor_parameters(beta_loss)

                q_next = self._next_q(next_state, absorbing)
                q = reward + (self.mdp_info.gamma ** smdp_length) * q_next

                self._critic_approximator.fit(state, action[:4], q, **self._critic_fit_params)

                self._update_target(self._critic_approximator, self._target_critic_approximator)

    def beta_loss(self, action, next_state, action_new_prime):
        # - 1/n * sum_n( beta(s_n', w_n') * A(s_n', w_n') )
        adv_func = self.adv_func(action, next_state, action_new_prime)
        state_prime_termination = np.concatenate((next_state, action_new_prime), axis=1)
        beta = self.termination_class.policy.compute_action_with_grad(state_prime_termination)
        return -(beta * adv_func.detach()).mean()

    def adv_func(self, action, next_state, action_new_prime):
        # A(s', w') = Q(s', w') - V(s') =  Q(s', w') - 1/n * [Q(s', w_0) + Q(s', w_1) + ... + Q(s', w_n)]
        q_0 = self._critic_approximator(next_state, action_new_prime, output_tensor=True, idx=0)
        q_1 = self._critic_approximator(next_state, action_new_prime, output_tensor=True, idx=0)
        q = torch.min(q_0, q_1)
        _q_sum = np.zeros(q)
        # sample w_n
        for i in range(self.num_adv_sample):
            # sample rule: Prob of w_old: 1-beta(s',w). Prob of w_new = beta(s',w) * policy_dist
            state_prime_termination = np.concatenate((next_state, action), axis=1)
            termination_prime, _ = self.termination_class.policy.compute_action_and_log_prob(state_prime_termination)
            if np.random.uniform() > termination_prime:
                sampled_action = action
            else:
                sampled_action = self.policy.compute_action_and_log_prob_t(next_state)
            _q_0 = self._critic_approximator(next_state, sampled_action, output_tensor=True, idx=0)
            _q_1 = self._critic_approximator(next_state, sampled_action, output_tensor=True, idx=0)
            _q = torch.min(_q_0, _q_1)
            _q_sum += _q
        v = _q_sum / self.num_adv_sample
        return q - v

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

    # overwrite the functions
    def _post_load(self):
        self._update_optimizer_parameters(self.policy.parameters())
        self.termination_class._update_optimizer_parameters(self.termination_class.policy.parameters())

    def draw_action(self, state):
        high_action = self.policy.draw_action(state)
        beta = self.termination_class.policy.draw_action(state)
        return np.concatenate((high_action, beta), axis=1)

    def episode_start(self):
        self.policy.reset()
        self.termination_class.policy.reset()

    @property
    def _alpha(self):
        return self._log_alpha.exp()

    @property
    def _alpha_np(self):
        return self._alpha.detach().cpu().numpy()
