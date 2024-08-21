import numpy as np
from copy import deepcopy
from .atacom_controller import ATACOMController
from mushroom_rl.core import Agent
from mushroom_rl.utils.preprocessors import StandardizationPreprocessor


class AgentWrapper(Agent):
    def __init__(self, atacom_controller: ATACOMController, learning_agent, filter_ratio=1, randomize_dynamics=False):
        self.atacom_controller = atacom_controller
        self.learning_agent = learning_agent
        self.filter_ratio = filter_ratio

        self.randomize_dynamics = randomize_dynamics
        super().__init__(self.learning_agent.mdp_info, None)

        self.prev_action = np.zeros_like(self.mdp_info.action_space.low)
        if self.filter_ratio < 1.:
            self._append_act_hist_prepro = AppendActionHistPrePro(self.mdp_info.action_space.shape[0])
            self.add_preprocessor(self._append_act_hist_prepro)

    def draw_action(self, state_orig):
        rl_action = self.learning_agent.draw_action(self.learning_agent_preprocess(state_orig.copy()))
        sampled_action = np.clip(rl_action, self.mdp_info.action_space.low, self.mdp_info.action_space.high)

        self.prev_action = (1 - self.filter_ratio) * self.prev_action + self.filter_ratio * sampled_action
        if self.filter_ratio < 1.:
            self._append_act_hist_prepro.update_action_hist(self.prev_action)

        q_x, x_dot = self._unwrap_state(state_orig)
        actual_action = self.atacom_controller.compose_action(q_x, self.prev_action, x_dot)
        actual_action = np.clip(actual_action, self.mdp_info.action_space.low, self.mdp_info.action_space.high)
        return [actual_action, rl_action]

    def episode_start(self):
        if self.randomize_dynamics:
            self.atacom_controller.system_dynamics.randomize()

        self.learning_agent.episode_start()
        self.prev_action = np.zeros_like(self.mdp_info.action_space.low)

        if self.filter_ratio < 1.:
            self._append_act_hist_prepro.update_action_hist(self.prev_action)

    def _unwrap_state(self, state):
        raise NotImplementedError

    def fit(self, dataset, **info):
        processed_dataset = self.process_dataset_before_fit(dataset, **info)
        self.learning_agent.fit(processed_dataset, **info)

    def learning_agent_preprocess(self, state):
        for p in self.learning_agent.preprocessors:
            if state.ndim == 2:
                state = np.array([p(s.copy()) for s in state])
            else:
                state = p(state)
        return state

    def process_dataset_before_fit(self, dataset, **dataset_info):
        dataset_new = list()
        for i in range(len(dataset)):
            temp = deepcopy(list(dataset[i]))
            temp[0] = self.learning_agent_preprocess(temp[0])
            temp[3] = self.learning_agent_preprocess(temp[3])
            dataset[i] = tuple(temp)
            dataset_new.append(tuple(temp))
        return dataset_new

    def stop(self):
        self.learning_agent.stop()


class AppendActionHistPrePro:
    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.prev_action = np.zeros(action_dim)

    def update_action_hist(self, prev_action):
        self.prev_action = prev_action.copy()

    def __call__(self, obs):
        return np.concatenate([obs, self.prev_action])
