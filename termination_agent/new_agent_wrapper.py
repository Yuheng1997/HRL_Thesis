import socket
import time
import os

import numpy as np

from mushroom_rl.core import Agent


class TournamentAgentWrapper(Agent):
    def draw_action(self, state):
        obs_1, obs_2 = np.split(state, 2)
        time_1, action_1 = self.get_action_1(obs_1)
        time_2, action_2 = self.get_action_2(obs_2)

        return action_1, action_2, time_1, time_2

    def episode_start(self):
        self.episode_start_1()
        self.episode_start_2()

    @property
    def preprocessors(self):
        def _preprocessor(obs):
            obs_1, obs_2 = np.split(obs, 2)
            normalized_obs_1 = self.preprocessor_1(obs_1)
            normalized_obs_2 = self.preprocessor_2(obs_2)
            return np.concatenate([normalized_obs_1, normalized_obs_2])

        return [_preprocessor]

    def get_action_1(self, obs_1):
        raise NotImplementedError

    def get_action_2(self, obs_2):
        raise NotImplementedError

    def episode_start_1(self):
        raise NotImplementedError

    def episode_start_2(self):
        raise NotImplementedError

    def preprocessor_1(self, obs_1):
        return obs_1

    def preprocessor_2(self, obs_2):
        return obs_2


class HRLTournamentAgentWrapper(TournamentAgentWrapper):
    def __init__(self, mdp_info, high_agent, low_agent, policy=None):
        super().__init__(mdp_info, policy)
        self.high_agent = high_agent
        self.low_agent = low_agent

        self.episode_start_1 = self.high_agent.episode_start
        self.episode_start_2 = self.low_agent.episode_start

    def draw_action(self, state):
        action_high = self.get_action_high(state)
        state_high_action = [state, action_high]
        action_low, _ = self.get_action_low(state_high_action)
        return action_low

    def get_action_high(self, obs_high):
        return self.high_agent.draw_action(obs_high)

    def get_action_low(self, obs_low):
        return self.low_agent.draw_action(obs_low)

    def reset(self):
        self.high_agent.reset()
        self.low_agent.reset()

    def stop(self):
        self.low_agent.stop()
        self.high_agent.stop()

    def episode_start(self):
        self.episode_start_1()
        self.episode_start_2()

    def preprocessor_1(self, obs_1):
        for p in self.high_agent.preprocessors:
            obs_1 = p(obs_1)
        return obs_1

    def preprocessor_2(self, obs_2):
        for p in self.low_agent.preprocessors:
            obs_2 = p(obs_2)
        return obs_2


if __name__ == "__main__":
    from t_sac import SACPlusTermination
    from high_level.train_curriculum_exp import SAC
    from high_level.hit_back_env import HitBackEnv
    from low_agent import build_low_agent
    import torch
    import os

    torch.manual_seed(1)
    np.random.seed(1)
    env = HitBackEnv()
    obs = env.reset()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    high_agent_dir = os.path.abspath(os.path.join(current_dir, os.pardir,
                                                  'trained_high_agent/hit_back_2024-06-11_15-59-15/parallel_seed___2/0/HitBackEnv_2024-06-11-16-00-59'))
    def get_file_by_postfix(parent_dir, postfix):
        file_list = list()
        for root, dirs, files in os.walk(parent_dir):
            for f in files:
                if f.endswith(postfix):
                    a = os.path.join(root, f)
                    file_list.append(a)
        return file_list
    rl_agent = SAC.load(get_file_by_postfix(high_agent_dir, 'agent-2.msh')[0])
    low_agent = build_low_agent(env.info, agent_1='Model_4250.pt', agent_2=None)
    agent = HRLTournamentAgentWrapper(env.info, high_agent=rl_agent, low_agent=low_agent)

    steps = 0
    while True:
        action = agent.draw_action(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        steps += 1

        if done or steps > env.info.horizon:
            steps = 0
            agent.reset()
            obs = env.reset()