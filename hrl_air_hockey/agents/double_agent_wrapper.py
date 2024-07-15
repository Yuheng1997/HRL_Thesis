import socket
import time
import os

import numpy as np

from mushroom_rl.core import Agent
from air_hockey_challenge.utils.tournament_agent_wrapper import SimpleTournamentAgentWrapper


class HRLTournamentAgentWrapper(SimpleTournamentAgentWrapper):
    def __init__(self, env_info, agent_1, agent_2):
        super().__init__(env_info, agent_1, agent_2)

    def epsisode_start_1(self):
        self.agent_1.epsiode_start()

    def epsisode_start_2(self):
        self.agent_2.epsiode_start()

    def fit(self, dataset, **info):
        dataset_1, info_1 = self._preprocess_dataset_agent_1(dataset, **info)
        self.agent_1.fit(dataset_1, **info_1)

        # if self.agent_2 != "baseline":
        #     dataset_2, info_2 = self._preprocess_dataset_agent_1(dataset, **info)
        #     self.agent_2.fit(dataset_2, info_2)

    def _preprocess_dataset_agent_1(self, dataset, **info):
        dataset_agent1 = list()
        for i, d in enumerate(dataset):
            state = d[0][:23]
            action = d[1][0]
            next_state = d[3][:23]
            dataset_agent1.append((state, action, d[2], next_state, d[4], d[5]))
        return dataset_agent1, info


if __name__ == "__main__":
    from t_sac import SACPlusTermination
    from high_level.train_curriculum_exp import SAC
    from hrl_air_hockey.envs.hit_back_env import HitBackEnv
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