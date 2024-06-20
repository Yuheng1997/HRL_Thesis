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
        self.agent_1.fit(dataset_1, info_1)

        # if self.agent_2 != "baseline":
        #     dataset_2, info_2 = self._preprocess_dataset_agent_1(dataset, **info)
        #     self.agent_2.fit(dataset_2, info_2)

    def _preprocess_dataset_agent_1(self, dataset, **info):
        # smdp_dataset
        smdp_dataset = []
        mdp_dataset = []
        for i in range(len(dataset)):
            self.sum_r += dataset[i][2] * self.power_gamma
            if self.save_initial_state:
                self.initial_states = np.array(dataset[i][0])
                self.smdp_length = 1
            if dataset[i][1][-1] or dataset[i][-2]:
                # smdp action terminate or game absorbing
                a_list = list(dataset[i])
                a_list[0] = self.initial_states
                a_list[1] = a_list[1][1]
                a_list[2] = self.sum_r
                a_list.append(self.smdp_length)
                smdp_dataset.append(a_list)
                # reset
                self.sum_r = 0.
                self.power_gamma = 1.
                self.save_initial_state = True
            else:
                self.smdp_length += 1
                self.save_initial_state = False
                self.power_gamma *= self.mdp_info.gamma
        for i in range(len(dataset)):
            a_list = list(dataset[i])
            a_list[1] = a_list[1][1]
            mdp_dataset.append(a_list)
        return [smdp_dataset, mdp_dataset], info


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