import socket
import time
import os

import numpy as np

from mushroom_rl.core import Agent
from baseline.baseline_agent.baseline_agent import BaselineAgent
from air_hockey_challenge.utils.tournament_agent_wrapper import SimpleTournamentAgentWrapper


class HRLTournamentAgentWrapper(SimpleTournamentAgentWrapper):
    def __init__(self, env_info, agent_1, agent_list):
        self.agent_list = agent_list
        self.agent_2 = self.agent_list[0]
        super().__init__(env_info, agent_1, self.agent_2)

    def episode_start(self):
        self.agent_1.episode_start()
        self.agent_2 = self.agent_list[np.random.randint(low=0, high=len(self.agent_list))]
        if isinstance(self.agent_2, BaselineAgent):
            self.agent_2.reset()
        else:
            self.agent_2.episode_start()

    def fit(self, dataset, **info):
        dataset_1, info_1 = self._preprocess_dataset_agent_1(dataset, **info)
        self.agent_1.fit(dataset_1, **info_1)

        # dataset_2, info_2 = self._preprocess_dataset_agent_1(dataset, **info)
        # self.agent_2.fit(dataset_2, info_2)

    def _preprocess_dataset_agent_1(self, dataset, **info):
        dataset_agent1 = list()
        for i, d in enumerate(dataset):
            state = d[0][:23]
            action = d[1][0]
            next_state = d[3][:23]
            dataset_agent1.append((state, action, d[2], next_state, d[4], d[5]))
        return dataset_agent1, info

    def _preprocess_dataset_agent_2(self, dataset, **info):
        dataset_agent2 = list()
        for i, d in enumerate(dataset):
            state = d[0][23:]
            action = d[1][1]
            next_state = d[3][23:]
            dataset_agent2.append((state, action, d[2], next_state, d[4], d[5]))
        return dataset_agent2, info