import numpy as np
import os
from mushroom_rl.utils.spaces import Box
from mushroom_rl.core import Core

from baseline.baseline_agent.baseline_agent import BaselineAgent
from hrl_air_hockey.envs.hit_back_env import HitBackEnv
from hrl_air_hockey.agents.double_agent_wrapper import HRLTournamentAgentWrapper
from hrl_air_hockey.agents.t_sac import SACPlusTermination

from hrl_air_hockey.utils.agent_builder import build_agent_T_SAC
from nn_planner_config import Config


def main(
        check_point: str = 'HitBackEnv_2024-07-14-22-38-26',
):
    env = HitBackEnv()

    def get_file_by_postfix(parent_dir, postfix):
        file_list = list()
        for root, dirs, files in os.walk(parent_dir):
            for f in files:
                if f.endswith(postfix):
                    a = os.path.join(root, f)
                    file_list.append(a)
        return file_list

    cur_path = os.path.abspath('.')
    parent_dir = os.path.dirname(cur_path)
    check_path = os.path.join(parent_dir, 'trained_high_agent', check_point)
    agent_1 = SACPlusTermination.load(get_file_by_postfix(check_path, 'agent-1.msh')[0])

    baseline_agent = BaselineAgent(env.env_info, agent_id=2)
    agent = HRLTournamentAgentWrapper(env.env_info, agent_1, baseline_agent)

    core = Core(agent, env)

    core.evaluate(n_episodes=10, render=True, record=False)


if __name__ == "__main__":
    main()
