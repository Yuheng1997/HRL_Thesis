import os
import numpy as np
from nn_agent import build_nn_agent
from air_hockey_challenge.framework.agent_base import AgentBase
from baseline.baseline_agent.baseline_agent import BaselineAgent
from air_hockey_challenge.utils.kinematics import jacobian, forward_kinematics


def get_file_by_postfix(parent_dir, postfix):
    file_list = list()
    for root, dirs, files in os.walk(parent_dir):
        for f in files:
            if f.endswith(postfix):
                file_list.append(os.path.join(root, f))
    return file_list


def get_file_by_prefix(parent_dir, prefix):
    file_list = list()
    for root, dirs, files in os.walk(parent_dir):
        for f in files:
            if f.startswith(prefix):
                file_list.append(os.path.join(root, f))
    return file_list


def load_agent(env_info, agent, id):
    if not agent:
        return BaselineAgent(env_info, agent_id=id), 'baseline'
    else:
        return build_nn_agent(env_info, agent), 'nn_planner'


class DoubleLowAgent(AgentBase):
    def __init__(self, env_info, agent_1, agent_2):
        self.traj_buffer = []
        self.training_agent, self.agent_type_1 = load_agent(env_info, agent_1, id=1)
        self.opponent_agent, self.agent_type_2 = load_agent(env_info, agent_2, id=2)
        # flag
        super().__init__(env_info)

    def draw_action(self, obs):
        if self.agent_type_1 == 'nn_planner':
            u_1, save_and_fit = self.training_agent.draw_action(obs[:23])
            u_2 = self.opponent_agent.draw_action(obs[23:46])
            u = np.array([u_1, u_2])
            return u, save_and_fit

    def reset(self, state=None):
        self.traj_buffer = []
        self.training_agent.reset()
        self.opponent_agent.reset()

    def update_ee_pos_vel(self, joint_pos, joint_vel, robot_model, robot_data):
        x_ee, _ = forward_kinematics(robot_model, robot_data, joint_pos)
        v_ee = jacobian(robot_model, robot_data, joint_pos)[:3, :self.base_env.env_info['robot']['n_joints']] @ joint_vel
        return x_ee, v_ee


if __name__ == "__main__":
    from base_env import BaseEnv
    np.random.seed(0)
    env = BaseEnv()
    # env.puck_pos = np.array([ 0.93455901, -0.11610415]) - [1.51, 0]
    obs = env.reset()
    # agent_1 = "trained_agent/atacom_exp_2024-01-18_17-55-40/check_point___.-logs-atacom_exp_2024-01-17_17-53-32-task_curriculum___True/task_curriculum___True/1"
    agent_1 = 'Model_2020.pt'
    agent_2 = None
    agent = DoubleLowAgent(env.env_info, agent_1, agent_2)
    # logger = Logger(log_name=f'{agent_1}_seed_0', results_dir='./test_logs', seed=0, use_timestamp=True, log_console=True)
    number = 0

    steps = 0
    while True:
        # np.array([0.6, -0.39105, 0, 0]), np.array([1.4, 0.39105, np.pi, 1])
        agent.training_agent.update_goal()
        action, _ = agent.draw_action(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        steps += 1
        env.update_number(number)

        if done or steps > 200:
            number += 1
            steps = 0
            obs = env.reset()
            agent.reset()
            # logger.epoch_info(number, puck_pos=env.initial_puck_pos)
