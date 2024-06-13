import numpy as np
from new_high_agent import HighAgent
import torch
import csv
import os
from air_hockey_challenge.utils.kinematics import forward_kinematics, jacobian
from new_double_low_agent import DoubleLowAgent
from air_hockey_challenge.framework.agent_base import AgentBase
from return_generator import ReturnGenerator


def build_warped_agent(env, rl_agent, agent_1, agent_2=None):
    return AgentWrapper(env, rl_high_agent=rl_agent, agent_1=agent_1, agent_2=agent_2)


class AgentWrapper(AgentBase):
    def __init__(self, env, rl_high_agent, agent_1, agent_2):
        self.sum_r = 0.
        self.smdp_length = 1
        self.power_gamma = 1.
        self.env = env
        self.termination_sleep = 20
        self.termination_count = 0
        self.termination = True
        self.initial_states = None
        self.save_initial_state = True
        self.initial_high_action = None
        self.return_traj = []
        self.in_return = False
        self.generator = ReturnGenerator(env_info=env.env_info)
        self.violate_data_path = os.path.join(os.path.abspath(os.getcwd()), "violate_data/violate_data.csv")

        self.huber = torch.nn.HuberLoss(reduction='none')
        self.high_agent = HighAgent(env, rl_high_agent=rl_high_agent)
        self.low_agent = DoubleLowAgent(env.env_info, agent_1=agent_1, agent_2=agent_2)
        super().__init__(env.env_info, None)


    def draw_action(self, obs):
        ee_pos, _vel = self.env.get_ee()
        ee_vel = _vel[-3:]
        if self.low_agent.training_agent.if_update_goal:
            high_action = self.high_agent.draw_action(obs)
            self.initial_high_action = high_action
            self.low_agent.training_agent.update_goal(high_action)
            traj = self.low_agent.training_agent.generate_whole_traj(obs)
            good_traj = self.check_traj_violation(traj, obs)
            # 检查violation, 如果违背则用baseline生成traj强制return。
            for i in range(10):
                high_action = np.random.uniform(low=[0.8, -0.39105, 0, 0.], high=[1.3, 0.39105, np.pi, 1], size=4)
                self.low_agent.training_agent.update_goal(high_action)
                traj = self.low_agent.training_agent.generate_whole_traj(obs)
                good_traj = self.check_traj_violation(traj, obs)
                if good_traj:
                    self.initial_high_action = high_action
                    break
            if not good_traj:
                # print('return_baseline')
                return_traj = self.generate_return_traj(ee_pos, ee_vel, obs)
                # self.low_agent.training_agent.traj_buffer = list(return_traj.reshape((-1, 14)))
                self.low_agent.training_agent.traj_buffer = return_traj
                self.in_return = True
        low_action, save_and_fit = self.low_agent.draw_action(obs)
        # termination
        if self.termination:
            if np.linalg.norm(ee_vel[:2]) > 0.9:
                terminate = False
            else:
                terminate = (np.random.rand() < 0.02)
            if terminate:
                # self.termination_count = 0
                # print('terminate')
                self.low_agent.training_agent.if_update_goal = True
                self.low_agent.training_agent.traj_buffer = []
                save_and_fit = True
        if self.in_return:
            save_and_fit = False
            self.in_return = False
        return [low_action, self.initial_high_action, save_and_fit]

    def fit(self, _dataset, **info):
        for i in range(len(_dataset)):
            self.sum_r += _dataset[i][2] * self.power_gamma
            if self.save_initial_state:
                self.initial_states = np.array(_dataset[i][0])
                self.smdp_length = 1

            # smdf action finished or game absorbing
            if _dataset[i][1][-1] or _dataset[i][-2]:
                a_list = list(_dataset[i])
                a_list[0] = self.initial_states
                a_list[1] = a_list[1][1]
                a_list[2] = self.sum_r
                a_list.append(self.smdp_length)
                self.high_agent.fit([a_list])
                # reset
                self.sum_r = 0.
                self.power_gamma = 1.
                self.save_initial_state = True
            else:
                self.smdp_length += 1
                self.save_initial_state = False
                self.power_gamma *= self.mdp_info.gamma

    def check_traj_violation(self, traj, obs):
        puck_pos = obs[:2]
        # check constraint, obstacle
        positions = []
        for i in range(len(traj)):
            position = self.low_agent.training_agent.forward_kinematics(traj[i][:7])[0][:3]
            positions.append(position)
        constraint_loss, x_loss, y_loss, z_loss = self.constraint_loss(positions, 0.02)
        # print('constraint_loss', constraint_loss)
        obstacle_loss = self.obstacle_loss(positions, 0.02, puck_pos)
        # save the violate datapoint
        if constraint_loss > 0.0001 or obstacle_loss > 0.1:
            with open(self.violate_data_path, 'a', newline='') as file:
                writer = csv.writer(file)
                data = np.array([*self.initial_high_action, *obs[6:20], *position])
                writer.writerow(data.tolist())
            return False
        else:
            return True

    def generate_return_traj(self, ee_pos, ee_vel, obs):
        ee_pos = ee_pos + np.array([1.51, 0., 0.])
        q, qd = self.get_joint_state(obs)
        x_home = np.array([0.65, 0., self.env_info['robot']['ee_desired_height']])
        qd_max = 0.8 * np.array([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562, 2.3562])
        t_stop = 1.0
        # for _ in range(10):
        #     self.generator.bezier_planner.compute_control_point(ee_pos[:2], ee_vel[:2], x_home[:2], np.zeros(2), t_stop)
        #     cart_traj = self.generator.generate_bezier_trajectory()
        #     success, traj = self.generator.optimize_trajectory(cart_traj, q, qd)
        #     if not success:
        #         t_stop *= 1.5
        #     else:
        #         return traj
        J = jacobian(mj_model=self.robot_model, mj_data=self.robot_data, q=q)[:3]
        # if np.linalg.norm(ee_vel[:2]) < 0.3:
        #     v_des = x_home - ee_pos
        #     print('home')
        # else:
        #     v_des = ee_vel * 0.02
        #     print('stop')
        v_des = x_home - ee_pos
        J_p = np.linalg.pinv(J)
        JJ = J_p.dot(J)
        I_n = np.eye(JJ.shape[0])
        qd_0 = (np.array([0., -0.1961, 0., -1.8436, 0., 0.9704, 0.]) - q) * 0.5
        qd_des = J_p.dot(v_des) + (I_n - JJ).dot(qd_0)
        gain = np.min(qd_max / np.abs(qd_des))
        qd_des = qd_des * gain
        q_des = q + qd_des * 0.02
        return [[*q_des, *qd_des]]


    def reset(self):
        self.return_traj = []
        self.in_force_return = False
        self.high_agent.reset()
        self.low_agent.reset()

    def stop(self):
        self.high_agent.stop()

    def episode_start(self):
        self.reset()

    def constraint_loss(self, position, dt):
        ee_pos = torch.tensor(position)
        huber_along_path = lambda x: dt * self.huber(x, torch.zeros_like(x))
        relu_huber_along_path = lambda x: huber_along_path(torch.relu(x))
        x_b = torch.tensor([0.58415, 1.51])
        y_b = torch.tensor([-0.47085, 0.47085])
        z = torch.tensor(0.1645)
        x_loss = relu_huber_along_path(x_b[0] - ee_pos[:, 0]) + relu_huber_along_path(ee_pos[:, 0] - x_b[1])
        y_loss = relu_huber_along_path(y_b[0] - ee_pos[:, 1]) + relu_huber_along_path(ee_pos[:, 1] - y_b[1])
        z_loss = relu_huber_along_path(z - ee_pos[:, 2]) + relu_huber_along_path(ee_pos[:, 2] - z)
        constraint_losses = torch.sum(x_loss + y_loss + z_loss)
        return constraint_losses, x_loss, y_loss, z_loss

    def obstacle_loss(self, position, dt, puck_pos):
        ee_pos_2d =  torch.tensor(np.array(position))[:, :2]
        dist_from_puck = torch.linalg.norm(torch.tensor(puck_pos) - ee_pos_2d[:, :2], axis=1)
        puck_loss = torch.sum(torch.relu(torch.tensor(0.0798) - dist_from_puck) * dt)
        return puck_loss


if __name__ == "__main__":
    from train_curriculum_exp import SAC
    from hit_back_env import HitBackEnv
    import os

    torch.manual_seed(3)
    np.random.seed(3)
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
    agent = AgentWrapper(env, rl_high_agent=rl_agent, agent_1='Model_4250.pt', agent_2=None)
    agent.termination = True

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
