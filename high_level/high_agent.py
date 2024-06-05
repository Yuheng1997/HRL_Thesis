from base_env import BaseEnv
from double_low_agent import DoubleLowAgent
from mushroom_rl.core import Agent
import numpy as np
from mushroom_rl.utils.frames import LazyFrames


def build_warped_agent(env, rl_agent, agent_1, agent_2=None):
    return HighAgent(env, rl_high_agent=rl_agent, agent_1=agent_1, agent_2=agent_2)


class HighAgent(Agent):
    def __init__(self, env, rl_high_agent, agent_1, agent_2):
        self.sum_r = 0.
        self.smdp_length = 1
        self.power_gamma = 1.
        self.env = env
        self.termination_sleep = 20
        self.termination_count = 0
        self.rl_agent = rl_high_agent
        self.termination = True
        self.initial_states = None
        self.save_initial_state = True
        self.initial_high_action = None
        self.double_low_agent = DoubleLowAgent(env.env_info, agent_1=agent_1, agent_2=agent_2)
        super().__init__(env.info, None)

    def draw_action(self, obs):
        ee_pos, _vel = self.env.get_ee()
        ee_vel = _vel[-2:]
        # print('vel', ee_vel, np.linalg.norm(ee_vel))
        high_action = self.rl_agent.draw_action(obs)
        if self.double_low_agent.training_agent.if_update_goal:
            self.initial_high_action = high_action
            self.double_low_agent.training_agent.update_goal(high_action)
        low_action, save_and_fit = self.double_low_agent.draw_action(obs)
        # termination
        if self.termination:
            if np.linalg.norm(ee_vel) > 0.9:
                terminate = False
            else:
                terminate = (np.random.rand() < 0.02)
            if terminate:
                # self.termination_count = 0
                print('terminate')
                self.double_low_agent.training_agent.if_update_goal = True
                self.double_low_agent.training_agent.traj_buffer = []
                save_and_fit = True
        return [low_action, self.initial_high_action, save_and_fit]

    def fit(self, _dataset, **info):
        for i in range(len(_dataset)):
            self.sum_r += _dataset[i][2] * self.power_gamma
            if self.save_initial_state:
                self.initial_states = np.array(_dataset[i][0])
                self.smdp_length = 1
            if _dataset[i][1][-1] or _dataset[i][-2]:
                # smdf action finished or game absorbing
                a_list = list(_dataset[i])
                a_list[0] = self.initial_states
                a_list[1] = a_list[1][1]
                a_list[2] = self.sum_r
                a_list.append(self.smdp_length)
                self.rl_agent.fit([a_list])
                # reset
                self.sum_r = 0.
                self.power_gamma = 1.
                self.save_initial_state = True
            else:
                self.smdp_length += 1
                self.save_initial_state = False
                self.power_gamma *= self.mdp_info.gamma

    def get_init_states(self, dataset):
        pick = True
        x_0 = list()
        for d in dataset:
            if pick:
                if isinstance(d[0], LazyFrames):
                    x_0.append(np.array(d[0]))
                else:
                    x_0.append(d[0])
            pick = d[-1]
        return np.array(x_0)

    def reset(self):
        self.rl_agent.policy.reset()
        self.double_low_agent.reset()

    def stop(self):
        self.rl_agent.stop()

    def episode_start(self):
        self.reset()


if __name__ == "__main__":
    env = BaseEnv()
    obs = env.reset()
    agent = HighAgent(env.env_info, rl_high_agent=None)

    steps = 0
    while True:
        action = agent.draw_action(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        steps += 1

        if done or steps > env.info.horizon:
            steps = 0
            env.reset()