import numpy as np
from new_high_agent import HighAgent
from new_double_low_agent import DoubleLowAgent
from air_hockey_challenge.framework.agent_base import AgentBase


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

        self.high_agent = HighAgent(env, rl_high_agent=rl_high_agent)
        self.low_agent = DoubleLowAgent(env.env_info, agent_1=agent_1, agent_2=agent_2)
        super().__init__(env.env_info, None)


    def draw_action(self, obs):
        ee_pos, _vel = self.env.get_ee()
        ee_vel = _vel[-2:]
        if self.low_agent.training_agent.if_update_goal:
            high_action = self.high_agent.draw_action(obs)
            self.initial_high_action = high_action
            self.low_agent.training_agent.update_goal(high_action)
        low_action, save_and_fit = self.low_agent.draw_action(obs)
        # termination
        if self.termination:
            if np.linalg.norm(ee_vel) > 0.9:
                terminate = False
            else:
                terminate = (np.random.rand() < 0.02)
            if terminate:
                # self.termination_count = 0
                print('terminate')
                self.low_agent.training_agent.if_update_goal = True
                self.low_agent.training_agent.traj_buffer = []
                save_and_fit = True
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

    def reset(self):
        self.high_agent.reset()
        self.low_agent.reset()

    def stop(self):
        self.high_agent.stop()

    def episode_start(self):
        self.reset()


if __name__ == "__main__":
    from curriculum_exp import SAC
    from experiments.agent.hit_back_env import HitBackEnv
    import os
    env = HitBackEnv()
    obs = env.reset()

    check_point = 'logs/high_level_2024-05-22_19-59-26/parallel_seed___0/0/BaseEnv_2024-05-22-19-59-51'
    def get_file_by_postfix(parent_dir, postfix):
        file_list = list()
        for root, dirs, files in os.walk(parent_dir):
            for f in files:
                if f.endswith(postfix):
                    a = os.path.join(root, f)
                    file_list.append(a)
        return file_list
    rl_agent = SAC.load(get_file_by_postfix(check_point, 'agent-0.msh')[0])
    agent = AgentWrapper(env, rl_high_agent=rl_agent, agent_1='Model_1010.pt', agent_2=None)

    steps = 0
    while True:
        action = agent.draw_action(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        steps += 1

        if done or steps > env.info.horizon:
            steps = 0
            env.reset()