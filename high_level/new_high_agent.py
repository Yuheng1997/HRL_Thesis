from base_env import BaseEnv
from mushroom_rl.core import Agent
import numpy as np


class HighAgent(Agent):
    def __init__(self, env, rl_high_agent):
        self.rl_agent = rl_high_agent
        super().__init__(env.info, None)

    def draw_action(self, obs):
        high_action = self.rl_agent.draw_action(obs)
        return high_action

    def fit(self, dataset, **info):
        self.rl_agent.fit(dataset)

    def reset(self):
        self.rl_agent.policy.reset()

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