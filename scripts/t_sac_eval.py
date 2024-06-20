import numpy as np
import os
from mushroom_rl.utils.spaces import Box
from mushroom_rl.core import Core

from baseline.baseline_agent.baseline_agent import BaselineAgent
from hrl_air_hockey.envs.hit_back_env import HitBackEnv
from hrl_air_hockey.agents.double_agent_wrapper import HRLTournamentAgentWrapper

from hrl_air_hockey.utils.agent_builder import build_agent_T_SAC
from nn_planner_config import Config


def main(
        batch_size: int = 64,
        initial_replay_size: int = 1000,
        warmup_transitions: int = 1000,
        max_replay_size: int = 200000,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        n_features: str = "128 128",
        tau: float = 0.003,
        lr_alpha: float = 1e-5,
        target_entropy: float = -2,
        use_cuda: bool = False,

):
    env = HitBackEnv()
    env.info.action_space = Box(np.array([0.8, -0.39105, -np.pi/2, 0.]), np.array([1.3, 0.39105, np.pi/2, 1]))

    current_dir = os.path.dirname(os.path.abspath(__file__))
    planner_path = os.path.abspath(os.path.join(current_dir, os.pardir, 'trained_low_agent/Model_4250.pt'))
    planner_path = "Transferred_model_4250.pt"

    planner_config = Config
    agent_1 = build_agent_T_SAC(mdp_info=env.info, env_info=env.env_info,
                                planner_path=planner_path, planner_config=planner_config,
                                actor_lr=actor_lr, critic_lr=critic_lr, n_features_actor=n_features,
                                n_features_critic=n_features, batch_size=batch_size,
                                initial_replay_size=initial_replay_size, max_replay_size=max_replay_size, tau=tau,
                                warmup_transitions=warmup_transitions, lr_alpha=lr_alpha, target_entropy=target_entropy,
                                dropout_ratio=0, layer_norm=False, use_cuda=use_cuda)
    baseline_agent = BaselineAgent(env.env_info, agent_id=2)
    agent = HRLTournamentAgentWrapper(env.env_info, agent_1, baseline_agent)

    core = Core(agent, env)

    core.evaluate(n_episodes=10, render=True)


if __name__ == "__main__":
    main()
