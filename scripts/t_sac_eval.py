import numpy as np
import os
from mushroom_rl.utils.spaces import Box
from mushroom_rl.core import Core

from baseline.baseline_agent.baseline_agent import BaselineAgent
from hrl_air_hockey.envs.hit_back_env import HitBackEnv
from hrl_air_hockey.envs.base_env import BaseEnv
from hrl_air_hockey.agents.double_agent_wrapper import HRLTournamentAgentWrapper
from hrl_air_hockey.agents.t_sac import SACPlusTermination

from hrl_air_hockey.utils.agent_builder import build_agent_T_SAC
from nn_planner_config import Config


def main(
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        termination_lr: float = 3e-6,
        num_adv_sample: int = 50,
        adv_bonus: float = 0.01,
        n_features_actor: str = '256 256 256',
        n_features_critic: str = '256 256 256',
        n_features_termination: str = '256 256 256',
        batch_size: int = 64,
        initial_replay_size: int = 8,
        max_replay_size: int = 200000,
        tau: float = 1e-3,
        warmup_transitions: int = 8,
        termination_warmup: int = 8,
        lr_alpha: float = 1e-5,
        target_entropy: float = -2,
        use_cuda: bool = True,
        dropout_ratio: float = 0.01,
        layer_norm: bool = False,

        # check_point: str = 'hit_back_2024-07-14_23-36-56/parallel_seed___0/0/HitBackEnv_2024-07-14-23-37-22',
        # check_point: str = 'static_hit_2024-07-15_16-27-51/parallel_seed___2/0/BaseEnv_2024-07-15-17-34-35',
        check_point: str = 'static_hit_2024-08-21_02-08-16/parallel_seed___0/0/BaseEnv_2024-08-21-02-08-38',
        # check_point=None
):
    # env = HitBackEnv(visual_target=True, horizon=1000)
    env = BaseEnv(visual_target=True, horizon=200)
    env.info.action_space = Box(np.array([-np.pi, 0]), np.array([np.pi, 2]))

    if check_point is None:
        agent_1 = build_agent_T_SAC(mdp_info=env.info, env_info=env.env_info, adv_bonus=adv_bonus,
                                    actor_lr=actor_lr, critic_lr=critic_lr, termination_lr=termination_lr,
                                    n_features_actor=n_features_actor, n_features_critic=n_features_critic,
                                    n_features_termination=n_features_termination, batch_size=batch_size,
                                    initial_replay_size=initial_replay_size, max_replay_size=max_replay_size, tau=tau,
                                    num_adv_sample=num_adv_sample, warmup_transitions=warmup_transitions,
                                    lr_alpha=lr_alpha, target_entropy=target_entropy, dropout_ratio=dropout_ratio,
                                    layer_norm=layer_norm, use_cuda=use_cuda, termination_warmup=termination_warmup)
    else:
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
        agent_1 = SACPlusTermination.load(get_file_by_postfix(check_path, 'agent-0.msh')[0])


    baseline_agent = BaselineAgent(env.env_info, agent_id=2)
    agent = HRLTournamentAgentWrapper(env.env_info, agent_1, baseline_agent)

    core = Core(agent, env)

    core.evaluate(n_episodes=8, render=True, record=True)


if __name__ == "__main__":
    main()
