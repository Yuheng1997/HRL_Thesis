import numpy as np
import os
from mushroom_rl.utils.spaces import Box
from mushroom_rl.core import Core

from baseline.baseline_agent.baseline_agent import BaselineAgent
from hrl_air_hockey.envs.hit_back_env import HitBackEnv
from hrl_air_hockey.envs.base_env import BaseEnv
from hrl_air_hockey.agents.double_agent_wrapper import HRLTournamentAgentWrapper
from hrl_air_hockey.agents.t_sac import SACPlusTermination
from hrl_air_hockey.agents.serve_agent import ServeAgent

from hrl_air_hockey.utils.agent_builder import build_agent_T_SAC
from nn_planner_config import Config


def main(
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        termination_lr: float = 3e-6,
        num_adv_sample: int = 50,
        adv_bonus: float = 0.01,
        load_nn_agent: str = 'Model_5600.pt',
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
        use_nn: bool = True,
        # 'two_days_origin_2024-09-11_12-59-00/two_days_origin/parallel_seed___0/0/HitBackEnv_2024-09-11-13-00-58',
        # 'two_days_selflearn_2024-09-12_01-25-49/two_days_selflearn/parallel_seed___0/0/HitBackEnv_2024-09-12-01-26-53',
        # 'cl_line_2024-09-12_00-48-29/cl_line/parallel_seed___0/0/HitBackEnv_2024-09-12-00-49-21',
        # 'cl_sl_line_2024-09-15_12-26-41/cl_sl_line/parallel_seed___0/0/HitBackEnv_2024-09-15-14-54-49',
        # 'cl_r_2024-09-16_13-38-36/cl_r/parallel_seed___0/0/HitBackEnv_2024-09-16-14-25-15',
        # 'cl_sl_r_2024-09-16_01-32-16/cl_sl_r/parallel_seed___0/0/HitBackEnv_2024-09-16-01-34-04',
        check_point: str = 'two_days_origin_2024-09-11_12-59-00/two_days_origin/parallel_seed___0/0/HitBackEnv_2024-09-11-13-00-58',
        # check_point=None
):
    env = HitBackEnv(visual_target=True, horizon=600, curriculum_steps=6, gamma=0.99)
    # env = BaseEnv(visual_target=True, horizon=200)
    if not use_nn:
        env.info.action_space = Box(np.array([-0.9 + 1.51, -0.45]), np.array([-0.2 + 1.51, 0.45]))
        env.info.observation_space = Box(np.ones(20), np.ones(20))
        planner_path = os.path.join('..', 'trained_low_agent', load_nn_agent)
        planner_config = Config
        if check_point is None:
            agent_1 = build_agent_T_SAC(mdp_info=env.info, env_info=env.env_info, adv_bonus=adv_bonus,
                                        planner_path=planner_path,
                                        planner_config=planner_config,
                                        actor_lr=actor_lr, critic_lr=critic_lr, termination_lr=termination_lr,
                                        n_features_actor=n_features_actor, n_features_critic=n_features_critic,
                                        n_features_termination=n_features_termination, batch_size=batch_size,
                                        initial_replay_size=initial_replay_size, max_replay_size=max_replay_size,
                                        tau=tau,
                                        num_adv_sample=num_adv_sample, warmup_transitions=warmup_transitions,
                                        lr_alpha=lr_alpha, target_entropy=target_entropy, dropout_ratio=dropout_ratio,
                                        layer_norm=layer_norm, use_cuda=use_cuda, termination_warmup=termination_warmup,
                                        use_nn=use_nn)
        else:
            agent_1 = SACPlusTermination.load(get_agent_path(check_point))
    else:
        env.info.action_space = Box(np.array([0.6, -0.39105, -np.pi, 0.]), np.array([1.3, 0.39105, np.pi, 1]))
        planner_path = os.path.join('..', 'trained_low_agent', load_nn_agent)
        planner_config = Config
        agent_1 = build_agent_T_SAC(mdp_info=env.info, env_info=env.env_info, adv_bonus=adv_bonus,
                                    planner_path=planner_path,
                                    planner_config=planner_config,
                                    actor_lr=actor_lr, critic_lr=critic_lr, termination_lr=termination_lr,
                                    n_features_actor=n_features_actor, n_features_critic=n_features_critic,
                                    n_features_termination=n_features_termination, batch_size=batch_size,
                                    initial_replay_size=initial_replay_size, max_replay_size=max_replay_size, tau=tau,
                                    num_adv_sample=num_adv_sample, warmup_transitions=warmup_transitions,
                                    lr_alpha=lr_alpha, target_entropy=target_entropy, dropout_ratio=dropout_ratio,
                                    layer_norm=layer_norm, use_cuda=use_cuda, termination_warmup=termination_warmup,
                                    use_nn=use_nn)

    baseline_agent = BaselineAgent(env.env_info, agent_id=2)
    agent_path_list = [
                       # 'two_days_origin_2024-09-11_12-59-00/two_days_origin/parallel_seed___0/0/HitBackEnv_2024-09-11-13-00-58',
                       # 'two_days_selflearn_2024-09-12_01-25-49/two_days_selflearn/parallel_seed___0/0/HitBackEnv_2024-09-12-01-26-53',
                       # 'cl_line_2024-09-12_00-48-29/cl_line/parallel_seed___0/0/HitBackEnv_2024-09-12-00-49-21',
                       # 'cl_sl_line_2024-09-15_12-26-41/cl_sl_line/parallel_seed___0/0/HitBackEnv_2024-09-15-14-54-49',
                       # 'cl_r_2024-09-16_13-38-36/cl_r/parallel_seed___0/0/HitBackEnv_2024-09-16-14-25-15',
                       # 'cl_sl_r_2024-09-16_01-32-16/cl_sl_r/parallel_seed___0/0/HitBackEnv_2024-09-16-01-34-04',
                       ]
    oppponent_agent_list = [SACPlusTermination.load(get_agent_path(agent_path)) for agent_path in agent_path_list]
    oppponent_agent_list.append(baseline_agent)

    agent = HRLTournamentAgentWrapper(env.info, agent_1, agent_list=oppponent_agent_list, use_nn=use_nn)

    # serve_agent = ServeAgent(env.info, agent_1)
    core = Core(agent, env)

    core.evaluate(n_episodes=2, render=True, record=True)

def get_file_by_postfix(parent_dir, postfix):
    file_list = list()
    for root, dirs, files in os.walk(parent_dir):
        for f in files:
            if f.endswith(postfix):
                a = os.path.join(root, f)
                file_list.append(a)
    return file_list

def get_agent_path(agent_path):
    cur_path = os.path.abspath('.')
    parent_dir = os.path.dirname(cur_path)
    check_path = os.path.join(parent_dir, 'trained_high_agent', agent_path)
    agent_name = f'agent-{agent_path.split("parallel_seed___")[1].split("/")[0]}.msh'
    return get_file_by_postfix(check_path, agent_name)[0]

if __name__ == "__main__":
    main()
