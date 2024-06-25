import os
import torch
import wandb
import inspect
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from datetime import datetime
from mushroom_rl.core import Core, Logger
from mushroom_rl.utils.spaces import Box
from experiment_launcher import single_experiment, run_experiment
from baseline.baseline_agent.baseline_agent import BaselineAgent

from hrl_air_hockey.envs.hit_back_env import HitBackEnv
from hrl_air_hockey.agents.double_agent_wrapper import HRLTournamentAgentWrapper
from hrl_air_hockey.utils.agent_builder import build_agent_T_SAC
from nn_planner_config import Config


@single_experiment
def experiment(env_name: str = 'HitBackEnv',
               n_epochs: int = 2,
               n_steps: int = 5,
               n_episodes: int = 1,
               quiet: bool = True,
               n_steps_per_fit: int = 1,
               render: bool = True,
               record: bool = False,
               n_eval_episodes: int = 1,
               mode: str = 'disabled',
               horizon: int = 1000,
               agent_1: str = 'Model_2600.pt',
               full_save: bool = False,

               group: str = None,

               actor_lr: float = 3e-4,
               critic_lr: float = 3e-4,
               termination_lr: float = 3e-4,
               num_adv_sample: int = 100,
               n_features_actor: str = '256 256 256',
               n_features_critic: str = '256 256 256',
               n_features_termination: str = '256 256 256',
               batch_size: int = 256,
               initial_replay_size: int = 1,
               max_replay_size: int = 200000,
               tau: float = 1e-3,
               warmup_transitions: int = 1,
               lr_alpha: float = 1e-5,
               target_entropy: float = -4,
               use_cuda: bool = False,
               dropout_ratio: float = 0.01,
               layer_norm: bool = False,

               # Continue training
               # check_point: str = 'logs/hit_back_2024-05-08_20-47-46/check_point___.-logs-high_level_2024-05-07_01-01-02-parallel_seed___0-0-BaseEnv_2024-05-07-01-01-21/parallel_seed___1/0/HitBackEnv_2024-05-08-21-18-54',
               # check_point: str = 'logs/hit_back_2024-05-08_20-09-58/parallel_seed___1/0/HitBackEnv_2024-05-08-20-18-50',
               # check_point: str = 'logs/hit_back_2024-05-09_10-15-30/check_point___.-logs-high_level_2024-05-07_01-01-02-parallel_seed___0-0-BaseEnv_2024-05-07-01-01-21/parallel_seed___1/0/HitBackEnv_2024-05-09-10-16-56',
               # check_point: str = 'logs/high_level_2024-05-15_23-16-22/parallel_seed___1/0/BaseEnv_2024-05-15-23-17-20',
               # check_point: str = 'logs/high_level_2024-05-22_19-59-26/parallel_seed___0/0/BaseEnv_2024-05-22-19-59-51',
               check_point: str = None,

               # curriculum config
               task_curriculum: bool = True,
               curriculum_steps: int = 10,

               parallel_seed: int = None,
               seed: int = 0,
               results_dir: str = './logs',
               ):
    if parallel_seed is None:
        parallel_seed = seed
    np.random.seed(parallel_seed)
    torch.manual_seed(parallel_seed)

    config = dict()
    for p in inspect.signature(experiment).parameters:
        config[p] = eval(p)

    logger = Logger(log_name=env_name, results_dir=results_dir, seed=parallel_seed, use_timestamp=True)

    current_time = datetime.now()
    current_time = current_time.strftime("%m-%d %H")

    os.environ["WANDB_API_KEY"] = "a903361ff1d9498b25c276d46a0dcc63fe596aca"
    wandb.init(project="LearnHitBack", dir=results_dir, config=config, name=f"{current_time}_seed{parallel_seed}",
               group=group, notes=f"logdir: {logger._results_dir}", mode=mode)

    eval_params = {
        "n_episodes": n_eval_episodes,
        "quiet": quiet,
        "render": render
    }

    env = HitBackEnv(horizon=horizon, gamma=0.99, task_curriculum=task_curriculum, curriculum_steps=curriculum_steps)

    env.info.action_space = Box(np.array([0.8, -0.39105, -np.pi, 0.]), np.array([1.3, 0.39105, np.pi, 1]))

    if check_point is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        planner_path = os.path.abspath(os.path.join(current_dir, os.pardir, 'trained_low_agent/Model_4250.pt'))
        planner_path = "Transferred_model_4250.pt"
        planner_config = Config
        agent_1 = build_agent_T_SAC(mdp_info=env.info, env_info=env.env_info,
                                    planner_path=planner_path, planner_config=planner_config,
                                    actor_lr=actor_lr, critic_lr=critic_lr, termination_lr=termination_lr,
                                    n_features_actor=n_features_actor, n_features_critic=n_features_critic,
                                    n_features_termination=n_features_termination, batch_size=batch_size,
                                    initial_replay_size=initial_replay_size, max_replay_size=max_replay_size, tau=tau,
                                    num_adv_sample=num_adv_sample, warmup_transitions=warmup_transitions,
                                    lr_alpha=lr_alpha, target_entropy=target_entropy, dropout_ratio=dropout_ratio,
                                    layer_norm=layer_norm, use_cuda=use_cuda)
        agent_1._log_alpha = torch.tensor(np.log(0.4)).to(agent_1._log_alpha).requires_grad_(True)
        agent_1._alpha_optim = optim.Adam([agent_1._log_alpha], lr=lr_alpha)
    else:
        raise NotImplemented
        # def get_file_by_postfix(parent_dir, postfix):
        #     file_list = list()
        #     for root, dirs, files in os.walk(parent_dir):
        #         for f in files:
        #             if f.endswith(postfix):
        #                 a = os.path.join(root, f)
        #                 file_list.append(a)
        #     return file_list
        #
        # agent = SAC.load(get_file_by_postfix(check_point, 'agent-0.msh')[0])
        # rl_agent._alpha_optim = optim.Adam([rl_agent._log_alpha], lr=lr_alpha)
        # rl_agent = SAC.load(check_point)

    baseline_agent = BaselineAgent(env.env_info, agent_id=2)
    wrapped_agent = HRLTournamentAgentWrapper(env.env_info, agent_1, baseline_agent)
    core = Core(wrapped_agent, env)

    best_R = -np.inf

    # initial evaluate
    # J, R, E, V, alpha, task_info = compute_metrics(core, record, eval_params)

    # logger.log_numpy(J=J, R=R, E=E, V=V, alpha=alpha, **task_info)
    # size_replay_memory = core.agent.agent_1._replay_memory.size
    # size_mdp_replay_memory = core.agent.agent_1._smdp_replay_memory.size
    # logger.epoch_info(0, J=J, R=R, E=E, V=V, alpha=alpha, size_replay_memory=size_replay_memory,
    #                   size_mdp_replay_memory=size_mdp_replay_memory, **task_info)
    #
    # log_dict = {"Reward/J": J, "Reward/R": R, "Training/E": E, "Training/V": V, "Training/alpha": alpha,
    #             "Training/size_replaymemory": size_replay_memory, "Training/size_mdp_replay_memory": size_mdp_replay_memory}

    # task_dict = {}
    # for key, value in task_info.items():
    #     if hasattr(value, '__iter__'):
    #         for i, v in enumerate(value):
    #             task_dict[key + f"_{i}"] = v
    #     else:
    #         task_dict[key] = value
    # log_dict.update(task_dict)
    # wandb.log(log_dict, step=0)

    for epoch in tqdm(range(n_epochs), disable=False):
        # core.agent.learning_agent.num_fits_left = n_steps
        # core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit, quiet=quiet)
        core.learn(n_episodes=n_episodes, n_steps_per_fit=n_steps_per_fit, quiet=quiet)

        J, R, E, V, alpha, task_info = compute_metrics(core, record, eval_params)
        size_replay_memory = core.agent.agent_1._replay_memory.size
        size_replay_memory = core.agent.agent_1._smdp_replay_memory.size

        if task_curriculum:
            if task_info['success_rate'] >= 0.7:
                core.mdp.update_task()
            task_info['task_id'] = env.task_curriculum_dict['idx']

        # Write logging
        logger.log_numpy(J=J, R=R, E=E, V=V, alpha=alpha, **task_info)
        logger.epoch_info(epoch + 1, J=J, R=R, E=E, V=V, alpha=alpha, **task_info)
        log_dict = {"Reward/J": J, "Reward/R": R, "Training/E": E, "Training/V": V, "Training/alpha": alpha,
                    "Training/size_replaymemory": size_replay_memory, "Training/size_mdp_replay_memory": size_mdp_replay_memory}

        task_dict = {}
        for key, value in task_info.items():
            if hasattr(value, '__iter__'):
                for i, v in enumerate(value):
                    task_dict[key + f"_{i}"] = v
            else:
                task_dict[key] = value
        log_dict.update(task_dict)
        wandb.log(log_dict, step=epoch + 1)

        logger.log_agent(agent_1, full_save=full_save)


def compute_metrics(core, record, eval_params, return_dataset=False):
    from mushroom_rl.utils.dataset import compute_J, compute_episodes_length
    from mushroom_rl.utils.frames import LazyFrames

    def get_init_states(dataset):
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

    def compute_V(agent, dataset):
        Q = list()
        inital_states = get_init_states(dataset)
        for state in inital_states:
            num = np.clip(len(inital_states), 0, 100)
            s = np.array([state for i in range(num)])
            a = np.array([agent.draw_action(state) for i in range(num)])
            Q.append(agent._critic_approximator(s, a).mean())
        return np.array(Q).mean()

    def parse_dataset(_dataset, features=None):
        assert len(_dataset) > 0

        shape = _dataset[0][0].shape if features is None else (features.size,)
        dataset = []

        sum_gamma = 1
        sum_r = 0
        save_initial_state = True
        for i in range(len(_dataset)):
            sum_r += _dataset[i][2] * sum_gamma
            if save_initial_state:
                initial_state = np.array(_dataset[i][0][:23])
            if _dataset[i][1][-1] or _dataset[i][-2]:
                a_list = list(_dataset[i])
                a_list[0] = initial_state
                a_list[1] = a_list[1][1]
                a_list[2] = sum_r
                dataset.append(a_list)
                sum_r = 0
                sum_gamma = 1
                save_initial_state = True
            else:
                save_initial_state = False
                sum_gamma *= 0.99
        if len(dataset) == 0:
            sum_gamma = 1
            sum_r = 0
            save_initial_state = True
            for i in range(len(_dataset)):
                sum_r += _dataset[i][2] * sum_gamma
                if save_initial_state:
                    initial_state = np.array(_dataset[i][0])
                if _dataset[i][-1]:
                    a_list = list(_dataset[i])
                    a_list[0] = initial_state
                    a_list[1] = a_list[1][1]
                    a_list[2] = sum_r
                    dataset.append(a_list)
                    sum_r = 0
                    sum_gamma = 1
                    save_initial_state = True
                else:
                    save_initial_state = False
                    sum_gamma *= 0.99

        state = np.ones((len(dataset),) + shape)
        action = np.ones((len(dataset),) + dataset[0][1].shape)
        reward = np.ones(len(dataset))
        next_state = np.ones((len(dataset),) + shape)
        absorbing = np.ones(len(dataset))
        last = np.ones(len(dataset))

        if features is not None:
            for i in range(len(dataset)):
                state[i, ...] = features(dataset[i][0])
                action[i, ...] = dataset[i][1]
                reward[i] = dataset[i][2]
                next_state[i, ...] = features(dataset[i][3])
                absorbing[i] = dataset[i][4]
                last[i] = dataset[i][5]
        else:
            for i in range(len(dataset)):
                state[i, ...] = dataset[i][0]
                action[i, ...] = dataset[i][1]
                reward[i] = dataset[i][2]
                next_state[i, ...] = dataset[i][3]
                absorbing[i] = dataset[i][4]
                last[i] = dataset[i][5]

        return np.array(state), np.array(action), np.array(reward), np.array(
            next_state), np.array(absorbing), np.array(last)

    dataset, dataset_info = core.evaluate(**eval_params, get_env_info=True, record=record)
    parsed_dataset = parse_dataset(dataset)

    rl_agent = core.agent.agent_1
    J = np.mean(compute_J(dataset, core.mdp.info.gamma))
    R = np.mean(compute_J(dataset))

    _, log_prob_pi = rl_agent.policy.compute_action_and_log_prob(parsed_dataset[0])
    E = -log_prob_pi.mean()

    V = compute_V(rl_agent, dataset)

    alpha = rl_agent._alpha.cpu().detach().numpy()

    task_info = get_dataset_info(core, dataset, dataset_info)
    task_info['episode_length'] = np.mean(compute_episodes_length(dataset))

    if hasattr(core.mdp, 'clear_task_info'):
        core.mdp.clear_task_info()

    if return_dataset:
        return J, R, E, V, alpha, task_info, parsed_dataset, dataset_info

    return J, R, E, V, alpha, task_info


def get_dataset_info(core, dataset, dataset_info):
    start_idx = 0

    epoch_info = {}
    success_list = []
    num_list = []
    for i, dataset in enumerate(dataset):
        last = dataset[-1]
        if last:
            success_list.append(dataset_info['success'][i])
            num_list.append(dataset_info['num_across_line'][i])
            start_idx = i + 1
    epoch_info['success_rate'] = np.sum(success_list) / len(success_list)
    epoch_info['num_across_line'] = np.sum(num_list)
    return epoch_info


if __name__ == '__main__':
    run_experiment(experiment)

    # import cProfile
    #
    # pr = cProfile.Profile()
    # pr.enable()
    #
    # run_experiment(experiment)
    #
    # pr.disable()
    # pr.dump_stats(file='profile_new.pstat')