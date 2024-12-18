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
from hrl_air_hockey.experiment_launcher import single_experiment, run_experiment
from baseline.baseline_agent.baseline_agent import BaselineAgent

from hrl_air_hockey.envs.hit_back_env import HitBackEnv
from hrl_air_hockey.envs.base_env import BaseEnv
from hrl_air_hockey.agents.t_sac import SACPlusTermination
from hrl_air_hockey.agents.double_agent_wrapper import HRLTournamentAgentWrapper
from hrl_air_hockey.utils.agent_builder import build_agent_T_SAC
from nn_planner_config import Config


@single_experiment
def experiment(env_name: str = 'HitBackEnv',
               n_epochs: int = 100,
               n_steps: int = 20000,
               quiet: bool = True,
               n_steps_per_fit: int = 1,
               render: bool = False,
               record: bool = False,
               n_eval_steps: int = 12000,
               mode: str = 'disabled',
               horizon: int = 12000,
               full_save: bool = False,

               group: str = None,

               gamma: float = 0.995,
               actor_lr: float = 3e-4,
               critic_lr: float = 3e-4,
               termination_lr: float = 0.00001,
               num_adv_sample: int = 50,
               adv_bonus: float = 0.1,
               n_features_actor: str = '256 256 256',
               n_features_critic: str = '256 256 256',
               n_features_termination: str = '256 256 256',
               batch_size: int = 256,
               initial_replay_size: int = 20000,
               max_replay_size: int = 1000000,
               tau: float = 1e-3,
               warmup_transitions: int = 20000,
               termination_warmup: int = 20000,
               lr_alpha: float = 1e-5,
               target_entropy: float = -2,
               use_cuda: bool = True,
               dropout_ratio: float = 0.01,
               layer_norm: bool = False,
               self_learn: bool = True,

               # Continue training
               check_point: str = 'cl_sl_line_2024-09-15_12-26-41/cl_sl_line/parallel_seed___0/0/HitBackEnv_2024-09-15-14-54-49',
               # check_point: str = None,

               # opponent agent
               agent_path_list: list = None,

               # curriculum config
               task_curriculum: bool = False,
               curriculum_steps: int = 6,

               parallel_seed: int = None,
               seed: int = 0,
               results_dir: str = './logs',
               ):
    if parallel_seed is None:
        parallel_seed = seed
    np.random.seed(parallel_seed)
    torch.manual_seed(parallel_seed)

    env = HitBackEnv(horizon=horizon, curriculum_steps=curriculum_steps, task_curriculum=task_curriculum, gamma=gamma)
    env.info.action_space = Box(np.array([-0.9 + 1.51, -0.45]), np.array([-0.2 + 1.51, 0.45]))
    env.info.observation_space = Box(np.ones(20), np.ones(20))

    if agent_path_list is None:
        agent_path_list = [
                           'two_days_origin_2024-09-11_12-59-00/two_days_origin/parallel_seed___0/0/HitBackEnv_2024-09-11-13-00-58',
                           # 'two_days_selflearn_2024-09-12_01-25-49/two_days_selflearn/parallel_seed___0/0/HitBackEnv_2024-09-12-01-26-53',
                           # 'cl_line_2024-09-12_00-48-29/cl_line/parallel_seed___0/0/HitBackEnv_2024-09-12-00-49-21',
                           # 'cl_sl_line_2024-09-15_12-26-41/cl_sl_line/parallel_seed___0/0/HitBackEnv_2024-09-15-14-54-49',
                           # 'cl_r_2024-09-16_13-38-36/cl_r/parallel_seed___0/0/HitBackEnv_2024-09-16-14-25-15',
                           # 'cl_sl_r_2024-09-16_01-32-16/cl_sl_r/parallel_seed___0/0/HitBackEnv_2024-09-16-01-34-04',
                           ]
        oppponent_agent_list = [SACPlusTermination.load(get_agent_path(agent_path)) for agent_path in agent_path_list]
        baseline_agent = BaselineAgent(env.env_info, agent_id=2)
        oppponent_agent_list.insert(0, baseline_agent)

    config = dict()
    for p in inspect.signature(experiment).parameters:
        config[p] = eval(p)

    logger = Logger(log_name=env_name, results_dir=results_dir, seed=parallel_seed, use_timestamp=True)

    current_time = datetime.now()
    current_time = current_time.strftime("%m-%d %H")

    os.environ["WANDB_API_KEY"] = Config.wandb.api_key
    wandb.init(project="LearnHitBack", dir=results_dir, config=config, name=f"{current_time}_seed{parallel_seed}",
               group=group, notes=f"logdir: {logger._results_dir}", mode=mode, id='3yqwd85o', resume='allow')

    eval_params = {
        "n_steps": n_eval_steps,
        "quiet": quiet,
        "render": render
    }

    if check_point is None:
        agent_1 = build_agent_T_SAC(mdp_info=env.info, env_info=env.env_info, adv_bonus=adv_bonus,
                                    actor_lr=actor_lr, critic_lr=critic_lr, termination_lr=termination_lr,
                                    n_features_actor=n_features_actor, n_features_critic=n_features_critic,
                                    n_features_termination=n_features_termination, batch_size=batch_size,
                                    initial_replay_size=initial_replay_size, max_replay_size=max_replay_size, tau=tau,
                                    num_adv_sample=num_adv_sample, warmup_transitions=warmup_transitions,
                                    lr_alpha=lr_alpha, target_entropy=target_entropy, dropout_ratio=dropout_ratio,
                                    layer_norm=layer_norm, use_cuda=use_cuda, termination_warmup=termination_warmup)
        agent_1._log_alpha = torch.tensor(np.log(0.4)).to(agent_1._log_alpha).requires_grad_(True)
        agent_1._alpha_optim = optim.Adam([agent_1._log_alpha], lr=lr_alpha)
    else:
        agent_1 = SACPlusTermination.load(get_agent_path(check_point))
        agent_1._alpha_optim = optim.Adam([agent_1._log_alpha], lr=lr_alpha)

    wrapped_agent = HRLTournamentAgentWrapper(env.env_info, agent_1, agent_list=oppponent_agent_list)
    core = Core(wrapped_agent, env)

    # initial evaluate
    J, R, E, V, alpha, max_Beta, mean_Beta, min_Beta, task_info = compute_metrics(core, eval_params, record)

    logger.log_numpy(J=J, R=R, E=E, V=V, alpha=alpha, max_Beta=max_Beta, mean_Beta=mean_Beta, min_Beta=min_Beta, **task_info)
    size_replay_memory = core.agent.agent_1._replay_memory.size
    adv_func_in_fit = np.mean(core.agent.agent_1.adv_list)

    logger.epoch_info(0, J=J, R=R, E=E, V=V, alpha=alpha, max_Beta=max_Beta, mean_Beta=mean_Beta, min_Beta=min_Beta,
                      size_replay_memory=size_replay_memory, **task_info)

    log_dict = {"Reward/J": J, "Reward/R": R, "Training/E": E, "Training/V": V, "Training/alpha": alpha,
                "Termination/max_beta": max_Beta, "Termination/mean_beta": mean_Beta, "Termination/min_beta":min_Beta,
                "size_replay_memory": size_replay_memory, "Termination/adv_value_in_fit(mean)": adv_func_in_fit}

    task_dict = {}
    for key, value in task_info.items():
        if hasattr(value, '__iter__'):
            for i, v in enumerate(value):
                task_dict[key + f"_{i}"] = v
        else:
            task_dict[key] = value
    log_dict.update(task_dict)
    wandb.log(log_dict, step=0)

    for epoch in tqdm(range(n_epochs), disable=False):
        if check_point is not None:
            epoch += 100
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit, quiet=quiet)

        J, R, E, V, alpha, max_Beta, mean_Beta, min_Beta, task_info = compute_metrics(core, eval_params, record)
        size_replay_memory = core.agent.agent_1._replay_memory.size
        adv_func_in_fit = np.mean(core.agent.agent_1.adv_list)

        if task_curriculum:
            if task_info['success_rate'] >= 0.7:
                core.mdp.update_task()
            task_info['task_id'] = env.task_curriculum_dict['idx']

        # Write logging
        logger.log_numpy(J=J, R=R, E=E, V=V, alpha=alpha, max_Beta=max_Beta, mean_Beta=mean_Beta, min_Beta=min_Beta, **task_info)
        logger.epoch_info(epoch + 1, J=J, R=R, E=E, V=V, alpha=alpha, max_Beta=max_Beta, mean_Beta=mean_Beta, min_Beta=min_Beta,
                          size_replay_memory=size_replay_memory, **task_info)
        log_dict = {"Reward/J": J, "Reward/R": R, "Training/E": E, "Training/V": V, "Training/alpha": alpha,
                    "Termination/max_beta": max_Beta, "Termination/mean_beta": mean_Beta, "Termination/min_beta":min_Beta,
                    "size_replay_memory": size_replay_memory, "Termination/adv_value_in_fit(mean)": adv_func_in_fit}

        task_dict = {}
        for key, value in task_info.items():
            if hasattr(value, '__iter__'):
                for i, v in enumerate(value):
                    task_dict[key + f"_{i}"] = v
            else:
                task_dict[key] = value
        log_dict.update(task_dict)
        wandb.log(log_dict, step=epoch + 1)
        core.agent.agent_1.epoch_start()
        logger.log_agent(agent_1, full_save=full_save)
        if self_learn:
            wrapped_agent.update_opponent_list(new_agent=agent_1)


def compute_metrics(core, eval_params, record=False, return_dataset=False):
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
            a = np.array([agent.policy.draw_action(state) for i in range(num)])
            Q.append(agent._critic_approximator(s, a).mean())
        return np.array(Q).mean()

    def sample_states_traj(dataset):
        states = list()
        options = list()
        for i in range(len(dataset)):
            states.append(dataset[i][0])
            options.append(dataset[i][1][14:16])
        return np.array(states), np.array(options)

    def compute_beta_metrics(agent, dataset):
        states_traj, options_traj = sample_states_traj(dataset)
        beta = np.array(
            [agent.termination_approximator.predict(states_traj[i], options_traj[i]) for i in range(len(states_traj))])
        return np.mean(beta), np.max(beta), np.min(beta)

    def spilt_dataset(_dataset):
        assert len(_dataset) > 0
        dataset = list()
        for i, d in enumerate(_dataset):
            state = d[0][:20]
            action = d[1][0]
            next_state = d[3][:20]
            dataset.append((state, action, d[2], next_state, d[4], d[5]))
        return dataset

    def parse_dataset(_dataset):
        assert len(_dataset) > 0

        state = np.ones((len(_dataset),) + _dataset[0][0].shape)
        option = np.ones((len(_dataset),) + (2,))
        reward = np.ones(len(_dataset))
        next_state = np.ones((len(_dataset),) + _dataset[0][0].shape)
        absorbing = np.ones(len(_dataset))
        last = np.ones(len(_dataset))

        for i in range(len(_dataset)):
            state[i, ...] = _dataset[i][0]
            option[i, ...] = _dataset[i][1][14:16]
            reward[i] = _dataset[i][2]
            next_state[i, ...] = _dataset[i][3]
            absorbing[i] = _dataset[i][4]
            last[i] = _dataset[i][5]

        return np.array(state), np.array(option), np.array(reward), np.array(
            next_state), np.array(absorbing), np.array(last)

    dataset, dataset_info = core.evaluate(**eval_params, record=record, get_env_info=True)
    dataset = spilt_dataset(dataset)
    parsed_dataset = parse_dataset(dataset)

    rl_agent = core.agent.agent_1
    J = np.mean(compute_J(dataset, core.mdp.info.gamma))
    R = np.mean(compute_J(dataset))

    _, log_prob_pi = rl_agent.policy.compute_action_and_log_prob(parsed_dataset[0])
    E = -log_prob_pi.mean()

    V = compute_V(rl_agent, dataset)

    mean_Beta, max_Beta, min_Beta = compute_beta_metrics(rl_agent, dataset)

    alpha = rl_agent._alpha.cpu().detach().numpy()

    task_info = get_dataset_info(core, dataset, dataset_info)
    task_info['episode_length'] = np.mean(compute_episodes_length(dataset))

    if hasattr(core.mdp, 'clear_task_info'):
        core.mdp.clear_task_info()

    if return_dataset:
        return J, R, E, V, alpha, max_Beta, mean_Beta, min_Beta, task_info, parsed_dataset, dataset_info

    return J, R, E, V, alpha, max_Beta, mean_Beta, min_Beta, task_info


def get_dataset_info(core, dataset, dataset_info):
    epoch_info = {}
    success_list = []
    termination_counts = 0
    episodes = 0
    adv_value = []
    sub_episodes_num = 0
    success_num = 0
    for i, d in enumerate(dataset):
        action = d[1]
        termination = action[16]
        adv_value.append(action[17])
        if termination == 1:
            termination_counts += 1
        last = d[-1]
        if last:
            episodes += 1
            # success_list.append(dataset_info['success'][i])
        if dataset_info['sub_episodes'][i] == 1:
            sub_episodes_num += 1
            success_num += dataset_info['success'][i]

    epoch_info['success_rate'] = success_num / sub_episodes_num
    epoch_info['adv_value_in_action(mean)'] = sum(adv_value) / len(adv_value)
    epoch_info['termination_num'] = termination_counts
    epoch_info['hit_num'] = dataset_info['hit_num'][-1]
    epoch_info['win'] = dataset_info['win'][-1]
    epoch_info['lose'] = dataset_info['lose'][-1]
    epoch_info['self_fault'] = dataset_info['self_faults'][-1]
    epoch_info['oppo_fault'] = dataset_info['oppo_faults'][-1]
    epoch_info['episodes_num'] = episodes
    epoch_info['serve_round'] = dataset_info['serve_round'][-1]
    epoch_info['serve_success'] = dataset_info['serve_success'][-1]
    epoch_info['attack_num'] =dataset_info['attack_num'][-1]
    epoch_info['undefended_num'] = dataset_info['undefended_num'][-1]
    return epoch_info


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
