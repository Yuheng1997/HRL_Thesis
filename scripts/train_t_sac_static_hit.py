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
def experiment(env_name: str = 'StaticHit',
               n_epochs: int = 10,
               n_steps: int = 300,
               n_episodes: int = 1,
               quiet: bool = True,
               n_steps_per_fit: int = 1,
               render: bool = False,
               record: bool = False,
               n_eval_episodes: int = 1,
               n_eval_steps: int = 10,
               mode: str = 'disabled',
               horizon: int = 300,
               load_nn_agent: str = 'Model_5600.pt',
               full_save: bool = False,

               group: str = None,

               actor_lr: float = 3e-4,
               critic_lr: float = 3e-4,
               termination_lr: float = 3e-6,
               num_adv_sample: int = 50,
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

               # Continue training
               # check_point: str = 'static_hit_2024-07-15_16-27-51/parallel_seed___2/0/BaseEnv_2024-07-15-17-34-35',
               check_point: str = None,

               # curriculum config
               task_curriculum: bool = False,
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

    os.environ["WANDB_API_KEY"] = Config.wandb.api_key
    wandb.init(project="LearnHitBack", dir=results_dir, config=config, name=f"{current_time}_seed{parallel_seed}",
               group=group, notes=f"logdir: {logger._results_dir}", mode=mode)

    eval_params = {
        "n_steps": n_eval_steps,
        "quiet": quiet,
        "render": render
    }

    env = BaseEnv(horizon=horizon)

    env.info.action_space = Box(np.array([0.6, -0.39105, -np.pi, 0.]), np.array([1.3, 0.39105, np.pi, 1]))

    if check_point is None:
        planner_path = os.path.join('..', 'trained_low_agent', load_nn_agent)
        planner_config = Config
        agent_1 = build_agent_T_SAC(mdp_info=env.info, env_info=env.env_info,
                                    planner_path=planner_path, planner_config=planner_config,
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
        agent_1 = SACPlusTermination.load(get_file_by_postfix(check_path, 'agent-1.msh')[0])
        agent_1._alpha_optim = optim.Adam([agent_1._log_alpha], lr=lr_alpha)

    baseline_agent = BaselineAgent(env.env_info, agent_id=2)
    wrapped_agent = HRLTournamentAgentWrapper(env.env_info, agent_1, baseline_agent)
    core = Core(wrapped_agent, env)

    best_R = -np.inf

    # initial evaluate
    J, R, E, V, alpha, max_Beta, mean_Beta, task_info = compute_metrics(core, eval_params, record)

    logger.log_numpy(J=J, R=R, E=E, V=V, alpha=alpha, max_Beta=max_Beta, mean_Beta=mean_Beta, **task_info)
    size_replay_memory = core.agent.agent_1._replay_memory.size
    num_violate_point = core.agent.agent_1.traj_planner.num_violate_point

    logger.epoch_info(0, J=J, R=R, E=E, V=V, alpha=alpha, max_Beta=max_Beta, mean_Beta=mean_Beta,
                      size_replay_memory=size_replay_memory, num_violate_point=num_violate_point, **task_info)

    log_dict = {"Reward/J": J, "Reward/R": R, "Training/E": E, "Training/V": V, "Training/alpha": alpha,
                "Training/max_beta": max_Beta, "Training/mean_beta": mean_Beta,
                "Training/num_violate_point": num_violate_point,
                "Training/size_replay_memory": size_replay_memory}

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
        # core.agent.learning_agent.num_fits_left = n_steps
        # core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit, quiet=quiet)
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit, quiet=quiet)

        J, R, E, V, alpha, max_Beta, mean_Beta, task_info = compute_metrics(core, eval_params, record)
        size_replay_memory = core.agent.agent_1._replay_memory.size
        num_violate_point = core.agent.agent_1.traj_planner.num_violate_point

        if task_curriculum:
            if task_info['success_rate'] >= 0.7:
                core.mdp.update_task()
            task_info['task_id'] = env.task_curriculum_dict['idx']

        # Write logging
        logger.log_numpy(J=J, R=R, E=E, V=V, alpha=alpha, max_Beta=max_Beta, mean_Beta=mean_Beta, **task_info)
        logger.epoch_info(epoch + 1, J=J, R=R, E=E, V=V, alpha=alpha, max_Beta=max_Beta, mean_Beta=mean_Beta,
                          size_replay_memory=size_replay_memory, num_violate_point=num_violate_point, **task_info)
        log_dict = {"Reward/J": J, "Reward/R": R, "Training/E": E, "Training/V": V, "Training/alpha": alpha,
                    "Training/max_beta": max_Beta, "Training/mean_beta": mean_Beta, "Training/num_violate_point": num_violate_point,
                    "Training/size_replay_memory": size_replay_memory}

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
        for i in range(len(dataset)):
            states.append(dataset[i][0])
            if dataset[i][5]:
                break
        return np.array(states)

    def compute_mean_beta(agent, dataset):
        initial_state = dataset[0][0]
        initial_option = agent.policy.draw_action(initial_state)
        states_traj = sample_states_traj(dataset)
        beta = np.array(
            [agent.termination_approximator.predict(states_traj[i], initial_option) for i in range(len(states_traj))])
        return np.mean(beta)

    def compute_max_beta(agent, dataset):
        initial_state = dataset[0][0]
        initial_option = agent.policy.draw_action(initial_state)
        states_traj = sample_states_traj(dataset)
        beta = np.array(
            [agent.termination_approximator.predict(states_traj[i], initial_option) for i in range(len(states_traj))])
        return np.max(beta)

    def spilt_dataset(_dataset):
        assert len(_dataset) > 0
        dataset = list()
        for i, d in enumerate(_dataset):
            state = d[0][:23]
            action = d[1][0]
            next_state = d[3][:23]
            dataset.append((state, action, d[2], next_state, d[4], d[5]))
        return dataset

    def parse_dataset(_dataset):
        assert len(_dataset) > 0

        state = np.ones((len(_dataset),) + _dataset[0][0].shape)
        action = np.ones((len(_dataset),) + (4,))
        reward = np.ones(len(_dataset))
        next_state = np.ones((len(_dataset),) + _dataset[0][0].shape)
        absorbing = np.ones(len(_dataset))
        last = np.ones(len(_dataset))

        for i in range(len(_dataset)):
            state[i, ...] = _dataset[i][0]
            action[i, ...] = _dataset[i][1][14:18]
            reward[i] = _dataset[i][2]
            next_state[i, ...] = _dataset[i][3]
            absorbing[i] = _dataset[i][4]
            last[i] = _dataset[i][5]

        return np.array(state), np.array(action), np.array(reward), np.array(
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

    max_Beta = compute_max_beta(rl_agent, dataset)

    mean_Beta = compute_mean_beta(rl_agent, dataset)

    alpha = rl_agent._alpha.cpu().detach().numpy()

    task_info = get_dataset_info(core, dataset, dataset_info)
    task_info['episode_length'] = np.mean(compute_episodes_length(dataset))

    if hasattr(core.mdp, 'clear_task_info'):
        core.mdp.clear_task_info()

    if return_dataset:
        return J, R, E, V, alpha, max_Beta, mean_Beta, task_info, parsed_dataset, dataset_info

    return J, R, E, V, alpha, max_Beta, mean_Beta, task_info


def get_dataset_info(core, dataset, dataset_info):
    epoch_info = {}
    success_list = []
    num_list = []
    termination_counts = 0
    num_traj = 0
    beta_termination = 0
    rest_traj_len = 0
    episodes = 0
    for i, d in enumerate(dataset):
        action = d[1]
        last_traj_length = action[19]
        termination = action[18]
        beta_t = action[21]
        if beta_t == 1:
            rest_traj_len += action[22]
        if termination == 1:
            num_traj += 1
            termination_counts += 1
        last = d[-1]
        if last:
            episodes += 1
            success_list.append(dataset_info['success'][i])
            if not termination == 1:
                num_traj += 1
    epoch_info['success_rate'] = np.sum(success_list) / len(success_list)
    epoch_info['num_termination'] = termination_counts
    epoch_info['mean_traj_length'] = len(dataset) / num_traj
    epoch_info['rest_traj_len'] = rest_traj_len / num_traj

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
