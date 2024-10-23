import os
import torch
import numpy as np
from torch import optim
import torch.nn.functional as F
from hrl_air_hockey.agents.t_sac import SACPlusTermination
from hrl_air_hockey.utils.sac_network import SACActorNetwork, SACCriticNetwork, TerminationNetwork


def build_agent_T_SAC(mdp_info, env_info, actor_lr, critic_lr, termination_lr, n_features_actor, n_features_critic, planner_config, planner_path,
                      n_features_termination, batch_size, initial_replay_size, max_replay_size, tau, num_adv_sample, adv_bonus,
                      warmup_transitions, lr_alpha, target_entropy, dropout_ratio, layer_norm, use_cuda, termination_warmup, use_nn):
    if type(n_features_actor) is str:
        n_features_actor = list(map(int, n_features_actor.split(" ")))

    if type(n_features_critic) is str:
        n_features_critic = list(map(int, n_features_critic.split(" ")))

    if type(n_features_termination) is str:
        n_features_termination = list(map(int, n_features_termination.split(" ")))

    actor_mu_params = dict(network=SACActorNetwork,
                           input_shape=mdp_info.observation_space.shape,
                           output_shape=mdp_info.action_space.shape,
                           n_features=n_features_actor,
                           use_cuda=use_cuda)
    actor_sigma_params = dict(network=SACActorNetwork,
                              input_shape=mdp_info.observation_space.shape,
                              output_shape=mdp_info.action_space.shape,
                              n_features=n_features_actor,
                              use_cuda=use_cuda)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': actor_lr}}
    critic_params = dict(network=SACCriticNetwork,
                         input_shape=(mdp_info.observation_space.shape[0] +
                                      mdp_info.action_space.shape[0],),
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': critic_lr}},
                         loss=F.mse_loss,
                         n_features=n_features_critic,
                         output_shape=(1,),
                         dropout_ratio=dropout_ratio,
                         dropout=dropout_ratio > 0,
                         layer_norm=layer_norm,
                         use_cuda=use_cuda)

    alg_params = dict(initial_replay_size=initial_replay_size,
                      max_replay_size=max_replay_size,
                      batch_size=batch_size,
                      warmup_transitions=warmup_transitions,
                      termination_warmup=termination_warmup,
                      tau=tau,
                      lr_alpha=lr_alpha,
                      critic_fit_params=None,
                      target_entropy=target_entropy,
                      )

    termination_params = dict(network=TerminationNetwork,
                              input_shape=(mdp_info.observation_space.shape[0] +
                                           mdp_info.action_space.shape[0],),
                              n_features=n_features_termination,
                              output_shape=(1,),
                              dropout_ratio=dropout_ratio,
                              dropout=dropout_ratio > 0,
                              layer_norm=layer_norm,
                              use_cuda=use_cuda)
    termination_optimizer = {'class': optim.Adam,
                             'params': {'lr': termination_lr}}

    device = 'cuda:0' if use_cuda else 'cpu'

    init_state = np.array([0., -0.1961, 0., -1.8436, 0., 0.9704, 0.])
    controller_info = dict(
        dim_q=env_info['robot']['n_joints'],
        pos_limit=env_info['robot']['joint_pos_limit'] * 0.95,
        vel_limit=env_info['robot']['joint_vel_limit'] * 0.95,
        slack_beta=[2] * 7 + [2., 2., 2., 2., 2., 2, 2, 2],
        slack_dynamics_type="exp",
        drift_compensation_type='vanilla',
        drift_clipping=True,
        slack_tol=1e-6,
        slack_vel_limit=50,
        lambda_c=0.5 / env_info['dt']
    )

    atacom_planner_params = dict(env_info=env_info, controller_info=controller_info, init_state=init_state)
    config = planner_config
    nn_planner_params = dict(planner_path=planner_path, env_info=env_info, config=config, device=device,
                             violate_path=config.data.violate_path)

    agent = SACPlusTermination(mdp_info, actor_mu_params=actor_mu_params, actor_sigma_params=actor_sigma_params, nn_planner_params=nn_planner_params,
                               atacom_planner_params=atacom_planner_params, termination_params=termination_params,
                               termination_optimizer=termination_optimizer, num_adv_sample=num_adv_sample, adv_bonus=adv_bonus,
                               actor_optimizer=actor_optimizer, critic_params=critic_params, device=device, use_nn=use_nn, **alg_params)

    return agent
