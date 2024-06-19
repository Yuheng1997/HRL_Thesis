import os
import torch
from torch import optim
import torch.nn.functional as F
from hrl_air_hockey.agents.t_sac import SACPlusTermination
from hrl_air_hockey.utils.sac_network import SACActorNetwork, SACCriticNetwork, TerminationNetwork


def build_agent_T_SAC(mdp_info, env_info, planner_path, planner_config,
                      actor_lr, critic_lr, n_features_actor, n_features_critic, batch_size,
                      initial_replay_size, max_replay_size, tau,
                      warmup_transitions, lr_alpha, target_entropy, dropout_ratio, layer_norm, use_cuda):
    if type(n_features_actor) is str:
        n_features_actor = list(map(int, n_features_actor.split(" ")))

    if type(n_features_critic) is str:
        n_features_critic = list(map(int, n_features_critic.split(" ")))

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
                      tau=tau,
                      lr_alpha=lr_alpha,
                      critic_fit_params=None,
                      target_entropy=target_entropy,
                      )

    termination_params = dict(network=TerminationNetwork,
                              input_shape=(mdp_info.observation_space.shape[0] +
                                           mdp_info.action_space.shape[0],),
                              optimizer={'class': optim.Adam,
                                         'params': {'lr': critic_lr}},
                              n_features=n_features_critic,
                              output_shape=(1,),
                              dropout_ratio=dropout_ratio,
                              dropout=dropout_ratio > 0,
                              layer_norm=layer_norm,
                              use_cuda=use_cuda)

    device = 'gpu' if use_cuda else 'cpu'
    config = planner_config
    nn_planner_params = dict(planner_path=planner_path, env_info=env_info, config=config, device=device)
    agent = SACPlusTermination(mdp_info, actor_mu_params=actor_mu_params, actor_sigma_params=actor_sigma_params,
                               nn_planner_params=nn_planner_params, termination_params=termination_params,
                               actor_optimizer=actor_optimizer, critic_params=critic_params, **alg_params)

    return agent


def build_nn_agent(env_info, model_name, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    Args:
        env_info (dict): The environment information
        kwargs (any): Additionally setting from agent_config.yml
    Returns:
         (AgentBase) An instance of the Agent
    """
    # load model in traned_agent
    device = 'cpu'
    cur_path = os.getcwd()
    par_path = os.path.dirname(cur_path)
    model_path = os.path.join(par_path, 'trained_low_agent', model_name)
    model = torch.load(model_path, map_location=torch.device(device))
    return NNAgent(env_info, model, **kwargs)
