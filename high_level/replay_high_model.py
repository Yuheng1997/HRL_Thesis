import os
import json
import numpy as np
from train_hrl_exp import compute_metrics, get_dataset_info
from sac import SAC
from mushroom_rl.core import Core, Agent
from high_agent import build_warped_agent
from hit_back_env import HitBackEnv


# high_agent_dir = "./logs/high_level_2024-05-07_22-20-51/parallel_seed___2/0/BaseEnv_2024-05-07-22-21-07"
current_dir = os.path.dirname(os.path.abspath(__file__))
high_agent_dir = os.path.abspath(os.path.join(current_dir, os.pardir,
                 'trained_high_agent/high_level_2024-05-22_19-59-26/parallel_seed___0/0/BaseEnv_2024-05-22-19-59-51'))

record = False
# agent_1 = os.path.abspath(os.path.join(current_dir, os.pardir, 'trained_low_agent/Model_2020.pt'))
agent_1 = os.path.abspath(os.path.join(current_dir, os.pardir, 'trained_low_agent/in1_3table.pt'))

def get_file_by_postfix(parent_dir, postfix):
    file_list = list()
    for root, dirs, files in os.walk(parent_dir):
        for f in files:
            if f.endswith(postfix):
                file_list.append(os.path.join(root, f))
    return file_list


def get_file_by_prefix(parent_dir, prefix):
    file_list = list()
    for root, dirs, files in os.walk(parent_dir):
        for f in files:
            if f.startswith(prefix):
                file_list.append(os.path.join(root, f))
    return file_list


def main():
    file_list = get_file_by_postfix(high_agent_dir, 'args.json')
    if len(file_list) > 0:
        f = open(file_list[0])
        cfg = json.load(f)
        print(cfg)

    # env = BaseEnv(horizon=300)
    env = HitBackEnv(horizon=1000)
    rl_agent = SAC.load(get_file_by_postfix(high_agent_dir, 'agent-0.msh')[0])
    rl_agent.policy._log_std_min._initial_value = -20
    rl_agent.policy._log_std_max._initial_value = -20

    wrapped_agent = build_warped_agent(env, rl_agent, agent_1=agent_1, agent_2=None)
    wrapped_agent.double_low_agent.training_agent.with_condition = True
    wrapped_agent.termination = True

    eval_params = {
        "n_episodes": 30,
        "quiet": True,
        "render": True
    }
    core = Core(wrapped_agent, env)

    violation_state_list = []
    for i in range(100):
        J, R, E, V, alpha, task_info, dataset, dataset_info = compute_metrics(core, eval_params=eval_params, record=record, return_dataset=True)
        print(f"J: {J:.4f}, R: {R:.4f}, E:{E:.4f}, V: {V:.4f}, \n {task_info}")

        # ee_pos_list = []
        # for s in dataset[0]:
        #     joint_pos = s[6:13]
        #     ee_pos, ee_rot = forward_kinematics(env.env_info['robot']['robot_model'],
        #                                         env.env_info['robot']['robot_data'], joint_pos)
        #     ee_pos_list.append(ee_pos)
        # ee_pos_list = np.array(ee_pos_list)
        # fig, axes = plt.subplots(1, 1, figsize=(6, 4))
        # im1 = axes.scatter(ee_pos_list[:, 0], ee_pos_list[:, 1], c=ee_pos_list[:, 2], s=2)
        # axes.set_xlim(0.5, 1.5)
        # axes.set_ylim(-0.5, 0.5)
        # divider = make_axes_locatable(axes)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # plt.colorbar(im1, cax=cax, orientation='vertical')
        # plt.tight_layout()
        # plt.show()


if __name__ == '__main__':
    main()