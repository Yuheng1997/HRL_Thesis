import wandb
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.size"] = 30
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"

# Spring Pastels from https://www.heavy.ai/blog/12-color-palettes-for-telling-better-stories-with-your-data
COLOR_PALETTE = ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7"]


def download_run_history(entity, project, save_path, group_name):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters={"group": group_name})

    for run in runs:
        history = run.scan_history()
        hist_list = list()
        for row in history:
            hist_list.append(row)
        run_hist = pd.DataFrame(hist_list)

        run_hist.to_csv(f"{save_path}/{run.id}.csv", index=False)


def get_dataset_summary(eval_dir):
    from experiments_new.replay_agent import load_agent_and_env

    atacom_rl_agent, env = load_agent_and_env(
        eval_dir + "/agents", deterministic_policy=False, agent_postfix='_3000.msh')
    constraints = atacom_rl_agent.atacom_controller.constraints

    for root, dirs, files in os.walk(os.path.join(eval_dir, "eval")):
        epoch_list = list()
        for file in files:
            if "dataset_eval" in file:
                epoch = file.split("_")[-1].split(".")[0]
                epoch_list.append(epoch)
        # epoch_list.sort()

        constraint_summary = pd.DataFrame()
        J_summary = pd.DataFrame()
        success_summary = pd.DataFrame()
        puck_cross_vel_summary = pd.DataFrame()
        init_state_summary = dict()
        for epoch in epoch_list:
            dataset = pickle.load(open(os.path.join(eval_dir, "eval", f"dataset_eval_{epoch}.pkl"), 'rb'))
            dataset_info = pickle.load(open(os.path.join(eval_dir, "eval", f"dataset_info_eval_{epoch}.pkl"), 'rb'))
            k_max_episode = []
            J_episode = []
            success_list = []
            puck_cross_vel_list = []
            puck_cross_vel = 0
            init_states = [dataset[0][0][:2]]
            k_max = 0
            J = 0
            epi_idx = 0
            for i, d in enumerate(dataset):
                state = d[0]
                J += d[2] * env.info.gamma ** epi_idx
                q, x = atacom_rl_agent._unwrap_state(state)
                k = constraints.k(q, None)
                k_max = max(k_max, k.max())
                epi_idx += 1
                if state[0] >= 1.51 and puck_cross_vel == 0:
                    puck_cross_vel = state[3]

                if d[-1]:
                    if (i + 1) < len(dataset):
                        init_states.append(dataset[i+1][0][:2])
                    k_max_episode.append(k_max)
                    J_episode.append(J)
                    success_list.append(dataset_info['success'][i])
                    puck_cross_vel_list.append(puck_cross_vel)
                    k_max = 0
                    J = 0
                    epi_idx = 0
                    puck_cross_vel = 0
            constraint_summary[epoch] = k_max_episode.copy()
            J_summary[epoch] = J_episode.copy()
            init_state_summary[epoch] = pd.DataFrame(init_states)
            success_summary[epoch] = success_list.copy()
            puck_cross_vel_summary[epoch] = puck_cross_vel_list.copy()

        return constraint_summary, J_summary, init_state_summary, success_summary, puck_cross_vel_summary


def plot_learning_curve(eval_metric, eval_constraint, metric_key, title, xlabel, ylabel, baseline_metric=None, save_dir=None):
    fig, axes = plt.subplots(int(len(metric_key) / 2), 2, figsize=(16, 12), sharex=True, )
    i = 0
    handles, labels = [], []
    for i, (key, y_label) in enumerate(zip(metric_key, ylabel)):
        ax = axes[i//2, i % 2]
        not_null_idx = eval_metric['eval/J'].notnull()
        steps = eval_metric['_step'][not_null_idx]
        if key == "constraint":
            data = eval_constraint[steps.astype(str)].mean(axis=0)
        else:
            data = eval_metric[key]
            data = data[not_null_idx]
        line, = ax.plot(steps, data, marker='s', markersize=20, color=COLOR_PALETTE[1], linewidth=3, label="RL")
        line_init = ax.axhline(data.iloc[0], color=COLOR_PALETTE[1],
                               linestyle='--', linewidth=3, label="Initial Policy")
        # ax.text(3000, data.iloc[0], f"{data.iloc[0]:.2f}", ha='right', va='bottom')
        if baseline_metric is not None:
            baseline = ax.axhline(baseline_metric[key].values[0], color=COLOR_PALETTE[0],
                                  linestyle='--', linewidth=3, label="Baseline")
        if i == 0:
            handles.append(line)
            labels.append("ATACOM + SAC")
            handles.append(line_init)
            labels.append("Initial Policy")
            handles.append(baseline)
            labels.append("Baseline")

        if i // 2 == 1:
            ax.set_xlabel(xlabel)
        ax.set_ylabel(y_label)
        ax.grid(True)

    fig.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir + f"/learning_curve.pdf")

    fig = plt.figure(figsize=(12, 2))
    plt.legend(handles, labels, loc='center', frameon=False, ncol=3)
    plt.axis('off')
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir + f"/legend.pdf")


def plot_heat_map(init_state_summary, success_summary, puck_cross_vel_summary, grid_size, save_dir):
    grid_size.reverse()
    for epoch, init_states in init_state_summary.items():
        fig = plt.figure(figsize=(8, 12))
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        success_state = init_states[success_summary[epoch]]
        hist_state, x, y = np.histogram2d(init_states[0], init_states[1],
                                          bins=grid_size, range=[[0.8, 1.2], [-0.4, 0.4]])
        hist_success, x, y = np.histogram2d(
            success_state[0], success_state[1], bins=grid_size, range=[[0.8, 1.2], [-0.4, 0.4]])
        hist = hist_success / hist_state
        # hist = np.arange(16).reshape(4, 4)
        # im = plt.imshow(hist[:, ::-1], origin="lower", extent=[-0.4, 0.4, 0.8, 1.2], cmap='Blues')
        im = plt.imshow(hist[:, ::-1], origin="lower", extent=[-0.4, 0.4, 0.8, 1.2], cmap='Blues', vmin=-1, vmax=1)
        y = y[::-1]
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                plt.text(y[j: j + 2].mean(), x[i: i + 2].mean(),
                         f"{hist[i, j]:.2f}", ha='center', va='center', color='white')
        plt.plot([-0.45, 0.45], [0.52, 0.52], lw=10, c='k')
        plt.plot([-0.55, 0.55], [1.50, 1.50], lw=10, c='k', ls='--')
        plt.plot([-0.45, -0.45], [0.52, 1.50], lw=10, c='k')
        plt.plot([0.45, 0.45], [0.52, 1.50], lw=10, c='k')
        # plt.text(0., 1.3, "Table", ha='center', va='center', color='black')
        if "baseline" == epoch:
            plt.text(0., 1.28, f"Planning Baseline", ha='center', va='center', color='black')
        else:
            plt.text(0., 1.28, f"Episode {epoch}", ha='center', va='center', color='black')
        plt.text(0., 0.68, f"Success Rate {len(success_state) / len(init_states)}",
                 ha='center', va='center', color='black')
        plt.text(0., 0.58, f"Puck Velocity {puck_cross_vel_summary[epoch].mean():.2f}",
                 ha='center', va='center', color='black')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"heat_map_{epoch}.pdf"))
        plt.close()


def main():
    entity = "puze-liu"
    project = "IJRR_ATACOM_ROS_REAL"
    group_name = "23.Mar_SAC"
    package_dir = os.path.dirname(os.path.dirname(__file__))
    import sys
    sys.path.append(os.path.join(package_dir, "experiments_new"))

    log_directory = package_dir + "/experiments_new/archieved_logs/air_hockey_logs/real_robot_exp"
    fig_directory = package_dir + "/figure/air_hockey/real_robot_exp"
    os.makedirs(fig_directory, exist_ok=True)

    # download_run_history(entity, project, log_directory, group_name)

    # eval_metric = pd.read_csv(log_directory + "/2024-03-23_13-21-04.csv", index_col=[0, 1])

    constraint_summary, J_summary, init_state_summary, success_summary, puck_cross_vel_summary = get_dataset_summary(
        os.path.join(log_directory, "RealRobotEpi3000"))
    # eval_metric_baseline = pd.DataFrame([{'eval/J': J_summary['baseline'].mean(),
    #                                       'eval/success': success_summary['baseline'].mean(),
    #                                       'eval/puck_cross_vel': puck_cross_vel_summary['baseline'].mean(),
    #                                       'constraint': constraint_summary['baseline'].mean()}])

    # plot_learning_curve(eval_metric, constraint_summary, ["eval/J", "eval/success", "eval/puck_cross_vel", "constraint"], "", "Episodes",
    #                     ["Return", "Success", "Puck Speed", "Max Violation"], baseline_metric=eval_metric_baseline, save_dir=fig_directory)

    plot_heat_map(init_state_summary, success_summary, puck_cross_vel_summary, [4, 4], fig_directory)


if __name__ == "__main__":
    main()
