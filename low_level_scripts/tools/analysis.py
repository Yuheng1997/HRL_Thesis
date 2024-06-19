import os
import numpy as np
import torch
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from config import Config
from differentiable_robot_model.differentiable_robot_model import DifferentiableRobotModel
from losses.constraint_functions import air_hockey_puck, air_hockey_table
from losses.hitting import HittingLoss
from utils.bspline import BSpline
from utils.constants import Limits


device = Config.train.device
air_hockey_dt = 0.02
bspline_q = BSpline(num_pts=Config.bspline_q.n_ctr_pts, degree=Config.bspline_q.degree,
                    num_T_pts=Config.bspline_q.num_T_pts, device=device)
bspline_t = BSpline(num_pts=Config.bspline_t.n_ctr_pts, degree=Config.bspline_t.degree,
                    num_T_pts=Config.bspline_t.num_T_pts, device=device)


def load_data(filename="test_data_challenge_defend.tsv", n=10000):
    dataset_path = Config.data.path
    dataset_path = dataset_path.replace("train", "test")
    dataset_path = dataset_path.replace("data.tsv", filename)
    data = np.loadtxt(dataset_path, delimiter='\t').astype(np.float32)

    # np.random.shuffle(data)
    # data = data[:n]
    features = torch.from_numpy(data).to(Config.train.device)

    return features


def compute_control_points(model, features):
    with torch.no_grad():
        q_cps, t_cps = model(features.to(torch.float32))
        q_cps, t_cps = q_cps.to(torch.float32), t_cps.to(torch.float32)

    return q_cps, t_cps


def interpolate_control_points(q_cps, t_cps):
        with torch.no_grad():
            q = torch.einsum('ijk,lkm->ljm', bspline_q.N, q_cps)
            q_dot_tau = torch.einsum('ijk,lkm->ljm', bspline_q.dN, q_cps)
            q_ddot_tau = torch.einsum('ijk,lkm->ljm', bspline_q.ddN, q_cps)

            dtau_dt = torch.einsum('ijk,lkm->ljm', bspline_t.N, t_cps)
            ddtau_dtt = torch.einsum('ijk,lkm->ljm', bspline_t.dN, t_cps)

            dt = 1. / dtau_dt[..., 0] / dtau_dt.shape[1]
            t_cumsum = torch.cumsum(dt, dim=-1)

            q_dot = q_dot_tau * dtau_dt
            q_ddot = q_ddot_tau * dtau_dt ** 2 + ddtau_dtt * q_dot_tau * dtau_dt

            dt = t_cumsum[0].cpu().numpy()
            q = q[0].cpu().numpy()
            q_dot = q_dot[0].cpu().numpy()
            q_ddot = q_ddot[0].cpu().numpy()

            q_interpol = [interp1d(dt, q[:, i], kind='linear', fill_value='extrapolate') for i in range(7)]
            q_dot_interpol = [interp1d(dt, q_dot[:, i], kind='linear', fill_value='extrapolate') for i in range(7)]
            q_ddot_interpol = [interp1d(dt, q_ddot[:, i], kind='linear', fill_value='extrapolate') for i in range(7)]

            ts = np.arange(0, dt[-1], air_hockey_dt / 20)
            pos = np.array([q_interpol[i](ts) for i in range(7)]).transpose()
            vel = np.array([q_dot_interpol[i](ts) for i in range(7)]).transpose()
            acc = np.array([q_ddot_interpol[i](ts) for i in range(7)]).transpose()

            return np.stack([pos, vel, acc], axis=1), np.round((air_hockey_dt / 20) * len(pos), 3)
        

def bar_plot_times(data: np.ndarray, save_name: str ="bar_plot"):
    """
    Plot a bar plot from the given data.

    Parameters:
    - data: A NumPy array containing the data to be plotted.
    - save_name: The name to save the plot (default is 'bar_plot').
    """
    print(np.mean(data))
    print(np.max(data))
    print(np.min(data))
    x = np.arange(len(data))
    plt.bar(x, data)

    plt.xlabel('X Axis Label')
    plt.ylabel('Y Axis Label')
    plt.title('Bar Plot')

    plt.xticks(x, [str(i) for i in range(len(data))])

    # plt.savefig(save_name + '.png')

    plt.show()


def compute_trajectory(q_cps, t_cps):
        q = torch.einsum('ijk,lkm->ljm', bspline_q.N, q_cps)
        q_dot_tau = torch.einsum('ijk,lkm->ljm', bspline_q.dN, q_cps)
        q_ddot_tau = torch.einsum('ijk,lkm->ljm', bspline_q.ddN, q_cps)
        q_dddot_tau = torch.einsum('ijk,lkm->ljm', bspline_q.dddN, q_cps)

        dtau_dt = torch.einsum('ijk,lkm->ljm', bspline_t.N, t_cps)
        ddtau_dtt = torch.einsum('ijk,lkm->ljm', bspline_t.dN, t_cps)
        dddtau_dttt = torch.einsum('ijk,lkm->ljm', bspline_t.ddN, t_cps)

        dt = 1. / dtau_dt[..., 0] / dtau_dt.shape[1]
        t_cumsum = torch.cumsum(dt, dim=-1)
        t = torch.sum(dt, dim=-1)

        dtau_dt2 = dtau_dt ** 2
        q_dot = q_dot_tau * dtau_dt
        q_ddot = q_ddot_tau * dtau_dt2 + ddtau_dtt * q_dot_tau * dtau_dt
        q_dddot = q_dddot_tau * dtau_dt ** 3 + 3 * q_ddot_tau * ddtau_dtt * dtau_dt2 + \
                  q_dot_tau * dtau_dt2 * dddtau_dttt + q_dot_tau * ddtau_dtt ** 2 * dtau_dt
        
        return q, q_dot, q_ddot, q_dddot, t_cumsum


def create_plot(mean_, max_, min_, title="title", fig=None, axs=None):
        if fig is None or axs is None:
            fig, axs = plt.subplots(7, 1, figsize=(10, 20), sharex=True)

        for i in range(7):
            axs[i].plot(mean_[:, i], label='Mean')
            # axs[i].plot(max_[:, i], label='Max')
            # axs[i].plot(min_[:, i], label='Min')
            axs[i].set_title(f'Dimension {i+1}')
            axs[i].set_ylabel('Loss')
            axs[i].grid(True)
            axs[i].legend()

        # Set common xlabel
        axs[-1].set_xlabel('Index')
        plt.suptitle(title)

        plt.tight_layout()


def plot_q_losses(q, q_dot, q_ddot, q_dddot):
    q_limits = Limits.q7
    q_dot_limits = Limits.q_dot7
    q_ddot_limits = Limits.q_ddot7
    q_dddot_limits = Limits.q_dddot7

    def compute_loss(loss_values, limits):
        loss_ = torch.relu(torch.abs(loss_values) - limits)
        return loss_

    q_loss = compute_loss(q, q_limits).cpu()
    q_dot_loss = compute_loss(q_dot, q_dot_limits).cpu()
    q_ddot_loss = compute_loss(q_ddot, q_ddot_limits).cpu()
    q_dddot_loss = compute_loss(q_dddot, q_dddot_limits).cpu()

    mean_q_loss = torch.mean(q_loss, dim=0)
    max_q_loss, _ = torch.max(q_loss, dim=0)
    min_q_loss, _ = torch.min(q_loss, dim=0)

    fig, axs = plt.subplots(7, 1, figsize=(10, 20), sharex=True)

    create_plot(mean_q_loss, max_q_loss, min_q_loss, "q", fig, axs)

    mean_q_dot_loss = torch.mean(q_dot_loss, dim=0)
    max_q_dot_loss, _ = torch.max(q_dot_loss, dim=0)
    min_q_dot_loss, _ = torch.min(q_dot_loss, dim=0)

    fig, axs = plt.subplots(7, 1, figsize=(10, 20), sharex=True)

    create_plot(mean_q_dot_loss, max_q_dot_loss, min_q_dot_loss, "q_dot", fig, axs)

    mean_q_ddot_loss = torch.mean(q_ddot_loss, dim=0)
    max_q_ddot_loss, _ = torch.max(q_ddot_loss, dim=0)
    min_q_ddot_loss, _ = torch.min(q_ddot_loss, dim=0)

    fig, axs = plt.subplots(7, 1, figsize=(10, 20), sharex=True)

    create_plot(mean_q_ddot_loss, max_q_ddot_loss, min_q_ddot_loss, "q_ddot", fig, axs)

    mean_q_dddot_loss = torch.mean(q_dddot_loss, dim=0)
    max_q_dddot_loss, _ = torch.max(q_dddot_loss, dim=0)
    min_q_dddot_loss, _ = torch.min(q_dddot_loss, dim=0)

    fig, axs = plt.subplots(7, 1, figsize=(10, 20), sharex=True)

    create_plot(mean_q_dddot_loss, max_q_dddot_loss, min_q_dddot_loss, "q_dddot", fig, axs)

    return {"q": torch.mean(mean_q_loss).item(), "q_dot": torch.mean(mean_q_dot_loss).item(), "q_ddot": torch.mean(mean_q_ddot_loss).item(), "q_dddot": torch.mean(mean_q_dddot_loss).item()}


def plot_times(t_cumsum_1, t_cumsum_10, t_cumsum_100):
    t_cumsum_1, t_cumsum_10, t_cumsum_100 = t_cumsum_1.cpu().numpy(), t_cumsum_10.cpu().numpy(), t_cumsum_100.cpu().numpy()

    t_cumsum_1 *= .52
    t_cumsum_10 *= .52
    t_cumsum_100 *= .52

    mean_1, std_1 = np.mean(t_cumsum_1), np.std(t_cumsum_1)
    mean_10, std_10 = np.mean(t_cumsum_10), np.std(t_cumsum_10)
    mean_100, std_100 = np.mean(t_cumsum_10), np.std(t_cumsum_10)

    # Models and their statistics
    models = ['Model 1', 'Model 10', 'Model 100']
    means = [mean_1, mean_10, mean_100]
    stds = [std_1, std_10, std_100]

    # Plotting with plt.errorbar
    plt.figure(figsize=(10, 6))

    plt.errorbar(models, means, yerr=stds, fmt='o', capsize=5, elinewidth=2, markeredgewidth=2)
    # plt.title('Mean and Standard Deviation of Execution Times per Model')
    plt.ylabel('Time (seconds)')
    plt.grid(axis='y')


def plot_loss(value, limit):
    def compute_loss(loss_values, limits):
       loss_ = torch.relu(torch.abs(loss_values) - limits)
       return loss_

    labels = ["model trained on 400 samples", "model trained on 4k samples", "model trained on 40k samples"]
    fig, axs = plt.subplots(2, 4, figsize=(20, 10), sharex=True)

    for j, v in enumerate(value):
        loss = compute_loss(v, limit).cpu()

        mean_q_loss = torch.mean(loss, dim=0)
        max_q_loss, _ = torch.max(loss, dim=0)
        min_q_loss, _ = torch.min(loss, dim=0)

        for i in range(4):
            axs[0, i].plot(mean_q_loss[:, i], label=labels[j] if i == 0 else None)
            axs[0, i].set_ylabel('Loss')
            axs[0, i].grid(True)

        for i in range(3):
            axs[1, i].plot(mean_q_loss[:, i+4], label=labels[j] if i == 1000 else None)
            axs[1, i].set_title(f'Dimension {i+5}')
            axs[1, i].set_ylabel('Loss')
            axs[1, i].grid(True)

    fig.legend(loc='center', shadow=True, ncol=1, bbox_to_anchor=(0.882, 0.3), fontsize='x-large')

    # Set common xlabel
    axs[-1, 0].set_xlabel('Index')
    axs[1, -1].set_visible(False)

    plt.tight_layout()
    plt.show()
    

     

def main():
    Limits.to_device(device)
    torch.device(device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor)
    features = load_data(filename="data.tsv", n=5000)
    
    manipulator = DifferentiableRobotModel(urdf_path=Config.manipulator.urdf_file,
                                           tensor_args={'device': device, 'dtype': torch.float32})

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Model_1%.pt")
    model_1 = torch.load(model_path, map_location=torch.device(device))

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Model_10%.pt")
    model_10 = torch.load(model_path, map_location=torch.device(device))

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Model_100%.pt")
    model_100 = torch.load(model_path, map_location=torch.device(device))

    # trajectories = [interpolate_control_points(q_cp.unsqueeze(0), t_cp.unsqueeze(0)) for q_cp, t_cp in zip(q_cps, t_cps)]
    # cmds = []
    # times = []
    # for entry in trajectories:
    #     for item in entry:
    #         if isinstance(item, np.ndarray):
    #             cmds.append(item)
    #         elif isinstance(item, float):
    #             times.append(item)
    # bar_plot_times(times)

    q_cps_1, t_cps_1 = compute_control_points(model_1, features[:, :42])
    q_cps_10, t_cps_10 = compute_control_points(model_10, features[:, :42])
    q_cps_100, t_cps_100 = compute_control_points(model_100, features[:, :42])

    loss_fn = HittingLoss(manipulator, air_hockey_table, air_hockey_puck, bspline_q, bspline_t, Limits.q7, Limits.q_dot7, Limits.q_ddot7, Limits.q_dddot7, device=device)

    loss = loss_fn(q_cps_10, t_cps_10, features[:, -2:])
    loss0 = loss[0]

    q1, q_dot1, q_ddot1, q_dddot1, t_cumsum_1 = compute_trajectory(q_cps_1, t_cps_1)
    #losses = plot_q_losses(q, q_dot, q_ddot, q_dddot)
    #print("1%: ", losses)

    q10, q_dot10, q_ddot10, q_dddot10, t_cumsum_10 = compute_trajectory(q_cps_10, t_cps_10)
    #losses = plot_q_losses(q, q_dot, q_ddot, q_dddot)
    #print("10%: ", losses)

    q100, q_dot100, q_ddot100, q_dddot100, t_cumsum_100 = compute_trajectory(q_cps_100, t_cps_100)
    #losses = plot_q_losses(q, q_dot, q_ddot, q_dddot)
    #print("100%: ", losses)

    # plot_times(t_cumsum_1[:, -1], t_cumsum_10[:, -1], t_cumsum_100[:, -1])

    plot_loss([q1, q10, q100], Limits.q7)

    plt.show()

    plot_loss([q_dot1, q_dot10, q_dot100], Limits.q_dot7)

    plt.show()

    plot_loss([q_ddot1, q_ddot10, q_ddot100], Limits.q_ddot7)

    plt.show()
    
    plot_loss([q_dddot1, q_dddot10, q_dddot100], Limits.q_ddot7)
    
    plt.show()


def plot_tactics():
    # Create figure and axis object
    fig, axes = plt.subplots(1, 5, figsize=(12, 3), sharey=True)

    # Define the x values for the vertical lines
    x_inner = 0.519
    x_outer = 0.609

    x_prepare_range = 0.39105
    x_defend_range = 0.45

    # Define the y value for the horizontal lines
    y_inner = 0.974
    y_outer = 1.034

    y_mid = 0.

    y_hit_range_min = 0.8 - 1.51
    y_hit_range_max = 1.3 - 1.51

    y_defend_range_min = 0.8 - 1.51
    y_defend_range_max = 1.0 - 1.51

    # Plot each subplot
    for ax in axes[:-1]:
        # plot horizontal lines at bottom and mid
        ax.hlines(y=-y_inner, color='dimgrey', linestyle='-', xmin=-x_inner, xmax=x_inner)
        ax.hlines(y=-y_outer, color='dimgrey', linestyle='-', xmin=-x_outer, xmax=x_outer)
        ax.hlines(y=-y_mid, color='silver', linestyle='--', xmin=-0.7, xmax=0.7)

        # plot vertical lines at table sides
        ax.vlines(x=-x_inner, color='dimgrey', ymin=-y_inner, ymax=y_inner)
        ax.vlines(x=-x_outer, color='dimgrey', ymin=-y_outer, ymax=y_outer)
        ax.vlines(x=x_inner, color='dimgrey', ymin=-y_inner, ymax=y_inner)
        ax.vlines(x=x_outer, color='dimgrey', ymin=-y_outer, ymax=y_outer)

        ax.set_xlim(-0.75, 0.75)
        ax.set_ylim(-1.2, 0.2)

        ax.axis('off')

    # plot first plot: tactic distribution

    axes[0].fill_between([-x_prepare_range, x_prepare_range], y_hit_range_min, y_hit_range_max, hatch='/', edgecolor='red', facecolor='none', joinstyle='miter', label='smash range')
    axes[0].fill_between([-x_defend_range, x_defend_range], y_defend_range_min, y_defend_range_max, hatch='\\', edgecolor='blue', facecolor='none', joinstyle='miter', label='defend range')

    axes[0].fill_between([-x_prepare_range, -x_inner], -y_inner, y_hit_range_max, hatch='/', edgecolor='green', facecolor='none', joinstyle='miter', label='prepare range')
    axes[0].fill_between([x_prepare_range, x_inner], -y_inner, y_hit_range_max, hatch='/', edgecolor='green', facecolor='none', joinstyle='miter')
    axes[0].fill_between([-x_prepare_range, x_prepare_range], -y_inner, y_hit_range_min, hatch='/', edgecolor='green', facecolor='none', joinstyle='miter')

    axes[0].fill_between([-x_inner, x_inner], y_hit_range_max, y_mid, hatch='/', edgecolor='silver', facecolor='none', joinstyle='miter', label='not reachable')

    axes[0].set_title("Tactic distribution")

    # plot second plot: hitting motion

    axes[1].hlines(y=y_hit_range_max, color='red', linestyle='--', xmin=-x_prepare_range, xmax=x_prepare_range)
    axes[1].hlines(y=y_hit_range_min, color='red', linestyle='--', xmin=-x_prepare_range, xmax=x_prepare_range)
    axes[1].vlines(x=-x_prepare_range, color='red', linestyle='--', ymin=y_hit_range_min, ymax=y_hit_range_max)
    axes[1].vlines(x=x_prepare_range, color='red', linestyle='--', ymin=y_hit_range_min, ymax=y_hit_range_max)

    x = np.arange(-x_prepare_range + 0.05, x_prepare_range - 0.05, (x_prepare_range + x_prepare_range) / 7)
    y = np.arange(y_hit_range_min + 0.01, y_hit_range_max - 0.01, .1)
    X, Y = np.meshgrid(x, y)

    common_point = np.array([0., 0.974])
    U = common_point[0] - X
    V = common_point[1] - Y

    axes[1].quiver(X, Y, U, V, label="direction of hit")
    axes[1].set_title("Tactic Smash")

    # plot third plot: defence motion

    axes[2].hlines(y=y_defend_range_max, color='blue', linestyle='--', xmin=-x_defend_range, xmax=x_defend_range)
    axes[2].hlines(y=y_defend_range_min, color='blue', linestyle='--', xmin=-x_defend_range, xmax=x_defend_range)
    axes[2].vlines(x=-x_defend_range, color='blue', linestyle='--', ymin=y_defend_range_min, ymax=y_defend_range_max)
    axes[2].vlines(x=x_defend_range, color='blue', linestyle='--', ymin=y_defend_range_min, ymax=y_defend_range_max)

    x = np.arange(-x_prepare_range + 0.05, 0, (x_prepare_range + x_prepare_range) / 7)
    y = np.arange(y_defend_range_min + 0.05, y_defend_range_max - 0.01, .1)
    X, Y = np.meshgrid(x, y)
    
    wind_direction = np.array([1, 0])
    U = wind_direction[0] * -np.ones_like(X)
    V = wind_direction[1] * -np.ones_like(Y)
    axes[2].quiver(X, Y, U, V)

    x = np.arange(0, x_prepare_range - 0.05, (x_prepare_range + x_prepare_range) / 7)
    y = np.arange(y_defend_range_min + 0.05, y_defend_range_max - 0.01, .1)
    X, Y = np.meshgrid(x, y)
    
    wind_direction = np.array([1, 0])
    U = wind_direction[0] * np.ones_like(X)
    V = wind_direction[1] * np.ones_like(Y)
    axes[2].quiver(X, Y, U, V)

    axes[2].set_title("Tactic Defend")

    # plot fourth plot: prepare motion

    axes[3].hlines(y=y_hit_range_min, color='green', linestyle='--', xmin=-x_inner, xmax=x_inner)
    axes[3].hlines(y=y_hit_range_max, color='green', linestyle='--', xmin=-x_inner, xmax=-x_prepare_range)
    axes[3].hlines(y=y_hit_range_max, color='green', linestyle='--', xmin=x_prepare_range, xmax=x_inner)
    axes[3].vlines(x=-x_prepare_range, color='green', linestyle='--', ymin=-y_inner, ymax=y_hit_range_max)
    axes[3].vlines(x=x_prepare_range, color='green', linestyle='--', ymin=-y_inner, ymax=y_hit_range_max)

    x = np.array([.425, -.425])
    y = np.array([-0.25, -0.35, -0.45, -0.55, -0.65, -0.75, -0.85])
    X, Y = np.meshgrid(x, y)
    
    wind_direction = np.array([1, 0])
    U = wind_direction[0] * np.ones_like(X)
    U[:, 1] *= -1
    V = np.zeros_like(Y)
    V[:2] = -.5
    V[-2:] = -1
    axes[3].quiver(X, Y, U, V)

    axes[3].hlines(y=y_hit_range_min, color='green', linestyle='--', xmin=-x_inner, xmax=x_inner)
    axes[3].hlines(y=y_hit_range_max, color='green', linestyle='--', xmin=-x_inner, xmax=-x_prepare_range)
    axes[3].hlines(y=y_hit_range_max, color='green', linestyle='--', xmin=x_prepare_range, xmax=x_inner)
    axes[3].vlines(x=-x_prepare_range, color='green', linestyle='--', ymin=-y_inner, ymax=y_hit_range_max)
    axes[3].vlines(x=x_prepare_range, color='green', linestyle='--', ymin=-y_inner, ymax=y_hit_range_max)

    x = np.array([.325, .225, -.225, -.325])
    y = np.array([-0.75, -0.85])
    X, Y = np.meshgrid(x, y)

    wind_direction = np.array([0, -1])
    U = wind_direction[0] * np.ones_like(X)
    V = wind_direction[1] * np.ones_like(Y)
    axes[3].quiver(X, Y, U, V)

    axes[3].fill_between([-.15, .15], -y_inner, y_hit_range_min, hatch='/', edgecolor='silver', facecolor='none', joinstyle='miter')

    axes[3].set_title("Tactic Prepare")

    # plot legend

    axes[4].axis('off')
    fig.legend(loc='center', shadow=True, ncol=1, bbox_to_anchor=(0.882, 0.5), fontsize='x-large')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def load_arrays_from_npz(filename):
    loaded_data = np.load(filename)
    arrays = [loaded_data[key] for key in loaded_data.files]
    return arrays


def plot_results():
    bars = [1, 2, 3]
    values = [886, 946, 837]
    height = 1000

    fig, ax = plt.subplots()

    for bar, value in zip(bars, values):
        ax.bar(bar, value, color='blue', alpha=.7, align='center')
        ax.bar(bar, height-value, color='red', alpha=.7, align='center', bottom=value)

    plt.grid(axis='y')

    ax.set_ylabel('Number of successes')

    ax.set_xticks(bars)
    ax.set_xticklabels(['Attack', 'Defend', 'Prepare'])

    plt.show()


    
if __name__ == '__main__':
    plot_results()
    # main()

    pass
