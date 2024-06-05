import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from config import Config

data_dir = os.path.abspath(os.path.join(parent_dir, "data"))
sys.path.append(data_dir)

from data.load_data import get_hitting_data

data_dir = os.path.abspath(os.path.join(parent_dir, "differentiable_robot_model"))
sys.path.append(data_dir)

from differentiable_robot_model.differentiable_robot_model import DifferentiableRobotModel

data_dir = os.path.abspath(os.path.join(parent_dir, "losses"))
sys.path.append(data_dir)

from losses.constraint_functions import air_hockey_table
from losses.hitting import HittingLoss

data_dir = os.path.abspath(os.path.join(parent_dir, "utils"))
sys.path.append(data_dir)

from utils.bspline import BSpline
from utils.constants import Limits, TableConstraint

data_dir = os.path.abspath(os.path.join(parent_dir, "train.py"))
sys.path.append(data_dir)

from train import train_epoch

data_dir = os.path.abspath(os.path.join(parent_dir, "utils"))
sys.path.append(data_dir)

from utils.plotter import plot_end_effector, plot_splines


def evaluate_on_own_data():
    device = Config.train.device
    print('Device: {}'.format(Config.train.device))
    torch.device(Config.train.device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor)

    train_valid_path = Config.data.path
    training_loader, validation_loader = get_hitting_data(batch_size=Config.train.batch_size, device=device, path=train_valid_path)
    Limits.to_device(device)
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Model_final.pt")
    model = torch.load(model_path, map_location=torch.device(device))
    optimizer = None
    
    manipulator = DifferentiableRobotModel(urdf_path=Config.manipulator.urdf_file,
                                           tensor_args={'device': device, 'dtype': torch.float32})
    
    bspline_q = BSpline(num_pts=Config.bspline_q.n_ctr_pts, degree=Config.bspline_q.degree,
                        num_T_pts=Config.bspline_q.num_T_pts, device=device)
    bspline_t = BSpline(num_pts=Config.bspline_t.n_ctr_pts, degree=Config.bspline_t.degree,
                        num_T_pts=Config.bspline_t.num_T_pts, device=device)
    
    loss_fn = HittingLoss(manipulator, air_hockey_table, bspline_q, bspline_t, Limits.q_dot7, Limits.q_ddot7, Limits.q_dddot7)

    losses = train_epoch(model, loss_fn, optimizer, training_loader, device, is_training=False)
    print("Training losses")
    print(losses)

    losses = train_epoch(model, loss_fn, optimizer, validation_loader, device, is_training=False)
    print("Validation losses")
    print(losses)

    test_path = train_valid_path.replace("data.tsv", "test_data.tsv")
    training_loader, validation_loader = get_hitting_data(batch_size=Config.train.batch_size, device=device, path=test_path, split=.99999)

    losses = train_epoch(model, loss_fn, optimizer, training_loader, device, is_training=False)
    print("Test losses own")
    print(losses)

    test_path = train_valid_path.replace("data.tsv", "test_data_challenge.tsv")
    training_loader, validation_loader = get_hitting_data(batch_size=Config.train.batch_size, device=device, path=test_path, split=.99999)

    losses = train_epoch(model, loss_fn, optimizer, training_loader, device, is_training=False)
    print("Test losses challenge")
    print(losses)


def plot_sample_distribution():
    dataset_path = Config.data.path
    data = np.loadtxt(dataset_path, delimiter='\t').astype(np.float32)[:20000]

    # Create a DataFrame with the specified column names
    column_names = ['q_0_i', 'q_1_i', 'q_2_i', 'q_3_i', 'q_4_i', 'q_5_i', 'q_6_i',
                    'qdot_0_i', 'qdot_1_i', 'qdot_2_i', 'qdot_3_i', 'qdot_4_i', 'qdot_5_i', 'qdot_6_i',
                    'qddot_0_i', 'qddot_1_i', 'qddot_2_i', 'qddot_3_i', 'qddot_4_i', 'qddot_5_i', 'qddot_6_i',
                    'q_0_f', 'q_1_f', 'q_2_f', 'q_3_f', 'q_4_f', 'q_5_f', 'q_6_f',
                    'qdot_0_f', 'qdot_1_f', 'qdot_2_f', 'qdot_3_f', 'qdot_4_f', 'qdot_5_f', 'qdot_6_f',
                    'qddot_0_f', 'qddot_1_f', 'qddot_2_f', 'qddot_3_f', 'qddot_4_f', 'qddot_5_f', 'qddot_6_f']

    df = pd.DataFrame(data, columns=column_names)

    test_path = dataset_path.replace("data.tsv", "test_data_challenge.tsv")
    data_challenge = np.loadtxt(test_path, delimiter='\t').astype(np.float32)[:2000]

    df_challenge = pd.DataFrame(data_challenge, columns=column_names)

    def plot_histogram(x, **kwargs):
        plt.hist(x, bins=20, **kwargs)

    for i in range(0, 42, 7):
        df_filtered = df.iloc[:, i:i+7]
        df_challenge_filtered = df_challenge.iloc[:, i:i+7]

        sns.set(style="ticks")

        # Create PairGrid for df
        g = sns.PairGrid(df_filtered)
        # g.map_lower(sns.scatterplot, alpha=.5, color='blue', size=5)
        # g.map_diag(plot_histogram, alpha=.5, color='blue')

        g.data = df_challenge_filtered
        g.map_lower(sns.scatterplot, alpha=.5, color='red', size=5)
        g.map_diag(plot_histogram, alpha=.5, color='red')

        # Combine PairGrids
        g.fig.suptitle('Comparison between df and df_challenge')
        plt.show()


def compare_closest_points():
    def calculate_l2_distance(point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    train_path = Config.data.path
    train_data = np.loadtxt(train_path, delimiter='\t').astype(np.float32)

    challenge_path = train_path.replace("data.tsv", "test_data_challenge.tsv")
    challenge_data = np.loadtxt(challenge_path, delimiter='\t').astype(np.float32)[:2000]
    
    distances = []
    max_distance = 0
    max_points = []
    min_distance = np.inf
    min_points = []
    for i, challenge_point in enumerate(challenge_data):
        print(i)
        l2_distances = np.array([calculate_l2_distance(challenge_point, train_point) for train_point in train_data])
        
        closest_index = np.argmin(l2_distances)
        
        distances.append(l2_distances[closest_index])

        if distances[-1] > max_distance:
            print(max_distance)
            max_distance = distances[-1]
            max_points = [train_data[closest_index], challenge_point]

        if distances[-1] < min_distance:
            min_distance = distances[-1]
            min_points = [train_data[closest_index], challenge_point]

    mean_distance = np.mean(distances)
    max_distance = np.max(distances)
    min_distance = np.min(distances)

    print(f"Mean Distance: {mean_distance}")
    print(f"Max Distance: {max_distance}")
    print(f"Min Distance: {min_distance}")

    print(f"Max Points: {max_points}")
    print(f"Min Points: {min_points}")


def plot_challenge_traj():
    device = Config.train.device
    print('Device: {}'.format(Config.train.device))
    torch.device(Config.train.device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor)

    train_valid_path = Config.data.path
    training_loader, validation_loader = get_hitting_data(batch_size=500, device=device, path=train_valid_path)
    Limits.to_device(device)
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Model_final.pt")
    model = torch.load(model_path, map_location=torch.device(device))
    optimizer = None
    
    manipulator = DifferentiableRobotModel(urdf_path=Config.manipulator.urdf_file,
                                           tensor_args={'device': device, 'dtype': torch.float32})
    
    bspline_q = BSpline(num_pts=Config.bspline_q.n_ctr_pts, degree=Config.bspline_q.degree,
                        num_T_pts=Config.bspline_q.num_T_pts, device=device)
    bspline_t = BSpline(num_pts=Config.bspline_t.n_ctr_pts, degree=Config.bspline_t.degree,
                        num_T_pts=Config.bspline_t.num_T_pts, device=device)
    
    loss_fn = HittingLoss(manipulator, air_hockey_table, bspline_q, bspline_t, Limits.q_dot7, Limits.q_ddot7, Limits.q_dddot7)


    plot_end_effector(manipulator, model, validation_loader, bspline_q, device=device)
    plot_splines(model, bspline_q, bspline_t, validation_loader, device=device)


def plot_puck_pos():
    plt.vlines(TableConstraint.XLB, TableConstraint.YLB, TableConstraint.YRT, colors='black', linestyles='solid', label='max')
    plt.vlines(TableConstraint.XRT, TableConstraint.YLB, TableConstraint.YRT, colors='black', linestyles='solid', label='max')
    plt.hlines(TableConstraint.YLB, TableConstraint.XLB, TableConstraint.XRT, colors='black', linestyles='solid', label='max')
    plt.hlines(TableConstraint.YRT, TableConstraint.XLB, TableConstraint.XRT, colors='black', linestyles='solid', label='max')

    data = np.loadtxt(Config.data.path, delimiter='\t').astype(np.float32)[30000:40000]
    qf = data[:, 21:28]

    device = Config.train.device
    manipulator = DifferentiableRobotModel(urdf_path=Config.manipulator.urdf_file,
                                           tensor_args={'device': device, 'dtype': torch.float32})
    
    qf_tensor = torch.from_numpy(qf).to(device)

    xyz = manipulator.compute_forward_kinematics(qf_tensor, torch.zeros_like(qf_tensor), link_name='iiwa_1/striker_tip')[0].cpu().numpy()
    print(xyz.shape)

    plt.scatter(xyz[:, 0], xyz[:, 1])
    plt.show()


if __name__ == '__main__':
    # compare_closest_points()
    # evaluate_on_own_data()
    # plot_sample_distribution()
    # plot_challenge_traj()
    plot_puck_pos()
