import os
import torch


class Manipulator:
    urdf_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hrl_air_hockey/bspline_planner/data/iiwa_only.urdf")


class BSplineT:
    n_ctr_pts = 20
    degree = 7
    num_T_pts = 200


class BSplineQ:
    n_ctr_pts = 20
    degree = 7
    num_T_pts = 200


class TrainConfig:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_epochs = 1200 * 2
    batch_size = 128 // 2 # 400: // 8, 4000: // 2, 40000: * 4
    learning_rate = 5e-6
    weight_decay = 1e-10
    epsilon = 1e-9


class Model:
    n_pts_fixed_begin = 3
    n_pts_fixed_end = 2
    n_dof = 7


class WandB:
    api_key = "a903361ff1d9498b25c276d46a0dcc63fe596aca"
    project_name = "neural_planner"
    # continue_id = 'lo8zjlre'
    continue_id = None


class Plots:
    enable = False
    interval = 100


class Save:
    enable = True
    interval = 1200 * 2
    current_folder = os.getcwd()
    parent_folder = os.path.dirname(current_folder)
    path = os.path.join(parent_folder, 'trained_low_agent', "model")


class Load:
    enable = True
    epoch = 1200


class Data:
    hit_path = os.path.join(os.path.abspath(os.getcwd()), "datasets/train/data.tsv")
    uniform_path = os.path.join(os.path.abspath(os.getcwd()), "datasets/uniform_train/data.tsv")


class Config:
    train = TrainConfig()
    bspline_q = BSplineQ()
    bspline_t = BSplineT()
    manipulator = Manipulator()
    model = Model()
    wandb = WandB()
    plots = Plots()
    save = Save()
    load = Load()
    data = Data()
