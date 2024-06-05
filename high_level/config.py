import os
import torch


class Manipulator:
    urdf_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iiwa_only.urdf")


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
    num_epochs = 35000
    batch_size = 128 // 2 # 400: // 8, 4000: // 2, 40000: * 4
    learning_rate = 5e-6
    weight_decay = 1e-10
    epsilon = 1e-9


class Model:
    n_pts_fixed_begin = 3
    n_pts_fixed_end = 2
    n_dof = 7


class WandB:
    os.environ["WANDB_API_KEY"] = 'a903361ff1d9498b25c276d46a0dcc63fe596aca'
    project_name = 'hrl'


class Plots:
    enable = False
    interval = 100


class Save:
    enable = True
    interval = 5000
    path = os.path.join(os.path.abspath(os.getcwd()), "model/checkpoints_defend/")


class Load:
    enable = False	
    epoch = 0


class Data:
    path = os.path.join(os.path.abspath(os.getcwd()), "datasets/train/data.tsv")


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
