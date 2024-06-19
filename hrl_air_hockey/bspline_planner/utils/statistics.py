import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from config import Config
from data.generate_data import load_data
from differentiable_robot_model import DifferentiableRobotModel
from losses.constraint_functions import air_hockey_table
from losses.hitting import HittingLoss
from utils.bspline import BSpline
from utils.constants import Limits


def iterate_data(model, loss_fn, data_loader):
    losses = torch.zeros(size=(0,))
    for i, data in enumerate(data_loader):
        q_cps, t_cps = model(data)
        model_loss, sum_constraint_loss, sum_q_dot_loss, sum_q_ddot_loss, sum_q_dddot_loss, q, q_dot, \
        q_ddot, q_dddot, t, t_cumsum, t_loss, dt, x_loss, y_loss, z_loss = loss_fn(q_cps, t_cps)

        iteration_loss = torch.stack([sum_q_dot_loss, sum_q_ddot_loss, sum_constraint_loss, x_loss, y_loss, z_loss, dt])
        losses = torch.cat((losses, iteration_loss), dim=1)

    return losses


def do_inference(model, loss_fn, data_loader):
    with torch.no_grad():
        for i, data in enumerate(data_loader):

            q_cps, t_cps = model(data)
            model_loss, sum_constraint_loss, sum_q_dot_loss, sum_q_ddot_loss, sum_q_dddot_loss, q, q_dot, q_ddot, \
            q_dddot, t, t_cumsum, t_loss, dt, x_loss, y_loss, z_loss = loss_fn(q_cps, t_cps)

    return q, q_dot, q_ddot, q_dddot, t, t_cumsum


def main():
    device = Config.train.device
    torch.device(Config.train.device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor)

    # Create manipulator model
    manipulator = DifferentiableRobotModel(urdf_path=Config.manipulator.urdf_file,
                                           tensor_args={'device': device, 'dtype': torch.float32})

    # Generate B-Spline Parameters
    bspline_q = BSpline(num_pts=Config.bspline_q.n_ctr_pts, degree=Config.bspline_q.degree,
                        num_T_pts=Config.bspline_q.num_T_pts, device=device)
    bspline_t = BSpline(num_pts=Config.bspline_t.n_ctr_pts, degree=Config.bspline_t.degree,
                        num_T_pts=Config.bspline_t.num_T_pts, device=device)

    # Generate data for training
    training_loader, validation_loader = load_data(200,
                                                   device,
                                                   path=os.path.join(
                                                       os.path.abspath(os.getcwd()),
                                                       "../datasets/train/data.tsv"),
                                                   shuffle=False)
    # training_loader, validation_loader = get_random_data(n_samples=20000, dev_ratio=.1, batch_size=128, device=device)
    Limits.to_device(device)

    # Initialize the model
    checkpoint_path = os.path.join(os.path.abspath(os.getcwd()), '..\\model\\checkpoints\\Model.pt')
    model = torch.load(checkpoint_path).to(device)

    loss_fn = HittingLoss(manipulator, air_hockey_table, bspline_q, bspline_t, Limits.q_dot7, Limits.q_ddot7, Limits.q_dddot7)

    validation_loss = iterate_data(model, loss_fn, validation_loader).cpu().detach().numpy()
    train_loss = iterate_data(model, loss_fn, training_loader).cpu().detach().numpy()

    for i, name in enumerate(["q_dot loss", "q_ddot loss", "constraint loss", "x loss", "y loss", "z loss"]):
        data = validation_loss[i]
        plt.hist(data, bins=30)
        plt.title("validation " + name)
        plt.savefig("../graphics/loss_" + "validation_" + name.replace(" ", "_"))
        plt.show()

    for i, name in enumerate(["q_dot loss", "q_ddot loss", "constraint loss", "x loss", "y loss", "z loss"]):
        data = train_loss[i]
        plt.hist(data, bins=30)
        plt.title("training " + name)
        plt.savefig("../graphics/loss_" + "training_" + name.replace(" ", "_"))
        plt.show()

    dt = train_loss[-1]
    for i, name in enumerate(["q_dot loss", "q_ddot loss", "constraint loss", "x loss", "y loss", "z loss"]):
        data = train_loss[i]
        for d, dti in zip(data, dt):
            plt.plot(dti, d, markersize=.1)
            plt.savefig("../graphics/loss_" + "training_" + name.replace(" ", "_"))
        plt.show()


if __name__ == '__main__':
    main()
