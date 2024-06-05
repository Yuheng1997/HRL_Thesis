import sys
import matplotlib.pyplot as plt
import torch
import os
import numpy as np


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

data_dir = os.path.abspath(os.path.join(parent_dir, "data"))
sys.path.append(data_dir)

from data.load_data import get_hitting_data, load_data
from differentiable_robot_model import DifferentiableRobotModel
from utils.bspline import BSpline
from utils.constants import Limits, TableConstraint


def plot_end_effector(manipulator, model, validation_loader, bsp, device='cpu'):
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            if i == 1:
                return
            
            data = data.to(device)
            q_cps, t_cps = model(data)
            q_cps, t_cps = q_cps.to(torch.float32), t_cps.to(torch.float32)

            q = torch.einsum('ijk,lkm->ljm', bsp.N, q_cps)

            batch_size, num_T_points, n_dof = q.shape
            positions = manipulator.my_compute_forward_kinematics(
                torch.reshape(q, (batch_size * num_T_points, n_dof)), link_name='iiwa_1/striker_tip')
            positions = torch.reshape(positions, (batch_size, num_T_points, 3))

            positions = positions.cpu().detach().numpy()

            savepath = os.path.join(os.getcwd(), './graphics/tableConstraints_{}_{}')

            for j, pos in enumerate(positions):
                fig, axes = plt.subplots(1, 2)
                axes[0].plot(pos[:, 0], pos[:, 1])
                axes[0].set_xlabel('x')
                axes[0].set_ylabel('y')
                axes[0].vlines(TableConstraint.XLB, TableConstraint.YLB, TableConstraint.YRT, colors='black', linestyles='solid', label='max')
                axes[0].vlines(TableConstraint.XRT, TableConstraint.YLB, TableConstraint.YRT, colors='black', linestyles='solid', label='max')
                axes[0].hlines(TableConstraint.YLB, TableConstraint.XLB, TableConstraint.XRT, colors='black', linestyles='solid', label='max')
                axes[0].hlines(TableConstraint.YRT, TableConstraint.XLB, TableConstraint.XRT, colors='black', linestyles='solid', label='max')
                axes[1].plot(pos[:, 0], pos[:, 2])
                axes[1].set_xlabel('x')
                axes[1].set_ylabel('z')
                axes[1].hlines(TableConstraint.Z, TableConstraint.XLB, TableConstraint.XRT, colors='black', linestyles='solid', label='max')

                plt.savefig(savepath.format(i, j))
                plt.close()


def plot_splines(model, bsp, bsp_t, validation_loader, device='cpu'):
    with torch.no_grad():
        for k, data in enumerate(validation_loader):
            if k == 1:
                return
            
            data = data.to(device)

            q_cps, t_cps = model(data)
            q_cps, t_cps = q_cps.to(torch.float32), t_cps.to(torch.float32)

            q = torch.einsum('ijk,lkm->ljm', bsp.N, q_cps)
            q_dot_tau = torch.einsum('ijk,lkm->ljm', bsp.dN, q_cps)
            q_ddot_tau = torch.einsum('ijk,lkm->ljm', bsp.ddN, q_cps)
            q_dddot_tau = torch.einsum('ijk,lkm->ljm', bsp.dddN, q_cps)

            dtau_dt = torch.einsum('ijk,lkm->ljm', bsp_t.N, t_cps)
            ddtau_dtt = torch.einsum('ijk,lkm->ljm', bsp_t.dN, t_cps)
            dddtau_dttt = torch.einsum('ijk,lkm->ljm', bsp_t.ddN, t_cps)

            dt = 1. / dtau_dt[..., 0] / dtau_dt.shape[1]
            t_cumsum = torch.cumsum(dt, dim=-1)
            t = torch.sum(dt, dim=-1)

            q_dot = q_dot_tau * dtau_dt
            q_ddot = q_ddot_tau * dtau_dt ** 2 + ddtau_dtt * q_dot_tau * dtau_dt
            q_dddot = q_dddot_tau * dtau_dt ** 3 + 3 * q_ddot_tau * ddtau_dtt * dtau_dt ** 2 + \
                      q_dot_tau * dtau_dt ** 2 * dddtau_dttt + q_dot_tau * ddtau_dtt ** 2 * dtau_dt

            dt_ = t_cumsum.cpu().detach().numpy()
            q_ = q[:, :].cpu().detach().numpy()
            q_dot_ = q_dot[:, :].cpu().detach().numpy()
            q_ddot_ = q_ddot[:, :].cpu().detach().numpy()
            q_dddot_ = q_dddot[:, :].cpu().detach().numpy()

            savepath = os.path.join(os.getcwd(), './graphics/Controls{}_{}')

            j = 0
            for dt, q, q_dot, q_ddot, q_dddot in zip(dt_, q_, q_dot_, q_ddot_, q_dddot_):
                dt, q, q_dot, q_ddot, q_dddot = np.transpose(dt), np.transpose(q), np.transpose(q_dot), \
                                                np.transpose(q_ddot), np.transpose(q_dddot)

                fig, axes = plt.subplots(4, 7)

                for i, y in enumerate(q):
                    axes[0, i].hlines(Limits.q7.cpu()[i], dt[0], dt[-1], colors=None, linestyles='solid', label='max')
                    axes[0, i].hlines(-Limits.q7.cpu()[i], dt[0], dt[-1], colors=None, linestyles='solid', label='min')
                    axes[0, i].plot(dt, y)

                for i, y in enumerate(q_dot):
                    axes[1, i].hlines(Limits.q_dot7.cpu()[i] / .8, dt[0], dt[-1], colors=None, linestyles='solid', label='max')
                    axes[1, i].hlines(-Limits.q_dot7.cpu()[i] / .8, dt[0], dt[-1], colors=None, linestyles='solid', label='min')
                    axes[1, i].plot(dt, y)

                for i, y in enumerate(q_ddot):
                    axes[2, i].hlines(Limits.q_ddot7.cpu()[i], dt[0], dt[-1], colors=None, linestyles='solid', label='max')
                    axes[2, i].hlines(-Limits.q_ddot7.cpu()[i], dt[0], dt[-1], colors=None, linestyles='solid', label='min')
                    axes[2, i].plot(dt, y)

                for i, y in enumerate(q_dddot):
                    axes[3, i].hlines(Limits.q_dddot7.cpu()[i], dt[0], dt[-1], colors=None, linestyles='solid', label='max')
                    axes[3, i].hlines(-Limits.q_dddot7.cpu()[i], dt[0], dt[-1], colors=None, linestyles='solid', label='min')
                    axes[3, i].plot(dt, y)

                plt.savefig(savepath.format(k, j))
                plt.close()
                j = j + 1


if __name__ == '__main__':
    device = torch.device('cpu')
    urdf_file = os.path.join(os.getcwd(), "iiwa_only.urdf")
    manipulator = DifferentiableRobotModel(urdf_path=urdf_file, tensor_args={'device': device, 'dtype': torch.float32})

    # set joint-spline parameters
    n_ctr_pts_q = 15
    degree_q = 7
    num_T_pts_q = 200

    # set time-spline parameters
    n_ctr_pts_t = 20
    degree_t = 7
    num_T_pts_t = 200

    n_pts_fixed_begin = 3
    n_pts_fixed_end = 2
    n_dof = 7

    bspline_q = BSpline(num_pts=n_ctr_pts_q, degree=degree_q, num_T_pts=num_T_pts_q, device=device)
    bspline_t = BSpline(num_pts=n_ctr_pts_t, degree=degree_t, num_T_pts=num_T_pts_t, device=device)

    checkpoint_path = os.path.join(os.path.abspath(os.getcwd()), 'model\\checkpoints\\Model_11000.pt')

    model = torch.load(checkpoint_path).to(device)
    model.to_device(device)
    # model = NNPlanner(n_ctr_pts_q, n_ctr_pts_t, bspline_q, bspline_t, n_pts_fixed_begin, n_pts_fixed_end, n_dof)

    # checkpoint = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    training_loader, validation_loader = get_hitting_data(batch_size=128, device='cpu', path=os.path.join(
                                                       os.path.abspath(os.getcwd()),
                                                       "datasets/train/defend_data_5000.tsv"), shuffle=True, split=.9)

    plot_end_effector(manipulator, model, validation_loader, bspline_q)
    plot_splines(model, bspline_q, bspline_t, validation_loader)
