import torch
from torch import nn
import numpy as np
import pinocchio as pino

from losses.huber import huber
from utils.manipulator import Iiwa


class FeasibilityLoss(nn.Module):
    def __init__(self, bsp, bsp_t, urdf_path, q_dot_limits, q_ddot_limits, q_dddot_limits, torque_limits):
        super(FeasibilityLoss, self).__init__()
        self.bsp_t = bsp_t
        self.bsp = bsp
        self.q_dot_limits = q_dot_limits
        self.q_ddot_limits = q_ddot_limits
        self.q_dddot_limits = q_dddot_limits
        self.torque_limits = torque_limits
        self.iiwa = Iiwa(urdf_path)
        self.model = pino.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

    # TODO: Check if @tf.custom_gradient is met
    def rnea(self, q, dq, ddq):
        torque = torch.zeros_like(q)
        n = q.shape[-1]
        dtorque_q = torch.zeros(q.shape + (n,))
        dtorque_dq = torch.zeros(q.shape + (n,))
        dtorque_ddq = torch.zeros(q.shape + (n,))
        q_ = q.detach().numpy() if isinstance(q, torch.Tensor) else q
        dq_ = dq.detach().numpy() if isinstance(dq, torch.Tensor) else dq
        ddq_ = ddq.detach().numpy() if isinstance(ddq, torch.Tensor) else ddq
        q_ = np.pad(q_, ((0, 0), (0, 0), (0, self.model.nq - q_.shape[-1])))
        dq_ = np.pad(dq_, ((0, 0), (0, 0), (0, self.model.nq - dq_.shape[-1])))
        ddq_ = np.pad(ddq_, ((0, 0), (0, 0), (0, self.model.nq - ddq_.shape[-1])))

        def grad_fn(upstream):
            g_q = torch.zeros_like(q)
            g_dq = torch.zeros_like(dq)
            g_ddq = torch.zeros_like(ddq)

            for i in range(q_.shape[0]):
                for j in range(q_.shape[1]):
                    g = pino.computeRNEADerivatives(self.model, self.data, q_[i, j], dq_[i, j], ddq_[i, j])
                    dtorque_q[i, j] = torch.from_numpy(g[0][:n, :n])
                    dtorque_dq[i, j] = torch.from_numpy(g[1][:n, :n])
                    dtorque_ddq[i, j] = torch.from_numpy(g[2][:n, :n])

            for i in range(q_.shape[0]):
                for j in range(q_.shape[1]):
                    g_q += (dtorque_q[i, j] @ upstream[..., None]).squeeze()
                    g_dq += (dtorque_dq[i, j] @ upstream[..., None]).squeeze()
                    g_ddq += (dtorque_ddq[i, j] @ upstream[..., None]).squeeze()

            return g_q, g_dq, g_ddq

        return torque, grad_fn

    def forward(self, q_cps, t_cps):
        q = torch.einsum('ijk,lkm->ljm', self.bsp.N, q_cps)
        q_dot_tau = torch.einsum('ijk,lkm->ljm', self.bsp.dN, q_cps)
        q_ddot_tau = torch.einsum('ijk,lkm->ljm', self.bsp.ddN, q_cps)
        q_dddot_tau = torch.einsum('ijk,lkm->ljm', self.bsp.dddN, q_cps)

        dtau_dt = torch.einsum('ijk,lkm->ljm', self.bsp_t.N, t_cps)
        ddtau_dtt = torch.einsum('ijk,lkm->ljm', self.bsp_t.dN, t_cps)
        dddtau_dttt = torch.einsum('ijk,lkm->ljm', self.bsp_t.ddN, t_cps)

        dt = 1. / dtau_dt[..., 0] / dtau_dt.shape[1]
        t_cumsum = torch.cumsum(dt, dim=-1)
        t = torch.sum(dt, dim=-1)

        dtau_dt2 = dtau_dt ** 2
        q_dot = q_dot_tau * dtau_dt
        q_ddot = q_ddot_tau * dtau_dt2 + ddtau_dtt * q_dot_tau * dtau_dt
        q_dddot = q_dddot_tau * dtau_dt ** 3 + 3 * q_ddot_tau * ddtau_dtt * dtau_dt2 + \
                  q_dot_tau * dtau_dt2 * dddtau_dttt + q_dot_tau * ddtau_dtt ** 2 * dtau_dt

        q_dot_limits = self.q_dot_limits
        q_ddot_limits = self.q_ddot_limits
        q_dddot_limits = self.q_dddot_limits
        torque_limits = self.torque_limits

        torque, grad = self.rnea(q, q_dot, q_ddot)

        torque_loss_ = torch.relu(torch.abs(torque) - torque_limits)
        torque_loss_ = huber(torque_loss_)
        torque_loss = torch.sum(torque_loss_ * dt.unsqueeze(-1), dim=1)

        q_dot_loss_ = torch.relu(torch.abs(q_dot) - q_dot_limits)
        q_dot_loss_ = huber(q_dot_loss_)
        q_dot_loss = torch.sum(q_dot_loss_ * dt.unsqueeze(-1), dim=1)

        q_ddot_loss_ = torch.relu(torch.abs(q_ddot) - q_ddot_limits)
        q_ddot_loss_ = huber(q_ddot_loss_)
        q_ddot_loss = torch.sum(q_ddot_loss_ * dt.unsqueeze(-1), dim=1)

        q_dddot_loss_ = torch.relu(torch.abs(q_dddot) - q_dddot_limits)
        q_dddot_loss_ = huber(q_dddot_loss_)
        q_dddot_loss = torch.sum(q_dddot_loss_ * dt.unsqueeze(-1), dim=1)

        model_losses = torch.cat([q_dot_loss, q_ddot_loss, q_dddot_loss], axis=-1)
        model_loss = torch.sum(model_losses)
        return model_loss, q_dot_loss, q_ddot_loss, q_dddot_loss, torque_loss, q, q_dot, q_ddot, q_dddot, torque, t, \
               t_cumsum, dt
