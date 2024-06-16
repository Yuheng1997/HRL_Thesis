import torch
from torch import nn


class HittingLoss(nn.Module):
    def __init__(self, manipulator, end_effector_constraints_distance_function, obstacle_distance_function, bsp, bsp_t, q_limits, q_dot_limits, q_ddot_limits, q_dddot_limits, device="cpu"):
        super(HittingLoss, self).__init__()
        self.bsp = bsp
        self.bsp_t = bsp_t
        self.q_limits = q_limits
        self.q_dot_limits = q_dot_limits
        self.q_ddot_limits = q_ddot_limits
        self.q_dddot_limits = q_dddot_limits

        self.alpha_obstacle = torch.log(torch.tensor([1e-1])).to(device)
        self.alpha_constraint = torch.log(torch.tensor([1e-1])).to(device)
        self.alpha_q = torch.log(torch.tensor([1e-5])).to(device)
        self.alpha_q_dot = torch.log(torch.tensor([1e-3])).to(device)
        self.alpha_q_ddot = torch.log(torch.tensor([1e-5])).to(device)
        self.alpha_q_dddot = torch.log(torch.tensor([1e-3])).to(device)
        self.gamma = torch.tensor(1e-2 * .7).to(device)
        # self.bar_obstacle = torch.tensor(1e-6).to(device)
        self.bar_obstacle = torch.tensor(3e-5).to(device)
        self.bar_constraint = torch.tensor(2e-6).to(device)
        # self.bar_constraint = torch.tensor(5e-5).to(device)
        self.bar_q = torch.tensor(6e-1).to(device)
        # self.bar_q_dot = torch.tensor(6e-3).to(device)
        self.bar_q_dot = torch.tensor(0.001).to(device)
        self.bar_q_ddot = torch.tensor(6e-2).to(device)
        # self.bar_q_dddot = torch.tensor(6e-1).to(device)
        self.bar_q_dddot = torch.tensor(100.).to(device)
        self.time_mul = torch.tensor(1e-0).to(device)

        self.min_alpha_value = -5
        self.max_alpha_value = 80

        self.man = manipulator
        self.end_effector_constraints_distance_function = end_effector_constraints_distance_function
        self.obstacle_distance_function = obstacle_distance_function
        self.huber = torch.nn.HuberLoss(reduction='none')

    def forward(self, q_cps, t_cps, puck_pos):
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

        q_limits = self.q_limits
        q_dot_limits = self.q_dot_limits
        q_ddot_limits = self.q_ddot_limits
        q_dddot_limits = self.q_dddot_limits

        def compute_loss(loss_values, limits):
            loss_ = torch.relu(torch.abs(loss_values) - limits)
            loss_ = self.huber(loss_, torch.zeros_like(loss_))
            return torch.sum(loss_ * dt.unsqueeze(-1), dim=1)

        q_loss = compute_loss(q, q_limits)
        q_dot_loss = compute_loss(q_dot, q_dot_limits)
        q_ddot_loss = compute_loss(q_ddot, q_ddot_limits)
        q_dddot_loss = compute_loss(q_dddot, q_dddot_limits)

        t_loss = self.huber(t.unsqueeze(-1), torch.zeros_like(t.unsqueeze(-1)))

        batch_size, num_T_points, n_dof = q.shape
        q_reshaped = torch.reshape(q, (batch_size * num_T_points, n_dof))
        positions = self.man.my_compute_forward_kinematics(q_reshaped, link_name='iiwa_1/striker_tip')
        positions = torch.reshape(positions, (batch_size, num_T_points, 3))

        constraint_loss, x_loss, y_loss, z_loss = self.end_effector_constraints_distance_function(positions, dt,
                                                                                                  self.huber)
        obstacle_loss = self.obstacle_distance_function(positions, dt, puck_pos)[:, None]

        losses = torch.cat([torch.exp(self.alpha_q) * q_loss,
                            torch.exp(self.alpha_q_dot) * q_dot_loss,
                            torch.exp(self.alpha_q_ddot) * q_ddot_loss,
                            torch.exp(self.alpha_q_dddot) * q_dddot_loss,
                            torch.exp(self.alpha_constraint) * constraint_loss,
                            torch.exp(self.alpha_obstacle) * obstacle_loss,
                            self.time_mul * t_loss], dim=-1)

        sum_q_loss = torch.sum(q_loss, dim=-1)
        sum_q_dot_loss = torch.sum(q_dot_loss, dim=-1)
        sum_q_ddot_loss = torch.sum(q_ddot_loss, dim=-1)
        sum_q_dddot_loss = torch.sum(q_dddot_loss, dim=-1)
        sum_constraint_loss = torch.sum(constraint_loss, dim=-1)
        sum_obstacle_loss = torch.sum(obstacle_loss, dim=-1)
        model_loss = torch.sum(losses, dim=-1)

        return model_loss, sum_constraint_loss, sum_obstacle_loss, sum_q_loss, sum_q_dot_loss, sum_q_ddot_loss, sum_q_dddot_loss, q, q_dot, q_ddot, q_dddot, t, t_cumsum, t_loss, dt, x_loss, y_loss, z_loss

    def alpha_update(self, q_loss, q_dot_loss, q_ddot_loss, q_dddot_loss, constraint_loss, obstacle_loss):
        max_alpha_update = 10.0
        alpha_q_update = self.gamma * torch.clamp(torch.log(q_loss / self.bar_q), -max_alpha_update, max_alpha_update)
        alpha_q_dot_update = self.gamma * torch.clamp(torch.log(q_dot_loss / self.bar_q_dot), -max_alpha_update, max_alpha_update)
        alpha_q_ddot_update = self.gamma * torch.clamp(torch.log(q_ddot_loss / self.bar_q_ddot), -max_alpha_update, max_alpha_update)
        alpha_q_dddot_update = self.gamma * torch.clamp(torch.log(q_dddot_loss / self.bar_q_dddot), -max_alpha_update, max_alpha_update)
        alpha_constraint_update = self.gamma * torch.clamp(torch.log(constraint_loss / self.bar_constraint), -max_alpha_update, max_alpha_update)
        alpha_obstacle_update = self.gamma * torch.clamp(torch.log(obstacle_loss / self.bar_obstacle), -max_alpha_update, max_alpha_update)

        self.alpha_q += alpha_q_update
        self.alpha_q_dot += alpha_q_dot_update
        self.alpha_q_ddot += alpha_q_ddot_update
        self.alpha_q_dddot += alpha_q_dddot_update
        self.alpha_constraint += alpha_constraint_update
        self.alpha_obstacle += alpha_obstacle_update

        self.alpha_q = torch.clamp(self.alpha_q, self.min_alpha_value, self.max_alpha_value)
        self.alpha_q_dot = torch.clamp(self.alpha_q_dot, self.min_alpha_value, self.max_alpha_value)
        self.alpha_q_ddot = torch.clamp(self.alpha_q_ddot, self.min_alpha_value, self.max_alpha_value)
        self.alpha_q_dddot = torch.clamp(self.alpha_q_dddot, self.min_alpha_value, self.max_alpha_value)
        self.alpha_constraint = torch.clamp(self.alpha_constraint, self.min_alpha_value, self.max_alpha_value)
        self.alpha_obstacle = torch.clamp(self.alpha_obstacle, self.min_alpha_value, self.max_alpha_value)
