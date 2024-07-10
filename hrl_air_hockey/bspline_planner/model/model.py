import torch
import torch.nn as nn
from hrl_air_hockey.bspline_planner.utils.constants import Limits


class NNPlanner(nn.Module):
    def __init__(self, n_ctr_pts, n_ctr_pts_t, bsp, bsp_t, n_pts_fixed_begin=3, n_pts_fixed_end=2, n_dof=7):
        super(NNPlanner, self).__init__()
        self.n_ctr_pts = n_ctr_pts - n_pts_fixed_begin - n_pts_fixed_end
        self.n_pts_fixed_begin = n_pts_fixed_begin
        self.n_pts_fixed_end = n_pts_fixed_end
        self.n_dof = n_dof

        self.qdd1 = bsp.ddN[0, 0, 0]
        self.qdd2 = bsp.ddN[0, 0, 1]
        self.qdd3 = bsp.ddN[0, 0, 2]
        self.qd1 = bsp.dN[0, 0, 1]
        self.td1 = bsp_t.dN[0, 0, 1]

        self.fc = nn.Sequential(
            nn.Linear(6 * self.n_dof, 2048),
            nn.Tanh(),
            nn.Linear(2048, 2048),
            nn.Tanh(),
            nn.Linear(2048, 2048),
            nn.Tanh(),
            nn.Linear(2048, 2048),
            nn.Tanh(),
            nn.Linear(2048, 2048),
            nn.Tanh()
        )

        self.q_est = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.Tanh(),
            nn.Linear(2048, self.n_dof * self.n_ctr_pts)
        )

        self.t_est = nn.Sequential(
            nn.Linear(2048, n_ctr_pts_t)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                gain = nn.init.calculate_gain('tanh' if isinstance(m, nn.Tanh) else 'relu')
                # gain = 5.0/3
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    # init.zeros_(m.bias)
                    m.bias.data.fill_(0.01)

    def forward(self, x):
        q0, q_dot_0, q_ddot_0, qd, q_dot_d, q_ddot_d = torch.split(x, self.n_dof, dim=1)
        expected_time = torch.max(torch.abs(qd - q0) / Limits.q_dot7.unsqueeze(0).to(q0.device), dim=-1)[0]

        y = self.fc(x)

        q_est = self.q_est(y)
        q_est = q_est.reshape(q_est.shape[0], self.n_dof, self.n_ctr_pts)

        dt_est = self.t_est(y)
        dtau_dt = torch.exp(dt_est)

        dtau_dt = dtau_dt / expected_time.unsqueeze(1)

        q = torch.pi * q_est.reshape(-1, self.n_ctr_pts, self.n_dof)
        s = torch.linspace(0., 1., q.shape[1] + 2).unsqueeze(0)[:, 1:-1, None].to(q.device)

        q1 = q_dot_0 / dtau_dt[:, :1] / self.qd1 + q0
        qm1 = qd - q_dot_d / dtau_dt[:, -1:] / self.qd1
        q2 = ((q_ddot_0 / dtau_dt[:, :1] -
               self.qd1 * self.td1 * (q1 - q0) * (dtau_dt[:, 1] - dtau_dt[:, 0])[:, None]) / dtau_dt[:, :1]
              - self.qdd1 * q0 - self.qdd2 * q1) / self.qdd3
        qm2 = ((q_ddot_d / dtau_dt[:, -1:] -
                self.qd1 * self.td1 * (qd - qm1) * (dtau_dt[:, -1] - dtau_dt[:, -2])[:, None]) / dtau_dt[:, -1:]
               - self.qdd1 * qd - self.qdd2 * qm1) / self.qdd3

        q0 = q0.unsqueeze(1)
        q1 = q1.unsqueeze(1)
        q2 = q2.unsqueeze(1)
        qm1 = qm1.unsqueeze(1)
        qm2 = qm2.unsqueeze(1)
        qd = qd.unsqueeze(1)

        q_begin = [q0]
        if self.n_pts_fixed_begin > 1:
            q_begin.append(q1)
        if self.n_pts_fixed_begin > 2:
            q_begin.append(q2)
        q_end = [qd]
        if self.n_pts_fixed_end > 1:
            q_end.append(qm1)
        if self.n_pts_fixed_end > 2:
            q_end.append(qm2)

        qb = q_begin[-1] * (1 - s) + q_end[-1] * s

        x = torch.cat(q_begin + [q + qb] + q_end[::-1], dim=-2)
        return x, dtau_dt.unsqueeze(-1)

    def to_device(self, device):
        self.qdd1 = self.qdd1.to(device)
        self.qdd2 = self.qdd2.to(device)
        self.qdd3 = self.qdd3.to(device)
        self.qd1 = self.qd1.to(device)
        self.td1 = self.td1.to(device)
        return self.to(device)
