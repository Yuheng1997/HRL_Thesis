import numpy as np
from typing import Union, Optional, List, Set, Dict, Tuple
from scipy import linalg, optimize
from .constraints import ConstraintList
from .system import ControlAffineSystem
from .slack import Slack
from .utils import smooth_basis


class ATACOMController:
    def __init__(self, constraints: ConstraintList, system: ControlAffineSystem, eq_constraints=None,
                 slack_beta=3., slack_dynamics_type="exp", drift_compensation_type='vanilla', drift_clipping=True,
                 slack_tol=1e-6, lambda_c=1, slack_vel_limit=1., second_order=False):
        self.constraints = constraints
        self.eq_constraints = eq_constraints
        self.system_dynamics = system
        self.dim_q = self.system_dynamics.dim_q
        self.dim_u = self.system_dynamics.dim_u
        self.dim_z = self.constraints.dim_z
        self.dim_k = self.constraints.dim_k
        self.dim_k_eq = 0
        if self.eq_constraints is not None:
            self.dim_k_eq = self.eq_constraints.dim_k
        self.second_order = second_order
        self.lambda_c = lambda_c
        self.drift_compensation_type = drift_compensation_type
        self.drift_clipping = drift_clipping
        self.slack = Slack(self.constraints.dim_k, beta=slack_beta, dynamics_type=slack_dynamics_type, tol=slack_tol,
                           vel_limit=slack_vel_limit)

    def get_q(self, s):
        """
        The state should be ordered as s = [q q_dot z]. 
        """
        return s[:self.dim_q]
    
    def get_q_dot(self, s):
        """
        The state should be ordered as s = [q q_dot z]. When not second_order system, return None
        """
        if self.second_order:
            return s[self.dim_q:self.dim_q * 2]
        else:
            return None

    def get_z(self, s):
        """
        The state should be ordered as s = [q q_dot z]. When dim_z == 0, return None
        """
        if self.dim_z != 0:
            return s[-self.dim_z:]
        else:
            return None

    def compose_action(self, s, u, z_dot=0.):
        # Get State
        q = self.get_q(s)
        z = self.get_z(s)
        q_dot = self.get_q_dot(s)

        # Get Constraint and Residual
        k = self.constraints.k(q, z)
        if self.second_order:
            assert hasattr(self.constraints, 'zeta')
            k = self.constraints.zeta * k + self.constraints.J_q(q, z) @ q_dot
        mu = self.get_mu(k)
        residual = k + mu
        if self.dim_k_eq > 0:
            l = self.eq_constraints.k(q, z)
            residual = np.concatenate([residual, l], axis=-1)

        # Get Drift
        psi = self.psi(q, q_dot, z, z_dot)
        J_G = self.J_G(q, z)
        J_u = self.J_u(J_G, mu)

        if self.drift_compensation_type == 'vanilla':
            u_drift_compensation = np.linalg.lstsq(J_u, -psi - self.lambda_c * residual, rcond=None)[0]
        elif self.drift_compensation_type == 'enforced':
            u_drift_compensation = np.linalg.lstsq(J_G, -psi, rcond=None)[0]
            u_drift_compensation = np.concatenate([u_drift_compensation, np.zeros(self.constraints.dim_k)])
            u_drift_compensation += np.linalg.lstsq(J_u, -self.lambda_c * residual, rcond=None)[0]
        elif self.drift_compensation_type == 'modified':
            # Check the compensation for each individual constriant
            u_comp_individual = [-np.linalg.pinv(J_G[i:i + 1]) @ psi[i:i + 1] for i in range(self.constraints.dim_k)]
            scale = np.clip((2 - 2 * np.abs(np.vstack(u_comp_individual)).max(axis=1)), 1e-3, 1)
            eff_scale_idx = scale < 1
            if np.sum(eff_scale_idx) > self.dim_u:
                # Sort by the closeness to the boundary mu of the effective one
                eff_scale = scale[eff_scale_idx]
                eff_scale[np.argsort(mu[eff_scale_idx])[self.dim_u:]] = 1
                scale[eff_scale_idx] = eff_scale
            J_u = np.hstack([J_G, scale * self.slack.alpha(mu)])
            u_drift_compensation = np.linalg.lstsq(J_u, -psi - self.lambda_c * residual, rcond=None)[0]

        B_u = smooth_basis(J_u)[:, :self.system_dynamics.dim_u]

        self.u_auxiliary = u_drift_compensation[:-self.constraints.dim_k]
        self.u_tangent = B_u[:-self.constraints.dim_k] @ u

        u_s = self.u_auxiliary + self.u_tangent
        return u_s

    def get_mu(self, k):
        """
        q is controllable state, z is uncontrollable state
        """
        return self.slack.mu(k)

    def G_aug(self, q, mu):
        """
        G_aug = [G(q) 0; 0 A(mu)]
        """
        return linalg.block_diag(self.system_dynamics.G(q), self.slack.alpha(mu))

    def J_c(self, q, z):
        """
        J_c = [J_k(q, z) I; J_l(q, z)]
        """
        J_c = np.hstack([self.constraints.J_q(q, z), np.eye(self.constraints.dim_k)])
        if self.dim_k_eq > 0:
            N = np.zeros((self.eq_constraints.dim_k, self.constraints.dim_k))
            J_c = np.vstack([self.eq_constraints.J_q(q, z), N])
        return J_c

    def J_G(self, q, z):
        J_q = self.constraints.J_q(q, z)
        if self.dim_k_eq > 0:
            J_q = np.vstack([J_q, self.eq_constraints.J_q(q, z)])

        G = self.system_dynamics.G(q)
        return J_q @ G

    def J_u(self, J_G, mu):
        """
        J_u = J_c @ G_aug = [J_k(q, z)G(q) A(mu); J_l(q, z)G(q) 0]
        """
        A_mu = self.slack.alpha(mu)
        if self.dim_k_eq > 0:
            N = np.zeros((self.eq_constraints.dim_k, self.constraints.dim_k))
            A_mu = np.vstack([A_mu, N])
        return np.hstack([J_G, A_mu])

    def psi(self, q, q_dot: Optional[List[float]], z: Optional[List[float]], z_dot: Optional[List[float]]):
        """
        psi = J_q(q, z)f(q) + J_zeta J_z(q, z)z_dot + J_zeta J_q(q, z)q_dot + J_q_dot(q, z, q_dot) q_dot
        """
        f = self.system_dynamics.f(q)
        psi = self.constraints.J_q(q, z) @ f
        if self.dim_z != 0:
            psi += self.constraints.J_z(q, z) @ z_dot

        if self.second_order:
            psi += (self.constraints.zeta[:, None] * self.constraints.J_q(q, z) + self.constraints.J_q_dot(q, q_dot, z)) @ q_dot
            # psi += (self.constraints.zeta[:, None] * self.constraints.J_q(q, z)) @ q_dot

        if self.drift_clipping:
            psi = np.maximum(psi, 0)

        if self.dim_k_eq > 0:
            psi_eq = self.eq_constraints.J_q(q, z) @ f
            if self.dim_z != 0:
                psi_eq += self.eq_constraints.J_z(q, z) @ z_dot

            psi = np.concatenate([psi, psi_eq], axis=-1)
        return psi
