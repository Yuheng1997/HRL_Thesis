from abc import ABC, abstractmethod
import numpy as np


class Constraint(ABC):
    def __init__(self, name, dim_q, dim_k, dim_z=0):
        self.name = name
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_z = dim_z

    @abstractmethod
    def fun(self, q, z=None) -> np.ndarray:
        pass

    @abstractmethod
    def df_dq(self, q, z=None) -> np.ndarray:
        pass

    def df_dz(self, q, z=None) -> np.ndarray:
        pass

    def k_(self, q, z=None):
        k_out = self.fun(q, z)
        assert k_out.shape[0] == self.dim_k
        return k_out

    def J_q(self, q, z=None):
        jac_out = self.df_dq(q, z)
        assert jac_out.shape == (self.dim_k, self.dim_q)
        return jac_out

    def J_z(self, q, z=None):
        if self.dim_z == 0:
            return 0.
        else:
            jac_out = self.df_dz(q, z)
            assert jac_out.shape == (self.dim_k, self.dim_z)
            return jac_out

    def zeta(self):
        raise NotImplementedError

class ConstraintList:
    def __init__(self, dim_q, dim_z=0):
        self.dim_q = dim_q
        self.dim_z = dim_z

        self.dim_k = 0
        self.constraints = []
        self.constraints_idx = {}

    def add_constraint(self, k: Constraint):
        assert self.dim_q == k.dim_q
        assert self.dim_z == k.dim_z
        self.constraints.append(k)
        self.constraints_idx.update({k.name: len(self.constraints)})
        self.dim_k += k.dim_k

    def k(self, q, z=None):
        k_tmp = np.array([])
        for k_i in self.constraints:
            k_tmp = np.concatenate([k_tmp, k_i.k_(q, z)])
        return k_tmp

    def J_q(self, q, z=None):
        J_tmp = list()
        for k_i in self.constraints:
            J_tmp.append(k_i.J_q(q, z))
        J_tmp = np.vstack(J_tmp)
        assert J_tmp.shape == (self.dim_k, self.dim_q)
        return J_tmp

    def J_z(self, q, z=None):
        if self.dim_z != 0:
            J_tmp = list()
            for k_i in self.constraints:
                J_tmp.append(k_i.J_z(q, z))
            J_tmp = np.vstack(J_tmp)

            assert J_tmp.shape == (self.dim_k, self.dim_z)
            return J_tmp
        else:
            return np.zeros(self.dim_k)

    @property
    def zeta(self):
        zeta_ = np.concatenate([k_i.zeta() for k_i in self.constraints]).flatten()
        assert zeta_.size == self.dim_k
        return zeta_
    
    def J_q_dot(self, q, q_dot=0, z=None):
        dt = 1e-8
        J = self.J_q(q, z)
        q_h = q + dt * q_dot
        J_h = self.J_q(q_h, z)
        return (J_h - J) / dt
