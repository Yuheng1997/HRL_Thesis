import numpy as np
import copy
import mujoco
# from atacom.system import VelocityControlSystem
# from atacom.constraints import ConstraintList, Constraint
# from atacom.atacom_controller import ATACOMController
# from atacom.utils import smooth_basis


from atacom.core.system import VelocityControlSystem
from atacom.core.constraints import ConstraintList, Constraint
from atacom.core.atacom_controller import ATACOMController
from atacom.core.utils import smooth_basis
from air_hockey_challenge.utils.kinematics import jacobian, forward_kinematics


class LinkConstraint(Constraint):
    def __init__(self, env_info, robot_model, robot_data):
        name = "ee_pos"
        self.n_joints = env_info['robot']['n_joints']
        x_low = - env_info['robot']['base_frame'][0][0, 3] - (
            env_info['table']['length'] / 2 - env_info['mallet']['radius'])
        x_high = 1.3
        y_low = - (env_info['table']['width'] / 2 - env_info['mallet']['radius'])
        y_high = env_info['table']['width'] / 2 - env_info['mallet']['radius']
        el_low = 0.35
        wr_low = 0.35
        self.bound = np.array([x_low, -x_high, y_low, -y_high, wr_low, el_low])
        self.bound += np.array([1, 1, 1, 1, 1, 1]) * 0.02
        self.k_range = np.array([x_high - x_low, x_high - x_low, y_high - y_low, y_high - y_low, 1., 1.])

        self.robot_model = copy.deepcopy(robot_model)
        self.robot_data = copy.deepcopy(robot_data)
        super().__init__(name, dim_q=self.n_joints, dim_k=6)

    def fun(self, q, x=None) -> np.ndarray:
        self.robot_data.qpos[:len(q)] = q
        mujoco.mj_fwdPosition(self.robot_model, self.robot_data)
        ee_pos = self.robot_data.body('iiwa_1/striker_joint_link').xpos.copy()
        wr_pos = self.robot_data.body('iiwa_1/link_6').xpos.copy()
        el_pos = self.robot_data.body('iiwa_1/link_4').xpos.copy()
        result = np.array([-ee_pos[0], ee_pos[0], -ee_pos[1], ee_pos[1], -wr_pos[2], -el_pos[2]]) + self.bound
        return result / self.k_range

    def df_dq(self, q, x=None) -> np.ndarray:
        jacobian = np.zeros((self.dim_k, self.dim_q))
        self.robot_data.qpos[:len(q)] = q
        mujoco.mj_fwdPosition(self.robot_model, self.robot_data)
        jac_pos = np.empty((3, self.robot_model.nv), dtype=self.robot_data.qpos.dtype)
        model_id = self.robot_model.body('iiwa_1/striker_joint_link').id
        mujoco.mj_jacBody(self.robot_model, self.robot_data, jac_pos, None, model_id)
        jacobian[0:4] = np.vstack([-jac_pos[0], jac_pos[0], -jac_pos[1], jac_pos[1]])

        model_id = self.robot_model.body('iiwa_1/link_6').id
        mujoco.mj_jacBody(self.robot_model, self.robot_data, jac_pos, None, model_id)
        jacobian[4] = -jac_pos[2].copy()

        model_id = self.robot_model.body('iiwa_1/link_4').id
        mujoco.mj_jacBody(self.robot_model, self.robot_data, jac_pos, None, model_id)
        jacobian[5] = -jac_pos[2].copy()
        return jacobian / self.k_range[:, None]


class TableConstraintInEq(Constraint):
    def __init__(self, env_info, robot_model, robot_data):
        name = "ee_height"
        self.n_joints = env_info['robot']['n_joints']
        z_low = env_info['robot']['ee_desired_height'] - 0.005
        z_high = env_info['robot']['ee_desired_height'] + 0.005
        self.bound = np.array([z_low, -z_high])
        self.k_range = np.array([1., 1.]) * 0.1

        self.robot_model = copy.deepcopy(robot_model)
        self.robot_data = copy.deepcopy(robot_data)
        super().__init__(name, dim_q=self.n_joints, dim_k=2)

    def fun(self, q, z=None) -> np.ndarray:
        self.robot_data.qpos[:len(q)] = q
        mujoco.mj_fwdPosition(self.robot_model, self.robot_data)
        ee_pos = self.robot_data.body('iiwa_1/striker_joint_link').xpos.copy()
        result = np.array([-ee_pos[2], ee_pos[2]]) + self.bound
        return result / self.k_range

    def df_dq(self, q, z=None) -> np.ndarray:
        jacobian = np.zeros((self.dim_k, self.dim_q))
        self.robot_data.qpos[:len(q)] = q
        mujoco.mj_fwdPosition(self.robot_model, self.robot_data)
        jac_pos = np.empty((3, self.robot_model.nv), dtype=self.robot_data.qpos.dtype)
        model_id = self.robot_model.body('iiwa_1/striker_joint_link').id
        mujoco.mj_jacBody(self.robot_model, self.robot_data, jac_pos, None, model_id)

        jacobian[0:2] = np.vstack([-jac_pos[2], jac_pos[2]])

        return jacobian / self.k_range[:, None]


class JointPosConstraint(Constraint):
    def __init__(self, n_joints, joint_limits):
        name = "joint_pos"
        self.n_joints = n_joints
        self.joint_limits = joint_limits
        self.q_m = (joint_limits[0] + joint_limits[1]) / 2
        self.q_delta_square = ((joint_limits[1] - joint_limits[0]) / 2) ** 2
        # super().__init__(name, dim_q=self.n_joints, dim_k=self.n_joints * 2, dim_x=0)
        super().__init__(name, dim_q=self.n_joints, dim_k=self.n_joints, dim_z=0)

    def fun(self, q, z=None) -> np.ndarray:
        pos = q[:self.n_joints]
        # result = np.concatenate([pos - self.joint_limits[1], self.joint_limits[0] - pos])
        result = (pos - self.q_m) ** 2 / self.q_delta_square - 1
        return result

    def df_dq(self, q, z=None):
        pos = q[:self.n_joints]
        # J_pos = np.vstack([np.eye(self.n_joints), -np.eye(self.n_joints)])
        J_pos = np.diag(2 * (pos - self.q_m) / self.q_delta_square)
        return J_pos


class AirHockeyController(ATACOMController):
    def __init__(self, env_info, controller_info, init_state):
        self.robot_model = copy.deepcopy(env_info['robot']['robot_model'])
        self.robot_data = copy.deepcopy(env_info['robot']['robot_data'])
        self.joint_pos_ids = env_info['joint_pos_ids']
        self.joint_vel_ids = env_info['joint_vel_ids']
        self._dt = env_info["dt"]
        self._double_integration = False

        self._pos_limit = controller_info['pos_limit']
        self._vel_limit = controller_info['vel_limit']
        self._acc_limit = controller_info['vel_limit'] * 10
        self._acc = np.zeros_like(self._acc_limit[0])
        self._k_pos = 10
        self._k_vel = 10
        self._q_init = init_state

        system = VelocityControlSystem(controller_info['dim_q'], np.ones(controller_info['dim_q']))
        constraints = ConstraintList(controller_info['dim_q'], 0)
        constraints.add_constraint(JointPosConstraint(controller_info['dim_q'], controller_info['pos_limit']))
        constraints.add_constraint(LinkConstraint(
            env_info, self.robot_model, self.robot_data))
        constraints.add_constraint(TableConstraintInEq(
            env_info, self.robot_model, self.robot_data))

        super().__init__(constraints=constraints, system=system, slack_beta=controller_info['slack_beta'],
                         slack_dynamics_type=controller_info['slack_dynamics_type'],
                         drift_compensation_type=controller_info['drift_compensation_type'],
                         drift_clipping=controller_info['drift_clipping'], slack_tol=controller_info['slack_tol'],
                         lambda_c=controller_info['lambda_c'],
                         slack_vel_limit=controller_info['slack_vel_limit'])

    def compute_control(self, action_2d, obs):
        q = obs[6:13]
        v_3d = np.concatenate([action_2d, [0]])
        jac_lin = jacobian(self.robot_model, self.robot_data, q)[:3]

        action, residual, rank, s = np.linalg.lstsq(jac_lin, v_3d, rcond=None)
        error = (self._q_init - q) * np.array([0, 1, 0, 1, 0, 1, 1])
        action += (np.eye(7) - np.linalg.pinv(jac_lin) @ jac_lin) @ error
        action = action / np.maximum(1., (np.abs(action) / self._vel_limit[1]).max())
        dq_des = self.compose_action(q, action)
        des_pos, des_vel = self.integrator(dq_des, obs)

        return np.stack([des_pos, des_vel], axis=0)

    def integrator(self, integrand, state_orig):
        """
        We first convert the state to the original state and get actual position and velocity.
        """
        pos = state_orig[self.joint_pos_ids]
        vel = state_orig[self.joint_vel_ids]

        # Compute the soft limit of the acceleration,
        # details can be found here: http://wiki.ros.org/pr2_controller_manager/safety_limits
        vel_soft_limit = np.clip(-self._k_pos * (pos - self._pos_limit), self._vel_limit[0], self._vel_limit[1])
        acc_soft_limit = np.clip(-self._k_vel * (vel - vel_soft_limit), self._acc_limit[0], self._acc_limit[1])
        if self._double_integration:
            clipped_acc = np.clip(integrand, *acc_soft_limit)

            coeffs = np.vstack([pos, vel, self._acc / 2, (clipped_acc - self._acc) / 6 / self._dt]).T
            pos = coeffs @ np.array([1, self._dt, self._dt ** 2, self._dt ** 3])
            vel = coeffs @ np.array([0., 1., 2 * self._dt, 3 * self._dt ** 2])
            self._acc = coeffs @ np.array([0., 0., 2, 6 * self._dt])
        else:
            # clipped_vel = np.clip(integrand, *vel_soft_limit)
            scale = np.maximum(1., (integrand / vel_soft_limit).max())
            clipped_vel = integrand / scale
            pos += clipped_vel * self._dt
            vel = clipped_vel.copy()

        return pos, vel