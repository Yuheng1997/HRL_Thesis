import os
import torch
import mujoco
import numpy as np
from config import Config
from air_hockey_challenge.utils.kinematics import forward_kinematics
from utils.bspline import BSpline
from scipy.interpolate import interp1d
from scipy.optimize import minimize, linprog
from air_hockey_challenge.framework import AgentBase


def build_nn_agent(env_info, model_name, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    Args:
        env_info (dict): The environment information
        kwargs (any): Additionally setting from agent_config.yml
    Returns:
         (AgentBase) An instance of the Agent
    """
    # load model in traned_agent
    device = 'cpu'
    cur_path = os.getcwd()
    par_path = os.path.dirname(cur_path)
    model_path = os.path.join(par_path, 'trained_low_agent', model_name)
    model = torch.load(model_path, map_location=torch.device(device))
    return NNAgent(env_info, model, **kwargs)


class NNAgent(AgentBase):
    def __init__(self, env_info, model, **kwargs):
        super().__init__(env_info, **kwargs)
        self.model = model
        device = 'cpu'
        self.goal = None
        self.hit_pos = None
        self.hit_vel = None
        self.air_hockey_dt = 0.02
        self.b_spline_q = BSpline(num_pts=Config.bspline_q.n_ctr_pts, degree=Config.bspline_q.degree,
                                  num_T_pts=Config.bspline_q.num_T_pts, device=device)
        self.b_spline_t = BSpline(num_pts=Config.bspline_t.n_ctr_pts, degree=Config.bspline_t.degree,
                                  num_T_pts=Config.bspline_t.num_T_pts, device=device)
        self.desired_height = 0.1645

        self.robot_model = mujoco.MjModel.from_xml_path(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/iiwa_only.xml"))
        self.robot_model.body('iiwa_1/base').pos = np.zeros(3)
        self.joint_anchor_pos = np.array([0., -0.1961, 0., -1.8436, 0., 0.9704, 0.])
        self.robot_data = mujoco.MjData(self.robot_model)
        self.traj_buffer = []
        self.scale = 1
        self.max_traj_length = 0
        self.resample_count = 0
        self.ee_pos = None
        # flag
        self.generate_traj = False
        self.if_update_goal = True

    def reset(self):
        self.if_update_goal = True
        self.generate_traj = False
        self.traj_buffer = []
        self.max_traj_length = 0

    def validate_condition(self, obs):
        puck_pos_2d = self.get_puck_pos(obs)[:2]
        puck_vel_2d = self.get_puck_vel(obs)[:2]
        self.ee_pos = self.get_ee_pose(obs)[0][:2]
        if len(self.traj_buffer):
            self.generate_traj = False
        else:
            self.generate_traj = True
        # if not np.linalg.norm(
        #     forward_kinematics(self.robot_model, self.robot_data, self.get_joint_pos(obs))[0][:2] - self.hit_pos) >= 0.15:
        #     self.generate_traj = False


    def update_goal(self, goal=np.zeros(4)):
        if goal.all() == np.zeros(4).all():
            hit_range = np.array([[0.8, 1.3], [-0.39105, 0.39105]])
            hit_pos = np.random.rand(2) * (hit_range[:, 1] - hit_range[:, 0]) + hit_range[:, 0]
            hit_dir_2d = np.array([2.49, 0]) - hit_pos
            self.hit_vel = hit_dir_2d / np.linalg.norm(hit_dir_2d)
            self.hit_pos = hit_pos - self.hit_vel * self.env_info['mallet']['radius']
        else:
            self.hit_pos = goal[:2]
            hit_angel = goal[2]
            self.scale = goal[-1]
            self.hit_vel = np.array([np.sin(hit_angel), np.cos(hit_angel)])

    def update_goal_for_analysis(self, goal):
        self.hit_pos = goal[:2]
        self.hit_vel = goal[2:]

    def draw_action(self, obs):
        self.validate_condition(obs)
        save_and_fit = False
        if self.generate_traj:
            q_0 = self.get_joint_pos(obs)
            dq_0 = self.get_joint_vel(obs)
            ddq_0 = np.zeros_like(q_0)
            q_f, dq_f = self.optimize_vel_and_pos(np.array([*self.hit_pos, self.desired_height]), np.array([*self.hit_vel, 0.]))
            ddq_f = np.zeros_like(q_f)

            features_hit = torch.tensor(np.concatenate([q_0, dq_0, ddq_0, q_f, dq_f, ddq_f]), dtype=torch.float32).unsqueeze(0)

            q_c_ps, t_c_ps = self.compute_control_points(self.model, features_hit)
            pos_traj, vel_traj = self.interpolate_control_points(q_c_ps, t_c_ps)
            hit_traj = np.concatenate((pos_traj, vel_traj), axis=2).squeeze().tolist()

            # h_q_0 = q_f
            # h_dq_0 = dq_f
            # h_ddq_0 = ddq_0
            # h_q_f = self.joint_anchor_pos
            # h_dq_f = np.zeros_like(h_q_0)
            # h_ddq_f = np.zeros_like(h_q_0)
            # features_home = torch.tensor(np.concatenate([h_q_0, h_dq_0, h_ddq_0, h_q_f, h_dq_f, h_ddq_f]), dtype=torch.float32).unsqueeze(0)
            #
            # q_c_ps, t_c_ps = self.compute_control_points(self.model, features_home)
            # pos_traj, vel_traj = self.interpolate_control_points(q_c_ps, t_c_ps)
            # home_traj = np.concatenate((pos_traj, vel_traj), axis=2).squeeze().tolist()
            # self.traj_buffer = hit_traj + home_traj
            self.traj_buffer = hit_traj

            self.generate_traj = False
            self.if_update_goal = False
            if len(self.traj_buffer) > self.max_traj_length:
                self.max_traj_length = len(self.traj_buffer)

        # self.resample_count += 1
        try:
            action = np.array(self.traj_buffer[0]).reshape(2, 7)
        except:
            print('traj', len(self.traj_buffer), self.traj_buffer[0], 'hit_pos', self.hit_pos, 'ee_pos', self.ee_pos)
        self.traj_buffer = self.traj_buffer[1:]
        if len(self.traj_buffer) == 0:
            self.if_update_goal = True
            save_and_fit = True
        # replan traj in next count
        # if self.with_condition:
        #     if self.resample_count == 40 and len(self.traj_buffer) > 10:
            # if self.resample_count > (self.len_hit_traj + 30) and len(self.traj_buffer) > 10:
            #     self.resample_count = 0
            #     self.traj_buffer = []
            #     self.if_update_goal = True
        return action, save_and_fit

    def generate_whole_traj(self, obs):
        q_0 = self.get_joint_pos(obs)
        dq_0 = self.get_joint_vel(obs)
        ddq_0 = np.zeros_like(q_0)
        q_f, dq_f = self.optimize_vel_and_pos(np.array([*self.hit_pos, self.desired_height]),
                                              np.array([*self.hit_vel, 0.]))
        ddq_f = np.zeros_like(q_f)

        features_hit = torch.tensor(np.concatenate([q_0, dq_0, ddq_0, q_f, dq_f, ddq_f]),
                                    dtype=torch.float32).unsqueeze(0)

        q_c_ps, t_c_ps = self.compute_control_points(self.model, features_hit)
        pos_traj, vel_traj = self.interpolate_control_points(q_c_ps, t_c_ps)
        hit_traj = np.concatenate((pos_traj, vel_traj), axis=2).squeeze().tolist()
        return hit_traj

    def draw_action_all_traj(self, obs):
        self.validate_condition(obs)

        if self.generate_traj:
            q_0 = self.get_joint_pos(obs)
            dq_0 = self.get_joint_vel(obs)
            ddq_0 = np.zeros_like(q_0)
            q_f, dq_f = self.optimize_vel_and_pos(np.array([*self.hit_pos, self.desired_height]),
                                                  np.array([*self.hit_vel, 0.]))
            ddq_f = np.zeros_like(q_f)

            features_hit = torch.tensor(np.concatenate([q_0, dq_0, ddq_0, q_f, dq_f, ddq_f]),
                                        dtype=torch.float32).unsqueeze(0)

            q_c_ps, t_c_ps = self.compute_control_points(self.model, features_hit)
            pos_traj, vel_traj = self.interpolate_control_points(q_c_ps, t_c_ps)
            hit_traj = np.concatenate((pos_traj, vel_traj), axis=2).squeeze().tolist()

            h_q_0 = q_f
            h_dq_0 = dq_f
            h_ddq_0 = ddq_0
            h_q_f = self.joint_anchor_pos
            h_dq_f = np.zeros_like(h_q_0)
            h_ddq_f = np.zeros_like(h_q_0)
            features_home = torch.tensor(np.concatenate([h_q_0, h_dq_0, h_ddq_0, h_q_f, h_dq_f, h_ddq_f]),
                                         dtype=torch.float32).unsqueeze(0)

            q_c_ps, t_c_ps = self.compute_control_points(self.model, features_home)
            pos_traj, vel_traj = self.interpolate_control_points(q_c_ps, t_c_ps)
            home_traj = np.concatenate((pos_traj, vel_traj), axis=2).squeeze().tolist()
            self.traj_buffer = hit_traj + home_traj
            self.generate_traj = False

        if len(self.traj_buffer):
            traj_buffer = self.traj_buffer
            self.traj_buffer = []
            return traj_buffer
        else:
            return [np.vstack([self.joint_anchor_pos, np.zeros_like(self.joint_anchor_pos)])]

    def draw_traj_for_analysis(self):
        q_0 = self.joint_anchor_pos
        dq_0 = np.zeros_like(q_0)
        ddq_0 = np.zeros_like(q_0)
        q_f, dq_f = self.optimize_vel_and_pos(np.array([*self.hit_pos, self.desired_height]), np.array([*self.hit_vel, 0.]))
        ddq_f = np.zeros_like(q_f)

        features_hit = torch.tensor(np.concatenate([q_0, dq_0, ddq_0, q_f, dq_f, ddq_f]), dtype=torch.float32).unsqueeze(0)
        q_c_ps, t_c_ps = self.compute_control_points(self.model, features_hit)
        hit_pos_traj, hit_vel_traj = self.interpolate_control_points(q_c_ps, t_c_ps)
        hit_traj = np.concatenate((hit_pos_traj, hit_vel_traj), axis=2).squeeze().tolist()

        h_q_0 = q_f
        h_dq_0 = dq_f
        h_ddq_0 = np.zeros_like(h_q_0)
        h_q_f = self.joint_anchor_pos
        h_dq_f = np.zeros_like(h_q_0)
        h_ddq_f = np.zeros_like(h_q_0)

        features_home = torch.tensor(np.concatenate([h_q_0, h_dq_0, h_ddq_0, h_q_f, h_dq_f, h_ddq_f]), dtype=torch.float32).unsqueeze(0)
        q_c_ps_home, t_c_ps_home = self.compute_control_points(self.model, features_home)
        home_pos_traj, home_vel_traj = self.interpolate_control_points(q_c_ps_home, t_c_ps_home)
        home_traj = np.concatenate((home_pos_traj, home_vel_traj), axis=2).squeeze().tolist()
        return hit_traj, hit_pos_traj, hit_vel_traj, home_traj, home_pos_traj, home_vel_traj

    def optimize_vel_and_pos(self, hitting_point, hitting_direction):
        v = hitting_direction
        initial_q = self.inverse_kinematics(hitting_point)[1]

        dq_min = -0.8 * np.array([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562, 2.3562])
        dq_max = 0.8 * np.array([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562, 2.3562])

        jacb_star = self.jacob(self.robot_model, self.robot_data, initial_q)[:3]
        # min{c.T @ x}
        vv = v.T.dot(v)
        c = [-vv, 0, 0, 0, 0, 0, 0, 0]
        J_dag = np.linalg.pinv(jacb_star)
        _JJ = J_dag.dot(jacb_star)
        N = np.eye(*_JJ.shape) - _JJ

        J_dag_v = J_dag
        J_dag_v = J_dag_v @ v

        A = np.c_[J_dag_v, N]
        A = np.r_[A, -A]
        _dq_max = 0.999 * dq_max
        _dq_min = 0.999 * dq_min
        _ub = np.c_[[*_dq_max, *(-_dq_min)]]
        result_eta = linprog(c=c, A_ub=A, b_ub=_ub, method='highs', x0=[0, 0, 0, 0, 0, 0, 0, 0], bounds=((0, None),
                             (dq_min[0], dq_max[0]), (dq_min[1], dq_max[1]), (dq_min[2], dq_max[2]), (dq_min[3], dq_max[3]),
                             (dq_min[4], dq_max[4]), (dq_min[5], dq_max[5]), (dq_min[6], dq_max[6])))

        if result_eta.success:
            optimal_eta = result_eta.x[0]
            optimal_alpha = result_eta.x[1:]
            optimal_v = optimal_eta * v
            _dq = (J_dag @ optimal_v + N @ optimal_alpha) * self.scale
            if (np.abs(_dq) > dq_max).any():
                _dq[np.abs(_dq) < 1e-8] = 1e-8
                beta = np.min(dq_max / (np.abs(_dq)))
                optimal_dq = _dq * beta
            else:
                optimal_dq = _dq
        else:
            _dq = np.linalg.lstsq(jacb_star, v, rcond=None)[0]
            beta = np.min(dq_max / np.abs(_dq))
            optimal_dq = _dq * self.scale * beta

        return initial_q, optimal_dq

    def link_to_xml_name(self, mj_model, link):
        try:
            mj_model.body('iiwa_1/base')
            link_to_frame_idx = {
                "1": "iiwa_1/link_1",
                "2": "iiwa_1/link_2",
                "3": "iiwa_1/link_3",
                "4": "iiwa_1/link_4",
                "5": "iiwa_1/link_5",
                "6": "iiwa_1/link_6",
                "7": "iiwa_1/link_7",
                "ee": "iiwa_1/striker_joint_link",
            }
        except:
            link_to_frame_idx = {
                "1": "planar_robot_1/body_1",
                "2": "planar_robot_1/body_2",
                "3": "planar_robot_1/body_3",
                "ee": "planar_robot_1/body_ee",
            }
        return link_to_frame_idx[link]

    def _mujoco_clik(self, desired_pos, desired_quat, initial_q, name, model, data, lower_limit, upper_limit):
        IT_MAX = 1000
        eps = 1e-4
        damp = 1e-3
        progress_thresh = 20.0
        max_update_norm = 0.1
        rot_weight = 1
        i = 0

        dtype = data.qpos.dtype

        data.qpos = initial_q

        neg_x_quat = np.empty(4, dtype=dtype)
        error_x_quat = np.empty(4, dtype=dtype)

        if desired_pos is not None and desired_quat is not None:
            jac = np.empty((6, model.nv), dtype=dtype)
            err = np.empty(6, dtype=dtype)
            jac_pos, jac_rot = jac[:3], jac[3:]
            err_pos, err_rot = err[:3], err[3:]
        else:
            jac = np.empty((3, model.nv), dtype=dtype)
            err = np.empty(3, dtype=dtype)
            if desired_pos is not None:
                jac_pos, jac_rot = jac, None
                err_pos, err_rot = err, None
            elif desired_quat is not None:
                jac_pos, jac_rot = None, jac
                err_pos, err_rot = None, err
            else:
                raise ValueError("Desired Position and desired rotation is None, cannot compute inverse kinematics")

        while True:
            # forward kinematics
            mujoco.mj_fwdPosition(model, data)

            x_pos = data.body(name).xpos
            x_quat = data.body(name).xquat

            error_norm = 0
            if desired_pos is not None:
                err_pos[:] = desired_pos - x_pos
                error_norm += np.linalg.norm(err_pos)

            if desired_quat is not None:
                mujoco.mju_negQuat(neg_x_quat, x_quat)
                mujoco.mju_mulQuat(error_x_quat, desired_quat, neg_x_quat)
                mujoco.mju_quat2Vel(err_rot, error_x_quat, 1)
                error_norm += np.linalg.norm(err_rot) * rot_weight

            if error_norm < eps:
                success = True
                break
            if i >= IT_MAX:
                success = False
                break

            mujoco.mj_jacBody(model, data, jac_pos, jac_rot, model.body(name).id)

            hess_approx = jac.T.dot(jac)
            joint_delta = jac.T.dot(err)

            hess_approx += np.eye(hess_approx.shape[0]) * damp
            update_joints = np.linalg.solve(hess_approx, joint_delta)

            update_norm = np.linalg.norm(update_joints)

            # Check whether we are still making enough progress, and halt if not.
            progress_criterion = error_norm / update_norm
            if progress_criterion > progress_thresh:
                success = False
                break

            if update_norm > max_update_norm:
                update_joints *= max_update_norm / update_norm

            mujoco.mj_integratePos(model, data.qpos, update_joints, 1)
            data.qpos = np.clip(data.qpos, lower_limit, upper_limit)
            i += 1
        q_cur = data.qpos.copy()

        return success, q_cur

    def inverse_kinematics(self, desired_position, desired_rotation=None, initial_q=None, link="ee"):
        q_init = np.zeros(self.robot_model.nq)
        if initial_q is None:
            q_init = self.robot_data.qpos
        else:
            q_init[:initial_q.size] = initial_q

        q_l = self.robot_model.jnt_range[:, 0]
        q_h = self.robot_model.jnt_range[:, 1]
        lower_limit = (q_l + q_h) / 2 - 0.95 * (q_h - q_l) / 2
        upper_limit = (q_l + q_h) / 2 + 0.95 * (q_h - q_l) / 2

        desired_quat = None
        if desired_rotation is not None:
            desired_quat = np.zeros(4)
            mujoco.mju_mat2Quat(desired_quat, desired_rotation.reshape(-1, 1))

        return self._mujoco_clik(desired_position, desired_quat, q_init, self.link_to_xml_name(self.robot_model, link),
                                 self.robot_model,
                                 self.robot_data, lower_limit, upper_limit)

    def forward_kinematics(self, q, link="ee"):
        """
        Compute the forward kinematics of the robots.

        IMPORTANT:
            For the iiwa we assume that the universal joint at the end of the end-effector always leaves the mallet
            parallel to the table and facing down. This assumption only makes sense for a subset of robot configurations
            where the mallet can be parallel to the table without colliding with the rod it is mounted on. If this is the
            case this function will return the wrong values.

        Coordinate System:
            All translations and rotations are in the coordinate frame of the Robot. The zero point is in the center of the
            base of the Robot. The x-axis points forward, the z-axis points up and the y-axis forms a right-handed
            coordinate system

        Args:
            mj_model (mujoco.MjModel):
                mujoco MjModel of the robot-only model
            mj_data (mujoco.MjData):
                mujoco MjData object generated from the model
            q (np.array):
                joint configuration for which the forward kinematics are computed
            link (string, "ee"):
                Link for which the forward kinematics is calculated. When using the iiwas the choices are
                ["1", "2", "3", "4", "5", "6", "7", "ee"]. When using planar the choices are ["1", "2", "3", "ee"]

        Returns
        -------
        position: numpy.ndarray, (3,)
            Position of the link in robot's base frame
        orientation: numpy.ndarray, (3, 3)
            Orientation of the link in robot's base frame
        """

        return self._mujoco_fk(q, self.link_to_xml_name(self.robot_model, link), self.robot_model, self.robot_data)

    def _mujoco_fk(self, q, name, model, data):
        data.qpos[:len(q)] = q
        mujoco.mj_fwdPosition(model, data)
        return data.body(name).xpos.copy(), data.body(name).xmat.reshape(3, 3).copy()

    def jacob(self, mj_model, mj_data, q, link="ee"):
        return self._mujoco_jac(q, self.link_to_xml_name(mj_model, link), mj_model, mj_data)

    def _mujoco_jac(self, q, name, model, data):
        data.qpos[:len(q)] = q
        dtype = data.qpos.dtype
        jac = np.empty((6, model.nv), dtype=dtype)
        jac_pos, jac_rot = jac[:3], jac[3:]
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_jacBody(model, data, jac_pos, jac_rot, model.body(name).id)
        return jac

    def optimize(self, hitting_point, hitting_direction):
        v = hitting_direction
        initial_q = self.inverse_kinematics(hitting_point)[1]

        def FKp(q):
            return self.forward_kinematics(q)[0]

        def hitting_objective(q, v):
            Jp = self.jacob(self.robot_model, self.robot_data, q)[:3]
            manipulability = np.linalg.norm(np.dot(v, Jp))
            return -manipulability

        constraints = ({'type': 'eq', 'fun': lambda q: FKp(q) - hitting_point})
        result = minimize(hitting_objective, initial_q, args=(v,), constraints=constraints)
        optimal_q = result.x

        return optimal_q

    def get_qd_max(self, hit_dir_2d, qf):
        hit_dir = np.concatenate([hit_dir_2d, [0]])
        jac = self.jacob(self.robot_model, self.robot_data, qf)[:3]
        qdf = np.linalg.lstsq(jac, hit_dir, rcond=None)[0]
        q_dot7 = 0.8 * torch.tensor([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562, 2.3562], dtype=torch.float32)
        max_gain = np.min(q_dot7.cpu().detach().numpy() / np.abs(qdf))
        qdf *= max_gain
        # print(jac @ qdf)
        # print(np.linalg.norm(jac @ qdf))
        return qdf

    def compute_control_points(self, model, features):
        with torch.no_grad():
            features_tensor = features.clone().detach()
            q_cps, t_cps = model(features_tensor)
            q_cps, t_cps = q_cps.to(torch.float32), t_cps.to(torch.float32)
        return q_cps, t_cps

    def interpolate_control_points(self, q_cps, t_cps):
        with torch.no_grad():
            q = torch.einsum('ijk,lkm->ljm', self.b_spline_q.N, q_cps)
            q_dot_tau = torch.einsum('ijk,lkm->ljm', self.b_spline_q.dN, q_cps)
            q_ddot_tau = torch.einsum('ijk,lkm->ljm', self.b_spline_q.ddN, q_cps)

            dtau_dt = torch.einsum('ijk,lkm->ljm', self.b_spline_t.N, t_cps)
            ddtau_dtt = torch.einsum('ijk,lkm->ljm',self.b_spline_t.dN, t_cps)

            dt = 1. / dtau_dt[..., 0] / dtau_dt.shape[1]
            t_cumsum = torch.cumsum(dt, dim=-1)

            q_dot = q_dot_tau * dtau_dt
            q_ddot = q_ddot_tau * dtau_dt ** 2 + ddtau_dtt * q_dot_tau * dtau_dt

            pos = list()
            vel = list()
            acc = list()

            t_cumsum = t_cumsum.numpy()
            q = q.numpy()
            q_dot = q_dot.numpy()
            q_ddot = q_ddot.numpy()
            for n in range(q_cps.shape[0]):
                _dt = t_cumsum[n]
                _q = q[n]
                _q_dot = q_dot[n]
                _q_ddot = q_ddot[n]

                q_interpol = [interp1d(_dt, _q[:, i], kind='linear', fill_value='extrapolate') for i in range(7)]
                q_dot_interpol = [interp1d(_dt, _q_dot[:, i], kind='linear', fill_value='extrapolate') for i in
                                  range(7)]
                q_ddot_interpol = [interp1d(_dt, _q_ddot[:, i], kind='linear', fill_value='extrapolate') for i in
                                   range(7)]

                # ts = np.arange(0, dt[-1], air_hockey_dt / 20)
                _end_t = t_cumsum[n, -1]
                _start_t = self.air_hockey_dt

                ts = np.arange(start=_start_t, stop=_end_t, step=self.air_hockey_dt)

                _pos = np.array([q_interpol[i](ts) for i in range(7)]).transpose()
                _vel = np.array([q_dot_interpol[i](ts) for i in range(7)]).transpose()
                _acc = np.array([q_ddot_interpol[i](ts) for i in range(7)]).transpose()
                pos.append(_pos)
                vel.append(_vel)
                acc.append(_acc)

        return pos, vel


