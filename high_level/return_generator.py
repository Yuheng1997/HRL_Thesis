import numpy as np
from scipy.interpolate import CubicSpline

from baseline.baseline_agent.bezier_planner_new import BezierPlanner
from baseline.baseline_agent.cubic_linear_planner import CubicLinearPlanner
from baseline.baseline_agent.optimizer import TrajectoryOptimizer
from baseline.baseline_agent.system_state import SystemState


class ReturnGenerator:
    def __init__(self, env_info):
        self.env_info = env_info
        self.dt = 1 / self.env_info['robot']['control_frequency']
        joint_anchor_pos = np.array([0., -0.1961, 0., -1.8436, 0., 0.9704, 0.])
        x_init = np.array([0.65, 0., self.env_info['robot']['ee_desired_height'] + 0.2])
        x_home = np.array([0.65, 0., self.env_info['robot']['ee_desired_height']])
        max_hit_velocity = 1.2
        self.agent_params = {'switch_tactics_min_steps': 15,
                             'max_prediction_time': 1.0,
                             'max_plan_steps': 5,
                             'static_vel_threshold': 0.4,
                             'transversal_vel_threshold': 0.1,
                             'joint_anchor_pos': joint_anchor_pos,
                             'default_linear_vel': 0.6,
                             'x_init': x_init,
                             'x_home': x_home,
                             'hit_range': [0.8, 1.3],
                             'max_hit_velocity': max_hit_velocity,
                             'defend_range': [0.8, 1.0],
                             'defend_width': 0.45,
                             'prepare_range': [0.8, 1.3]}
        self.bezier_planner = self._init_bezier_planner()
        self.cubic_linear_planner = CubicLinearPlanner(self.env_info['robot']['n_joints'], self.dt)
        self.optimizer = TrajectoryOptimizer(self.env_info)

    def generate_stop_trajectory(self, ee_pos, ee_vel, t_stop):
        x_stop = ee_pos[:2] + ee_vel[:2] * 0.2
        x_stop = np.clip(x_stop, self.bound_points[0] + 0.05, self.bound_points[2] - 0.05)
        self.bezier_planner.compute_control_point(ee_pos[:2], ee_vel[:2], x_stop, ee_vel * 0.0001, t_stop)
        cart_traj = self.generate_bezier_trajectory()
        return cart_traj

    def _init_bezier_planner(self):
        self.bound_points = np.array([[-(self.env_info['table']['length'] / 2 - self.env_info['mallet']['radius']),
                                       -(self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius'])],
                                      [-(self.env_info['table']['length'] / 2 - self.env_info['mallet']['radius']),
                                       (self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius'])],
                                      [-0.1, (self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius'])],
                                      [-0.1, -(self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius'])]
                                      ])
        self.bound_points = self.bound_points + np.tile([1.51, 0.], (4, 1))
        self.boundary_idx = np.array([[0, 1], [1, 2], [0, 3]])

        table_bounds = np.array([[self.bound_points[0], self.bound_points[1]],
                                 [self.bound_points[1], self.bound_points[2]],
                                 [self.bound_points[2], self.bound_points[3]],
                                 [self.bound_points[3], self.bound_points[0]]])
        return BezierPlanner(table_bounds, self.dt)

    def plan_cubic_linear_motion(self, start_pos, start_vel, end_pos, end_vel, t_total=None):
        if t_total is None:
            t_total = np.linalg.norm(end_pos - start_pos) / self.agent_params['default_linear_vel']

        return self.cubic_linear_planner.plan(start_pos, start_vel, end_pos, end_vel, t_total)

    def generate_bezier_trajectory(self, max_steps=-1):
        if max_steps > 0:
            t_plan = np.minimum(self.bezier_planner.t_final, max_steps * self.dt)
        else:
            t_plan = self.bezier_planner.t_final
        res = np.array([self.bezier_planner.get_point(t_i) for t_i in np.arange(self.dt, t_plan + 1e-6, self.dt)])
        p = res[:, 0]
        dp = res[:, 1]
        ddp = res[:, 2]

        p = np.hstack([p, np.ones((p.shape[0], 1)) * self.env_info['robot']["ee_desired_height"]])
        dp = np.hstack([dp, np.zeros((p.shape[0], 1))])
        ddp = np.hstack([ddp, np.zeros((p.shape[0], 1))])
        return np.hstack([p, dp, ddp])

    def optimize_trajectory(self, cart_traj, q_start, dq_start):
        success, joint_pos_traj = self.optimizer.optimize_trajectory(cart_traj, q_start, dq_start, self.agent_params['joint_anchor_pos'])
        if len(joint_pos_traj) > 1:
            t = np.linspace(0, joint_pos_traj.shape[0], joint_pos_traj.shape[0] + 1) * 0.02
            f = CubicSpline(t, np.vstack([q_start, joint_pos_traj]), axis=0, bc_type=((1, dq_start),
                                                                                      (2, np.zeros_like(dq_start))))
            df = f.derivative(1)
            return success, np.stack([f(t[1:]), df(t[1:])]).swapaxes(0, 1)
        else:
            return success, []

    def solve_anchor_pos(self, hit_pos_2d, hit_dir_2d, q_0):
        hit_pos = np.concatenate([hit_pos_2d, [self.env_info['robot']["ee_desired_height"]]])
        hit_dir = np.concatenate([hit_dir_2d, [0]])
        success, q_star = self.optimizer.solve_hit_config(hit_pos, hit_dir, q_0)
        if not success:
            q_star = q_0
        return q_star

    def solve_anchor_pos_ik_null(self, hit_pos_2d, hit_dir_2d, q_0):
        hit_pos = np.concatenate([hit_pos_2d, [self.env_info['robot']["ee_desired_height"]]])
        hit_dir = np.concatenate([hit_dir_2d, [0]])
        success, q_star = self.optimizer.solve_hit_config_ik_null(hit_pos, hit_dir, q_0)
        if not success:
            q_star = q_0
        return q_star
