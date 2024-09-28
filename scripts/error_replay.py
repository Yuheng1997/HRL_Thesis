from hrl_air_hockey.bspline_planner.planner import TrajectoryPlanner
import os
import numpy as np
from hrl_air_hockey.envs.base_env import BaseEnv
from nn_planner_config import Config


env = BaseEnv(horizon=200)

planner_path = os.path.join('..', 'trained_low_agent', 'Model_5600.pt')
planner_config = Config
config = planner_config
device = 'cpu'
nn_planner_params = dict(planner_path=planner_path, env_info=env.env_info, config=config, device=device,
                         violate_path=os.path.join(os.path.abspath(os.getcwd()), "violate_data/violate_data_1_3.tsv"))
traj_planner = TrajectoryPlanner(**nn_planner_params)

q_0 = np.array([ 1.29419183, 0.17180398, -1.69057944, -2.08379478 , -0.01234609, 1.19106001, 0.08808502])
dq_0 = np.array([-10.7781536, -0.0560243,  6.23165233, 1.28206261, 7.99141574, 1.66055999, 23.57457681]) * 0.8
ddq_0 = np.array( [0., 0., 0., 0., 0., 0., 0.])
q_f = np.array([ 1.27957926,  0.19464438, -1.71155261, -1.71732696, -0.06076008,  0.93477441, 0.08808502])
dq_f = np.array([0.01214383,  0.00507045, -0.00071457, -0.00583698, -0.01307594, -0.00451885, -0.02942002])
ddq_f = np.array([0., 0., 0., 0., 0., 0., 0.])
hit_pos = np.array([ 0.61845756, -0.36572334])
hit_dir = np.array([-0.9999977,  -0.00215736])
hit_scale = 0.015623391

# [ 2.53162061 -2.04099675 -0.47458706 -0.47332711 -0.56899058  1.18493248
#   0.36778008]
#  [  2.13767798   2.3299892  -15.69675564  10.17743677  24.89913201
#   14.04725918   0.29565096]
#  [0. 0. 0. 0. 0. 0. 0.]
#  [ 2.6130444  -1.98968    -0.81249941 -0.46686324 -0.82299284  1.46850693
#   0.36778008]
#  [-0.15048657 -0.15048657  0.0061539  -0.13278526  0.03489527  0.23901345
#  -0.23901345]
#  [0. 0. 0. 0. 0. 0. 0.]
#  [0.91808426 0.36603543]
#  [-0.9863435  -0.16470115]
#  0.1269272


traj_planner.plan_trajectory(q_0=q_0, dq_0=dq_0, hit_pos=hit_pos, hit_dir=hit_dir, hit_scale=hit_scale)