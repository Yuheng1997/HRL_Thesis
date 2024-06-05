import os
import numpy as np
import torch
from config import Config
from differentiable_robot_model.differentiable_robot_model import DifferentiableRobotModel
from generate_hitting_data import get_defend_sample, get_hitting_sample, get_prepare_sample
from losses.constraint_functions import air_hockey_puck, air_hockey_table
from losses.hitting import HittingLoss
from mushroom_rl.distributions import GaussianDistribution
from utils.creps import ConstrainedREPS
from utils.bspline import BSpline
from utils.constants import Limits
    

def get_sample(puck_position_2d, v):

    if 0.8 < puck_position_2d[0] < 1.3 and abs(puck_position_2d[1]) < 0.39105 and v <= .15:  # If: Hit
        s1, s2 = get_hitting_sample(puck_position_2d)
    elif 0.8 < puck_position_2d[0] < 1.0 and abs(puck_position_2d[1]) < 0.45:  # Else if: Defend
        s1, s2 = get_defend_sample(puck_position_2d)
    elif 0.57 < puck_position_2d[0] < 0.8 and abs(puck_position_2d[1]) < 0.48535 or \
         0.57 < puck_position_2d[0] < 1.3 and 0.39105 < abs(puck_position_2d[1]) < 0.48535:  # Else if: Prepare
        s1, s2 = get_prepare_sample(puck_position_2d)
    else:  # Else: position not in range of action
        raise RuntimeError("Invalid sample")

    return s1, s2
    

def main():
    device = Config.train.device
    Limits.to_device(device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor)

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/checkpoints_1%/Model_31000.pt")
    model = torch.load(model_path, map_location=torch.device(Config.train.device))

    # Generate B-Spline Parameters
    bspline_q = BSpline(num_pts=Config.bspline_q.n_ctr_pts, degree=Config.bspline_q.degree,
                        num_T_pts=Config.bspline_q.num_T_pts, device=device)
    bspline_t = BSpline(num_pts=Config.bspline_t.n_ctr_pts, degree=Config.bspline_t.degree,
                        num_T_pts=Config.bspline_t.num_T_pts, device=device)
    
    manipulator = DifferentiableRobotModel(urdf_path=Config.manipulator.urdf_file,
                                           tensor_args={'device': device, 'dtype': torch.float32})
    loss_fn = HittingLoss(manipulator, air_hockey_table, air_hockey_puck, bspline_q, bspline_t, Limits.q7, Limits.q_dot7, Limits.q_ddot7, Limits.q_dddot7, device=device)

    mu = np.array([1., 0., 0.15])
    sigma = np.array([[0.2, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])
    
    distribution = GaussianDistribution(mu, sigma)

    creps = ConstrainedREPS(distribution, .1, .1)

    while True:
        num_samples = 20
        data = []
        thetas = []

        while len(data) < num_samples:
            x, y, v = distribution.sample()
            if not (0.8 < x < 1.3 and abs(y) < .48535):
                continue

            puck_pos_2d = np.array([x, y])
            try:
                s1, s2 = get_sample(puck_pos_2d, v)
            except RuntimeError:
                continue

            thetas.append([x, y, v])
            thetas.append([x, y, v])

            data.append(s1)
            data.append(s2)

        data = torch.from_numpy(np.array(data)).to(torch.float32).to(device)
        q_cps, t_cps = model(data[:, :42])
        loss_tuple = loss_fn(q_cps, t_cps, data[:, -2:])

        # TODO: Update distribution using ConstrainedREPS
        Jep = loss_tuple[0].detach().cpu().numpy()
        theta = np.array(thetas)
        creps._update(Jep, theta, None)


if __name__ == '__main__':
    main()
