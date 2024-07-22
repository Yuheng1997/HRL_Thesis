import os

import numpy as np
import torch
import wandb
import time

from air_hockey_challenge.utils import forward_kinematics
from nn_planner_config import Config
from hrl_air_hockey.bspline_planner.losses.hitting import HittingLoss
from hrl_air_hockey.bspline_planner.utils.constants import Limits, RobotEnvInfo
from hrl_air_hockey.bspline_planner.utils.bspline import BSpline
from hrl_air_hockey.bspline_planner.data.load_data import get_hitting_data, get_violate_data
from hrl_air_hockey.bspline_planner.model.model import NNPlanner
from hrl_air_hockey.bspline_planner.losses.constraint_functions import air_hockey_puck, air_hockey_table
from hrl_air_hockey.bspline_planner.differentiable_robot_model import DifferentiableRobotModel
from hrl_air_hockey.bspline_planner.utils.model_utils import save_model, load_model, only_load_model


def setup_device(device_name):
    device = torch.device(device_name)
    torch.set_default_tensor_type(torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor)
    return device


def validate_pairs(q_1, pos_2):
    return True
    q_1 = q_1.cpu().numpy()
    pos_1 = forward_kinematics(mj_model=RobotEnvInfo.robot_model, mj_data=RobotEnvInfo.robot_data, q=q_1)[0][:2]
    pos_2 = pos_2.cpu().numpy()
    dist = np.linalg.norm(pos_1-pos_2)
    return dist > 1.2 * (RobotEnvInfo.puck_radius + RobotEnvInfo.mallet_radius)


def train_epoch(model, loss_fn, optimizer, data_loader, violate_loader, device, is_training=True):
    model_losses = torch.tensor([])
    constraint_losses = torch.tensor([])
    obstacle_losses = torch.tensor([])
    q_losses = torch.tensor([])
    q_dot_losses = torch.tensor([])
    q_ddot_losses = torch.tensor([])
    q_dddot_losses = torch.tensor([])
    t_losses = torch.tensor([])
    time_elapsed = torch.tensor([])
    x_loss = torch.tensor([])
    y_loss = torch.tensor([])
    z_loss = torch.tensor([])
    prefix = 'validation/' if not is_training else 'train/'

    model.train(is_training)
    with torch.set_grad_enabled(is_training):
        for i, data in enumerate(data_loader):
            if is_training:
                optimizer.zero_grad()

            pairs = torch.zeros((data.shape[0], data.shape[1] * 2 - 2))
            for j in range(data.shape[0]):
                r = np.random.randint(low=0, high=data.shape[0])
                while not validate_pairs(data[j, :7], data[r, -2:]):
                    r = np.random.randint(low=0, high=data.shape[0])
                combined = torch.cat((data[j, :21], data[r, :]), dim=0)
                pairs[j, :] = combined
            q_cps, t_cps = model(pairs[:, :42])
            loss_tuple = loss_fn(q_cps, t_cps, pairs[:, -2:])
            model_loss = torch.mean(loss_tuple[0])

            if is_training:
                model_loss.backward()
                optimizer.step()

            model_losses = torch.cat((model_losses, model_loss.unsqueeze(0)))
            constraint_losses = torch.cat((constraint_losses, torch.mean(loss_tuple[1]).unsqueeze(0)))
            obstacle_losses = torch.cat((obstacle_losses, torch.mean(loss_tuple[2]).unsqueeze(0)))
            q_losses = torch.cat((q_losses, torch.mean(loss_tuple[3]).unsqueeze(0)))
            q_dot_losses = torch.cat((q_dot_losses, torch.mean(loss_tuple[4]).unsqueeze(0)))
            q_ddot_losses = torch.cat((q_ddot_losses, torch.mean(loss_tuple[5]).unsqueeze(0)))
            q_dddot_losses = torch.cat((q_dddot_losses, torch.mean(loss_tuple[6]).unsqueeze(0)))
            t_losses = torch.cat((t_losses, torch.mean(loss_tuple[13]).unsqueeze(0)))
            time_elapsed = torch.cat((time_elapsed, torch.mean(loss_tuple[12]).unsqueeze(0)))
            x_loss = torch.cat((x_loss, torch.mean(loss_tuple[-3]).unsqueeze(0)))
            y_loss = torch.cat((y_loss, torch.mean(loss_tuple[-2]).unsqueeze(0)))
            z_loss = torch.cat((z_loss, torch.mean(loss_tuple[-1]).unsqueeze(0)))

        # use violate point to train
        for _, data in enumerate(violate_loader):
            if is_training:
                optimizer.zero_grad()
            q_cps, t_cps = model(data[:, :42])
            loss_tuple = loss_fn(q_cps, t_cps, data[:, -2:])
            model_loss = torch.mean(loss_tuple[0])
            if is_training:
                model_loss.backward()
                optimizer.step()

            model_losses = torch.cat((model_losses, model_loss.unsqueeze(0)))
            constraint_losses = torch.cat((constraint_losses, torch.mean(loss_tuple[1]).unsqueeze(0)))
            obstacle_losses = torch.cat((obstacle_losses, torch.mean(loss_tuple[2]).unsqueeze(0)))
            q_losses = torch.cat((q_losses, torch.mean(loss_tuple[3]).unsqueeze(0)))
            q_dot_losses = torch.cat((q_dot_losses, torch.mean(loss_tuple[4]).unsqueeze(0)))
            q_ddot_losses = torch.cat((q_ddot_losses, torch.mean(loss_tuple[5]).unsqueeze(0)))
            q_dddot_losses = torch.cat((q_dddot_losses, torch.mean(loss_tuple[6]).unsqueeze(0)))
            t_losses = torch.cat((t_losses, torch.mean(loss_tuple[13]).unsqueeze(0)))
            time_elapsed = torch.cat((time_elapsed, torch.mean(loss_tuple[12]).unsqueeze(0)))
            x_loss = torch.cat((x_loss, torch.mean(loss_tuple[-3]).unsqueeze(0)))
            y_loss = torch.cat((y_loss, torch.mean(loss_tuple[-2]).unsqueeze(0)))
            z_loss = torch.cat((z_loss, torch.mean(loss_tuple[-1]).unsqueeze(0)))

    return {
        prefix + 'model_loss':      torch.mean(model_losses),
        prefix + 'constraint_loss': torch.mean(constraint_losses),
        prefix + 'obstacle_loss':   torch.mean(obstacle_losses),
        prefix + 'q_loss':          torch.mean(q_losses),
        prefix + 'q_dot_loss':      torch.mean(q_dot_losses),
        prefix + 'q_ddot_loss':     torch.mean(q_ddot_losses),
        prefix + 'q_dddot_loss':    torch.mean(q_dddot_losses),
        prefix + 't_losses':        torch.mean(t_losses),
        prefix + 'time_elapsed':    torch.mean(time_elapsed),
        prefix + 'x_loss':          torch.mean(x_loss),
        prefix + 'y_loss':          torch.mean(y_loss),
        prefix + 'z_loss':          torch.mean(z_loss)
    }


def train():
    # Use cuda if available
    device = Config.train.device
    print('Device: {}'.format(Config.train.device))
    torch.set_default_device(Config.train.device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor)

    # Create manipulator model
    manipulator = DifferentiableRobotModel(urdf_path=Config.manipulator.urdf_file,
                                           tensor_args={'device': device, 'dtype': torch.float32})

    # Generate B-Spline Parameters
    bspline_q = BSpline(num_pts=Config.bspline_q.n_ctr_pts, degree=Config.bspline_q.degree,
                        num_T_pts=Config.bspline_q.num_T_pts, device=device)
    bspline_t = BSpline(num_pts=Config.bspline_t.n_ctr_pts, degree=Config.bspline_t.degree,
                        num_T_pts=Config.bspline_t.num_T_pts, device=device)

    # Generate data for training
    # training_loader, validation_loader = load_data(Config.train.batch_size, device)
    train_path = Config.data.uniform_path
    # train_path = train_path.replace("data.tsv", "prepare_data_5000.tsv")
    violate_path = Config.data.violate_path

    training_loader, validation_loader = get_hitting_data(batch_size=Config.train.batch_size, device=device, path=train_path, shuffle=True)
    violate_loader = get_violate_data(batch_size=Config.train.batch_size, device=device, path=violate_path, shuffle=True)
    Limits.to_device(device)

    # Initialize the model
    if Config.load.enable:
        load_epoch = Config.load.epoch
        model = only_load_model(load_epoch, config=Config)
    else:
        model = NNPlanner(Config.bspline_q.n_ctr_pts, Config.bspline_t.n_ctr_pts, bspline_q, bspline_t,
                          Config.model.n_pts_fixed_begin, Config.model.n_pts_fixed_end, Config.model.n_dof)

    # Define Optimizer and Loss Function
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=Config.train.learning_rate,
                                  weight_decay=Config.train.weight_decay,
                                  eps=Config.train.epsilon)
    loss_fn = HittingLoss(manipulator, air_hockey_table, air_hockey_puck, bspline_q, bspline_t, Limits.q7, Limits.q_dot7, Limits.q_ddot7, Limits.q_dddot7, device=device)

    # set up wandb
    os.environ["WANDB_API_KEY"] = Config.wandb.api_key
    wandb.init(project=Config.wandb.project_name, id=Config.wandb.continue_id, resume=Config.wandb.resume)
    wandb.watch(model, log='gradients', log_freq=Config.train.batch_size)
    # wandb.watch(model, log='parameters', log_freq=Config.train.batch_size)

    load_epoch = 0
    if Config.load.enable:
        load_epoch = Config.load.epoch
        _, loss_fn, optimizer = load_model(loss_fn, optimizer, load_epoch, config=Config)

    for epoch in range(load_epoch+1, Config.train.num_epochs+1):
        start_time = time.time()
        print('EPOCH {}:'.format(epoch))

        train_metrics = train_epoch(model, loss_fn, optimizer, training_loader, violate_loader, device, is_training=True)

        loss_fn.alpha_update(train_metrics['train/q_loss'].item(),
                             train_metrics['train/q_dot_loss'].item(),
                             train_metrics['train/q_ddot_loss'].item(),
                             train_metrics['train/q_dddot_loss'].item(),
                             train_metrics['train/constraint_loss'].item(),
                             train_metrics['train/obstacle_loss'].item())

        train_metrics.update({
            'alpha/alpha_constraint': loss_fn.alpha_constraint,
            'alpha/alpha_obstacle': loss_fn.alpha_obstacle,
            'alpha/alpha_q': loss_fn.alpha_q,
            'alpha/alpha_q_dot': loss_fn.alpha_q_dot,
            'alpha/alpha_q_ddot': loss_fn.alpha_q_ddot,
            'alpha/alpha_q_dddot': loss_fn.alpha_q_dddot,
        })

        with torch.no_grad():
            validation_metrics = train_epoch(model, loss_fn, None, validation_loader, device, is_training=False)
            train_metrics.update(validation_metrics)

        # if device.type == 'cuda':
        #     torch.cuda.empty_cache()

        epoch_time = time.time() - start_time

        wandb.log(train_metrics)
        print("T: %s" % epoch_time)

        if Config.save.enable:
            if epoch % Config.save.interval == 0:
                save_model(model, loss_fn, optimizer, epoch, config=Config)

    wandb.finish()


if __name__ == '__main__':
    train()
