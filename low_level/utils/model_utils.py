import os
import torch

from config import Config


def save_model(model, loss_fn, optimizer, epoch):
    model_path = os.path.join(Config.save.path, 'Model_{}.pt'.format(epoch))
    misc_path = os.path.join(Config.save.path, 'Misc_{}.pt'.format(epoch))
    opt_path = os.path.join(Config.save.path, 'Opt_{}.pt'.format(epoch))
    torch.save(model, model_path)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'alpha_q': loss_fn.alpha_q,
        'alpha_q_dot': loss_fn.alpha_q_dot,
        'alpha_q_ddot': loss_fn.alpha_q_ddot,
        'alpha_q_dddot': loss_fn.alpha_q_dddot,
        'alpha_constraint': loss_fn.alpha_constraint,
        'alpha_obstacle': loss_fn.alpha_obstacle
    }, misc_path)
    torch.save(optimizer, opt_path)


def load_model(loss_fn, optimizer, epoch):
    model_path = os.path.join(Config.save.path, 'Model_{}.pt'.format(epoch))
    misc_path = os.path.join(Config.save.path, 'Misc_{}.pt'.format(epoch))
    model = torch.load(model_path)
    checkpoint = torch.load(misc_path)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss_fn.alpha_q = checkpoint['alpha_q']
    loss_fn.alpha_q_dot = checkpoint['alpha_q_dot']
    loss_fn.alpha_q_ddot = checkpoint['alpha_q_ddot']
    loss_fn.alpha_q_dddot = checkpoint['alpha_q_dddot']
    loss_fn.alpha_constraint = checkpoint['alpha_constraint']
    loss_fn.alpha_obstacle = checkpoint['alpha_obstacle']

    return model, loss_fn, optimizer


def only_load_model(epoch):
    model_path = os.path.join(Config.save.path, 'Model_{}.pt'.format(epoch))
    model = torch.load(model_path)
    return model