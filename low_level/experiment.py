import os
from train_sample_points import train
# from train import train
import wandb

from experiment_launcher import run_experiment, single_experiment


# This decorator creates results_dir as results_dir/seed, and saves the experiment arguments into a file.
@single_experiment
def experiment(
    # MANDATORY
    seed: int = 0,
    results_dir: str = '../logs',
    **kwargs
):
    filename = os.path.join(results_dir, 'log_' + str(seed) + '.txt')
    out_str = f'Running experiment with seed {seed}'
    print(out_str)
    with open(filename, 'w') as file:
        file.write('Some logs in a log file.\n')
        file.write(out_str)
    wandb.init(project="neural_planner", id='s68svumr', resume='must')
    wandb.log({'seed': seed}, step=1)

    train()


if __name__ == '__main__':
    # Leave unchanged
    run_experiment(experiment)
