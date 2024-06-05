from itertools import product

from experiment_launcher import Launcher, is_local

LOCAL = is_local()
TEST = False
USE_CUDA = True

PARTITION = 'stud'  # 'amd', 'rtx'
GRES = 'gpu:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
CONDA_ENV = 'neural_planner'  # None

launcher = Launcher(
    exp_name='neural_planner',
    exp_file='experiment',
    # project_name='project01234',  # for hrz cluster
    n_seeds=1,
    n_exps_in_parallel=1,
    n_cores=2,
    memory_per_core=900,
    days=0,
    hours=23,
    minutes=59,
    seconds=59,
    partition=PARTITION,
    gres=GRES,
    conda_env=CONDA_ENV,
    use_timestamp=True,
    compact_dirs=False
)

wandb_options = dict(
  wandb_enabled=True,  # If True, runs and logs to wandb.
  wandb_entity='yuheng_ouyang',
  wandb_project='neural_planner',
)

launcher.add_experiment(
    # A subdirectory will be created for parameters with a trailing double underscore.
    debug=False,
    wandb_options=wandb_options,
)

launcher.run(LOCAL, TEST)
