from hrl_air_hockey.experiment_launcher import Launcher, is_local
from itertools import product
import yaml


config_file = './exp_configs/curriculum_exp.yaml'
with open(config_file) as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)

exp_params = cfg['experiments']
del cfg['experiments']

LOCAL = is_local()
TEST = False

cfg['use_cuda'] = exp_params['use_cuda']
exp_params['memory_per_core'] = exp_params['n_exps_in_parallel'] * exp_params['memory_single_job'] // exp_params['n_cores']
exp_params['partition'] = 'gpu' if exp_params['use_cuda'] else 'stud'
exp_params['gres'] = 'gpu:rtx3080:1' if exp_params['use_cuda'] else None  # gpu:rtx2080:1, gpu:rtx3080:1
del exp_params['memory_single_job']
del exp_params['use_cuda']

launcher = Launcher(**exp_params)

exp_list = list()

if 'sweep_params' in cfg.keys():
    sweep_param = cfg['sweep_params']
    del cfg['sweep_params']

    for param_tuple in product(*sweep_param.values()):
        exp_dict = dict()
        for j, key in enumerate(sweep_param.keys()):
            exp_dict[key] = param_tuple[j]
        exp_list.append(exp_dict)

if 'listed_exps' in cfg.keys():
    listed_exps = cfg['listed_exps']
    del cfg['listed_exps']

    for exp in listed_exps:
        exp_list.append(exp)


for exp_config in exp_list:
    launcher.add_experiment(
        **cfg,
        **exp_config,
    )

launcher.run(LOCAL, TEST)
