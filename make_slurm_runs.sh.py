from pathlib import Path

run_sources = Path.cwd() / 'benchmarks/run_sources/'

config4ib = Path.cwd() / 'benchmarks/run_sources/IB/config4ib.yaml'

config4mtc = Path.cwd() / 'benchmarks/run_sources/MTC/config4mtc.yaml'
config4mtc_rel = Path.cwd() / 'benchmarks/run_sources/MTC/config4mtc_relative.yaml'

plagih_startpy = Path.cwd() / 'start.py'

SLURM_RUNS = Path.cwd() / 'run_examples/slurm_runs/'
if not SLURM_RUNS.is_dir():
    Path.mkdir(SLURM_RUNS)  # just to be sure

print('Make sure that this file is executed on top level')

# a_velocity,a_gain,a_shift
run_starts = ['IB_MSE_50_0', 'IB_MSE_50_1', 'IB_MSE_50_2',
              'IB_MSE_rel_50_0', 'IB_MSE_rel_50_1', 'IB_MSE_rel_50_2',
              'IB_MSE_tanh_50_0', 'IB_MSE_tanh_50_1', 'IB_MSE_tanh_50_2',

              # 'IB_mean_0', 'IB_mean_1', 'IB_mean_2',
              'IB_MSE_udluft_0', 'IB_MSE_udluft_1', 'IB_MSE_udluft_2'

              'IB_MSE_scratch_0', 'IB_MSE_scratch_1', 'IB_MSE_scratch_2',
              # 'IB_MSE_tanh_udluft_0', 'IB_MSE_tanh_udluft_1', 'IB_MSE_tanh_udluft_2',

              'IB_MSE_sim1_0', 'IB_MSE_sim1_1', 'IB_MSE_sim1_2',


              # 'MTC200_scratch',
              # 'MTC200_gpFfriendly',
              # 'MTC200_preset',
              # 'MTC200_simple',
              # 'MTC200_simple_fix',
              # 'MTC200_simplePlus_fix',
              # 'MTC200_simplePlus',
              #
              # 'MTC75_scratch',
              # 'MTC75_simple',
              # 'MTC75_simple_fix',
              #
              # 'MTC200_rel_simple',
              # 'MTC200_rel_gpfriendly_fix',
              'MTC200_rel_preset_fix',
              'MTC200_rel_simple_fix',

              'MTC200tan_simple_fix',
              'MTC200_rel_tan_simple_fix',
              ]

complete_params = []

for name in run_starts:
    out_path = str(SLURM_RUNS / name)
    final_line = f'python3 {plagih_startpy} -config_lookup {name}'

    # if len(str(param[2])) > 0:
    #     origin_param = f' -origin_tree {param[2]}'
    #     final_line += origin_param

    single_sh = '#!/usr/bin/env bash\n' \
                '#-*- coding:utf-8 -*-\n' \
        f'echo Starting run in {out_path}\n' \
        f'{final_line}'
    with Path.open(SLURM_RUNS / f'{name}.sh', 'w') as sh_file:
        sh_file.write(single_sh)

    complete_params.append(final_line)

print('\n'.join(complete_params))
print('\n')

all_sbatchs = '\n'.join(['sbatch --partition=All {}'.format(SLURM_RUNS / f'{x}.sh') for x in run_starts])
sbatch_sh = '#!/usr/bin/env bash\n' + all_sbatchs
with Path.open(Path('start_all_slurm_sbatch_jobs.sh'), 'w') as sh_file:
    sh_file.write(sbatch_sh)
print(sbatch_sh)
