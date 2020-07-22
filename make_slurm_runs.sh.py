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
run_starts = {
    'IB_50_0': ['ib', 'a_velocity', 'IB/ib_tree_50s_0.csv'],
    'IB_50_1': ['ib', 'a_gain', 'IB/ib_tree_50s_1.csv'],
    'IB_50_2': ['ib', 'a_shift', 'IB/ib_tree_50s_2.csv'],
    # 'IB_mean_0': ['ib', 'a_velocity', 'IB/ib_tree_mean_0.csv'],
    # 'IB_mean_1': ['ib', 'a_gain', 'IB/ib_tree_mean_1.csv'],
    # 'IB_mean_2': ['ib', 'a_shift', 'IB/ib_tree_mean_2.csv'],
    'IB_udluft_0': ['ib', 'a_velocity', 'IB/ib_tree_udluft_0.csv'],
    'IB_udluft_1': ['ib', 'a_gain', 'IB/ib_tree_udluft_1.csv'],
    'IB_udluft_2': ['ib', 'a_shift', 'IB/ib_tree_udluft_2.csv'],

    'IB_scratch_0': ['ib', 'a_velocity', ''],
    'IB_scratch_1': ['ib', 'a_gain', ''],
    'IB_scratch_2': ['ib', 'a_shift', ''],

    'IB_tanh_50_0': ['ibtanh', 'a_velocity', 'IB/ib_tree_50s_0.csv'],
    'IB_tanh_50_1': ['ibtanh', 'a_gain', 'IB/ib_tree_50s_1.csv'],
    'IB_tanh_50_2': ['ibtanh', 'a_shift', 'IB/ib_tree_50s_2.csv'],
    'IB_tanh_udluft_0': ['ibtanh', 'a_velocity', 'IB/ib_tree_udluft_0.csv'],
    'IB_tanh_udluft_1': ['ibtanh', 'a_gain', 'IB/ib_tree_udluft_1.csv'],
    'IB_tanh_udluft_2': ['ibtanh', 'a_shift', 'IB/ib_tree_udluft_2.csv'],

    'IB_sim1_0': ['', '', ''],
    'IB_sim1_1': ['', '', ''],
    'IB_sim1_2': ['', '', ''],

    'IB_rel_50_0': ['ibrel', 'a_velocity', 'IB/ib_tree_50s_0.csv'],
    'IB_rel_50_1': ['ibrel', 'a_gain', 'IB/ib_tree_50s_1.csv'],
    'IB_rel_50_2': ['ibrel', 'a_shift', 'IB/ib_tree_50s_2.csv'],

    'MTC200_scratch': ['mtc200', None, ''],
    'MTC200_gpFfriendly': ['mtc200', None, 'MTC/tree_gpFriendly_fix.csv'],
    'MTC200_preset': ['mtc200', None, 'MTC/tree_preset_fix.csv'],
    'MTC200_simple': ['mtc200', None, 'MTC/tree_simple.csv'],
    'MTC200_simple_fix': ['mtc200', None, 'MTC/tree_simple_fix.csv'],
    'MTC200_simplePlus_fix': ['mtc200', None, 'MTC/tree_simplePlus_fix.csv'],
    'MTC200_simplePlus': ['mtc200', None, 'MTC/tree_simplePlus.csv'],

    'MTC75_scratch': ['mtc75', None, ''],
    'MTC75_simple': ['mtc75', None, 'MTC/tree_simple.csv'],
    'MTC75_simple_fix': ['mtc75', None, 'MTC/tree_simple_fix.csv'],

    'MTC200_rel_simple': ['mtc200rel', None, 'MTC/tree_simple.csv'],
    'MTC200_rel_gpfriendly_fix': ['mtc200rel', None, 'MTC/tree_gpFriendly_fix.csv'],
    'MTC200_rel_preset_fix': ['mtc200rel', None, 'MTC/tree_preset_fix.csv'],
    'MTC200_rel_simple_fix': ['mtc200rel', None, 'MTC/tree_simple_fix.csv'],

    'MTC200tan_simple_fix': ['mtc200tanh', None, 'MTC/tree_simple_fix.csv'],
    'MTC200_rel_tan_simple_fix': ['mtc200tanhrel', None, 'MTC/tree_simple_fix.csv'],
}

complete_params = []

for name, param in run_starts.items():
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

all_sbatchs = '\n'.join(['sbatch --partition=All {}'.format(SLURM_RUNS / f'{x}.sh') for x in run_starts.keys()])
sbatch_sh = '#!/usr/bin/env bash\n' + all_sbatchs
with Path.open(Path('start_all_slurm_sbatch_jobs.sh'), 'w') as sh_file:
    sh_file.write(sbatch_sh)
print(sbatch_sh)
