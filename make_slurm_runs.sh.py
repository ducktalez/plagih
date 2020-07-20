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
    'IB_50s_0': ['ib', 'ib', 'a_velocity', run_sources / 'IB/ib_tree_50s_0.csv'],
    'IB_50s_1': ['ib', 'ib', 'a_gain', run_sources / 'IB/ib_tree_50s_1.csv'],
    'IB_50s_2': ['ib', 'ib', 'a_shift', run_sources / 'IB/ib_tree_50s_2.csv'],
    'IB_mean_0': ['ib', 'ib', 'a_velocity', run_sources / 'IB/ib_tree_mean_0.csv'],
    'IB_mean_1': ['ib', 'ib', 'a_gain', run_sources / 'IB/ib_tree_mean_1.csv'],
    'IB_mean_2': ['ib', 'ib', 'a_shift', run_sources / 'IB/ib_tree_mean_2.csv'],
    'IB_udluft_0': ['ib', 'ib', 'a_velocity', run_sources / 'IB/ib_tree_udluft_0.csv'],
    'IB_udluft_1': ['ib', 'ib', 'a_gain', run_sources / 'IB/ib_tree_udluft_1.csv'],
    'IB_udluft_2': ['ib', 'ib', 'a_shift', run_sources / 'IB/ib_tree_udluft_2.csv'],
    'IB_scratch_0': ['ib', 'ib', 'a_velocity', ''],
    'IB_scratch_1': ['ib', 'ib', 'a_gain', ''],
    'IB_scratch_2': ['ib', 'ib', 'a_shift', ''],

    'MTC200_scratch': ['mtc', 'mtc200', None, ''],
    'MTC200_GP_friendly': ['mtc', 'mtc200', None, run_sources / 'MTC/tree_gpFriendly_fix.csv'],
    'MTC200_preset': ['mtc', 'mtc200',None, run_sources / 'MTC/tree_preset_fix.csv'],
    'MTC200_tree_simple': ['mtc', 'mtc200', None, run_sources / 'MTC/tree_simple.csv'],
    'MTC200_tree_simple_fix': ['mtc', 'mtc200', None, run_sources / 'MTC/tree_simple_fix.csv'],
    'MTC200_tree_simplePlus_fix': ['mtc', 'mtc200', None, run_sources / 'MTC/tree_simplePlus_fix.csv'],
    # 'MTC200_tree_simplePlus': [config4mtc, 'mtc200', None, 'MTC/tree_simplePlus.csv'],

    'MTC75_scratch': ['mtc', 'mtc75', None, ''],
    'MTC75_tree_simple': ['mtc', 'mtc75', None, run_sources / 'MTC/tree_simple.csv'],
    'MTC75_tree_simple_fix': ['mtc', 'mtc75', None, run_sources / 'MTC/tree_simple_fix.csv'],

    'MTC200rel_tree_simple': ['mtc_rel', 'mtc200', None, run_sources / 'MTC/tree_simple.csv'],
    'MTC200rel_gpfriendly_fix': ['mtc_rel', 'mtc200', None, run_sources / 'MTC/tree_gpFriendly_fix.csv'],
    'MTC200rel_preset_fix': ['mtc_rel', 'mtc200', None, run_sources / 'MTC/tree_preset_fix.csv'],
}

main_format = 'python3 ' + str(plagih_startpy) + ' -config_lookup {} -out_dir {} -samples_lookup {} -action {}'
complete_params = []

for name, param in run_starts.items():
    param0 = param[0]
    param1 = param[1]
    param2 = param[2]
    out_path = str(SLURM_RUNS / name)
    final_line = main_format.format(param0, out_path, param1, str(param2))

    if len(str(param[3])) > 0:
        param3 = param[3]
        origin_param = f' -origin_tree {param3}'
        final_line += origin_param

    single_sh = '#!/usr/bin/env bash\n' \
                '#-*- coding:utf-8 -*-\n' \
                f'echo Starting run in {out_path}\n' \
                f'{final_line}'
    with Path.open(SLURM_RUNS / f'{name}.sh', 'w') as sh_file:
        sh_file.write(single_sh)

    complete_params.append(final_line)

print('\n'.join(complete_params))
print('\n')

allstuff = '\n'.join(['sbatch --partition=All {}'.format(SLURM_RUNS / f'{x}.sh') for x in run_starts.keys()])
sbatch_sh = '#!/usr/bin/env bash\n' + allstuff
with Path.open(Path('start_all_slurm_sbatch_jobs.sh'), 'w') as sh_file:
    sh_file.write(sbatch_sh)
print(sbatch_sh)
