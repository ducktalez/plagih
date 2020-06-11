from pathlib import Path

samples4ib = 'IB/samples_ready.p'
config4ib = 'IB/config4ib.yaml'
config4mtc = 'MTC/config4mtc.yaml'
samples4mtc200 = 'MTC/MTC200_data_prepared.p'
samples4mtc75 = 'MTC/MTC75_data_prepared.p'
PLAGIH_ROOT = '../../../'
copypaste_path = PLAGIH_ROOT + 'benchmarks/run_load_files/'

SLURM_TASKS = 'slurm_tasks/'

run_starts = {
    'IB_50s_0': [config4ib, samples4ib, 0, 'IB/ib_tree_50s_0.csv'],
    'IB_50s_1': [config4ib, samples4ib, 1, 'IB/ib_tree_50s_1.csv'],
    'IB_50s_2': [config4ib, samples4ib, 2, 'IB/ib_tree_50s_2.csv'],
    'IB_mean_0': [config4ib, samples4ib, 0, 'IB/ib_tree_mean_0.csv'],
    'IB_mean_1': [config4ib, samples4ib, 1, 'IB/ib_tree_mean_1.csv'],
    'IB_mean_2': [config4ib, samples4ib, 2, 'IB/ib_tree_mean_2.csv'],
    'IB_udluft_0': [config4ib, samples4ib, 0, 'IB/ib_tree_udluft_0.csv'],
    'IB_udluft_1': [config4ib, samples4ib, 1, 'IB/ib_tree_udluft_1.csv'],
    'IB_udluft_2': [config4ib, samples4ib, 2, 'IB/ib_tree_udluft_2.csv'],
    'IB_scratch_0': [config4ib, samples4ib, 0, ''],
    'IB_scratch_1': [config4ib, samples4ib, 1, ''],
    'IB_scratch_2': [config4ib, samples4ib, 2, ''],

    'MTC200_scratch': [config4mtc, samples4mtc200, 0, ''],
    'MTC200_GP_friendly': [config4mtc, samples4mtc200, 0, 'MTC/tree_gpFriendly_fix.csv'],
    'MTC200_preset': [config4mtc, samples4mtc200, 0, 'MTC/tree_preset_fix.csv'],
    'MTC200_tree_simple': [config4mtc, samples4mtc200, 0, 'MTC/tree_simple.csv'],
    'MTC200_tree_simple_fix': [config4mtc, samples4mtc200, 0, 'MTC/tree_simple_fix.csv'],
    'MTC200_tree_simplePlus_fix': [config4mtc, samples4mtc200, 0, 'MTC/tree_simplePlus_fix.csv'],
    # 'MTC200_tree_simplePlus': [config4mtc, samples4mtc200, 0, 'MTC/tree_simplePlus.csv'],

    'MTC75_scratch': [config4mtc, samples4mtc75, 0, ''],
    'MTC75_tree_simple': [config4mtc, samples4mtc75, 0, 'MTC/tree_simple.csv'],
    'MTC75_tree_simple_fix': [config4mtc, samples4mtc75, 0, 'MTC/tree_simple_fix.csv'],
}

main_format = 'python3 ' + PLAGIH_ROOT + 'start.py -config {} -out_dir {} -samples_ready {} -action {}'
complete_params = []

for name, param in run_starts.items():
    folder = 'benchmarks/run_load_files/'
    param0 = copypaste_path + param[0]
    param1 = copypaste_path + param[1]
    param2 = param[2]
    out_path = 'run_examples/{}/'.format(name)
    rootpath = 'run_files/{}'.format(name)
    final_line = main_format.format(param0, out_path, param1, str(param2))

    if len(param[3]) > 0:
        param3 = copypaste_path + param[3]
        origin_param = ' -origin_tree {}'.format(param3)
        final_line += origin_param

    with Path.open(Path('slurm_tasks/{}.sh'.format(name)), 'w') as sh_file:
        fghj = '#!/usr/bin/env bash\n' \
               '#-*- coding:utf-8 -*-\n' \
               'echo Starting run in {}\n' \
               '{}'.format(out_path, final_line)
        sh_file.write(fghj)
    complete_params.append(final_line)

print('\n'.join(complete_params))
print('\n')
sbatch_sh = '#!/usr/bin/env bash\n' +\
            ('\n'.join(['sbatch --partition=All slurm_tasks/{}.sh'.format(x) for x in run_starts.keys()]))
with Path.open(Path('start_all_slurm_sbatch_jobs.sh'), 'w') as sh_file:
    sh_file.write(sbatch_sh)
print(sbatch_sh)
