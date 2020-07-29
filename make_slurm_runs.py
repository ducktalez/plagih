from pathlib import Path

SLURM_RUNS = Path.cwd() / 'benchmarks/slurm_runs/'  # sfeh lel? not needed?
if not SLURM_RUNS.is_dir():
    Path.mkdir(SLURM_RUNS)  # just to be sure

print('Make sure that this file is executed on top level')

# a_velocity,a_gain,a_shift
run_starts = [
    # 'IB_RMSE_50_0', 'IB_RMSE_50_1', 'IB_RMSE_50_2',
    # 'IB_RMSE_explun_50_0', 'IB_RMSE_explun_50_1', 'IB_RMSE_explun_50_2',
    # 'IB_RMSE_tanh_50_0', 'IB_RMSE_tanh_50_1', 'IB_RMSE_tanh_50_2',

    # 'IB_RMSE_mean_0', 'IB_RMSE_mean_1', 'IB_RMSE_mean_2',
    # 'IB_RMSE_udluft_0', 'IB_RMSE_udluft_1', 'IB_RMSE_udluft_2'
    # 'IB_RMSE_scratch_0', 'IB_RMSE_scratch_1', 'IB_RMSE_scratch_2',

    # 'IB_RMSE_tanh_udluft_0', 'IB_RMSE_tanh_udluft_1', 'IB_RMSE_tanh_udluft_2',


    # 'IB_MSE_sim2_0', 'IB_MSE_sim2_1', 'IB_MSE_sim2_2',
    # 'IB_RMSE_sim2_0', 'IB_RMSE_sim2_1', 'IB_RMSE_sim2_2',
    'IB_RMSE_explun_tanh_sim2_0', 'IB_RMSE_explun_tanh_sim2_1', 'IB_RMSE_explun_tanh_sim2_2',
    'IB_MSE_tanh_sim2_0', 'IB_MSE_tanh_sim2_1', 'IB_MSE_tanh_sim2_2',
    'IB_MAE_tanh_sim2_0', 'IB_MAE_tanh_sim2_1', 'IB_MAE_tanh_sim2_2',
    'IB_MAE_explun_tanh_sim2_0', 'IB_MAE_explun_tanh_sim2_1', 'IB_MAE_explun_tanh_sim2_2',
    'IB_MAE_sim2_0', 'IB_MAE_sim2_1', 'IB_MAE_sim2_2',

    # 'MTC200_MAE_scratch',
    # 'MTC200_MAE_gpFfriendly',
    # 'MTC200_MAE_preset',
    # 'MTC200_MAE_simple',
    # 'MTC200_MAE_simple_fix',
    # 'MTC200_MAE_simplePlus_fix',
    # 'MTC200_MAE_simplePlus',
    #
    'MTC75_MAE_scratch',
    'MTC75_MAE_simple',
    'MTC75_MAE_simple_fix',

    'MTC200_MAE_explun_simple',
    'MTC200_MAE_explun_gpfriendly_fix',
    # 'MTC200_MAE_explun_preset_fix',
    # 'MTC200_MAE_explun_simple_fix',
    #
    # 'MTC200_MAE_tanh_simple_fix',
    # 'MTC200_MAE_explun_tanh_simple_fix',
    # 'MTC200_RMSE_explun_tanh_simple_fix',

    ### NOPE ###

    # 'IB_RMSE_sim1_0', 'IB_RMSE_sim1_1', 'IB_RMSE_sim1_2',
]

complete_params = []

# todo -D, --chdir=<directory>
#     Set the working directory of the batch script to directory before it is executed. The path can be specified as full path or relative path to the directory where the command is executed.
# sfeh run more beautiful?

# for run_name in run_starts:
#     Path.mkdir(Path.cwd() / f'benchmarks/slurm_runs/{run_name}')  # todo hate this done here

# sfeh --output=./benchmarks/slurm_runs/{run_name}/slurm-%j.out # not used anymore cause its shit
sbatch_sh = "#!/usr/bin/env bash\n" + '\n'.join(
    [f'sbatch --partition=All ./benchmarks/linux_start_slurm_run.sh {run_name} $1 $2 $3 $4' for run_name in run_starts])

with Path.open(Path('start_all_slurm_sbatch_jobs.sh'), 'w') as sh_file:
    sh_file.write(sbatch_sh)
print(sbatch_sh)
