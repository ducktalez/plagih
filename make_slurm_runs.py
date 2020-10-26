from pathlib import Path

SLURM_RUNS = Path.cwd() / 'benchmarks/slurm_runs/'  # sfeh lel? not needed?
if not SLURM_RUNS.is_dir():
    Path.mkdir(SLURM_RUNS)  # just to be sure

print('Make sure that this file is executed on top level')

# a_velocity,a_gain,a_shift
run_starts = [
    'IB_RMSE_s3m_0', 'IB_RMSE_s3m_1', 'IB_RMSE_s3m_2',
    'IB_RMSE_tanh_s3m_0', 'IB_RMSE_tanh_s3m_1', 'IB_RMSE_tanh_s3m_2',
    'IB_RMSE_explun01_s3m_0', 'IB_RMSE_explun01_tanh_s3m_1', 'IB_RMSE_explun01_tanh_s3m_2',

    'IB_MSE_s3m_0', 'IB_MSE_s3m_1', 'IB_MSE_s3m_2',
    'IB_MSE_tanh_s3m_0', 'IB_MSE_tanh_s3m_1', 'IB_MSE_tanh_s3m_2',
    'IB_MSE_explun01_tanh_s3m_0', 'IB_MSE_explun01_tanh_s3m_1', 'IB_MSE_explun01_tanh_s3m_2',

    # 'IB_MAE_scratch_0', 'IB_MAE_scratch_1', 'IB_MAE_scratch_2',
    'IB_MSE_scratch_0', 'IB_MSE_scratch_1', 'IB_MSE_scratch_2',
    # 'IB_MAE_tanh_scratch_0', 'IB_MAE_tanh_scratch_1', 'IB_MAE_tanh_scratch_2',

    # # # sim2 # #
    # 'IB_MAE_sim2_0', 'IB_MAE_sim2_1', 'IB_MAE_sim2_2',
    # 'IB_MAE_explun01_sim2_0', 'IB_MAE_explun01_sim2_1', 'IB_MAE_explun01_sim2_2',
    # 'IB_MAE_tanh_sim2_0', 'IB_MAE_tanh_sim2_1', 'IB_MAE_tanh_sim2_2',
    # 'IB_MAE_explun01_tanh_sim2_0', 'IB_MAE_explun01_tanh_sim2_1', 'IB_MAE_explun01_tanh_sim2_2',
    #
    # 'IB_MSE_sim2_0', 'IB_MSE_sim2_1', 'IB_MSE_sim2_2',
    # 'IB_MSE_explun01_sim2_0', 'IB_MSE_explun01_sim2_1', 'IB_MSE_explun01_sim2_2',
    # 'IB_MSE_tanh_sim2_0', 'IB_MSE_tanh_sim2_1', 'IB_MSE_tanh_sim2_2',
    # 'IB_MSE_explun01_tanh_sim2_0', 'IB_MSE_explun01_tanh_sim2_1', 'IB_MSE_explun01_tanh_sim2_2',

    # # RMSE (Root mean sqare error)
    # 'IB_RMSE_sim2_0', 'IB_RMSE_sim2_1', 'IB_RMSE_sim2_2',
    # 'IB_RMSE_explun01_sim2_0', 'IB_RMSE_explun01_sim2_1', 'IB_RMSE_explun01_sim2_2',
    # 'IB_RMSE_tanh_sim2_0', 'IB_RMSE_tanh_sim2_1', 'IB_RMSE_tanh_sim2_2',
    # 'IB_RMSE_explun01_tanh_sim2_0', 'IB_RMSE_explun01_tanh_sim2_1', 'IB_RMSE_explun01_tanh_sim2_2',

    # _50
    'IB_MSE_50_0', 'IB_MSE_50_1', 'IB_MSE_50_2',
    # 'IB_RMSE_explun01_50_0', 'IB_RMSE_explun01_50_1', 'IB_RMSE_explun01_50_2',
    # 'IB_RMSE_tanh_50_0', 'IB_RMSE_tanh_50_1', 'IB_RMSE_tanh_50_2',

    # 'IB_RMSE_mean_0', 'IB_RMSE_mean_1', 'IB_RMSE_mean_2',
    # 'IB_RMSE_scratch_0', 'IB_RMSE_scratch_1', 'IB_RMSE_scratch_2',

    # 'IB_RMSE_udluft_0', 'IB_RMSE_udluft_1', 'IB_RMSE_udluft_2'
    # 'IB_MAE_tanh_udluft_0', 'IB_MAE_tanh_udluft_1', 'IB_MAE_tanh_udluft_2',

    ### NOPE ###

    # 'IB_RMSE_sim1_0', 'IB_RMSE_sim1_1', 'IB_RMSE_sim1_2',

    # """
    # MC200 (SARSA-Agent after 200 training steps)
    # """
    # 'MTC200_MAE_explun01_simpleFix',
    # 'MTC200_MAE_tanh_simpleFix',
    # 'MTC200_MAE_explun01_tanh_simpleFix',
    # 'MTC200_RMSE_explun01_tanh_simpleFix',

    'MTC200_MAE_scratch',
    'MTC200_MAE_gpFriendly',
    'MTC200_MAE_gpfriendlyFix',
    'MTC200_MAE_preset',
    'MTC200_MAE_simple',
    'MTC200_MAE_simpleFix',
    'MTC200_MAE_simplePlus',
    'MTC200_MAE_simplePlusFix',
    'MTC200_MAE_simonBest',
    'MTC200_MAE_simonBad',

    'MTC200_MSE_scratch',
    'MTC200_MSE_gpFriendly',
    'MTC200_MSE_gpfriendlyFix',
    'MTC200_MSE_preset',
    'MTC200_MSE_simple',
    'MTC200_MSE_simpleFix',
    'MTC200_MSE_simplePlusFix',
    'MTC200_MSE_simplePlus',
    'MTC200_MSE_simonBest',
    'MTC200_MSE_simonBad',
    #
    # 'MTC200_MAE_explun01_simple',
    # 'MTC200_MAE_explun01_gpfriendlyFix',
    # 'MTC200_MAE_explun01_presetFix',
    #
    # # """
    # # MC75 (SARSA-Agent after 75 training steps)
    # # """
    'MTC75_MAE_scratch',
    'MTC75_MAE_simple',
    'MTC75_MAE_simpleFix',
    'MTC75_MAE_gpFriendly',

    'MTC75_MSE_scratch',
    'MTC75_MSE_simple',
    'MTC75_MSE_simpleFix',
]

complete_params = []

# sfeh -D, --chdir=<directory>
#     Set the working directory of the batch script to directory before it is executed. The path can be specified as full path or relative path to the directory where the command is executed.
# sfeh --output=./benchmarks/slurm_runs/{run_name}/slurm-%j.out # not used anymore cause its shit
# --exclusive (only you are allowed to work on the machine)
sbatch_sh = "#!/usr/bin/env bash\n" + '\n'.join(
    [f'sbatch --partition=All --exclusive ./benchmarks/linux_start_slurm_run.sh {run_name} $1 $2 $3 $4' for run_name in run_starts])  # --cpus-per-task=8 is better, but everyone uses these engines

with Path.open(Path('start_all_slurm_sbatch_jobs.sh'), 'w') as sh_file:
    sh_file.write(sbatch_sh)
print(sbatch_sh)
