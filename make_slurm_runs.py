from pathlib import Path

print('Make sure that this file is executed on top level')

"""
gpFriendly: CORRUPT TREE CORRUPT RUNS
preset: CORRUPT RUNS
"""
run_starts = [

    # """
    # MC200 (SARSA-Agent after 200 training steps)
    # """
    # 'MTC200_MAE_explun01_simpleFix',
    # 'MTC200_MAE_tanh_simpleFix',
    # 'MTC200_MAE_explun01_tanh_simpleFix',
    # 'MTC200_RMSE_explun01_tanh_simpleFix',

    'MTC200_MAE_scratch',
    'MTC200_MAE_simple',
    'MTC200_MAE_simpleFix',
    # 'MTC200_MAE_simplePlus',
    # 'MTC200_MAE_simplePlusFix',
    'MTC200_MAE_simonBest',
    'MTC200_MAE_simonBestFix',
    'MTC200_MAE_simonBestFix2',
    'MTC200_MAE_xiao',
    'MTC200_MAE_xiaoFix',
    # 'MTC200_MAE_simonOkay',
    # 'MTC200_MAE_simonOkayFix',

    'MTC200_MAE_explun01_simple',
    'MTC200_MAE_explun01_simonBestFix2',

    'MTC200_MSE_scratch',
    'MTC200_MSE_simple',
    'MTC200_MSE_simpleFix',
    # 'MTC200_MSE_simplePlus',
    # 'MTC200_MSE_simplePlusFix',
    'MTC200_MSE_simonBest',
    'MTC200_MSE_simonBestFix',
    'MTC200_MSE_simonBestFix2',
    'MTC200_MSE_xiao',
    'MTC200_MSE_xiaoFix',

    'MTC200_MSE_explun01_simple',
    'MTC200_MSE_explun01_simonBestFix2',

    # # # """
    # # # MC75 (SARSA-Agent after 75 training steps)
    # # # """
    'MTC75_MAE_scratch',
    'MTC75_MAE_simple',
    'MTC75_MAE_simpleFix',

    'MTC75_MSE_scratch',
    'MTC75_MSE_simple',
    'MTC75_MSE_simpleFix',

    'MTC75_MAE_explun01_simple',
    'MTC75_MSE_explun01_simple',

    # """
    # IB
    # """
    'IB_MAE_scratch_0', 'IB_MAE_scratch_1', 'IB_MAE_scratch_2',
    # 'IB_MAE_tanh_scratch_0', 'IB_MAE_tanh_scratch_1', 'IB_MAE_tanh_scratch_2',
    # 'IB_MAE_sim2_0', 'IB_MAE_sim2_1', 'IB_MAE_sim2_2',
    # 'IB_MAE_explun01_sim2_0', 'IB_MAE_explun01_sim2_1', 'IB_MAE_explun01_sim2_2',
    # 'IB_MAE_tanh_sim2_0', 'IB_MAE_tanh_sim2_1', 'IB_MAE_tanh_sim2_2',
    # 'IB_MAE_explun01_tanh_sim2_0', 'IB_MAE_explun01_tanh_sim2_1', 'IB_MAE_explun01_tanh_sim2_2',
    # 'IB_MAE_tanh_udluft_0', 'IB_MAE_tanh_udluft_1', 'IB_MAE_tanh_udluft_2',

    'IB_RMSE_scratch_0', 'IB_RMSE_scratch_1', 'IB_RMSE_scratch_2',
    # 'IB_RMSE_s3m_0', 'IB_RMSE_s3m_1', 'IB_RMSE_s3m_2',
    # 'IB_RMSE_tanh_s3m_0', 'IB_RMSE_tanh_s3m_1', 'IB_RMSE_tanh_s3m_2',
    # 'IB_RMSE_explun01_s3m_0', 'IB_RMSE_explun01_s3m_1', 'IB_RMSE_explun01_s3m_2',
    # # 'IB_RMSE_sim2_0', 'IB_RMSE_sim2_1', 'IB_RMSE_sim2_2',
    # # 'IB_RMSE_explun01_sim2_0', 'IB_RMSE_explun01_sim2_1', 'IB_RMSE_explun01_sim2_2',
    # # 'IB_RMSE_tanh_sim2_0', 'IB_RMSE_tanh_sim2_1', 'IB_RMSE_tanh_sim2_2',
    # # 'IB_RMSE_explun01_tanh_sim2_0', 'IB_RMSE_explun01_tanh_sim2_1', 'IB_RMSE_explun01_tanh_sim2_2',
    # # 'IB_RMSE_explun01_50_0', 'IB_RMSE_explun01_50_1', 'IB_RMSE_explun01_50_2',
    # # 'IB_RMSE_tanh_50_0', 'IB_RMSE_tanh_50_1', 'IB_RMSE_tanh_50_2',
    # # 'IB_RMSE_mean_0', 'IB_RMSE_mean_1', 'IB_RMSE_mean_2',
    # # 'IB_RMSE_scratch_0', 'IB_RMSE_scratch_1', 'IB_RMSE_scratch_2',
    # # 'IB_RMSE_udluft_0', 'IB_RMSE_udluft_1', 'IB_RMSE_udluft_2'

    'IB_MSE_scratch_0', 'IB_MSE_scratch_1', 'IB_MSE_scratch_2',
    # 'IB_MSE_mean_0', 'IB_MSE_mean_1', 'IB_MSE_mean_2',
    'IB_MSE_s3m_0', 'IB_MSE_s3m_1', 'IB_MSE_s3m_2',
    'IB_MSE_50_0', 'IB_MSE_50_1', 'IB_MSE_50_2',

    # 'IB_MSE_tanh_s3m_0', 'IB_MSE_tanh_s3m_1', 'IB_MSE_tanh_s3m_2',
    # 'IB_MSE_explun01_s3m_0', 'IB_MSE_explun01_s3m_1', 'IB_MSE_explun01_s3m_2',
    # # 'IB_MSE_sim2_0', 'IB_MSE_sim2_1', 'IB_MSE_sim2_2',
    # # 'IB_MSE_explun01_sim2_0', 'IB_MSE_explun01_sim2_1', 'IB_MSE_explun01_sim2_2',
    # # 'IB_MSE_tanh_sim2_0', 'IB_MSE_tanh_sim2_1', 'IB_MSE_tanh_sim2_2',
    # # 'IB_MSE_explun01_tanh_sim2_0', 'IB_MSE_explun01_tanh_sim2_1', 'IB_MSE_explun01_tanh_sim2_2',

]

complete_params = []

# sfeh -D, --chdir=<directory>
#     Set the working directory of the batch script to directory before it is executed. The path can be specified as full path or relative path to the directory where the command is executed.
# sfeh --output=./benchmarks/slurm_runs/{run_name}/slurm-%j.out # not used anymore cause its shit
# --exclusive (only you are allowed to work on the machine)
# --cpus-per-task=8 is better, but everyone uses these engines
sbatch_sh = "#!/usr/bin/env bash\n" + '\n'.join(
    [f'sbatch --partition=All ./benchmarks/linux_start_slurm_run.sh {run_name} $1 $2 $3 $4' for run_name in run_starts]
    + [f'sbatch --partition=All ./benchmarks/linux_start_slurm_run.sh {run_name} -slurm_runs_folder slurm_runs2 $1 $2 $3 $4' for run_name in run_starts]
    + [f'sbatch --partition=All ./benchmarks/linux_start_slurm_run.sh {run_name} -slurm_runs_folder slurm_runs3_easy -sfeh_no_crazyops $1 $2 $3 $4' for run_name in run_starts])

with Path.open(Path('start_all_slurm_sbatch_jobs.sh'), 'w') as sh_file:
    sh_file.write(sbatch_sh)
print(sbatch_sh)
