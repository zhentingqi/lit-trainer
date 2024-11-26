#!/bin/bash
#SBATCH --job-name=eval
#SBATCH -c 16
#SBATCH -N 1
#SBATCH -t 2-00:00              # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu_requeue                  # Partition to submit to
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=256GB

#SBATCH -o /n/home06/zhentingqi/slurm_out/eval_%j.out      # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/home06/zhentingqi/slurm_out/eval_%j.err      # File to which STDERR will be written, %j inserts jobid

#SBATCH --mail-type=FAIL,END    # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=zhentingqi@g.harvard.edu

# --- load env here ---
module load python/3.10.12-fasrc01
source activate
source activate lit
# ---------------------

python -c 'print("Hi Zhenting. Your job is running!")'

# --- run your code here ---
bash jobs/evaluation/collect_responses-0.sh
bash jobs/evaluation/collect_responses-1.sh
bash jobs/evaluation/collect_responses-2.sh
bash jobs/evaluation/collect_responses-3.sh
bash jobs/evaluation/collect_responses-4.sh
bash jobs/evaluation/collect_responses-5.sh
bash jobs/evaluation/collect_responses-6.sh
# --------------------------

python -c 'print("Hi Zhenting. Everything is done!")'
