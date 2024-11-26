#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH -c 32
#SBATCH -N 1
#SBATCH -t 7-00:00                                              # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu                                             # Partition to submit to
#SBATCH --gres=gpu:4
#SBATCH --constraint=h100
#SBATCH --mem=256GB

#SBATCH -o /n/home06/zhentingqi/slurm_out/pretrain_%j.out       # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/home06/zhentingqi/slurm_out/pretrain_%j.err       # File to which STDERR will be written, %j inserts jobid

#SBATCH --mail-type=FAIL,END    # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=zhentingqi@g.harvard.edu

# --- load env here ---
module load python/3.10.12-fasrc01
source activate
source activate lit
# ---------------------

python -c 'print("Hi Zhenting. Your job is running!")'

# --- run your code here ---
litgpt pretrain --config config_hub/my_configs/myllama-500M-10BT.yaml
# --------------------------

python -c 'print("Hi Zhenting. Everything is done!")'
