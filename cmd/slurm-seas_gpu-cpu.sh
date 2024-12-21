#!/bin/bash
#SBATCH --job-name=data
#SBATCH -c 32
#SBATCH -N 1
#SBATCH -t 2-00:00                                              # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu                                             # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=256GB

#SBATCH -o /n/home06/zhentingqi/slurm_out/data_%j.out       # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/home06/zhentingqi/slurm_out/data_%j.err       # File to which STDERR will be written, %j inserts jobid

#SBATCH --mail-type=FAIL,END    # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=zhentingqi@g.harvard.edu

# --- load env here ---
module load python/3.10.12-fasrc01
source activate
source activate lit
# ---------------------

python -c 'print("Hi Zhenting. Your job is running!")'

# --- run your code here ---
bash jobs/prepare_data/prepare_fineweb.sh
# --------------------------

python -c 'print("Hi Zhenting. Everything is done!")'
