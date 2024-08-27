#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=m4633
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --module=gpu,nccl-2.15

module load cudatoolkit/12.0
module load cudnn/8.9.3_cuda12
module load python
conda activate icon

task=$1
# cmd="python main_hessian.py --task $task --filter ema"
cmd="python main_hessian.py --task $task"

set -x
srun -l \
    bash -c "
    $cmd
    "

