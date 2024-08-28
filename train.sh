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
# cmd="python main_hessian.py --task $task"
# cmd="python main_hessian.py --explicit_hessian_regularization 0.01"

# Multiplication task: easy to grok for split_ratio=0.5
cmd="python main_hessian.py --task multiplication --nsm --nsm_sigma 0.0"
cmd="python main_hessian.py --task multiplication --nsm --nsm_sigma 0.01"
cmd="python main_hessian.py --task multiplication --nsm --nsm_sigma 0.03"
cmd="python main_hessian.py --task multiplication --nsm --nsm_sigma 0.1"
# Quad1 task: set split_ratio=0.75
cmd="CUDA_VISIBLE_DEVICES=0 python main_hessian.py --task quad1 --nsm --nsm_sigma 0.0 --split_ratio 0.75"
cmd="CUDA_VISIBLE_DEVICES=1 python main_hessian.py --task quad1 --nsm --nsm_sigma 0.01 --split_ratio 0.75"
cmd="CUDA_VISIBLE_DEVICES=2 python main_hessian.py --task quad1 --nsm --nsm_sigma 0.03 --split_ratio 0.75"
cmd="CUDA_VISIBLE_DEVICES=3 python main_hessian.py --task quad1 --nsm --nsm_sigma 0.1 --split_ratio 0.75"

set -x
srun -l \
    bash -c "
    $cmd
    "

