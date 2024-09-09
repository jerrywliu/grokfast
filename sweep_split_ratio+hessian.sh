#!/bin/bash

split_ratios=$(seq 0.95 -0.05 0.1)
gpus=(0 1 2 3)
hessian_regularizations=(0.0 0.01 0.03 0.1)
cmd="python main.py --task parity_division --budget 300000 --seed 1 --nsm"

for ratio in $split_ratios; do
    for hessian_regularization in "${hessian_regularizations[@]}"; do
        while true; do
            for gpu in "${gpus[@]}"; do
                if ! screen -list | grep -q "gpu${gpu}"; then
                    full_cmd="source $(conda info --base)/etc/profile.d/conda.sh && conda activate grokking && CUDA_VISIBLE_DEVICES=${gpu} ${cmd} --split_ratio ${ratio} --nsm_sigma ${hessian_regularization} && exit"
                    screen -dmS "gpu${gpu}_ratio=${ratio}_nsm=${hessian_regularization}" bash -c "${full_cmd}"
                    echo "Started job with split_ratio ${ratio} and nsm_sigma ${hessian_regularization} on GPU ${gpu} in screen session gpu${gpu}_ratio=${ratio}_nsm=${hessian_regularization}"
                    sleep 1  # Small delay to avoid potential race conditions
                    break 2  # Exit both loops and continue with the next split_ratio
                fi
            done
            sleep 5  # Wait for a short period before checking again
        done
    done
done

# Wait for all screens to finish before exiting the script
while screen -list | grep -q "gpu"; do
    sleep 10
done

echo "All jobs completed."
