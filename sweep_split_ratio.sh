#!/bin/bash

# split_ratios=$(seq 0.1 0.05 0.95)
split_ratios=$(seq 0.95 -0.05 0.1)
gpus=(0 1 2 3)
cmd="python main.py --task quad1 --budget 300000 --seed 1"

for ratio in $split_ratios; do
    while true; do
        for gpu in "${gpus[@]}"; do
            if ! screen -list | grep -q "gpu${gpu}"; then
                full_cmd="source $(conda info --base)/etc/profile.d/conda.sh && conda activate grokking && CUDA_VISIBLE_DEVICES=${gpu} ${cmd} --split_ratio ${ratio} && exit"
                screen -dmS "gpu${gpu}_task_${ratio}" bash -c "${full_cmd}"
                echo "Started job with split_ratio ${ratio} on GPU ${gpu} in screen session gpu${gpu}_task_${ratio}"
                sleep 1  # Small delay to avoid potential race conditions
                break 2  # Exit both loops and continue with the next split_ratio
            fi
        done
        sleep 5  # Wait for a short period before checking again
    done
done

# Wait for all screens to finish before exiting the script
while screen -list | grep -q "gpu"; do
    sleep 10
done

echo "All jobs completed."
