#!/bin/bash

tasks=("multiplication" "addition" "subtraction" "division" "parity_division" "sum_of_squares" "quad1" "quad2" "cubic1" "cubic2")

for task in "${tasks[@]}"
do
    sbatch ./train.sh $task
done