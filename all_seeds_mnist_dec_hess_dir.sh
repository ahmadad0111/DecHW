#!/bin/bash

# Check if dirichlet-alpha is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <dirichlet-alpha>"
    exit 1
fi

# Define dirichlet-alpha from the argument
DIRICHLET_ALPHA=$1
# Define GPU index from the argument
GPU=$2


# Define the seeds
SEEDS=(42 48 123 2023 9999)

# Iterate over each seed and run the command
for SEED in "${SEEDS[@]}"; do
    echo "Running with seed $SEED and dirichlet-alpha $DIRICHLET_ALPHA..."
    python main.py --run_dec_hess \
        --exp-name fed_diff_hess \
        --dataset "mnist" \
        --model "cnn" \
        --dirichlet \
        --dirichlet-alpha $DIRICHLET_ALPHA \
        --aggregation-func fed_diff_hessian_diag \
        --communication-rounds 1000 \
        --local-ep 5 \
        --local-distil-ep 40 \
        --outfolder "fed_diff_hess_cnn_mnist_dir_$DIRICHLET_ALPHA" \
        --gpu $GPU \
        --val-split 0.2 \
        --zipf-alpha 1.6 \
        --lr 0.001 \
        --momentum 0.5 \
        --local-bs 100 \
        --graph-synth-type "barabasi_albert_graph" \
        --graph-synth-args "50,5,7" \
        --seed $SEED
    echo "Finished running with seed $SEED and dirichlet-alpha $DIRICHLET_ALPHA"
done
