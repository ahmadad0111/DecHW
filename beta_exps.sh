#!/bin/bash

# Check if dirichlet-alpha is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <dirichlet-alpha> <hessian-beta>"
    exit 1
fi

# Check if hessian-beta is provided
if [ -z "$2" ]; then
    echo "Usage: $0 <dirichlet-alpha> <hessian-beta>"
    exit 1
fi

# Define dirichlet-alpha and hessian-beta from the arguments
DIRICHLET_ALPHA=$1
HESSIAN_BETA=$2
# Define GPU index from the argument
GPU=$3

# Define the seeds
SEEDS=(42)

# Iterate over each seed and run the command
for SEED in "${SEEDS[@]}"; do
    echo "Running with seed $SEED, dirichlet-alpha $DIRICHLET_ALPHA, and hessian-beta $HESSIAN_BETA..."
    python main.py --run_dec_hess \
        --exp-name fed_diff_hess \
        --dataset "cifar10" \
        --model "cnn" \
        --dirichlet \
        --dirichlet-alpha $DIRICHLET_ALPHA \
        --hessian-beta $HESSIAN_BETA \
        --aggregation-func fed_diff_hessian_diag \
        --communication-rounds 500 \
        --local-ep 5 \
        --local-distil-ep 40 \
        --outfolder "fed_diff_hess_cnn_cifar10_dir_${DIRICHLET_ALPHA}_beta_${HESSIAN_BETA}" \
        --gpu $GPU \
        --val-split 0.2 \
        --zipf-alpha 1.6 \
        --lr 0.01 \
        --momentum 0.9 \
        --local-bs 100 \
        --graph-synth-type "barabasi_albert_graph" \
        --graph-synth-args "50,5,7" \
        --seed $SEED
    echo "Finished running with seed $SEED, dirichlet-alpha $DIRICHLET_ALPHA, and hessian-beta $HESSIAN_BETA"
done
