#!/bin/bash

# Check if dirichlet-alpha is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <dirichlet-alpha>"
    exit 1
fi
DIRICHLET_ALPHA=$1
# Define the seeds
SEEDS=(42 48 123 2023 9999)
# Define GPU index from the argument
GPU=$2
# Iterate over each seed and run the command
for SEED in "${SEEDS[@]}"; do
    echo "Running heterogeneous federated learning with Dirichlet distribution and seed $SEED..."
    python main.py --run_dec \
        --exp-name fed_heter \
        --dataset "fashion_mnist" \
        --model "cnn" \
        --dirichlet \
        --dirichlet-alpha $DIRICHLET_ALPHA \
        --scale_weights \
        --communication-rounds 1000 \
        --local-ep 5 \
        --local-distil-ep 40 \
        --outfolder "fed_heter_scale_cnn_fmnist_dir_$DIRICHLET_ALPHA" \
        --gpu $GPU \
        --val-split 0.2 \
        --zipf-alpha 1.6 \
        --lr 0.001 \
        --momentum 0.9 \
        --local-bs 100 \
        --graph-synth-type "barabasi_albert_graph" \
        --graph-synth-args "50,5,7" \
        --seed $SEED
    echo "Finished heterogeneous federated learning with Dirichlet distribution run for seed $SEED"
done
