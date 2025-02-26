#!/bin/bash

# Define the seeds
SEEDS=(42 48 123 2023 9999)

# Iterate over each seed and run the command
for SEED in "${SEEDS[@]}"; do
    echo "Running Hessian-based fed_diff with seed $SEED..."
    python main.py --run_dec_hess \
        --exp-name fed_diff_hess \
        --dataset "cifar10" \
        --model "cnn" \
        --noniid \
        --aggregation-func fed_diff_hessian_diag \
        --communication-rounds 1000 \
        --local-ep 5 \
        --local-distil-ep 40 \
        --outfolder "fed_diff_hess_cnn_cifar10_noniid" \
        --gpu 0 \
        --val-split 0.2 \
        --zipf-alpha 1.6 \
        --lr 0.01 \
        --momentum 0.9 \
        --local-bs 100 \
        --graph-synth-type "barabasi_albert_graph" \
        --graph-synth-args "50,5,7" \
        --seed $SEED
    echo "Finished Hessian-based fed_diff run with seed $SEED"
done
