#!/bin/bash
# Check if dirichlet-alpha is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <dirichlet-alpha>"
    exit 1
fi

DIRICHLET_ALPHA=$1


OUTFOLDER="fed_diff_kd_vt_cnn_cifar100_dir_${DIRICHLET_ALPHA}"
# Define GPU index from the argument
GPU=$2
# Define the seeds
SEEDS=(42 48 123 2023 9999)

# Iterate over each seed and run the command
for SEED in "${SEEDS[@]}"; do
    echo "Running distillation with seed $SEED..."
    python main.py --run_dec_distillation \
        --exp-name fed_diff_kd_vt \
        --dataset "cifar100" \
        --model "cnn" \
        --dirichlet \
        --dirichlet-alpha $DIRICHLET_ALPHA \
        --aggregation-func fed_diff \
        --kd-alpha 1 \
        --skd-beta 0.99 \
        --vteacher_generator fixed \
        --communication-rounds 2000 \
        --local-ep 5 \
        --local-distil-ep 40 \
        --outfolder  $OUTFOLDER \
        --gpu $GPU \
        --val-split 0.2 \
        --zipf-alpha 1.6 \
        --lr 0.0001 \
        --momentum 0.9 \
        --local-bs 100 \
        --graph-synth-type "barabasi_albert_graph" \
        --graph-synth-args "50,5,7" \
        --seed $SEED
    echo "Finished distillation run with seed $SEED"
done
