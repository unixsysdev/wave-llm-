#!/bin/bash
# Run the grokking comparison: Standard Attention vs Wave variants

cd "$(dirname "$0")"

python train_comparison.py \
    --p 53 \
    --d_model 128 \
    --n_heads 4 \
    --d_mlp 512 \
    --train_frac 0.3 \
    --lr 1e-3 \
    --weight_decay 1.0 \
    --n_epochs 30000 \
    --wave_modes fnet learned_gate phase_shift full \
    --output_dir results
