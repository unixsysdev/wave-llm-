#!/bin/bash
# Run with visualizations - saves frames every 500 epochs

cd "$(dirname "$0")"

python train_visual.py \
    --p 113 \
    --d_model 128 \
    --n_heads 4 \
    --d_mlp 512 \
    --train_frac 0.3 \
    --lr 1e-3 \
    --weight_decay 1.0 \
    --n_epochs 30000 \
    --snapshot_every 500 \
    --wave_modes fnet learned_gate \
    --output_dir results_visual
