#!/bin/bash
# Distill Qwen3-0.6B attention into wave layers

cd "$(dirname "$0")"

# Default: learned_gate (fastest to train, proven on grokking)
python distill_wave_qwen.py \
    --teacher "Qwen/Qwen3-0.6B" \
    --wave_type learned_gate \
    --n_samples 1000 \
    --batch_size 4 \
    --n_epochs 3 \
    --lr 1e-4 \
    --max_seq_len 256 \
    --temperature 2.0 \
    --alpha 0.5 \
    --output_dir results

# Uncomment to try other wave types:
# --wave_type fnet
# --wave_type wave_network
# --wave_type frequency_band
