#!/bin/bash
# Train wave adapter and compare with baseline

source ../venv/bin/activate

# Quick test run (small dataset)
echo "Running training experiment..."
echo "This trains:"
echo "  1. Baseline: Frozen Qwen3 + simple classifier"
echo "  2. Wave Adapter: Frozen Qwen3 + wave attention layers + classifier"
echo ""

python train_wave_adapter.py \
    --model "Qwen/Qwen3-0.6B" \
    --epochs 3 \
    --batch_size 8 \
    --lr 1e-4 \
    --train_samples 5000 \
    --test_samples 1000 \
    --num_adapter_layers 2

echo ""
echo "Results saved to: results/wave_adapter_results.json"
