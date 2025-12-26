#!/bin/bash
# Run phase attention comparison experiment

python phase_attention_experiment.py --model "Qwen/Qwen3-0.6B"

echo ""
echo "Results:"
echo "  - results/attention_comparison.png: Visual comparison of attention patterns"
echo "  - results/correlation_comparison.png: Which method best captures semantics"
echo "  - results/attention_analysis.json: Detailed numerical results"
