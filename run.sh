#!/bin/bash
# Run wave token analysis on Qwen3 0.6B

# Activate virtual environment
source ../venv/bin/activate

# Run analysis
python visualize_waves.py --model "Qwen/Qwen3-0.6B" --n_tokens 1000

echo ""
echo "Results saved to: results/"
echo "Key files:"
echo "  - results/summary.json: Numerical results"
echo "  - results/pair_*.png: Token pair comparisons"
echo "  - results/frequency_clusters.png: Category clustering"
echo "  - results/dominant_frequencies.png: Global frequency analysis"
