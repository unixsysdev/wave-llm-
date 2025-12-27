#!/bin/bash
# Generate from the trained wave model

# Find the most recent checkpoint
CHECKPOINT=$(ls -t results/wave_qwen_*.pt 2>/dev/null | head -1)

if [ -z "$CHECKPOINT" ]; then
    echo "No checkpoint found in results/"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT"
echo ""

# Run generation with comparison to teacher
python generate_wave.py \
    --model "$CHECKPOINT" \
    --compare \
    --max_tokens 30 \
    "$@"
