#!/bin/bash

# Run training with different rollout values in parallel
echo "Starting parallel training runs..."

uv run python -m tinkering2.train rollouts=5 &
uv run python -m tinkering2.train rollouts=8 &
uv run python -m tinkering2.train rollouts=16 &

wait

echo "All training runs completed!"
