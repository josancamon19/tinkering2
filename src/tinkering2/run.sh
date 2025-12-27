#!/bin/bash

# Run training with different rollout values in parallel
echo "Starting parallel training runs..."

uv run python -m tinkering2.train rollouts=5 reward_type="full_loose" &
uv run python -m tinkering2.train rollouts=8 reward_type="full_loose" &
uv run python -m tinkering2.train rollouts=16 reward_type="full_loose" &

wait

echo "All training runs completed!"