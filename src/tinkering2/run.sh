#!/bin/bash

# Run training with different rollout values in parallel
echo "Starting parallel training runs..."

# uv run python -m tinkering2.train rollouts=8 reward_type="partial_strict" &
uv run python -m tinkering2.train rollouts=8 reward_type="partial_loose" batch_size=32 &
uv run python -m tinkering2.train rollouts=8 reward_type="partial_loose" batch_size=48 &
uv run python -m tinkering2.train rollouts=8 reward_type="partial_loose" batch_size=64 &

wait

echo "All training runs completed!"