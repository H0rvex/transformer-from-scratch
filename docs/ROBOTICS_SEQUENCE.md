# Robotics Sequence Modeling Extension

This extension adds a small robotics-adjacent sequence modeling task without claiming to be a robotics benchmark.

`src/transformer/data/robotics.py` generates synthetic 2D grid trajectories as token sequences. A tiny GPT predicts the next discretized position token from previous positions. The task is useful for demonstrating sequence modeling mechanics that map to robotics work: state tokenization, autoregressive rollout, deterministic synthetic data, and CPU-friendly smoke tests.

Run:

```bash
python scripts/train_robotics_sequence.py --dry-run
```

Limitations:

- Synthetic grid trajectories are not real robot demonstrations.
- There are no dynamics, controls, sensors, collision geometry, or sim-to-real claims.
- Use this as an engineering extension and test fixture, not as evidence of robotics model performance.
