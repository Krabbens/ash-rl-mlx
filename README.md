# Self-Improving RL Loop with MLX

This project implements a reinforcement learning loop (Expert Iteration / Rejection Sampling Fine-Tuning) for small LLMs on Mac Silicon.

## Algorithm
1.  **Generate**: For each problem in the dataset, generate 128 responses using `Qwen2.5-0.5B-Instruct`.
2.  **Evaluate**: Check each response against a ground-truth verifier (Reward Function).
3.  **Select**: Pick the best response (correct answer).
4.  **Train**: Fine-tune the model (LoRA) on the selected best responses.
5.  **Repeat**: Improve the model iteratively.

## Setup
```bash
uv init
uv add mlx mlx-lm numpy tqdm
```

## Run
```bash
uv run python main.py
```

## Configuration
Edit `main.py` to change:
- `NUM_SAMPLES`: Number of exploration paths (default 128).
- `MODEL_NAME`: Model to use (default `Qwen/Qwen2.5-0.5B-Instruct`).
