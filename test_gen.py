import mlx_lm
from mlx_lm import load, generate

model, tokenizer = load("Qwen/Qwen2.5-0.5B-Instruct")
response = generate(model, tokenizer, prompt="Q: 2+2=?. A:", max_tokens=10)
print(response)
