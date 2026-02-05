
import mlx.core as mx
from mlx_lm import load, generate

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "adapters"

print(f"Loading model with trained adapters: {ADAPTER_PATH}...")
model, tokenizer = load(MODEL_NAME, adapter_path=ADAPTER_PATH)

def ask_bot(prompt):
    sys_prompt = "You are an expert Linux terminal assistant. Output only the bash command to solve the user request. Put the command inside a code block."
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    response = generate(model, tokenizer, prompt=formatted_prompt, max_tokens=128)
    return response

challenge = "Find all .py files in current directory, count how many of them contain the word 'import', and save the count to 'stats.txt'."

print(f"\nCHALLENGE: {challenge}")
print("-" * 50)
response = ask_bot(challenge)
print(f"BOT RESPONSE:\n{response}")
print("-" * 50)
