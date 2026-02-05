
import mlx.core as mx
from mlx_lm import load, generate, sample_utils
from terminal_bench import TerminalBench
import re

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "adapters"

model, tokenizer = load(MODEL_NAME, adapter_path=ADAPTER_PATH)
bench = TerminalBench()

task_id = "complex_search"
task_idx = 16 # Check index in terminal_bench

prompt = "Count files in current directory containing the word 'secret', and save count to 'res.txt'."
sys_prompt = "You are an expert Linux terminal assistant. Output only the bash command. Put the command inside a code block."
formatted = tokenizer.apply_chat_template([{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

print(f"Sampling 32 plans for: {prompt}\n")

found = False
for i in range(32):
    res = generate(model, tokenizer, prompt=formatted, max_tokens=64, sampler=sample_utils.make_sampler(temp=0.9))
    cmd_match = re.search(r'```(?:bash|sh)?\n(.*?)```', res, re.DOTALL)
    cmd = cmd_match.group(1).strip() if cmd_match else res.strip()
    
    success, out, err = bench.evaluate_task(task_idx, {'command': cmd})
    status = "✅" if success else "❌"
    print(f"[{i+1:02d}] {status} -> {cmd[:60]}")
    if success:
        print(f"    SUCCESS! Output: {out}")
        found = True
        break

if not found:
    print("\nNo solution found in 32 samples.")
