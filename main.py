import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import random
import re
import os
import json
import subprocess
import shutil
import time
from tqdm import tqdm

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
NUM_SAMPLES = 4      # Reduced for faster iterations
NUM_ITERATIONS = 50  # Long training loop
ADAPTER_PATH = "adapters"
TRAIN_DATA_FILE = "bash_trajectories.jsonl"
SANDBOX_DIR = "./sandbox"
METRICS_FILE = "metrics.jsonl"

# Safety: We just ensure we run in sandbox.

def reset_sandbox():
    if os.path.exists(SANDBOX_DIR):
        subprocess.run(f"rm -rf {SANDBOX_DIR}", shell=True)
    os.makedirs(SANDBOX_DIR)

def check_bash_syntax(cmd):
    """Checks if the command is syntactically valid bash."""
    if not cmd:
        return False
    try:
        # Using bash -n to check syntax without executing
        res = subprocess.run(["bash", "-n"], input=cmd, text=True, capture_output=True, timeout=1)
        return res.returncode == 0
    except:
        return False

def calculate_reward(success, eval_meta, cmd, prompt):
    """
    Calculates a shaped reward instead of binary 0/1.
    """
    if not cmd:
        return 0.0
        
    reward = 0.0
    
    # 1. Base success - Increase weight
    if success:
        reward += 2.0
    
    # 2. Syntax correctness (bash -n)
    if check_bash_syntax(cmd):
        reward += 0.3
    
    # 3. Execution status
    if eval_meta.get("exit_code") == 0:
        reward += 0.2
    elif eval_meta.get("exit_code") == 124: # Timeout
        reward -= 0.5
        
    # 4. Keyword/File matching (Progress reward)
    # If the command mentions files or tools from the prompt, it's on the right track.
    potential_files = re.findall(r'/?app/([\w\.\-]+)', prompt)
    for f in potential_files:
        if f in cmd:
            reward += 0.1
            
    # 5. Penalize common 0.5B hallucinations
    if "<<" in cmd and "jq" in cmd: # Model often hallucinates jq syntax
        reward -= 0.5
    
    if cmd.strip().endswith("|"):
        reward -= 0.5
        
    # 6. Brevity penalty
    len_penalty = max(0, (len(cmd) - 100) * 0.0001)
    reward -= len_penalty
    
    return max(0.0, reward)

def log_metric(data):
    """Appends a metric entry to the JSONL file."""
    with open(METRICS_FILE, "a") as f:
        json.dump(data, f)
        f.write("\n")

def create_bash_tasks():
    """
    Returns a comprehensive list of tasks with increasing difficulty.
    """
    tasks = [
        # Level 1: Basic File Ops
        {
            "id": "pwd_01",
            "level": 1,
            "prompt": "Print the current working directory.",
            "verifier": "pwd"
        },
        {
            "id": "ls_01",
            "level": 1,
            "prompt": "List all files in the current directory.",
            "verifier": "ls"
        },
        {
            "id": "touch_01",
            "level": 1,
            "prompt": "Create an empty file named 'blank.txt'.",
            "verifier": "test -f blank.txt"
        },
        {
            "id": "mkdir_01",
            "level": 1,
            "prompt": "Create a directory named 'data'.",
            "verifier": "test -d data"
        },
        
        # Level 2: Content & Pipes
        {
            "id": "echo_01",
            "level": 2,
            "prompt": "Create a file named 'hello.txt' containing the text 'world'.",
            "verifier": "grep -q 'world' hello.txt"
        },
        {
            "id": "cat_pipe_01",
            "level": 2,
            "prompt": "Count the number of lines in 'data.txt'.",
            "setup": "printf 'a\nb\nc\n' > data.txt",
            "verifier": "wc -l data.txt"
        },
        {
            "id": "grep_01",
            "level": 2,
            "prompt": "Find lines containing 'error' in 'log.txt'.",
            "setup": "printf 'info: ok\nerror: fail\ninfo: done\n' > log.txt",
            "verifier": "grep 'error' log.txt"
        },

        # Level 3: Advanced Manipulation (sed, find, complex pipes)
        {
            "id": "sed_01",
            "level": 3,
            "prompt": "Replace all occurrences of 'foo' with 'bar' in 'file.txt' and print the result.",
            "setup": "echo 'foo is foo' > file.txt",
            "verifier": "sed 's/foo/bar/g' file.txt | grep 'bar is bar'"
        },
        {
            "id": "find_01",
            "level": 3,
            "prompt": "Find all text files (.txt) in the current directory.",
            "setup": "touch a.txt b.jpg c.txt",
            "verifier": "find . -name '*.txt'"
        },
        {
            "id": "sort_uniq_01",
            "level": 3,
            "prompt": "Sort 'names.txt' and remove duplicates, saving output to 'sorted.txt'.",
            "setup": "printf 'bob\nalice\nbob\n' > names.txt",
            "verifier": "test -f sorted.txt && cat sorted.txt | grep -q 'alice' && [ $(wc -l < sorted.txt) -eq 2 ]"
        },
        {
            "id": "compound_01",
            "level": 4,
            "prompt": "List all files, count them, and save the number to 'count.log'.",
            "setup": "touch a b c",
            "verifier": "test -f count.log && [ $(cat count.log) -ge 3 ]"
        }
    ]
    return tasks

def extract_command(response):
    """
    Extracts the command from the LLM response.
    """
    # 1. Try finding code blocks first
    code_blocks = re.findall(r'```(?:bash|sh)?\n(.*?)```', response, re.DOTALL)
    if code_blocks:
        full_block = code_blocks[0].strip()
        lines = [line for line in full_block.split('\n') if line.strip() and not line.strip().startswith('#')]
        return "\n".join(lines)
    
    # 2. Try to find the block after Reasoning/Thinking
    if "</thinking>" in response:
        after_thinking = response.split("</thinking>")[-1].strip()
        lines = [l.strip() for l in after_thinking.split('\n') if l.strip()]
        # Take everything until we hit a non-command looking line if possible
        cmd_lines = []
        for l in lines:
            if any(l.startswith(c) for c in ["mkdir", "touch", "cp", "mv", "rm", "grep", "sed", "awk", "find", "cat", "echo", "python", "gcc", "ls", "cd", "./"]):
                cmd_lines.append(l)
            else:
                break
        if cmd_lines:
            return "\n".join(cmd_lines)

    # 3. Fallback: specific "Command: ..." pattern or just last line
    lines = response.strip().split('\n')
    for line in reversed(lines):
        cl = line.strip()
        if cl and not cl.startswith(('#', 'Here', 'To', 'Step', 'This', 'The', 'Note')):
            return cl
    return response.strip()

# Removed unused execute_and_evaluate (logic is in terminal_bench.py)

from terminal_bench import TerminalBench

# ... Config ...

def main():
    # Initial Metrics Reset
    if os.path.exists(METRICS_FILE):
        os.remove(METRICS_FILE)
        
    print(f"Loading model: {MODEL_NAME}")
    
    current_adapter_path = None
    if os.path.exists(os.path.join(ADAPTER_PATH, "adapters.safetensors")):
        print(f"Loading using adapters: {ADAPTER_PATH}")
        current_adapter_path = ADAPTER_PATH

    model, tokenizer = load(MODEL_NAME, adapter_path=current_adapter_path)
    
    # Initialize Bench with full dataset
    FULL_DATASET_PATH = "terminal_bench_full.json"
    bench = TerminalBench(task_file=FULL_DATASET_PATH)
    all_tasks = bench.get_tasks()
    
    print(f"Loaded {len(all_tasks)} total tasks for Generalist Training.")

    for iteration in range(NUM_ITERATIONS):
        print(f"\n=== Iteration {iteration+1}/{NUM_ITERATIONS} ===")
        
        # Curriculum Learning: Filter tasks by difficulty
        if iteration < 5:
            filtered_tasks = [t for t in all_tasks if t.get("category", "").lower() in ["easy", "basic", "files", "text"]]
            diff_label = "EASY ONLY"
        elif iteration < 15:
            filtered_tasks = [t for t in all_tasks if t.get("category", "").lower() in ["easy", "basic", "files", "text", "medium"]]
            diff_label = "EASY + MEDIUM"
        else:
            filtered_tasks = all_tasks
            diff_label = "ALL DIFFICULTIES"

        print(f"Curriculum: {diff_label} ({len(filtered_tasks)} available tasks)")
        
        # Shuffle tasks for diversity
        import random
        random.shuffle(filtered_tasks)
        
        # We can run a subset to keep iterations fast.
        # Micro-batch: 5 tasks per training cycle to reduce memory
        current_tasks = filtered_tasks[:5]
        
        candidates_dataset = []
        iter_stats = {"solved": 0, "total_reward": 0.0, "attempted": 0}
        
        # Stream successful trajectories to disk to reduce RAM
        data_dir = "data_bash"
        os.makedirs(data_dir, exist_ok=True)
        train_file_path = os.path.join(data_dir, "train.jsonl")
        # Clear previous iteration's data
        open(train_file_path, "w").close()
        
        for i, task in enumerate(tqdm(current_tasks, desc="Running Benchmark")):
            prompt = task["prompt"]
            task_id = task["id"]
            # Map difficulty back to categories if needed
            category = task.get("category", "basic").lower()
            
            iter_stats["attempted"] += 1
            
            sys_prompt = (
                "You are an expert Linux terminal assistant. "
                "The current working directory is /app. All files in the instructions are strictly in /app or subdirectories. "
                "Always use relative paths (e.g. 'data.txt' instead of '/app/data.txt').\n"
                "First, reason about the task inside <thinking>...</thinking> tags. "
                "Then, provide the final bash script inside a ```bash\n...``` code block."
            )
            messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            best_cmd = None
            best_reward = 0.0
            first_response = None
            
            # Determine number of samples (Reduced for memory)
            sampling_limit = 1  # Start with single sample for low memory
            
            # Dynamic token limit: 256 for bash, 512 for coding tasks (reduced from 1024)
            is_coding = any(tag in task.get("tags", []) for tag in ["coding", "python", "numpy"]) or \
                        any(word in prompt.lower() for word in ["write a script", "create a file", "implement", "function"])
            
            max_gen_tokens = 512 if is_coding else 256
            
            if category == "medium":
                sampling_limit = 2
            elif category == "hard" or category == "extra":
                sampling_limit = 2  # Capped at 2 for memory
                
            # Explore Samples
            for sample_idx in range(sampling_limit):
                print(f"  Sample {sample_idx+1}/{sampling_limit}...", end="\r", flush=True)
                response = generate(
                    model, 
                    tokenizer, 
                    prompt=formatted_prompt, 
                    max_tokens=max_gen_tokens, 
                    verbose=False, 
                    sampler=make_sampler(temp=0.9)
                )
                
                if sample_idx == 0:
                    first_response = response  # Save for debug
                
                cmd = extract_command(response)
                
                # Evaluation
                success, eval_meta = bench.evaluate_task(task, {'command': cmd})
                
                reward = calculate_reward(success, eval_meta, cmd, prompt)
                
                if reward > best_reward:
                    best_reward = reward
                    best_cmd = cmd
                
                if success:
                    break # Success found
            
            # Cleanup memory per task instead of per sample
            import gc
            gc.collect()
            mx.clear_cache()
            
            # Log Task Result
            is_solved = (best_reward >= 1.0)
            status = "✅" if is_solved else "❌"
            print(f"\n  {status} [{task_id}] (Reward: {best_reward:.2f}) {prompt[:50]}...")
            if best_cmd:
                print(f"     → {best_cmd}", flush=True)
            else:
                # Show what model actually generated
                cmd_from_first = extract_command(first_response) if first_response else "?"
                print(f"     → FAIL (tried: {cmd_from_first[:60]}...)", flush=True)
            
            log_metric({
                "type": "task_result",
                "iteration": iteration + 1,
                "task_id": task_id,
                "level": category,
                "success": is_solved,
                "command": best_cmd if best_cmd else "N/A",
                "reward": best_reward
            })
            
            iter_stats["total_reward"] += best_reward
            if is_solved:
                iter_stats["solved"] += 1
                
                # Stream to disk immediately instead of buffering
                full_messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": f"```bash\n{best_cmd}\n```"}
                ]
                full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
                with open(train_file_path, "a") as f:
                    json.dump({"text": full_text}, f)
                    f.write("\n")
                candidates_dataset.append(1)  # Just count, don't store

        # End of Iteration Stats
        print(f"Iteration {iteration+1}: Solved {iter_stats['solved']}/{len(current_tasks)}")
        log_metric({
            "type": "iteration_summary",
            "iteration": iteration + 1,
            "solved_count": iter_stats["solved"],
            "total_tasks": len(current_tasks),
            "success_rate": iter_stats["solved"] / len(current_tasks)
        })

        # Training Step
        if candidates_dataset:
            print(f"Fine-tuning on {len(candidates_dataset)} trajectories...")
            
            # Validation set (copy first few lines)
            with open(os.path.join(data_dir, "valid.jsonl"), "w") as vf:
                with open(train_file_path, "r") as tf:
                    for i, line in enumerate(tf):
                        if i >= max(1, len(candidates_dataset) // 5):
                            break
                        vf.write(line)
            
            # Unload model BEFORE training to free VRAM
            del model
            del tokenizer
            gc.collect()
            mx.clear_cache()
            
            # MLX LoRA Command with gradient accumulation
            cmd = [
                "uv", "run", "-m", "mlx_lm.lora",
                "--model", MODEL_NAME,
                "--data", data_dir,
                "--train",
                "--batch-size", "1",
                "--grad-accumulation-steps", "4",  # Correct flag
                "--iters", "20",
                "--learning-rate", "1e-5",
                "--adapter-path", ADAPTER_PATH,
                "--save-every", "30"
            ]
            if current_adapter_path:
                 cmd.extend(["--resume-adapter-file", os.path.join(ADAPTER_PATH, "adapters.safetensors")])
            
            try:
                subprocess.run(cmd, check=True)
                print("Adapters updated. Reloading...")
            except Exception as e:
                print(f"Training error: {e}")
            
            # Reload model after training
            model, tokenizer = load(MODEL_NAME, adapter_path=ADAPTER_PATH)
            current_adapter_path = ADAPTER_PATH
        else:
            print("No solutions found. No training.")

if __name__ == "__main__":
    reset_sandbox()
    main()
