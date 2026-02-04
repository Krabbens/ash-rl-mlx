import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import random
import re
import os
import json
import subprocess
import shutil
from tqdm import tqdm

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

NUM_SAMPLES = 16   # Reduced for speed in demo, scale to 128 for real run
NUM_ITERATIONS = 5
ADAPTER_PATH = "adapters"
TRAIN_DATA_FILE = "bash_trajectories.jsonl"
SANDBOX_DIR = "./sandbox"

# Safety: We just ensure we run in sandbox, but "rm -rf /" protection is limited here.
# For this demo, we assume the model is weak and we only ask for safe tasks.

def reset_sandbox():
    if os.path.exists(SANDBOX_DIR):
        shutil.rmtree(SANDBOX_DIR)
    os.makedirs(SANDBOX_DIR)

def create_bash_tasks():
    """
    Returns a list of tasks.
    Each task has a prompt and a verification command (truth).
    """
    tasks = [
        {
            "prompt": "List all files in the current directory.",
            "verifier": "ls" # If this runs and returns 0, likely okay? verified by output?
             # Better verification: we check if the command executed successfully.
             # For 'ls', simply running it is success.
        },
        {
            "prompt": "Create a new directory named 'backup'.",
            "verifier": "test -d backup"
        },
        {
            "prompt": "Create a file named 'hello.txt' containing the text 'world'.",
            "verifier": "grep -q 'world' hello.txt"
        },
        {
            "prompt": "Print the current working directory.",
            "verifier": "pwd"
        },
        {
            "prompt": "Count the number of lines in 'hello.txt' (assuming it exists).",
            "setup": "echo 'line1\nline2' > hello.txt",
            "verifier": "wc -l hello.txt" # Weak verify
        },
        {
            "prompt": "Renaming 'hello.txt' to 'greeting.txt'.",
            "setup": "echo 'hi' > hello.txt",
            "verifier": "test -f greeting.txt && test ! -f hello.txt"
        }
    ]
    return tasks

def extract_command(response):
    """
    Extracts the command from the LLM response.
    Looks for code blocks ```bash ... ``` or just takes the whole text if short.
    """
    # Try finding code blocks first
    code_blocks = re.findall(r'```(?:bash|sh)?\n(.*?)```', response, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    
    # Fallback: strict regex for common commands or just single line
    lines = response.strip().split('\n')
    # If the last line looks like a command, take it?
    # Or just take the whole thing if it's short.
    # Let's try to grab the last non-empty line as the command if no blocks
    if len(lines) > 0:
        return lines[-1].strip()
    return response.strip()

def execute_and_evaluate(command, verifier_cmd, setup_cmd=None):
    """
    Executes the command in the sandbox.
    Returns score (1.0 for success, 0.0 for fail/crash).
    """
    # 1. Reset/Prepare Sandbox for this specific task attempt
    # ideally we clean up between attempts to keep them independent
    # ignoring full isolation for speed, but `setup_cmd` helps
    
    # We want to run the generated command, THEN the verifier.
    # If Generated Command exits 0 AND Verifier exits 0 -> Success.
    
    try:
        # Run setup if needed
        if setup_cmd:
            subprocess.run(setup_cmd, shell=True, cwd=SANDBOX_DIR, check=True, capture_output=True)
            
        # Run Model Command
        # Timeout to prevent hanging
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=SANDBOX_DIR, 
            timeout=2, 
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return 0.0, f"Command failed: {result.stderr}"
            
        # Run Verifier
        v_result = subprocess.run(
            verifier_cmd, 
            shell=True, 
            cwd=SANDBOX_DIR, 
            timeout=2, 
            capture_output=True
        )
        
        if v_result.returncode == 0:
            return 1.0, result.stdout
        else:
            return 0.0, "Verifier failed"
            
    except Exception as e:
        return 0.0, str(e)

def main():
    print(f"Loading model: {MODEL_NAME}")
    
    current_adapter_path = None
    if os.path.exists(os.path.join(ADAPTER_PATH, "adapters.safetensors")):
        print(f"Loading using adapters: {ADAPTER_PATH}")
        current_adapter_path = ADAPTER_PATH

    model, tokenizer = load(MODEL_NAME, adapter_path=current_adapter_path)
    
    tasks = create_bash_tasks()
    print(f"Loaded {len(tasks)} bash tasks.")

    for iteration in range(NUM_ITERATIONS):
        print(f"\n=== Iteration {iteration + 1}/{NUM_ITERATIONS} ===")
        
        candidates_dataset = []
        total_solved = 0
        
        for task in tqdm(tasks, desc="Solving Tasks"):
            prompt = task["prompt"]
            verifier = task["verifier"]
            setup = task.get("setup", None)
            
            # Construct input prompt
            # We explicitly ask for a command wrapped in code block to help extraction
            sys_prompt = "You are an expert Linux terminal assistant. Output only the bash command to solve the user request. Put the command inside a code block."
            
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            best_response = None
            solved = False
            
            # Generate N samples (Parallel universes)
            for _ in range(NUM_SAMPLES):
                # Clean sandbox for each attempt to be fair? 
                # Doing it inside execute_and_evaluate roughly.
                reset_sandbox() 
                
                response = generate(
                    model, 
                    tokenizer, 
                    prompt=formatted_prompt, 
                    max_tokens=64, 
                    verbose=False, 
                    sampler=make_sampler(temp=0.8) 
                )
                
                cmd = extract_command(response)
                
                score, feedback = execute_and_evaluate(cmd, verifier, setup)
                
                if score == 1.0:
                    best_response = response # The full response that contained the working command
                    solved = True
                    # Optimization: Stop once we find *a* solution? 
                    # Or keep searching for the "shortest" / "most efficient"?
                    # User asked for "best path".
                    # For now, first correct is good enough.
                    break
            
            if solved:
                total_solved += 1
                
                # Format for MLX LoRA (pre-apply template to avoid tokenizer issues during training)
                # We want the model to learn to output the assistant response given the user input.
                # mlx_lm.lora supports {"text": "..."} where it trains on the whole sequence.
                # Ideally we mask the user part, but for simple "text" format it trains on everything.
                # This is "fine" for this demo.
                
                full_messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": best_response}
                ]
                full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
                
                entry = {"text": full_text}
                candidates_dataset.append(entry)
                # print(f"  [SOLVED] {prompt} -> {cmd}")
            # else:
                # print(f"  [FAILED] {prompt}")

        print(f"Iteration {iteration+1}: Solved {total_solved}/{len(tasks)}")

        # Train Step
        if candidates_dataset:
            print(f"Fine-tuning on {len(candidates_dataset)} trajectories...")
            
            data_dir = "data_bash"
            os.makedirs(data_dir, exist_ok=True)
            with open(os.path.join(data_dir, "train.jsonl"), "w") as f:
                for entry in candidates_dataset:
                    json.dump(entry, f)
                    f.write("\n")
            
            # Create dummy validation set (copy of train or subset)
            # mlx_lm requires it.
            with open(os.path.join(data_dir, "valid.jsonl"), "w") as f:
                # Use first few examples or all
                for entry in candidates_dataset[:max(1, len(candidates_dataset)//5)]:
                     json.dump(entry, f)
                     f.write("\n")
            
            # MLX LoRA Command
            cmd = [
                "uv", "run", "-m", "mlx_lm.lora",
                "--model", MODEL_NAME,
                "--data", data_dir,
                "--train",
                "--batch-size", "1",
                "--iters", "30",
                "--learning-rate", "1e-5",
                "--adapter-path", ADAPTER_PATH,
                "--save-every", "30"
            ]
            if current_adapter_path:
                 cmd.extend(["--resume-adapter-file", os.path.join(ADAPTER_PATH, "adapters.safetensors")])
            
            try:
                subprocess.run(cmd, check=True)
                print("Adapters updated. Reloading...")
                
                del model
                del tokenizer
                mx.metal.clear_cache()
                
                model, tokenizer = load(MODEL_NAME, adapter_path=ADAPTER_PATH)
                current_adapter_path = ADAPTER_PATH
            except Exception as e:
                print(f"Training error: {e}")
        else:
            print("No solutions found. No training.")

if __name__ == "__main__":
    reset_sandbox()
    main()
