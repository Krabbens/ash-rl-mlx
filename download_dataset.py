
from datasets import load_dataset
import json

def download_and_convert():
    print("Downloading TerminalBench from Hugging Face (ia03/terminal-bench)...")
    try:
        dataset = load_dataset("ia03/terminal-bench", split="test")
    except Exception as e:
        print(f"Error downloading: {e}")
        return

    tasks = []
    print(f"Found {len(dataset)} potential tasks.")
    
    import yaml # Add yaml to dependencies
    
    # Map them to our format: id, prompt, setup, verifier
    for i, item in enumerate(dataset):
        if i == 0:
            print(f"Sample task_yaml:\n{item['task_yaml'][:200]}...")
            
        try:
            config = yaml.safe_load(item["task_yaml"])
            task = {
                "id": item["task_id"],
                "category": item.get("category", "all"),
                "prompt": config.get("goal", ""),
                "setup": "", # Setup is usually in the archive, but maybe there's a command
                "verifier": config.get("test", "")
            }
            
            # Simple setup extraction if available
            if "setup" in config:
                task["setup"] = config["setup"]

            if task["prompt"] and task["verifier"]:
                tasks.append(task)
        except Exception as e:
            print(f"Skipping {item['task_id']} due to YAML error: {e}")

    print(f"Saving {len(tasks)} tasks to terminal_bench_hf.jsonl")
    with open("terminal_bench_hf.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

if __name__ == "__main__":
    download_and_convert()
