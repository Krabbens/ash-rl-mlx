
import os
import yaml
import json

def aggregate_tasks(root_dir):
    tasks = []
    original_tasks_dir = os.path.join(root_dir, "original-tasks")
    
    if not os.path.exists(original_tasks_dir):
        print(f"Error: {original_tasks_dir} not found.")
        return

    print(f"Aggregating tasks from {original_tasks_dir}...")
    for task_name in os.listdir(original_tasks_dir):
        task_path = os.path.join(original_tasks_dir, task_name)
        if os.path.isdir(task_path):
            yaml_path = os.path.join(task_path, "task.yaml")
            if os.path.exists(yaml_path):
                try:
                    with open(yaml_path, "r") as f:
                        config = yaml.safe_load(f)
                        # Improved Verifier Detection
                        verifier = config.get("test", "")
                        if not verifier:
                            # Look for common test scripts in the folder
                            if os.path.exists(os.path.join(task_path, "run-tests.sh")):
                                verifier = "bash run-tests.sh"
                            elif os.path.exists(os.path.join(task_path, "tests")):
                                verifier = "python3 -m pytest tests/"
                            elif os.path.exists(os.path.join(task_path, "test.py")):
                                verifier = "python3 test.py"
                        
                        task_data = {
                            "task_id": task_name,
                            "category": config.get("category", "unknown"),
                            "difficulty": config.get("difficulty", "unknown"),
                            "instruction": config.get("instruction", ""),
                            "verifier": verifier,
                            "setup": config.get("setup", ""),
                            "tags": config.get("tags", []),
                            "path": f"original-tasks/{task_name}",
                            "task_dir": task_path # Store for copying files
                        }
                        if task_data["instruction"]:
                            tasks.append(task_data)
                except Exception as e:
                    print(f"Error parsing {yaml_path}: {e}")

    output_file = "/Users/kosmagasiorowski/ash-rl-mlx/terminal_bench_full.json"
    print(f"Saving {len(tasks)} tasks to {output_file}")
    with open(output_file, "w") as f:
        json.dump(tasks, f, indent=2)

if __name__ == "__main__":
    aggregate_tasks("/Users/kosmagasiorowski/ash-rl-mlx/terminal-bench")
