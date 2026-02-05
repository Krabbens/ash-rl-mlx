
from datasets import load_dataset
import json

def download_bash_dataset():
    print("Downloading aelhalili/bash-commands-dataset from Hugging Face...")
    try:
        # This dataset contains (explanation, command) pairs
        dataset = load_dataset("aelhalili/bash-commands-dataset", split="train")
    except Exception as e:
        print(f"Error downloading: {e}")
        return

    tasks = []
    print(f"Found {len(dataset)} examples.")
    
    # We will treat each 'explanation' as a prompt and 'command' as the target
    # Since we don't have verifiers for these, we'll use them as 'bootstrap' data
    # or try to simple verifiers like checking if command contains the right binary.
    
    for i, item in enumerate(dataset):
        prompt = item.get("prompt", "")
        command = item.get("response", "")
        
        if prompt and command:
            tasks.append({
                "id": f"bash_data_{i}",
                "category": "imported",
                "prompt": prompt,
                "gold_command": command,
                "setup": "",
                "verifier": f"echo 'Correct if matches gold command'; exit 0" # Placeholder
            })

    print(f"Saving {len(tasks)} items to bash_expert_data.jsonl")
    with open("bash_expert_data.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

if __name__ == "__main__":
    download_bash_dataset()
