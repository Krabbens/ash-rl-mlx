import os
import subprocess
import shutil
import json
import re

class TerminalBench:
    def __init__(self, task_file=None):
        self.sandbox_dir = os.path.abspath("sandbox_rl")
        self.tasks = self._define_local_tasks()
        if task_file and os.path.exists(task_file):
            print(f"Loading tasks from {task_file}... (Path mapping /app -> ./ enabled)")
            with open(task_file, "r") as f:
                raw_tasks = json.load(f)
                # Map full dataset keys to internal ones
                for t in raw_tasks:
                    self.tasks.append({
                        "id": t.get("task_id", "unknown"),
                        "category": t.get("difficulty", t.get("category", "all")),
                        "prompt": t.get("instruction", t.get("prompt", "")),
                        "setup": t.get("setup", ""),
                        "verifier": t.get("verifier", ""),
                        "task_dir": t.get("task_dir", "")
                    })
        print(f"Total tasks loaded: {len(self.tasks)}")
        self.reset_sandbox()

    def reset_sandbox(self):
        if os.path.exists(self.sandbox_dir):
            # Using rm -rf is more robust for sandboxes with read-only/weird files
            subprocess.run(f"rm -rf {self.sandbox_dir}", shell=True)
        os.makedirs(self.sandbox_dir, exist_ok=True)

    def _map_paths(self, text):
        if not text: return text
        # Map absolute paths like /app/ or /root/ to relative ./ for sandbox
        return re.sub(r'(^|\s|["\'])/(?:app|root|home)(/|\b)', r"\1.\2", text)

    def _run(self, cmd, timeout=2):
        """Run command with timeout using subprocess."""
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                cwd=self.sandbox_dir, 
                timeout=timeout,
                capture_output=True,
                text=True
            )
            return result
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(args=cmd, returncode=124, stdout="", stderr="Timeout")
        except Exception as e:
            return subprocess.CompletedProcess(args=cmd, returncode=-1, stdout="", stderr=str(e))

    def _define_local_tasks(self):
        """Reliable local tasks that work without external dependencies."""
        tasks = [
            # BASIC (4)
            {"id": "ls_all", "category": "basic", "prompt": "List all files including hidden ones.", "setup": "touch .hidden file.txt", "verifier": "ls -a | grep .hidden"},
            {"id": "pwd", "category": "basic", "prompt": "Print the current working directory.", "setup": "", "verifier": "pwd"},
            {"id": "echo_hello", "category": "basic", "prompt": "Echo 'Hello World' to stdout.", "setup": "", "verifier": "echo 'Hello World'"},
            {"id": "whoami", "category": "basic", "prompt": "Print the current username.", "setup": "", "verifier": "whoami"},
            
            # FILE OPS (4)
            {"id": "mkdir_nested", "category": "files", "prompt": "Create nested directories: project/src/tests", "setup": "", "verifier": "test -d project/src/tests"},
            {"id": "touch_file", "category": "files", "prompt": "Create an empty file called 'output.log'.", "setup": "", "verifier": "test -f output.log"},
            {"id": "cp_file", "category": "files", "prompt": "Copy 'source.txt' to 'dest.txt'.", "setup": "echo 'data' > source.txt", "verifier": "diff source.txt dest.txt"},
            {"id": "mv_file", "category": "files", "prompt": "Rename 'old.txt' to 'new.txt'.", "setup": "echo 'test' > old.txt", "verifier": "test -f new.txt && test ! -f old.txt"},
            
            # TEXT PROCESSING (4)
            {"id": "cat_file", "category": "text", "prompt": "Display contents of 'info.txt'.", "setup": "echo 'hello' > info.txt", "verifier": "cat info.txt | grep hello"},
            {"id": "grep_word", "category": "text", "prompt": "Find lines containing 'error' in 'log.txt'.", "setup": "printf 'info\\nerror\\nwarn\\n' > log.txt", "verifier": "grep error log.txt"},
            {"id": "wc_lines", "category": "text", "prompt": "Count the number of lines in 'data.txt'.", "setup": "printf 'a\\nb\\nc\\n' > data.txt", "verifier": "wc -l data.txt"},
            {"id": "sed_replace", "category": "text", "prompt": "Replace 'foo' with 'bar' in 'text.txt' and print result.", "setup": "echo 'foo is foo' > text.txt", "verifier": "sed 's/foo/bar/g' text.txt | grep bar"},

            # CURRICULUM BRIDGES (2)
            {"id": "pipe_count", "category": "medium", "prompt": "Count how many times 'fox' appears in 'story.txt' using pipes.", "setup": "echo 'fox fox dog' > story.txt", "verifier": "grep -o 'fox' story.txt | wc -l | grep 2"},
            {"id": "find_redirect", "category": "medium", "prompt": "Find all .txt files and save their names to 'list.log'.", "setup": "touch a.txt b.txt c.jpg", "verifier": "grep -q 'a.txt' list.log && grep -q 'b.txt' list.log"},
            
            # SEARCH + SCRIPT + GIT (4)
            {"id": "find_txt", "category": "search", "prompt": "Find all .txt files in the current directory.", "setup": "touch a.txt b.txt c.log", "verifier": "find . -name '*.txt' | wc -l | grep 2"},
            {"id": "chmod_exec", "category": "script", "prompt": "Make 'run.sh' executable.", "setup": "echo '#!/bin/bash' > run.sh", "verifier": "test -x run.sh"},
            {"id": "env_var", "category": "script", "prompt": "Print the value of the HOME environment variable.", "setup": "", "verifier": "echo $HOME"},
            {"id": "git_init", "category": "git", "prompt": "Initialize a new git repository.", "setup": "", "verifier": "test -d .git"},

            # HARD CHALLENGES (2)
            {
                "id": "complex_search", 
                "category": "hard", 
                "prompt": "Count files in current directory containing the word 'secret', and save count to 'res.txt'.", 
                "setup": "echo 'secret' > f1.txt; echo 'nothing' > f2.txt; echo 'secret' > f3.py", 
                "verifier": "grep -q '2' res.txt"
            },
            {
                "id": "backup_tar", 
                "category": "hard", 
                "prompt": "Create a directory 'archive', move all .txt files there, and create 'all.tar.gz' from it.", 
                "setup": "touch a.txt b.txt c.jpg", 
                "verifier": "test -f all.tar.gz && tar -tf all.tar.gz | grep a.txt"
            },
        ]
        return tasks

    def evaluate_task(self, task, command_output_dict):
        cmd = self._map_paths(command_output_dict['command'])
        
        self.reset_sandbox()
        
        # 0. Copy resource files from task_dir if available
        task_dir = task.get("task_dir")
        if task_dir and os.path.exists(task_dir):
            for item in os.listdir(task_dir):
                s = os.path.join(task_dir, item)
                d = os.path.join(self.sandbox_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
        
        # Block dangerous/hanging commands
        BLOCKED_PATTERNS = [
            'ssh ', 'shutdown', 'reboot', 'rm -rf /'
        ]
        
        if cmd:
            cmd_lower = cmd.lower()
            for pattern in BLOCKED_PATTERNS:
                if pattern in cmd_lower:
                    return False, {"error": f"Blocked command: {pattern}", "exit_code": -1}
        
        # 1. Setup (skip if contains blocked patterns)
        if task["setup"]:
            setup_cmd = self._map_paths(task["setup"])
            setup_lower = setup_cmd.lower()
            skip_setup = any(p in setup_lower for p in BLOCKED_PATTERNS)
            if not skip_setup:
                self._run(setup_cmd, timeout=2)
            
        # 2. Agent
        if not cmd:
            return False, {"error": "No command", "exit_code": -1}
        
        result = self._run(cmd, timeout=30)  # Increased timeout for installs
        
        # 3. Verify
        verifier_cmd = self._map_paths(task["verifier"])
        try:
            if not verifier_cmd:
                success = False
            elif verifier_cmd.startswith("bash ") or verifier_cmd.startswith("python3 "):
                v_result = self._run(verifier_cmd, timeout=30)
                success = (v_result.returncode == 0)
            else:
                verifier_path = os.path.join(self.sandbox_dir, "verify_task.sh")
                with open(verifier_path, "w") as f:
                    f.write(verifier_cmd)
                v_result = self._run("bash verify_task.sh", timeout=5)
                success = (v_result.returncode == 0)
        except Exception:
            success = False
        
        return success, {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode
        }

    def get_tasks(self):
        return self.tasks

