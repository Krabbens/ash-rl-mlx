from terminal_bench import TerminalBench
import json

bench = TerminalBench(task_file="terminal_bench_full.json")
tasks = bench.get_tasks()
task = tasks[0]
print(f"Testing task: {task['id']}")
success, meta = bench.evaluate_task(task, {'command': 'ls'})
print(f"Success: {success}")
print(f"Meta: {meta}")
