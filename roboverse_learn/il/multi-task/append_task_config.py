import argparse, json, os, sys, time

def load_existing(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Corrupted JSON file: {path}")
    if not isinstance(data, list):
        raise ValueError(f"Existing file must contain a JSON list: {path}")
    return data

def main():
    parser = argparse.ArgumentParser(description="Append a task entry to a multi-task config JSON.")
    parser.add_argument("--config_path", type=str, default="roboverse_learn/il/task_config.json")
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--max_demo_idx", type=int, required=True)
    parser.add_argument("--expert_data_num", type=int, required=True)
    parser.add_argument("--metadata_dir", required=True,
                   help="Relative or absolute path to metadata root (contains demo_0000 etc.)")
    parser.add_argument("--downsample_ratio", type=int, default=1)
    parser.add_argument("--force_replace", action="store_true",
                   help="If a task with same task_name exists, replace it.")
    parser.add_argument("--show", action="store_true", help="Print resulting config then exit.")
    args = parser.parse_args()

    config_path = args.config_path
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    tasks = load_existing(config_path)

    new_entry = {
        "task_name": args.task_name,
        "max_demo_idx": args.max_demo_idx,
        "expert_data_num": args.expert_data_num,
        "metadata_dir": args.metadata_dir,
        "downsample_ratio": args.downsample_ratio,
        "timestamp": int(time.time())
    }

    existing_index = next((i for i, t in enumerate(tasks) if t.get("task_name") == args.task_name), None)

    if existing_index is not None:
        if args.force_replace:
            tasks[existing_index] = new_entry
            action = "replaced"
        else:
            # Append with suffix to distinguish
            suffix = sum(1 for t in tasks if t.get("task_name", "").startswith(args.task_name))
            new_entry["task_name"] = f"{args.task_name}_v{suffix}"
            tasks.append(new_entry)
            action = "appended (renamed)"
    else:
        tasks.append(new_entry)
        action = "added"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)

    print(f"Config path: {config_path}")
    print(f"Task {action}: {new_entry['task_name']}")
    if args.show:
        print(json.dumps(tasks, indent=2))

if __name__ == "__main__":
    main()