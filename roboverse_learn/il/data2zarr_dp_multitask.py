import argparse
import json
import logging
import os
import shutil

import imageio.v2 as iio
import numpy as np
import torch
import zarr
from tqdm import tqdm

try:
    from pytorch3d import transforms
except ImportError:
    transforms = None


def load_tasks(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Task config file must contain a JSON list.")
    # Normalize entries
    tasks = []
    for t in data:
        tasks.append({
            "task_name": t["task_name"],
            "metadata_dir": os.path.expanduser(t["metadata_dir"]),
            "max_demo_idx": int(t.get("max_demo_idx", 0)),
            "expert_data_num": t.get("expert_data_num", None),
            "downsample_ratio": int(t.get("downsample_ratio", 1)),
        })
    return tasks


def create_datasets_if_needed(zarr_data, zarr_meta, compressor, batch_size,
                              rgb_sample, state_dim, action_dim):
    if "head_camera" in zarr_data:
        return
    zarr_data.create_dataset(
        "head_camera",
        shape=(0, *rgb_sample.shape),
        chunks=(batch_size, *rgb_sample.shape),
        dtype=rgb_sample.dtype,
        compressor=compressor,
        overwrite=True,
    )
    zarr_data.create_dataset(
        "state",
        shape=(0, state_dim),
        chunks=(batch_size, state_dim),
        dtype="float32",
        compressor=compressor,
        overwrite=True,
    )
    zarr_data.create_dataset(
        "action",
        shape=(0, action_dim),
        chunks=(batch_size, action_dim),
        dtype="float32",
        compressor=compressor,
        overwrite=True,
    )
    zarr_data.create_dataset(
        "task_id",
        shape=(0,),
        chunks=(batch_size,),
        dtype="int32",
        compressor=compressor,
        overwrite=True,
    )
    zarr_meta.create_dataset(
        "episode_ends",
        shape=(0,),
        chunks=(batch_size,),
        dtype="int64",
        compressor=compressor,
        overwrite=True,
    )


def compute_state(i, meta, args):
    if args.observation_space == "joint_pos":
        state = meta["joint_qpos"][i]
        if args.joint_pos_padding > 0 and len(state) < args.joint_pos_padding:
            pad = np.zeros(args.joint_pos_padding - len(state))
            state = np.concatenate([state, pad])
        return state
    elif args.observation_space == "ee":
        if transforms is None:
            raise ImportError("pytorch3d is required for ee observation space.")
        robot_pos = torch.tensor(meta["robot_root_state"][i][0:3])
        robot_quat = torch.tensor(meta["robot_root_state"][i][3:7])
        local_ee_pos = transforms.quaternion_apply(
            transforms.quaternion_invert(robot_quat),
            torch.tensor(meta["robot_ee_state"][i][0:3]) - robot_pos,
        )
        local_ee_quat = transforms.quaternion_multiply(
            transforms.quaternion_invert(robot_quat),
            torch.tensor(meta["robot_ee_state"][i][3:7]),
        )
        gripper_state = meta["joint_qpos"][i][-2:]
        state = np.concatenate([local_ee_pos.numpy(), local_ee_quat.numpy(), gripper_state])
        assert state.shape == (9,)
        return state
    else:
        raise ValueError(f"Unknown observation space {args.observation_space}")


def compute_action(i, meta, args):
    if args.action_space == "joint_pos":
        action = meta["joint_qpos_target"][i]
        if args.joint_pos_padding > 0 and len(action) < args.joint_pos_padding:
            pad = np.zeros(args.joint_pos_padding - len(action))
            action = np.concatenate([action, pad])
        return action
    elif args.action_space == "ee":
        if transforms is None:
            raise ImportError("pytorch3d is required for ee action space.")
        robot_pos = torch.tensor(meta["robot_root_state"][i][0:3])
        robot_quat = torch.tensor(meta["robot_root_state"][i][3:7])

        local_ee_pos = transforms.quaternion_apply(
            transforms.quaternion_invert(robot_quat),
            torch.tensor(meta["robot_ee_state"][i][0:3]) - robot_pos,
        )
        local_next_ee_pos = transforms.quaternion_apply(
            transforms.quaternion_invert(robot_quat),
            torch.tensor(meta["robot_ee_state_target"][i][0:3]) - robot_pos,
        )

        local_ee_quat = transforms.quaternion_multiply(
            transforms.quaternion_invert(robot_quat),
            torch.tensor(meta["robot_ee_state"][i][3:7]),
        )
        local_next_ee_quat = transforms.quaternion_multiply(
            transforms.quaternion_invert(robot_quat),
            torch.tensor(meta["robot_ee_state_target"][i][3:7]),
        )
        gripper_action = meta["joint_qpos_target"][i][-2:]

        if not args.delta_ee:
            action = np.concatenate([
                local_next_ee_pos.numpy(),
                local_next_ee_quat.numpy(),
                gripper_action
            ])
        else:
            delta_pos = (local_next_ee_pos - local_ee_pos).numpy()
            delta_quat = transforms.quaternion_multiply(
                transforms.quaternion_invert(local_ee_quat),
                local_next_ee_quat
            ).numpy()
            action = np.concatenate([delta_pos, delta_quat, gripper_action])
        assert action.shape == (9,), f"Action shape {action.shape} != (9,)"
        return action
    else:
        raise ValueError(f"Unknown action space {args.action_space}")


def flush_batch(zarr_data, zarr_meta, arrays, batch_index):
    if len(arrays["head_camera"]) == 0:
        return batch_index
    head_camera = np.array(arrays["head_camera"])
    head_camera = np.moveaxis(head_camera, -1, 1)  # NHWC -> NCHW
    state = np.array(arrays["state"])
    action = np.array(arrays["action"])
    task_id = np.array(arrays["task_id"])
    episode_ends = np.array(arrays["episode_ends"])

    zarr_data["head_camera"].append(head_camera)
    zarr_data["state"].append(state)
    zarr_data["action"].append(action)
    zarr_data["task_id"].append(task_id)
    zarr_meta["episode_ends"].append(episode_ends)

    # print(f"Batch {batch_index + 1} written with {len(head_camera)} samples.")
    arrays["head_camera"].clear()
    arrays["state"].clear()
    arrays["action"].clear()
    arrays["task_id"].clear()
    arrays["episode_ends"].clear()
    return batch_index + 1


def main():
    parser = argparse.ArgumentParser(description="Multi-task metadata to ZARR for Diffusion Policy.")
    parser.add_argument("--task_config", type=str, required=True, help="Path to JSON list of task dicts.")
    parser.add_argument("--output_name", type=str, default="multitask", help="Output base name.")
    parser.add_argument("--observation_space", type=str, default="joint_pos", choices=["joint_pos", "ee"])
    parser.add_argument("--action_space", type=str, default="joint_pos", choices=["joint_pos", "ee"])
    parser.add_argument("--delta_ee", type=int, choices=[0, 1], default=0)
    parser.add_argument("--joint_pos_padding", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=100)
    args = parser.parse_args()

    if args.joint_pos_padding > 0 and args.observation_space == "ee" and args.action_space == "ee":
        logging.warning("Padding ignored for ee spaces.")

    tasks = load_tasks(args.task_config)
    save_dir = f"data_policy/{args.output_name}.zarr"
    print("ZARR save dir:", save_dir)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    arrays = {
        "head_camera": [],
        "state": [],
        "action": [],
        "task_id": [],
        "episode_ends": [],
    }

    total_count = 0
    batch_index = 0
    task_summaries = []

    # Infer dimensions lazily from first valid sample.
    state_dim = None
    action_dim = None
    rgb_sample = None

    for task_idx, task in enumerate(tasks):
        task_name = task["task_name"]
        metadata_dir = task["metadata_dir"]
        max_demo_idx = task["max_demo_idx"]
        expert_limit = task["expert_data_num"]
        downsample_ratio = task["downsample_ratio"]

        print(f"\nProcessing task {task_idx} ({task_name}) path={metadata_dir}")
        episode_count = 0
        start_samples = total_count

        for current_ep in tqdm(range(max_demo_idx), desc=f"{task_name} demos"):
            demo_id = str(current_ep).zfill(4)
            demo_dir = os.path.join(metadata_dir, f"demo_{demo_id}")
            if not os.path.isdir(demo_dir):
                print(f"Skipping episode {current_ep} as directory {demo_dir} does not exist.")
                continue
            meta_path = os.path.join(demo_dir, "metadata.json")
            if not os.path.isfile(meta_path):
                print(f"Skipping episode {current_ep} as metadata file {meta_path} does not exist.")
                continue
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            episode_count += 1

            rgbs = iio.mimread(os.path.join(demo_dir, "rgb.mp4"))
            for i, rgb in enumerate(rgbs):
                if i % downsample_ratio != 0:
                    continue

                if state_dim is None:
                    # First valid sample determines dataset shapes
                    tmp_state = compute_state(i, meta, args)
                    tmp_action = compute_action(i, meta, args)
                    rgb_sample = np.moveaxis(rgb, -1, 0)  # HWC->CHW
                    state_dim = len(tmp_state)
                    action_dim = len(tmp_action)
                    create_datasets_if_needed(
                        zarr_data,
                        zarr_meta,
                        compressor,
                        args.batch_size,
                        rgb_sample,
                        state_dim,
                        action_dim,
                    )

                state = compute_state(i, meta, args)
                action = compute_action(i, meta, args)

                arrays["head_camera"].append(rgb)
                arrays["state"].append(state)
                arrays["action"].append(action)
                arrays["task_id"].append(task_idx)
                total_count += 1


            arrays["episode_ends"].append(total_count)

            if len(arrays["head_camera"]) >= args.batch_size:
                batch_index = flush_batch(zarr_data, zarr_meta, arrays, batch_index)

            if expert_limit is not None and episode_count >= expert_limit:
                break

        task_summaries.append({
            "task_id": task_idx,
            "task_name": task_name,
            "num_episodes": episode_count,
            "num_samples": total_count - start_samples,
        })
        print(f"Task {task_name} done. Episodes: {episode_count}, Samples: {total_count - start_samples}")
    # Flush remaining
    if len(arrays["head_camera"]) > 0:
        batch_index = flush_batch(zarr_data, zarr_meta, arrays, batch_index)

    meta_attrs = {
        "observation_space": args.observation_space,
        "action_space": args.action_space,
        "delta_ee": args.delta_ee,
        "joint_pos_padding": args.joint_pos_padding,
        "num_tasks": len(tasks),
        "tasks": task_summaries,
        "total_samples": total_count,
        "batch_size": args.batch_size,
    }
    for k, v in meta_attrs.items():
        zarr_meta.attrs[k] = v

    meta_json_path = os.path.join(save_dir, "metadata.json")
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(meta_attrs, f, indent=4)

    print("\nMulti-task conversion complete.")
    print(f"Total samples: {total_count}")
    print(f"Tasks: {len(tasks)}")
    print(f"Metadata saved to: {meta_json_path}")


if __name__ == "__main__":
    main()