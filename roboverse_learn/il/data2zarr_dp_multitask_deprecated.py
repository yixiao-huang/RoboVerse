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
    pass

def process_single_task(
    task_config,
    zarr_data,
    zarr_meta,
    compressor,
    batch_size,
    global_offset,
    task_idx,
    args
):
    """Process a single task and return the updated global offset."""
    task_name = task_config["task_name"]
    max_demo_idx = task_config.get("max_demo_idx", 200)
    expert_data_num = task_config.get("expert_data_num", None)
    metadata_dir = os.path.expanduser(task_config["metadata_dir"])
    downsample_ratio = task_config.get("downsample_ratio", 1)

    print(f"\n{'='*60}")
    print(f"Processing Task {task_idx + 1}: {task_name}")
    print(f"Metadata load dir: {metadata_dir}")
    print(f"{'='*60}\n")

    head_camera_arrays = []
    action_arrays = []
    state_arrays = []
    episode_ends_arrays = []
    task_id_arrays = []
    total_count = global_offset
    current_demo_index = 0

    for current_ep in tqdm(range(max_demo_idx), desc=f"Processing {task_name}"):
        demo_id = str(current_ep).zfill(4)
        demo_dir = os.path.join(metadata_dir, f"demo_{demo_id}")

        if not os.path.isdir(demo_dir):
            continue

        metadata_path = os.path.join(demo_dir, "metadata.json")
        if not os.path.isfile(metadata_path):
            continue

        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        current_demo_index += 1

        rgbs = iio.mimread(os.path.join(demo_dir, "rgb.mp4"))
        for i, rgb in enumerate(rgbs):
            if i % downsample_ratio != 0:
                continue

            # Process observation
            if args.observation_space == "joint_pos":
                state = metadata["joint_qpos"][i]
                if args.joint_pos_padding > 0 and len(state) < args.joint_pos_padding:
                    padding = np.zeros(args.joint_pos_padding - len(state))
                    state = np.concatenate([state, padding])
            elif args.observation_space == "ee":
                robot_pos, robot_quat = (
                    torch.tensor(metadata["robot_root_state"][i][0:3]),
                    torch.tensor(metadata["robot_root_state"][i][3:7]),
                )

                local_ee_pos = transforms.quaternion_apply(
                    transforms.quaternion_invert(robot_quat),
                    torch.tensor(metadata["robot_ee_state"][i][0:3]) - robot_pos,
                )
                local_ee_quat = transforms.quaternion_multiply(
                    transforms.quaternion_invert(robot_quat),
                    torch.tensor(metadata["robot_ee_state"][i][3:7])
                )

                gripper_state = metadata["joint_qpos"][i][-2:]
                state = np.concatenate([local_ee_pos, local_ee_quat, gripper_state])
                assert state.shape == (9,)
            else:
                raise ValueError(f"Unknown observation space: {args.observation_space}")

            # Process action
            if args.action_space == "joint_pos":
                action = metadata["joint_qpos_target"][i]
                if args.joint_pos_padding > 0 and len(action) < args.joint_pos_padding:
                    padding = np.zeros(args.joint_pos_padding - len(action))
                    action = np.concatenate([action, padding])
            elif args.action_space == "ee":
                robot_pos, robot_quat = (
                    torch.tensor(metadata["robot_root_state"][i][0:3]),
                    torch.tensor(metadata["robot_root_state"][i][3:7]),
                )

                local_ee_pos = transforms.quaternion_apply(
                    transforms.quaternion_invert(robot_quat),
                    torch.tensor(metadata["robot_ee_state"][i][0:3]) - robot_pos,
                )
                local_next_ee_pos = transforms.quaternion_apply(
                    transforms.quaternion_invert(robot_quat),
                    torch.tensor(metadata["robot_ee_state_target"][i][0:3]) - robot_pos,
                )

                local_ee_quat = transforms.quaternion_multiply(
                    transforms.quaternion_invert(robot_quat),
                    torch.tensor(metadata["robot_ee_state"][i][3:7])
                )
                local_next_ee_quat = transforms.quaternion_multiply(
                    transforms.quaternion_invert(robot_quat),
                    torch.tensor(metadata["robot_ee_state_target"][i][3:7])
                )
                gripper_action = metadata["joint_qpos_target"][i][-2:]

                if not args.delta_ee:
                    action = np.concatenate([local_next_ee_pos, local_next_ee_quat, gripper_action])
                else:
                    local_ee_delta_pos = local_next_ee_pos - local_ee_pos
                    local_ee_delta_quat = transforms.quaternion_multiply(
                        transforms.quaternion_invert(local_ee_quat), local_next_ee_quat
                    )
                    action = np.concatenate([local_ee_delta_pos, local_ee_delta_quat, gripper_action])

                assert action.shape == (9,), f"Action shape is {action.shape}, expected (9,)"
            else:
                raise ValueError(f"Unknown action space: {args.action_space}")

            action = list(action)
            head_camera_arrays.append(rgb)
            state_arrays.append(state)
            action_arrays.append(action)
            task_id_arrays.append(task_idx)
            total_count += 1

        episode_ends_arrays.append(total_count)

        if expert_data_num is not None and current_demo_index >= expert_data_num:
            break

    # Write remaining data for this task
    if len(head_camera_arrays) > 0:
        head_camera_arrays = np.array(head_camera_arrays)
        head_camera_arrays = np.moveaxis(head_camera_arrays, -1, 1)  # NHWC -> NCHW
        action_arrays = np.array(action_arrays)
        state_arrays = np.array(state_arrays)
        task_id_arrays = np.array(task_id_arrays)
        episode_ends_arrays = np.array(episode_ends_arrays)

        zarr_data["head_camera"].append(head_camera_arrays)
        zarr_data["state"].append(state_arrays)
        zarr_data["action"].append(action_arrays)
        zarr_data["task_id"].append(task_id_arrays)
        zarr_meta["episode_ends"].append(episode_ends_arrays)

        print(f"Task {task_name}: Written {len(head_camera_arrays)} samples, {current_demo_index} episodes")

    return total_count, current_demo_index


def main():
    parser = argparse.ArgumentParser(description="Process Multi-Task Meta Data To ZARR For Diffusion Policy.")
    parser.add_argument(
        "--task_config",
        type=str,
        required=True,
        help="Path to JSON file containing task configurations",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="multitask",
        help="Name for the output zarr file (e.g., 'multitask' -> multitask.zarr)",
    )
    parser.add_argument(
        "--observation_space",
        type=str,
        default="joint_pos",
        choices=["joint_pos", "ee"],
        help="The observation space to use",
    )
    parser.add_argument(
        "--action_space",
        type=str,
        default="joint_pos",
        choices=["joint_pos", "ee"],
        help="The action space to use",
    )
    parser.add_argument("--delta_ee", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--joint_pos_padding",
        type=int,
        default=0,
        help="If > 0, pad joint positions to this length",
    )

    args = parser.parse_args()

    # Load task configuration
    with open(args.task_config, encoding="utf-8") as f:
        task_configs = json.load(f)

    if not isinstance(task_configs, list):
        raise ValueError("Task configuration must be a list of task dictionaries")

    save_dir = f"data_policy/{args.output_name}.zarr"
    print(f"ZARR save dir: {save_dir}")

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    batch_size = 100

    if args.joint_pos_padding > 0 and args.observation_space == "ee" and args.action_space == "ee":
        logging.warning("Padding is not supported for ee observation and action spaces.")

    # Initialize datasets with first task to determine dimensions
    print("Initializing datasets...")
    first_task = task_configs[0]
    metadata_dir = os.path.expanduser(first_task["metadata_dir"])
    demo_dir = os.path.join(metadata_dir, "demo_0000")
    
    # Get sample data to determine dimensions
    with open(os.path.join(demo_dir, "metadata.json"), encoding="utf-8") as f:
        sample_metadata = json.load(f)
    
    sample_rgb = iio.mimread(os.path.join(demo_dir, "rgb.mp4"))[0]
    sample_rgb = np.moveaxis(sample_rgb, -1, 0)  # HWC -> CHW
    
    if args.observation_space == "joint_pos":
        state_dim = len(sample_metadata["joint_qpos"][0])
        if args.joint_pos_padding > 0:
            state_dim = args.joint_pos_padding
    else:
        state_dim = 9  # ee: 3 pos + 4 quat + 2 gripper
    
    if args.action_space == "joint_pos":
        action_dim = len(sample_metadata["joint_qpos_target"][0])
        if args.joint_pos_padding > 0:
            action_dim = args.joint_pos_padding
    else:
        action_dim = 9  # ee: 3 pos + 4 quat + 2 gripper

    # Create zarr datasets
    zarr_data.create_dataset(
        "head_camera",
        shape=(0, *sample_rgb.shape),
        chunks=(batch_size, *sample_rgb.shape),
        dtype=sample_rgb.dtype,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "state",
        shape=(0, state_dim),
        chunks=(batch_size, state_dim),
        dtype="float32",
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action",
        shape=(0, action_dim),
        chunks=(batch_size, action_dim),
        dtype="float32",
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "task_id",
        shape=(0,),
        chunks=(batch_size,),
        dtype="int32",
        compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends",
        shape=(0,),
        chunks=(batch_size,),
        dtype="int64",
        compressor=compressor,
    )

    # Process each task
    global_offset = 0
    task_info = []
    
    for task_idx, task_config in enumerate(task_configs):
        new_offset, num_episodes = process_single_task(
            task_config,
            zarr_data,
            zarr_meta,
            compressor,
            batch_size,
            global_offset,
            task_idx,
            args
        )
        
        task_info.append({
            "task_id": task_idx,
            "task_name": task_config["task_name"],
            "num_episodes": num_episodes,
            "num_samples": new_offset - global_offset,
        })
        
        global_offset = new_offset

    # Save metadata
    metadata = {
        "observation_space": args.observation_space,
        "action_space": args.action_space,
        "delta_ee": args.delta_ee,
        "joint_pos_padding": args.joint_pos_padding,
        "num_tasks": len(task_configs),
        "tasks": task_info,
        "total_samples": global_offset,
    }

    for key, value in metadata.items():
        zarr_meta.attrs[key] = value

    metadata_path = os.path.join(save_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n{'='*60}")
    print(f"Multi-task dataset created successfully!")
    print(f"Total samples: {global_offset}")
    print(f"Total tasks: {len(task_configs)}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()