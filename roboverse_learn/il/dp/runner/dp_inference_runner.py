
from dp.runner.dp_runner import DPRunner
import copy
import datetime
import os
import pathlib
import random
import time
from dataclasses import dataclass
from typing import Literal
from omegaconf import OmegaConf

import hydra
import imageio.v2 as iio
import numpy as np
import torch
import tqdm
import wandb
from diffusion_policy.model.diffusion.ema_model import EMAModel
from loguru import logger as log
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.constants import SimType
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_robot
from dp.base.base_eval_runner import BaseEvalRunner
from dp.base.base_runner import BaseRunner
from roboverse_learn.il.utils.common.eval_args import Args
from roboverse_learn.il.utils.common.eval_runner_getter import get_runner
from roboverse_learn.il.utils.common.json_logger import JsonLogger
from roboverse_learn.il.utils.common.lr_scheduler import get_scheduler
from roboverse_learn.il.utils.common.pytorch_util import dict_apply, optimizer_to
from torch.utils.data import DataLoader

from roboverse_pack.randomization import (
    CameraPresets,
    CameraRandomizer,
    LightPresets,
    LightRandomizer,
    MaterialPresets,
    MaterialRandomizer,
    ObjectPresets,
    ObjectRandomizer,
)
# from roboverse_pack.randomization.presets.light_presets import LightScenarios

from metasim.task.registry import get_task_class

class DistillDPRunner(DPRunner):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.train_config.training_params.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # configure model
        self.model = hydra.utils.instantiate(cfg.model_config)

        self.ema_model = None
        if cfg.train_config.training_params.use_ema:
            self.ema_model = copy.deepcopy(self.model)



        # configure training state
        self.global_step = 0
        self.epoch = 0

        self.distill_args = hydra.utils.instantiate(cfg.eval_config.distill_args)

    def distill(self, ckpt_path=None):
        args = self.distill_args

        # Setup Domain Randomization Config
        # self.dr_cfg = DomainRandomizationCfg(
        #     enable=False,
        #     seed=args.dr_seed if hasattr(args, "dr_seed") else 42,
        #     use_unified_object_randomizer=True,
        #     lighting_scenario=args.lighting_scenario if hasattr(args, "lighting_scenario") else "default",
        #     camera_scenario=args.camera_scenario if hasattr(args, "camera_scenario") else "combined",
        #     camera_name="camera0"
        # )

        num_envs: int = args.num_envs
        log.info(f"Using GPU device: {args.gpu_id}")
        task_cls = get_task_class(args.task)
        print("dp camera is ", args.dp_camera)
        if args.dp_camera:
            import warnings
            warnings.warn("Using dp camera position!")
            dp_pos = (1.0, 0.0, 0.75)
        else:
            dp_pos = (1.5, 0.0, 1.5)
        camera = PinholeCameraCfg(
            name="camera0",
            pos=dp_pos,
            look_at=(0.0, 0.0, 0.0)
        )

        scenario = task_cls.scenario.update(
            robots=[args.robot],
            simulator=args.sim,
            num_envs=args.num_envs,
            headless=args.headless,
            cameras=[camera]
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tic = time.time()
        env = task_cls(scenario, device=device)
        robot = get_robot(args.robot)

        # Initialize Domain Randomization Manager
        # if self.dr_cfg.enable:
        #     self.randomization_manager = DomainRandomizationManager(
        #         cfg=self.dr_cfg,
        #         scenario=scenario,
        #         sim_handler=env.handler
        #     )
        #     log.info("Domain Randomization Manager initialized successfully")
        # else:
        #     self.randomization_manager = None
        #     log.info("Domain Randomization is disabled")

        toc = time.time()
        log.trace(f"Time to launch: {toc - tic:.2f}s")

        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint = self.get_checkpoint_path()
        # checkpoint = ckpt_path if checkpoint is None else checkpoint
        checkpoint = ckpt_path if ckpt_path is None else checkpoint
        if checkpoint is None:
            raise ValueError(
                "No checkpoint found, please provide a valid checkpoint path."
            )
        args.checkpoint_path = checkpoint
        ckpt_name = args.checkpoint_path.name + "_" + time_str
        ckpt_name = f"{args.task}/{args.algo}/{args.robot}/{ckpt_name}"
        os.makedirs(f"tmp/{ckpt_name}", exist_ok=True)
        runnerCls = get_runner(args.algo)
        policyRunner: BaseEvalRunner = runnerCls(
            self,
            scenario=scenario,
            num_envs=num_envs,
            checkpoint_path=args.checkpoint_path,
            device=f"cuda:{args.gpu_id}",
            task_name=args.task,
            subset=args.subset,
        )

        action_set_steps = (
            2 if policyRunner.policy_cfg.action_config.action_type == "ee" else 1
        )
        ## Data
        tic = time.time()
        assert os.path.exists(env.traj_filepath), (
            f"Trajectory file: {env.traj_filepath} does not exist."
        )
        init_states, all_actions, all_states = get_traj(env.traj_filepath, robot, env.handler)
        num_demos = len(init_states)
        toc = time.time()
        log.trace(f"Time to load data: {toc - tic:.2f}s")

        total_success = 0
        total_completed = 0
        if args.max_demo is None:
            max_demos = args.task_id_range_high - args.task_id_range_low
        else:
            max_demos = args.max_demo
        max_demos = min(max_demos, num_demos)

        all_episodes = {}   # Dict: env_id -> list of episodes

        for demo_start_idx in range(
            args.task_id_range_low, args.task_id_range_low + max_demos, num_envs
        ):
            demo_end_idx = min(demo_start_idx + num_envs, num_demos)
            current_demo_idxs = list(range(demo_start_idx, demo_end_idx))

            # For trajectory saving
            current_episode_actions = {}  # Dict: env_id -> current episode actions
            current_episode_states = {}  # Dict: env_id -> current episode states
            current_episode_init_state = {}  # Dict: env_id -> init state
            episode_step_count = {}  # Dict: env_id -> step count in current episode
            for i in range(demo_start_idx, demo_end_idx):
                all_episodes[i] = []
                current_episode_actions[i] = []
                current_episode_states[i] = []
                episode_step_count[i] = 0
                current_episode_init_state[i] = init_states[i]

            ## Randomize environment for current batch of demos
            if self.randomization_manager is not None:
                for demo_idx in current_demo_idxs:
                    self.randomization_manager.randomize_for_demo(demo_idx)

            ## Reset before first step
            tic = time.time()
            obs, extras = env.reset(states=init_states[demo_start_idx:demo_end_idx])
            policyRunner.reset()
            toc = time.time()
            # reset rendering
            # env.handler.refresh_render()
            log.trace(f"Time to reset: {toc - tic:.2f}s")

            step = 0
            MaxStep = args.max_step
            SuccessOnce = [False] * num_envs
            TimeOut = [False] * num_envs
            images_list = []
            print(policyRunner.policy_cfg)
            # env.handler.refresh_render()

            dynamic_dr_interval = 20
            while step < MaxStep:
                log.debug(f"Step {step}")

                ## DR after dynamic_dr_interval steps
                # if self.randomization_manager is not None and step % dynamic_dr_interval == 0 and step > 0:
                #     log.info(f"Step {step}: Executing dynamic domain randomization")
                #     self.randomization_manager.randomize_for_demo(demo_idx=demo_start_idx + step//dynamic_dr_interval)

                new_obs = {
                    "rgb": obs.cameras["camera0"].rgb,
                    "joint_qpos": obs.robots[args.robot].joint_pos,
                }
                # import pdb; pdb.set_trace()
                # if len(images_list) == 0:
                #     iio.imwrite(f"tmp/{ckpt_name}/picture_{demo_start_idx}.png", np.array(new_obs["rgb"].cpu()).squeeze(0))
                images_list.append(np.array(new_obs["rgb"].cpu()))
                action = policyRunner.get_action(new_obs)

                for round_i in range(action_set_steps):
                    obs, reward, success, time_out, extras = env.step(action)

                # Record trajectory data (with downsampling)
                # Get states from handler for trajectory recording
                # handler_states = None
                # if hasattr(env, 'handler') and env.handler is not None:
                #     handler_states = env.handler.get_states(mode="tensor")

                # Only record for envs that haven't finished all episodes
                # if not finished_envs[i] and not done_masks[i] and (episode_step_count[i] % save_every_n_steps == 0):
                if step % args.save_every_n_steps == 0:
                    current_state = {
                        'objects': obs.objects,
                        'robots': obs.robots,
                        # 'cameras': obs.cameras,
                    }
                    """

                    """
                    current_action = action
                    # # Get robot joint positions as actions from handler states
                    # robot_name = scenario.robots[0].name
                    # joint_names = sorted(scenario.robots[0].actuators.keys())

                    # if handler_states is not None and hasattr(handler_states, 'robots') and robot_name in handler_states.robots:
                    #     # Use handler states (preferred)
                    #     robot_state = handler_states.robots[robot_name]
                    #     joint_positions = robot_state.joint_pos[i].cpu().numpy()
                    # else:
                    #     # Fallback to obs if handler not available
                    #     robot_state = obs.robots[robot_name]
                    #     joint_positions = robot_state.joint_pos[i].cpu().numpy()
                    # joint_qpos = obs.robots[args.robot].joint_pos,
                    # action_record = {
                    #     "dof_pos_target": {name: float(pos) for name, pos in zip(joint_names, joint_positions)},
                    # }
                    # current_episode_actions[i].append(action_record)

                    # # Record state if requested
                    # if save_states and current_episode_states[i] is not None:
                    #     # Extract state for this specific env using handler
                    #     current_state = extract_state_dict(env, scenario, env_idx=i)
                    #     current_episode_states[i].append(current_state)


                # eval
                SuccessOnce = [SuccessOnce[i] or success[i] for i in range(num_envs)]
                TimeOut = [TimeOut[i] or time_out[i] for i in range(num_envs)]
                step += 1
                if all(SuccessOnce):
                    break

            SuccessEnd = success.tolist()
            total_success += SuccessOnce.count(True)
            total_completed += len(SuccessOnce)

            for i, demo_idx in enumerate(range(demo_start_idx, demo_end_idx)):
                demo_idx_str = str(demo_idx).zfill(4)
                if i % args.save_video_freq == 0:
                    iio.mimwrite(
                        f"tmp/{ckpt_name}/{demo_idx}.mp4",
                        [images[i] for images in images_list],
                    )
                with open(f"tmp/{ckpt_name}/{demo_idx_str}.txt", "w") as f:
                    f.write(f"Demo Index: {demo_idx}\n")
                    f.write(f"Num Envs: {num_envs}\n")
                    f.write(f"SuccessOnce: {SuccessOnce[i]}\n")
                    f.write(f"SuccessEnd: {SuccessEnd[i]}\n")
                    f.write(f"TimeOut: {TimeOut[i]}\n")
                    f.write(f"Domain Randomization Enabled: {self.dr_cfg.enable}\n")  # Record DR status
                    f.write(
                        f"Cumulative Average Success Rate: {total_success / total_completed:.4f}\n"
                    )
                if SuccessEnd[i]:
                    # saving success trajectory
                    episode_data = {
                        "init_state": current_episode_init_state[i],
                        "actions": current_episode_actions[i],
                        "states": current_episode_states[i]
                    }
                    all_episodes[i].append(episode_data)
                    # log.info(f"Env {i} Episode {episodes_per_env[i].item()}: Saved trajectory ({len(current_episode_actions[i])} steps, return: {current_returns[i].item():.2f})")

                # Reset trajectory tracking for this env
                current_episode_actions[i] = []
                current_episode_states[i] = []
                episode_step_count[i] = 0
            log.info("Demo Indices: ", range(demo_start_idx, demo_end_idx))
            log.info("Num Envs: ", num_envs)
            log.info(f"SuccessOnce: {SuccessOnce}")
            log.info(f"SuccessEnd: {SuccessEnd}")
            log.info(f"TimeOut: {TimeOut}")
        log.info(f"FINAL RESULTS: Average Success Rate = {total_success / total_completed:.4f}")
        with open(f"tmp/{ckpt_name}/final_stats.txt", "w") as f:
            f.write(f"Total Success: {total_success}\n")
            f.write(f"Total Completed: {total_completed}\n")
            f.write(f"Average Average Success Rate: {total_success / total_completed:.4f}\n")
            f.write(f"Domain Randomization Config: {self.dr_cfg}\n")  # save DR config
        env.close()
