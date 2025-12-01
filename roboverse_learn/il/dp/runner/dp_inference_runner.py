
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
from tqdm.rich import tqdm_rich as tqdm
import wandb
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
from scripts.advanced.collect_demo import DemoCollector, DemoIndexer
from scripts.advanced.collect_demo_utils import ensure_clean_state
from metasim.utils.state import state_tensor_to_nested



class DistillDPRunner(BaseRunner):
    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.distill_config.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # configure model
        self.model = hydra.utils.instantiate(cfg.model_config)
        self.ema_model = None
        if cfg.train_config.training_params.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.train_config.optimizer, params=self.model.parameters()
        )
        self.distill_args = hydra.utils.instantiate(cfg.distill_config.distill_args)

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
            cameras=[camera],
            env_spacing=args.env_spacing,
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
        # os.makedirs(f"tmp/{ckpt_name}", exist_ok=True)
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

        # total_success = 0
        # total_completed = 0
        # global global_step, tot_success, tot_give_up
        tot_success = 0
        tot_give_up = 0
        global_step = 0
        if args.max_demo is None:
            max_demos = args.task_id_range_high - args.task_id_range_low
        else:
            max_demos = args.max_demo
        max_demos = min(max_demos, num_demos)

        # setup collector
        task_desc = getattr(env, 'task_desc', "")
        if args.custom_save_dir:
            save_root_dir = args.custom_save_dir
        else:
            additional_str = f"{args.cust_name}" if args.cust_name else ""
            save_root_dir = f"roboverse_demo/demo_{args.sim}/distill-{args.task}{additional_str}/robot-{args.robot}"
        log.info(f"Saving demos to {save_root_dir}")
        os.makedirs(save_root_dir, exist_ok=True)
        os.makedirs(f"{save_root_dir}/tmp", exist_ok=True)
        collector = DemoCollector(env.handler, robot, save_root_dir, task_desc)

        # pbar = tqdm(total=max_demo - args.demo_start_idx, desc="Collecting demos")
        pbar = tqdm(total=args.num_demo_success, desc="Collecting successful demos")

        demo_indexer = DemoIndexer(
            save_root_dir=save_root_dir,
            start_idx=args.task_id_range_low,
            end_idx=max_demos,
            pbar=pbar,
        )
        ## Main Loop
        stop_flag = False

        # for demo_start_idx in range(
        #     args.task_id_range_low, args.task_id_range_low + max_demos, num_envs
        # ):
        while not stop_flag:
            demo_idxs = []
            if tot_success >= args.num_demo_success:
                log.info(f"Reached target number of successful demos ({args.num_demo_success}).")
                stop_flag = True

            if demo_indexer.next_idx >= max_demos:
                if not stop_flag:
                    log.warning(f"Reached maximum demo index ({max_demos}), finishing in-flight demos.")
                stop_flag = True

            # demo_end_idx = min(demo_start_idx + num_envs, num_demos)
            # current_demo_idxs = list(range(demo_start_idx, demo_end_idx))
            for demo_idx in range(env.handler.num_envs):
                demo_idxs.append(demo_indexer.next_idx)
                demo_indexer.move_on()
            log.info(f"Collecting rollouts with demo idxs: {demo_idxs}")
            ## Randomize environment for current batch of demos
            # if self.randomization_manager is not None:
            #     for demo_idx in current_demo_idxs:
            #         self.randomization_manager.randomize_for_demo(demo_idx)


            ## Reset before first step
            tic = time.time()
            obs, extras = env.reset(states=[init_states[demo_idx] for demo_idx in demo_idxs])

            ## Wait for environment to stabilize after reset (before counting demo steps)
            # For initial setup, we can't validate individual states easily, so just ensure stability
            ensure_clean_state(env.handler)
            ## Reset episode step counters AFTER stabilization
            # this is only used for checking timeouts in some envs
            if hasattr(env, "_episode_steps"):
                for env_id in range(env.handler.num_envs):
                    env._episode_steps[env_id] = 0
            ## Now record the clean, stabilized initial state
            obs = env.handler.get_states()
            nested_obs = state_tensor_to_nested(env.handler, obs)
            for env_id, demo_idx in enumerate(demo_idxs):
                log.info(f"Starting Demo {demo_idx} in Env {env_id}")
                collector.create(demo_idx, nested_obs[env_id])
            # reset policy runner state
            policyRunner.reset()
            toc = time.time()
            # reset rendering
            # env.handler.refresh_render()
            log.trace(f"Time to reset: {toc - tic:.2f}s")

            step = 0
            MaxStep = args.max_step
            ## State variables
            SuccessOnce = [False] * num_envs
            TimeOut = [False] * num_envs
            failure_count = [0] * env.handler.num_envs
            steps_after_success = [0] * env.handler.num_envs
            finished = [False] * env.handler.num_envs

            images_list = []
            print(policyRunner.policy_cfg)


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

                # actions = get_actions(all_actions, env, demo_idxs, robot)
                # obs, reward, success, time_out, extras = env.step(actions)
                nested_obs = state_tensor_to_nested(env.handler, obs)
                # run_out = get_run_out(all_actions, env, demo_idxs)

                for env_id in range(env.handler.num_envs):
                    if finished[env_id]:
                        continue

                    demo_idx = demo_idxs[env_id]
                    collector.add(demo_idx, nested_obs[env_id])

                for env_id in success.nonzero().squeeze(-1).tolist():
                    if finished[env_id]:
                        continue

                    demo_idx = demo_idxs[env_id]
                    if steps_after_success[env_id] == 0:
                        log.info(f"Demo {demo_idx} in Env {env_id} succeeded!")
                        tot_success += 1
                        pbar.update(1)
                        pbar.set_description(f"Frame {global_step} Success {tot_success} Giveup {tot_give_up}")

                    if steps_after_success[env_id] < args.tot_steps_after_success:
                        steps_after_success[env_id] += 1
                    else:
                        steps_after_success[env_id] = 0
                        collector.save(demo_idx, status="success")
                        collector.delete(demo_idx)

                        # if (not stop_flag) and (demo_indexer.next_idx < max_demos):
                        #     new_demo_idx = demo_indexer.next_idx
                        #     demo_idxs[env_id] = new_demo_idx
                        #     log.info(f"Transitioning Env {env_id}: Demo {demo_idx} to Demo {new_demo_idx}")

                        #     randomization_manager.randomize_for_demo(new_demo_idx)
                        #     force_reset_to_state(env, init_states[new_demo_idx], env_id)

                        #     obs = env.handler.get_states()
                        #     obs = state_tensor_to_nested(env.handler, obs)
                        #     collector.create(new_demo_idx, obs[env_id])
                        #     demo_indexer.move_on()
                        #     run_out[env_id] = False
                        # else:
                        finished[env_id] = True

                for env_id in time_out.nonzero().squeeze(-1).tolist():
                # for env_id in (time_out | torch.tensor(run_out, device=time_out.device)).nonzero().squeeze(-1).tolist():
                    if finished[env_id]:
                        continue

                    demo_idx = demo_idxs[env_id]
                    log.info(f"Demo {demo_idx} in Env {env_id} timed out!")
                    collector.save(demo_idx, status="failed")
                    collector.delete(demo_idx)
                    # failure_count[env_id] += 1

                    # if failure_count[env_id] < try_num:
                    #     log.info(f"Demo {demo_idx} failed {failure_count[env_id]} times, retrying...")
                    #     randomization_manager.randomize_for_demo(demo_idx)
                    #     force_reset_to_state(env, init_states[demo_idx], env_id)

                    #     obs = env.handler.get_states()
                    #     obs = state_tensor_to_nested(env.handler, obs)
                    #     collector.create(demo_idx, obs[env_id])
                    # else:
                    log.error(f"Demo {demo_idx} failed too many times, giving up")
                    # failure_count[env_id] = 0
                    tot_give_up += 1
                    # pbar.update(1)
                    pbar.set_description(f"Frame {global_step} Success {tot_success} Giveup {tot_give_up}")

                    # if demo_indexer.next_idx < max_demo:
                    #     new_demo_idx = demo_indexer.next_idx
                    #     demo_idxs[env_id] = new_demo_idx
                    #     randomization_manager.randomize_for_demo(new_demo_idx)
                    #     force_reset_to_state(env, init_states[new_demo_idx], env_id)

                    #     obs = env.handler.get_states()
                    #     obs = state_tensor_to_nested(env.handler, obs)
                    #     collector.create(new_demo_idx, obs[env_id])
                    #     demo_indexer.move_on()
                    # else:
                    finished[env_id] = True
                step += 1
                if all(finished):
                    log.info("All environments finished their demos, breaking out of step loop.")
                    break
            global_step += 1
            for i, demo_idx in enumerate(demo_idxs):
                if i % args.save_video_freq == 0:
                    iio.mimwrite(
                        f"{save_root_dir}/tmp/{demo_idx}.mp4",
                        [images[i] for images in images_list],
                    )
        log.info("Finalizing")
        collector.final()
        env.close()

    def run(self, ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = pathlib.Path(self.cfg.distill_config.ckpt_path)
        self.distill(ckpt_path=ckpt_path)

    def eval(self):
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
                if SuccessEnd[i] and steps_after_success[i] == 0:
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


# @hydra.main(
#     version_base=None,
#     config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
#     config_name=pathlib.Path(__file__).stem,
# )
# def main(cfg):
#     workspace = DistillDPRunner(cfg)
#     workspace.run()
