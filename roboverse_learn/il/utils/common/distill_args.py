from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from loguru import logger as log

# from metasim.cfg.randomization import RandomizationCfg


@dataclass
class Args:
    task: str
    """Task name"""
    # random: RandomizationCfg = field(default_factory=RandomizationCfg)
    """Domain randomization options"""
    robot: str = "franka"
    """Robot name"""
    num_envs: int = 1
    """Number of parallel environments, find a proper number for best performance on your machine"""
    sim: Literal["isaaclab", "mujoco", "isaacgym"] = "isaacsim"
    """Simulator backend"""
    dp_camera: bool = False
    """Whether to use DP camera settings"""
    max_demo: int | None = None
    """Maximum number of demos to collect, None for all demos"""
    headless: bool = True
    """Run in headless mode"""
    env_spacing: float = 1.0
    """Spacing between parallel environments"""
    table: bool = True
    """Try to add a table"""
    task_id_range_low: int = 0
    """Low end of the task id range"""
    task_id_range_high: int = 1000
    """High end of the task id range"""
    algo: str = "diffusion_policy"
    """Algorithm to use"""
    subset: str = "pickcube_l0"
    """Subset your ckpt trained on"""
    action_set_steps: int = 1
    """Number of steps to take for each action set"""
    save_video_freq: int = 1
    """Frequency of saving videos"""
    max_step: int = 250
    """Maximum number of steps to collect"""
    gpu_id: int = 0
    """GPU ID to use"""
    cust_name: str | None = None
    """Custom name for the dataset"""
    custom_save_dir: str | None = None
    """Custom base path for saving demos. If None, use default structure."""

    # Distillation specific args
    num_demo_success: int = 100
    """Target number of successful demos to collect"""
    tot_steps_after_success: int = 20
    """Maximum number of steps to collect after success, or until run out of demo"""

    # copied from collect_demo.py
    run_all: bool = True
    """Rollout all trajectories, overwrite existing demos"""
    run_unfinished: bool = False
    """Rollout unfinished trajectories"""
    run_failed: bool = False
    """Rollout unfinished and failed trajectories"""
    def __post_init__(self):
        # if self.random.table and not self.table:
        #     log.warning("Cannot enable table randomization without a table, disabling table randomization")
        #     self.random.table = False
        log.info(f"Args: {self}")
