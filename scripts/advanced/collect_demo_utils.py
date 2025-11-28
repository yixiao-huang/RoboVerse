from __future__ import annotations

from loguru import logger as log
from rich.logging import RichHandler

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


import multiprocessing as mp

import torch

from metasim.scenario.robot import RobotCfg


def get_actions(all_actions, env, demo_idxs: list[int], robot: RobotCfg):
    action_idxs = env._episode_steps

    actions = []
    for env_id, (demo_idx, action_idx) in enumerate(zip(demo_idxs, action_idxs)):
        if action_idx < len(all_actions[demo_idx]):
            action = all_actions[demo_idx][action_idx]
        else:
            action = all_actions[demo_idx][-1]

        actions.append(action)

    return actions


def get_run_out(all_actions, env, demo_idxs: list[int]) -> list[bool]:
    action_idxs = env._episode_steps
    run_out = [action_idx >= len(all_actions[demo_idx]) for demo_idx, action_idx in zip(demo_idxs, action_idxs)]
    return run_out


def save_demo_mp(save_req_queue: mp.Queue, robot_cfg: RobotCfg, task_desc: str):
    from metasim.utils.save_util import save_demo

    while (save_request := save_req_queue.get()) is not None:
        demo = save_request["demo"]
        save_dir = save_request["save_dir"]
        log.info(f"Received save request, saving to {save_dir}")
        save_demo(save_dir, demo, robot_cfg=robot_cfg, task_desc=task_desc)


def ensure_clean_state(handler, expected_state=None):
    """Ensure environment is in clean initial state with intelligent validation."""
    prev_state = None
    stable_count = 0
    max_steps = 10
    min_steps = 2

    for step in range(max_steps):
        handler.simulate()
        current_state = handler.get_states()

        # Only start checking after minimum steps
        if step >= min_steps:
            if prev_state is not None:
                # Check if key states are stable (focus on articulated objects)
                is_stable = True
                if hasattr(current_state, "objects") and hasattr(prev_state, "objects"):
                    for obj_name, obj_state in current_state.objects.items():
                        if obj_name in prev_state.objects:
                            # Check DOF positions for articulated objects
                            curr_dof = getattr(obj_state, "dof_pos", None)
                            prev_dof = getattr(prev_state.objects[obj_name], "dof_pos", None)
                            if curr_dof is not None and prev_dof is not None:
                                if not torch.allclose(curr_dof, prev_dof, atol=1e-5):
                                    is_stable = False
                                    break

                # Additional validation: check if we're stable at the RIGHT state
                if is_stable and expected_state is not None:
                    is_correct_state = _validate_state_correctness(current_state, expected_state)
                    if not is_correct_state:
                        # We're stable but at wrong state - force more simulation
                        log.debug(f"State stable but incorrect at step {step}, continuing simulation...")
                        stable_count = 0
                        is_stable = False
                        # Continue simulating to let physics settle properly

                if is_stable:
                    stable_count += 1
                    if stable_count >= 2:  # Stable for 2 consecutive steps at correct state
                        break
                else:
                    stable_count = 0

            prev_state = current_state

    # Final validation if we ran out of steps
    if expected_state is not None:
        final_state = handler.get_states()
        is_final_correct = _validate_state_correctness(final_state, expected_state)
        if not is_final_correct:
            log.warning(f"State validation failed after {max_steps} steps - reset may not have taken full effect")

    # Final state refresh
    handler.get_states()


def _validate_state_correctness(current_state, expected_state):
    """Validate that current state matches expected initial state for critical objects."""
    if not hasattr(current_state, "objects") or not hasattr(expected_state, "objects"):
        return True  # Can't validate, assume correct

    # Focus on articulated objects which are most prone to reset issues
    critical_objects = []
    for obj_name, expected_obj in expected_state.objects.items():
        if hasattr(expected_obj, "dof_pos") and getattr(expected_obj, "dof_pos", None) is not None:
            critical_objects.append(obj_name)

    if not critical_objects:
        return True  # No critical objects to validate

    tolerance = 5e-3  # Reasonable tolerance for DOF positions

    for obj_name in critical_objects:
        if obj_name not in current_state.objects:
            continue

        expected_obj = expected_state.objects[obj_name]
        current_obj = current_state.objects[obj_name]

        # Check DOF positions for articulated objects (most critical for demo consistency)
        expected_dof = getattr(expected_obj, "dof_pos", None)
        current_dof = getattr(current_obj, "dof_pos", None)

        if expected_dof is not None and current_dof is not None:
            if not torch.allclose(current_dof, expected_dof, atol=tolerance):
                # Log the specific difference for debugging
                diff = torch.abs(current_dof - expected_dof).max().item()
                log.debug(f"DOF mismatch for {obj_name}: max diff = {diff:.6f} (tolerance = {tolerance})")
                return False

    return True


def force_reset_to_state(env, state, env_id):
    """Force reset environment to specific state with validation."""
    env.reset(states=[state], env_ids=[env_id])
    # Pass expected state for validation
    ensure_clean_state(env.handler, expected_state=state)
    # Reset episode counter AFTER stabilization to ensure demo starts from action 0
    if hasattr(env, "_episode_steps"):
        env._episode_steps[env_id] = 0
