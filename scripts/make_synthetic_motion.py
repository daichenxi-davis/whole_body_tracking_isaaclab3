"""Generate a synthetic, format-valid motion .npz for whole_body_tracking (Isaac Lab 3.0).

This is a pipeline-validation artifact, NOT a real motion. It drives forward kinematics on the G1
asset for a short "stand still" clip so that ``MotionLoader``, the command term, the reward terms
and the RL loop can all be exercised end-to-end. Replace it with a real retargeted motion
(``scripts/csv_to_npz.py`` on a LAFAN1-style CSV) for any actual training.

Run with the Isaac Lab venv, from the Isaac Lab checkout:

    .venv\\Scripts\\python.exe <this file> --output <path.npz>

Quaternion order: Isaac Lab 3.0 is XYZW, matching the ``quat_order`` marker written into the file.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate a synthetic G1 motion npz.")
parser.add_argument("--output", type=str, required=True, help="Destination .npz path.")
parser.add_argument("--num_frames", type=int, default=200, help="Number of output frames.")
parser.add_argument("--output_fps", type=int, default=50, help="Output frame rate.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

# Allow running against an uninstalled source tree:
#   set WBT_SOURCE=<...>\whole_body_tracking\source\whole_body_tracking
_source = os.environ.get("WBT_SOURCE")
if _source and _source not in sys.path:
    sys.path.insert(0, _source)

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG  # noqa: E402


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / args_cli.output_fps, device=args_cli.device))
    sim.set_camera_view(eye=(2.0, 2.0, 1.5), target=(0.0, 0.0, 0.8))
    scene = InteractiveScene(SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5))
    sim.reset()
    robot: Articulation = scene["robot"]
    print(f"[INFO] joint names ({len(robot.data.joint_names)}): {robot.data.joint_names}")
    print(f"[INFO] body names ({len(robot.data.body_names)}): {robot.data.body_names}")

    # Static "stand still" clip: hold the configured default joint pose with a settled floating base.
    root_state = robot.data.default_root_state.torch.clone()
    joint_pos = robot.data.default_joint_pos.torch.clone()
    joint_vel = torch.zeros_like(joint_pos)

    # Write the flat base state, then let forward kinematics populate the body data.
    robot.write_root_link_pose_to_sim_index(root_pose=root_state[:, :7])
    robot.write_root_com_velocity_to_sim_index(root_velocity=root_state[:, 7:])
    robot.write_joint_position_to_sim_index(position=joint_pos)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())

    joint_pos_np = robot.data.joint_pos.torch[0].cpu().numpy().copy()
    joint_vel_np = robot.data.joint_vel.torch[0].cpu().numpy().copy()
    body_pos_w = robot.data.body_pos_w.torch[0].cpu().numpy().copy()
    body_quat_w = robot.data.body_quat_w.torch[0].cpu().numpy().copy()  # XYZW in Isaac Lab 3.0
    body_lin_vel_w = robot.data.body_lin_vel_w.torch[0].cpu().numpy().copy()
    body_ang_vel_w = robot.data.body_ang_vel_w.torch[0].cpu().numpy().copy()

    n = args_cli.num_frames
    log = {
        "fps": np.array(args_cli.output_fps),
        "quat_order": np.array("xyzw"),
        "joint_pos": np.tile(joint_pos_np, (n, 1)).astype(np.float32),
        "joint_vel": np.tile(joint_vel_np, (n, 1)).astype(np.float32),
        "body_pos_w": np.tile(body_pos_w, (n, 1, 1)).astype(np.float32),
        "body_quat_w": np.tile(body_quat_w, (n, 1, 1)).astype(np.float32),
        "body_lin_vel_w": np.tile(body_lin_vel_w, (n, 1, 1)).astype(np.float32),
        "body_ang_vel_w": np.tile(body_ang_vel_w, (n, 1, 1)).astype(np.float32),
    }
    out = os.path.abspath(args_cli.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, **log)
    print(f"[INFO] wrote {out}")
    print(f"[INFO] frames={n} fps={args_cli.output_fps} bodies={body_pos_w.shape[0]} joints={joint_pos_np.shape[0]}")
    print(f"[INFO] body_quat_w[0] (xyzw) = {body_quat_w[0]}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
