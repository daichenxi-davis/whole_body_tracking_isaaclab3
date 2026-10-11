"""Export a trained whole-body tracking policy to ONNX, with optional Weights & Biases upload.

This restores two things the Isaac Lab 3.0 unified CLI does not provide for this project:

1. The **motion** ONNX contract. ``isaaclab play`` exports a plain ``policy.onnx`` whose only output
   is ``actions``. Deployment for BeyondMimic needs the reference-motion outputs as well
   (``joint_pos``, ``body_pos_w``, ...), which is what the upstream ``MotionOnPolicyRunner.save``
   produced on every checkpoint.
2. The **W&B** side of that: uploading the exported ONNX as a run artifact linked to the training
   run identifier.

Usage (run from the Isaac Lab checkout, with the extension installed)::

    .venv\\Scripts\\python.exe scripts\\export_policy.py \\
        --task Tracking-Flat-G1-v0 \\
        --checkpoint latest \\
        --num_envs 2 \\
        --output_dir logs/rsl_rl/g1_flat/exported \\
        --visualizer none physics=isaacsim_physx \\
        "env.commands.motion.motion_file=C:\\path\\to\\motion.npz"

Add ``--wandb_project <project>`` to upload instead of only writing locally. Set ``WANDB_MODE=offline``
to exercise the upload path without network access.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_SOURCE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source", "whole_body_tracking"))
if REPO_SOURCE not in sys.path:
    sys.path.insert(0, REPO_SOURCE)

# Register the downstream tasks before any task resolution happens.
import whole_body_tracking.tasks  # noqa: F401, E402

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Export a trained tracking policy to ONNX.")
parser.add_argument("--task", type=str, required=True, help="Registered Gym task identifier.")
parser.add_argument("--checkpoint", type=str, default="latest", help="Checkpoint path, or 'latest'/'best'.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments (the export ignores physics).")
parser.add_argument("--seed", type=int, default=None, help="Seed used by the environment.")
parser.add_argument("--output_dir", type=str, default=None, help="Destination directory for the ONNX file.")
parser.add_argument("--output_name", type=str, default="policy.onnx", help="Output ONNX file name.")
parser.add_argument("--agent", type=str, default=None, help="Optional agent config entry point override.")
parser.add_argument("--wandb_project", type=str, default=None, help="Upload the export to this W&B project.")
parser.add_argument("--wandb_run_path", type=str, default=None, help="W&B run identifier recorded as provenance.")
parser.add_argument("--wandb_artifact_name", type=str, default="motion_policy", help="Name of the W&B artifact.")
parser.add_argument("--skip_metadata", action="store_true", help="Do not attach the deployment metadata.")
AppLauncher.add_app_launcher_args(parser)
# Everything the parser does not recognize is forwarded to Hydra (env/agent overrides, presets).
args_cli, hydra_overrides = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks.utils  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_rl.rsl_rl.utils import (  # noqa: E402
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

from whole_body_tracking.utils.exporter import (  # noqa: E402
    attach_onnx_metadata,
    export_motion_policy_as_onnx,
    upload_to_wandb,
)


def _resolve_checkpoint(task: str, agent_cfg, env_cfg, checkpoint: str) -> str:
    """Resolve a checkpoint selector (``latest``/``best``/path) to a local file path."""
    from isaaclab_rl.entrypoints.backends.play_rsl_rl import _resolve_checkpoint as resolve

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    # The resolver reads ``task`` (published-checkpoint lookup), ``agent`` (run manifest match) and
    # ``checkpoint`` off the namespace, so mirror what the play entrypoint passes.
    args = argparse.Namespace(checkpoint=checkpoint, task=task, agent=None)
    path = resolve(args, agent_cfg, env_cfg, log_root_path)
    if path is None:
        raise FileNotFoundError(
            f"Could not resolve checkpoint '{checkpoint}' under '{log_root_path}'. "
            "Pass an explicit path, or make sure the run's log directory is the one holding the model."
        )
    return path


def main() -> int:
    # Let Hydra read the overrides through argv, matching the unified CLI behaviour.
    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + hydra_overrides
    try:
        env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent or "rsl_rl_cfg_entry_point")
    finally:
        sys.argv = original_argv

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    # Normalize the agent config exactly as the unified training entrypoint does. This matters when
    # exporting a checkpoint produced by the older ``policy=`` config: its saved model cfg carries the
    # legacy ``stochastic``/``init_noise_std`` keys, which the current ``MLPModel`` no longer accepts.
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())

    checkpoint_path = _resolve_checkpoint(args_cli.task, agent_cfg, env_cfg, args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    log_dir = os.path.dirname(checkpoint_path)

    # Build the environment. Importing the env module here keeps Kit launch ordering explicit.
    from isaaclab.envs import ManagerBasedRLEnv

    env = ManagerBasedRLEnv(cfg=env_cfg)

    output_dir = args_cli.output_dir or os.path.join(log_dir, "exported")
    os.makedirs(output_dir, exist_ok=True)

    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(wrapped, agent_cfg)
        runner.load(checkpoint_path, map_location=env_cfg.sim.device)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        onnx_path = export_motion_policy_as_onnx(
            env,
            policy,
            path=output_dir,
            filename=args_cli.output_name,
        )
        print(f"[INFO] Wrote motion ONNX policy: {onnx_path}")

        if not args_cli.skip_metadata:
            attach_onnx_metadata(env, args_cli.wandb_run_path or checkpoint_path, output_dir, args_cli.output_name)
            print(f"[INFO] Attached deployment metadata (joints, gains, action scale) to {onnx_path}")

        if args_cli.wandb_project:
            import wandb

            wandb.init(project=args_cli.wandb_project, name=os.path.basename(log_dir), job_type="export")
            upload_to_wandb(onnx_path, run_path=args_cli.wandb_run_path, artifact_name=args_cli.wandb_artifact_name)
            wandb.finish()
            print(f"[INFO] Uploaded {onnx_path} to W&B project '{args_cli.wandb_project}'")

        # Report the ONNX contract so a failed export is obvious.
        import onnx

        model = onnx.load(onnx_path)
        input_shapes = {
            i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in model.graph.input
        }
        output_names = [o.name for o in model.graph.output]
        print(f"[INFO] ONNX inputs : {input_shapes}")
        print(f"[INFO] ONNX outputs: {output_names}")
        print(f"[INFO] metadata keys: {sorted(p.key for p in model.metadata_props)}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    _rc = 1
    try:
        _rc = main()
        print(f"[INFO] export finished with status {_rc}", flush=True)
    except BaseException as exc:  # surface failures before the app closes and swallows output
        import traceback

        traceback.print_exc()
        print(f"[ERROR] export failed: {type(exc).__name__}: {exc}", flush=True)
        _rc = 1
    finally:
        simulation_app.close()
    raise SystemExit(_rc)
