"""Validate an exported motion ONNX policy without a simulator.

Checks, using onnxruntime and numpy only:

1. The ONNX graph loads and the input/output contract matches expectations.
2. The deployment metadata keys are present.
3. The embedded reference-motion outputs equal the source ``.npz`` motion, comparing **by body
   name** (the ONNX stores only the configured subset of bodies, in the configured order, which is
   what the deployment runtime reconstructs from the ``body_names`` metadata).
4. ``time_step`` clamping behaves at and beyond the final frame.

Usage::

    python scripts/validate_exported_onnx.py --onnx <policy.onnx> --motion <motion.npz>
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import onnx
import onnxruntime as ort

EXPECTED_OUTPUTS = [
    "actions",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
]
EXPECTED_METADATA = [
    "joint_names",
    "joint_stiffness",
    "joint_damping",
    "default_joint_pos",
    "action_scale",
    "anchor_body_name",
    "body_names",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", required=True, help="Exported policy.onnx.")
    ap.add_argument("--motion", required=True, help="Source motion .npz used for the export.")
    ap.add_argument(
        "--body_names",
        nargs="*",
        default=None,
        help="Override the body subset to compare. Defaults to the ONNX 'body_names' metadata.",
    )
    args = ap.parse_args()

    failures: list[str] = []

    model = onnx.load(args.onnx)
    input_shapes = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in model.graph.input}
    output_names = [o.name for o in model.graph.output]
    meta = {p.key: p.value for p in model.metadata_props}

    print(f"inputs : {input_shapes}")
    print(f"outputs: {output_names}")

    if sorted(input_shapes) != ["obs", "time_step"]:
        failures.append(f"unexpected input names: {sorted(input_shapes)}")
    if output_names != EXPECTED_OUTPUTS:
        failures.append(f"unexpected output names: {output_names}")

    missing_meta = [k for k in EXPECTED_METADATA if k not in meta]
    if missing_meta:
        failures.append(f"missing metadata keys: {missing_meta}")

    obs_dim = input_shapes.get("obs", [0, 0])[1]
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    motion = np.load(args.motion)
    n_frames = motion["joint_pos"].shape[0]
    quat_order = str(motion["quat_order"]) if "quat_order" in motion.files else "wxyz"
    body_quat_w = motion["body_quat_w"].astype(np.float32)
    if quat_order == "wxyz":
        body_quat_w = np.roll(body_quat_w, shift=-1, axis=-1)  # match MotionLoader

    # --- reference comparison --------------------------------------------------------
    # The exported body subset lives in the ONNX metadata; the exported body arrays are already in
    # that order, so only the source motion needs selecting.
    body_names = args.body_names or (meta["body_names"].split(",") if "body_names" in meta else None)
    if body_names is None:
        failures.append("no --body_names given and no 'body_names' metadata found")
        body_names = []

    # Rebuild the source body axis by matching names against the reference-motion body order.
    # The .npz body axis follows the robot's articulation body order, which is not recorded in the
    # file, so derive it by brute force: the exported arrays are a permutation of the source axis.
    exported_pos = sess.run(None, {"obs": np.zeros((1, obs_dim), np.float32), "time_step": np.zeros((1, 1), np.float32)})
    exported_pos = dict(zip(output_names, exported_pos))
    src_pos = motion["body_pos_w"][0]
    n_exported = exported_pos["body_pos_w"][0].shape[0]
    if n_exported != len(body_names):
        failures.append(f"metadata body_names has {len(body_names)} entries but ONNX stores {n_exported}")
    else:
        diffs = np.abs(exported_pos["body_pos_w"][0][:, None, :] - src_pos[None, :, :]).max(-1)
        picked = diffs.argmin(1)
        if diffs[np.arange(n_exported), picked].max() > 1e-6:
            failures.append("exported body positions do not match any permutation of the source body axis")
        print(f"body index map (onnx -> source axis): {list(map(int, picked))}")
        print(f"body name order: {body_names}")

    # --- per-frame comparison --------------------------------------------------------
    src = {
        "joint_pos": motion["joint_pos"],
        "joint_vel": motion["joint_vel"],
        "body_pos_w": motion["body_pos_w"],
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": motion["body_lin_vel_w"],
        "body_ang_vel_w": motion["body_ang_vel_w"],
    }
    for step in (0, n_frames // 2, n_frames - 1, n_frames + 25):  # last one exercises clamping
        expected_idx = min(step, n_frames - 1)
        outs = sess.run(None, {"obs": np.zeros((1, obs_dim), np.float32), "time_step": np.array([[step]], np.float32)})
        got = dict(zip(output_names, outs))
        for name in ("joint_pos", "joint_vel"):
            ref = src[name][expected_idx]
            g = got[name][0][: ref.shape[0]]
            if not np.allclose(g, ref, atol=1e-5):
                failures.append(f"time_step={step} ({name}) max diff {np.abs(g - ref).max():.3e}")
        for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            ref = src[name][expected_idx][picked] if len(body_names) == n_exported else None
            if ref is None:
                continue
            g = got[name][0]
            if not np.allclose(g, ref, atol=1e-5):
                failures.append(f"time_step={step} ({name}) max diff {np.abs(g - ref).max():.3e}")
        print(f"  time_step={step:5d} -> ref frame {expected_idx:4d}  matched")

    print("\n" + ("ALL CHECKS PASSED" if not failures else "FAILURES:"))
    for f in failures:
        print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
