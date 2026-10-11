# Isaac Lab 3.0 adapter for whole_body_tracking

Adaptation layer that gets
[HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)
(BeyondMimic motion tracking) training on **Isaac Lab 3.0** — it was written for Isaac Lab 2.1.0.

Upstream targets **Isaac Lab 2.1.0 / Isaac Sim 4.5 / Python 3.10**.
This targets **Isaac Lab 3.0.0-dev / Isaac Sim 6.1 / Python 3.12**, with **Isaac Sim PhysX** physics
and a **Newton GL** viewer.

This is a **patch repository**, not a fork — it never duplicates upstream source.

```
patches/isaaclab3-migration.patch   the code adaptation (13 files, +488 −184)
scripts/                            standalone tools
docs/RUNNING.md                     full runbook — start here
docs/MIGRATION.md                   what changed and why
docs/KNOWN-ISSUES.md                what is broken and what was never tested
```

---

## Status

Everything below was executed on Windows 11 / NVIDIA T1200 (4 GB VRAM) / Isaac Sim 6.1.0-rc.26.

| Capability | Status |
|---|---|
| Task import, env construction | ✅ verified |
| End-to-end training loop | ✅ verified, `exit 0` |
| Newton GL viewer + PhysX physics | ✅ verified |
| Config drift vs upstream | ✅ **PASS — no unintended drift** |
| Quaternion WXYZ→XYZW (code + data) | ✅ verified, legacy `.npz` loads identically |
| Capsule collision restoration | ✅ both variants train, `exit 0` |
| Motion ONNX export | ✅ `ALL CHECKS PASSED` |
| W&B upload | ✅ verified offline (including external-weight fix) |
| W&B upload to cloud | ❌ needs credentials |
| `--checkpoint latest` | ❌ broken, use an explicit path |
| Real LAFAN1 training data | ❌ absent — only a synthetic stand-still clip |

---

## Quick start

Full runbook: **[docs/RUNNING.md](docs/RUNNING.md)**.

```powershell
# Windows: without this the GBK console crashes on Isaac Lab's box-drawing output
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"

.venv\Scripts\python.exe -m isaaclab train --rl_library rsl_rl `
  --task Tracking-Flat-G1-v0 `
  --external_callback whole_body_tracking.tasks.register.register `
  --num_envs 64 --max_iterations 30000 --logger tensorboard `
  --visualizer newton_gl physics=isaacsim_physx `
  "env.commands.motion.motion_file=<motion.npz>"
```

Three arguments are mandatory on Isaac Lab 3.0 and are easy to get wrong:

- **`--external_callback ...register.register`** — the unified entrypoint resolves tasks through the
  `isaaclab.tasks` entry-point group, so a downstream package must register first.
- **`physics=isaacsim_physx`** — `SimulationCfg.physics` defaults to `None` and silently resolves to
  `PhysxCfg()`. A typed `PresetCfg` is what makes the selector work.
- **`env.commands.motion.motion_file=`** — the W&B-registry motion channel was removed with no
  replacement; motions load from local files.

Also note: **`--headless` no longer exists** in 3.0 (the parser rejects it). Use `--visualizer none`.

---

## Install

1. **Isaac Lab 3.0** — follow the official guide. Verify Isaac Sim is actually present, because the
   framework installs fine *without* it and only fails later:

   ```powershell
   .venv\Scripts\python.exe -c "import isaacsim, omni.kit.app; print('ok')"
   ```

2. **Upstream project at the patch baseline:**

   ```bash
   git clone https://github.com/HybridRobotics/whole_body_tracking.git
   cd whole_body_tracking
   git checkout cd65172
   ```

3. **Apply the patch:**

   ```bash
   git apply --check <this-repo>/patches/isaaclab3-migration.patch   # verify
   git apply         <this-repo>/patches/isaaclab3-migration.patch
   ```

4. **Robot assets** (not in either repository):

   ```bash
   curl -L -o unitree_description.tar.gz \
     https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz
   tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/
   ```

   Verify: **55,579,791 bytes**, **130 files**, `urdf/g1/main.urdf` present, all 35 referenced meshes
   resolvable under `meshes/g1/`.

5. **Install the extension and W&B** (the venv is uv-managed, so there is no `pip` module):

   ```powershell
   uv pip install --python <IsaacLab>\.venv\Scripts\python.exe -e source\whole_body_tracking
   uv pip install --python <IsaacLab>\.venv\Scripts\python.exe "wandb>=0.19"
   ```

6. **Generate the capsule collision variant** (importer 3.0 dropped the built-in conversion):

   ```powershell
   python <this-repo>\scripts\make_capsule_urdf.py
   ```

---

## What changed

Detail in **[docs/MIGRATION.md](docs/MIGRATION.md)**.

**Three import-time blockers absent from the official migration guide** — found by scanning all 114
framework symbols the project touches:

| Symbol | Issue |
|---|---|
| `AdditiveUniformNoiseCfg` | removed entirely (0 hits in 3.0) → `UniformNoiseCfg` |
| `ArticulationData.default_joint_pos_nominal` | removed (0 hits) → cache on the term instance |
| `dump_pickle`, `hydra_task_config` | removed → `pickle`, `resolve_task_config` |

**Silent-wrongness traps:**

- **Quaternion order flipped WXYZ → XYZW.** Both literals and reference-motion data need conversion;
  a mismatch corrupts every orientation reward and termination *without erroring*.
- **`ProxyArray`**: `.data.*` no longer returns `Tensor`, and `@torch.jit.script` math helpers do not
  honour `__torch_function__` — they need an explicit `.torch`.
- **`sim.physx` no longer exists**, and leaving `sim.physics` unset falls back silently.
- A **project-local bug**: `observations.py` referenced `command.robot_anchor_vel_w`, which never
  existed in `MotionCommand`.

**Restored:** capsule collision geometry (via URDF preprocessing) and the motion-ONNX export path.

---

## Impact on training results

In order of magnitude:

1. **`num_envs` 4096 → 64.** Batch per PPO update drops from 98304 to 1536 steps. A 4 GB VRAM
   constraint, not a migration artifact — and larger in effect than every API difference combined.
2. **rsl_rl 2.x → 5.5.1.** *This project's* configuration was proven equivalent; the library's
   internal numerical behaviour was **not** (no 2.x source available to diff).
3. **PhysX 5.x → 5.4+** (Isaac Sim 4.5 → 6.1).

What does **not** change: observation/reward semantics (verified line-by-line against Isaac Lab
2.3.2), Hydra command-line syntax, and quaternion handling once converted.

---

## Tooling

| Script | Purpose |
|---|---|
| `export_policy.py` | Export the **motion** ONNX contract (actions + reference motion), optional W&B upload. Restores what `MotionOnPolicyRunner.save` did upstream. |
| `validate_exported_onnx.py` | Validate an export against the source motion — no simulator needed. |
| `make_capsule_urdf.py` | Generate the capsule-collision URDF variant. |
| `diff_against_upstream.py` | Prove the resolved config has no unintended drift. |
| `check_agent_cfg_equivalence.py` | Prove the modern `actor`/`critic` config equals the deprecated `policy` config. |
| `make_synthetic_motion.py` | Produce a format-valid `.npz` for pipeline validation. |

---

## Credits and licence

Upstream [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)
is MIT licensed by its authors — paper [arXiv:2508.08241](https://arxiv.org/abs/2508.08241). This
repository contains only a patch against it plus independently written tooling; no upstream source is
redistributed here.
