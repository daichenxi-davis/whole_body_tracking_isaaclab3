# Changelog

Newest first. This repository is a work log as much as a deliverable: the entries record what was
verified, and — where a claim later turned out to be wrong — what the correction was. The
"Corrected" notes are deliberate; several early conclusions were misattributions to Isaac Lab that
turned out to be bugs in this repository's own tooling.

---

## Unreleased

### Added

- `LICENSE` (MIT, with an explicit scope note: no upstream source is redistributed).
- `CHANGELOG.md`.
- `docs/VERIFICATION.md` — the evidence table for every verification claim in the README.

### Fixed

- Stray `scripts/__pycache__/` and `scripts/play_hang_traceback.txt` removed from the working tree
  (both were already git-ignored; they were simply left on disk).

---

## 4 — Full-chain harness reaches 13/13 PASS

Commit `c3cf409`. **13/13 steps pass in 98 s**, process table clean afterwards.

### Added

- README section *How the harness bounds itself*, documenting each bounding mechanism and the two
  PowerShell traps that were hit.
- RUNNING.md section 8, *Run the whole chain at once*.

### Fixed

Four harness bugs, each of which had produced a wrong result:

- **Termination used `Process.Kill($true)`**, which is a .NET Core 3.0+ overload. Windows PowerShell
  5.1 exposes only the parameterless `Kill`, so the call failed with *"Cannot find an overload for
  Kill"* and nothing was killed — one leftover simulator process accumulated **550 s of CPU**. Now
  `taskkill /PID <pid> /T /F`.
- **A timeout was treated as failure unconditionally.** If the log already satisfies a step's
  assertions, the work completed and only teardown failed to exit; that is now reported distinctly.
- **The generic failure-marker scan was removed.** `MISSING_MODULES: [` matched the *success* line
  `MISSING_MODULES: []`, and a raw `Traceback` would have flagged the step whose documented outcome
  *is* a traceback. Acceptance is explicit patterns plus the real process exit code.
- **W&B step 12 searched the wrong directory.** wandb nests offline runs one further `wandb` level
  below `WANDB_DIR`; the check now searches recursively.

Also fixed a PowerShell variable-shadowing bug that corrupted every result row: `foreach ($name in …)`
inside a function that also takes a `$Name` parameter clobbers it, because PowerShell variable names
are case-insensitive.

### Changed

- Steps hold a hard wall-clock timeout and are stopped as soon as their success markers reach the
  log, with a short grace period for trailing lines. Previously each of steps 09 and 13 burned its
  entire 300 s timeout; they now finish in **5.7 s**.

---

## 3 — URDF path resolution, harness correctness, play diagnostics

Commit `628aabb`.

### Added

- `scripts/diagnose_play_hang.py` — timestamps every stage of the play sequence and arms
  `faulthandler` to dump all thread stacks if a stage blocks.
- `scripts/diagnose_actor_cfg.py` — shows the agent config before and after
  `handle_deprecated_rsl_rl_cfg`.
- KNOWN-ISSUES §5c: `isaaclab play` dies on the first `env.step` with
  `RuntimeError: Can't get __cuda_array_interface__ on Variable that requires grad`.

### Fixed

- **`make_capsule_urdf.py` could not find the URDF.** Its default path was relative to the script,
  which only resolves when the adapter is copied *inside* the upstream checkout. Resolution is now
  `--urdf` → `WBT_SOURCE` → the installed `whole_body_tracking` package → script-relative, and the
  error lists every candidate tried.
- **The harness never checked exit codes**, so a step that exited 1 was reported PASS. Each step now
  emits an explicit `exitcode: N` and the harness reads it.
- **Assertions read a captured pipeline variable**, which cannot see `Write-Host` output, so steps
  that succeeded were reported FAIL. Assertions now read the step's log file.
- **Steps ran in one session**, so a subprocess blocking in teardown held the run open with no way to
  bound it. Each step now runs as its own child process with a timeout.

### Corrected

Two earlier conclusions were wrong and are now recorded as such in KNOWN-ISSUES:

- *"`isaaclab play` hangs at policy loading."* It loads in 0.02 s and then raises on the first
  `env.step`; the process blocks in teardown afterwards, which is why it looked like a hang with no
  traceback.
- *"`create_rsl_rl_runner` hangs."* The diagnostic that reported this had omitted
  `handle_deprecated_rsl_rl_cfg`, reproducing the `stochastic` keyword `TypeError`. The real play
  entrypoint calls it (`play_rsl_rl.py:113`). With normalisation, runner creation is 0.02 s.

---

## 2 — W&B setup documented

Commit `64934a2`.

### Added

- README section *Weights & Biases setup*: the `wandb<0.29` requirement, the `WANDB_USERNAME` vs
  `WANDB_ENTITY` inconsistency, full remote and offline commands, the CLI mapping, resume via
  `--checkpoint wandb:…`, and a table of the upstream W&B features Isaac Lab 3.0 dropped.
- KNOWN-ISSUES §5b: `--logger wandb` aborts with
  `AttributeError: module 'wandb.util' has no attribute 'generate_id'`.

### Fixed

- `--checkpoint latest` in `export_policy.py`. The wrapper passed the raw CLI task name and
  `agent=None` to the framework resolver, so every run-manifest metadata comparison failed and
  `resolve_checkpoint_selector` silently found zero compatible runs. The framework's own
  `isaaclab play --checkpoint latest` was never affected.

### Notes

- `wandb.util.generate_id` was bisected: present ≤ 0.28.0, removed in 0.29.0.
  `wandb.sdk.lib.runid.generate_id` still exists. Pinning `wandb==0.28.0` is the recommended fix.

---

## 1 — Adapt `whole_body_tracking` to Isaac Lab 3.0

Commit `07c87e6`. Baseline: upstream `whole_body_tracking` at `cd65172`
(Isaac Lab 2.1.0 / Isaac Sim 4.5 / Python 3.10). Target: Isaac Lab 3.0.0-dev / Isaac Sim 6.1 /
Python 3.12, PhysX physics with a Newton GL viewer.

### Added

- `patches/isaaclab3-migration.patch` — 13 files, +488 −184.
- `scripts/` — export, validation, URDF preparation, and config self-checks.
- `tasks/register.py` (in the patch) — the `--external_callback` hook the unified CLI needs.
- Documentation: `README.md`, `docs/RUNNING.md`, `docs/MIGRATION.md`, `docs/KNOWN-ISSUES.md`.

### Fixed — import-time blockers absent from the official migration guide

Found by scanning all 114 framework symbols the project touches rather than by reading the guide:

| symbol | issue |
|---|---|
| `AdditiveUniformNoiseCfg` | removed entirely (0 hits) → `UniformNoiseCfg` |
| `ArticulationData.default_joint_pos_nominal` | removed (0 hits) → cache on the term instance |
| `dump_pickle`, `hydra_task_config` | removed → `pickle`, `resolve_task_config` |

### Fixed — semantic traps

- **Quaternion order flipped WXYZ → XYZW.** Both literals and reference-motion data need conversion;
  a mismatch corrupts every orientation reward and termination without erroring. A `quat_order`
  marker keeps legacy `.npz` files loadable — verified: legacy and modern files load identically.
- **`ProxyArray`**: `.data.*` no longer returns `Tensor`, and `@torch.jit.script` math helpers do
  **not** honour `__torch_function__`, so they need an explicit `.torch`. A scan of all 33
  jit-scripted math functions found exactly one affected call site.
- **`sim.physx` no longer exists**, and leaving `sim.physics` unset silently falls back to
  `PhysxCfg()` — so a `PresetCfg` is required for the `physics=` selector to work at all.
- **`CommandManager` passes `slice(None)`** where the command term expected an index tensor;
  `len(slice)` raised `TypeError`.
- **`replace_cylinders_with_capsules` was removed** from the URDF importer. The G1's collision
  geometry is 28 cylinders + 1 sphere; the conversion is restored by pre-processing the URDF.

### Fixed — a project-local bug unrelated to the version jump

`observations.py` referenced `command.robot_anchor_vel_w`, which `MotionCommand` never defined.

### Verified

Training loop, Newton GL viewer, config drift (PASS, no unintended drift), capsule variants,
motion ONNX export (`ALL CHECKS PASSED`), and the W&B upload path including the external-weight
sibling `policy.onnx.data`.
