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
tools/run_full_chain.ps1            13-step end-to-end harness with per-step timeouts
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
| W&B export upload (offline) | ✅ verified, including external-weight fix |
| W&B training logger | ⚠️ works only with **`wandb<0.29`** — see [Weights & Biases setup](#weights--biases-setup) |
| W&B upload to cloud | ❌ not run — needs credentials |
| `--checkpoint latest` / `best` | ✅ verified (a wrapper bug was found and fixed) |
| `isaaclab play` end-to-end | ⚠️ fails on the first `env.step` — framework issue, see [KNOWN-ISSUES §5c](docs/KNOWN-ISSUES.md) |
| Full-chain harness | ✅ `tools/run_full_chain.ps1`, 13 steps, per-step timeouts |
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

## Weights & Biases setup

### ⚠️ Pin `wandb<0.29` first

Isaac Lab 3.0's `announce_new_run` calls `wandb.util.generate_id()`, but **wandb removed that symbol
in 0.29.0**, so `--logger wandb` crashes before training starts:

```
File ".../isaaclab_rl/utils/wandb.py", line 107, in announce_new_run
    os.environ["WANDB_RUN_ID"] = wandb.util.generate_id()
AttributeError: module 'wandb.util' has no attribute 'generate_id'
```

Isaac Lab declares `wandb>=0.13.11` with no upper bound, so a current wandb resolves in and breaks it.
Bisected removal point:

| wandb | `wandb.util.generate_id` |
|---|---|
| ≤ 0.28.0 | present |
| **0.29.0** | **removed** |
| 0.30.0 | removed |

`wandb.sdk.lib.runid.generate_id` still exists — only the `wandb.util` alias is gone.

Fix by pinning:

```powershell
uv pip install --python <IsaacLab>\.venv\Scripts\python.exe "wandb==0.28.0"
```

Two alternatives if you need a current wandb:

- **Pre-set `WANDB_RUN_ID`.** `announce_new_run` only generates an id when the variable is unset, so
  setting it skips the failing call. You then own the run id — which is also what
  `--checkpoint wandb:...` needs:

  ```powershell
  $env:WANDB_RUN_ID = [guid]::NewGuid().ToString().Replace('-','').Substring(0,8)
  ```

- **Patch the framework:** `wandb.util.generate_id()` → `wandb.sdk.lib.runid.generate_id()`. This is
  an Isaac Lab change and must be reapplied on every upgrade, which is why pinning is preferred here.

### Entity goes in `WANDB_USERNAME`, not `WANDB_ENTITY`

The two code paths disagree:

| Location | Environment variable read |
|---|---|
| `resolve_wandb_entity` (prints the resume hint) | `WANDB_ENTITY` **or** `WANDB_USERNAME` |
| `WandbLogWriter` (the actual `wandb.init`) | **only `WANDB_USERNAME`** |

Setting only `WANDB_ENTITY` makes the printed hint correct while the run silently lands under your
default entity. **Use `WANDB_USERNAME`** — both paths read it.

### Remote run

```powershell
cd <IsaacLab>
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"

$env:WANDB_API_KEY="<your API key>"       # or run `wandb login` and omit this line
$env:WANDB_USERNAME="<your entity>"        # account or team
Remove-Item Env:\WANDB_MODE -ErrorAction SilentlyContinue   # ensure it is not offline

.venv\Scripts\python.exe -m isaaclab train --rl_library rsl_rl `
  --task Tracking-Flat-G1-v0 `
  --external_callback whole_body_tracking.tasks.register.register `
  --num_envs 64 --max_iterations 30000 `
  --logger wandb `
  --log_project_name wbt-g1-tracking `
  --run_name first-run `
  --visualizer newton_gl physics=isaacsim_physx `
  "env.commands.motion.motion_file=<motion.npz>"
```

Verify with a short run first (`--num_envs 4 --max_iterations 2`, a few tens of seconds). Success looks
like:

```
[INFO] Logging to wandb project 'wbt-g1-tracking', run id 'xxxxxxxx'.
[INFO] Once your entity is known, resume or play it with: --checkpoint wandb:<entity>/wbt-g1-tracking/xxxxxxxx
```

**Keep that run id** — it is how you resume or play the run later.

### Offline run (no credentials, no network)

```powershell
$env:WANDB_MODE="offline"
# same command as above
```

Artifacts land in `<repo>\wandb\offline-run-*/`. Push them later with:

```powershell
wandb sync <repo>\wandb\offline-run-*
```

### CLI mapping

| Flag | Maps to | Notes |
|---|---|---|
| `--logger wandb` | `agent_cfg.logger` | choices: `{wandb, tensorboard, neptune}`; default `tensorboard` |
| `--log_project_name X` | `agent_cfg.wandb_project` **and** `neptune_project` | only applied when the logger is wandb/neptune; defaults to `isaaclab` |
| `--run_name Y` | `agent_cfg.run_name` | a **log-directory suffix**, not the W&B run name — that becomes `{timestamp}_{Y}` |

Entity and credentials are environment-only; there is no CLI flag for either.

### Resuming and playing from W&B

```powershell
--checkpoint wandb:<entity>/<project>/<run_id>
# specific iteration:  ...?checkpoint=500
```

### What the logger gives you

`WandbLogWriter` subclasses TensorBoard's `SummaryWriter`, so every scalar that reaches TensorBoard
also reaches W&B (`Loss/*`, `Episode/*`, `Policy/mean_std`, `Perf/*`), and `save_model` uploads each
checkpoint automatically.

### What is gone from upstream

| Upstream W&B feature | Status |
|---|---|
| `--logger wandb`, `--log_project_name` | ✅ present |
| Automatic metric + checkpoint upload | ✅ present |
| `--checkpoint wandb:<entity>/<project>/<run_id>` | ✅ present |
| `--wandb_path` | 🔄 renamed to the `--checkpoint wandb:...` form |
| `--registry_name` (motion from a W&B artifact registry) | ❌ removed, no replacement |
| `wandb.run.use_artifact(...)` (link run to motion artifact) | ❌ removed, no replacement |
| Motion ONNX export + upload on every checkpoint | ❌ lost → use `scripts/export_policy.py` |

Note also that `announce_new_run` **forces** `WANDB_RUN_ID` on first launch (unless you pre-set it),
so run ids are generated for you rather than chosen.

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
| `diagnose_play_hang.py` | Instrument every stage of the play sequence with timestamps; `faulthandler` dumps all thread stacks if a stage blocks. |
| `diagnose_actor_cfg.py` | Show the agent config before and after `handle_deprecated_rsl_rl_cfg`. |

Run everything at once:

```powershell
powershell -File tools\run_full_chain.ps1 -IL <IsaacLab> -WBT <whole_body_tracking>
```

Each of the 13 steps runs in its own job with a hard wall-clock timeout, writes a per-step log, and
asserts against that log. Two steps legitimately do not exit — training with `--logger wandb` and
`isaaclab play` both block in teardown — so they declare success patterns and are reported as
`PASS (timed out in teardown, after success markers were present)` rather than being counted as
failures.

---

## Credits and licence

Upstream [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)
is MIT licensed by its authors — paper [arXiv:2508.08241](https://arxiv.org/abs/2508.08241). This
repository contains only a patch against it plus independently written tooling; no upstream source is
redistributed here.
