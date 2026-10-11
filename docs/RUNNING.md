# Running

Every command here was executed against Isaac Lab 3.0.0-dev / Isaac Sim 6.1.0-rc.26 on Windows 11.
Substitute your own paths where marked.

**Set these two variables in every new terminal.** Isaac Lab's loading screen draws Unicode box
characters; on a non-UTF-8 Windows console (GBK) that raises `UnicodeEncodeError` and kills the run
before training starts:

```powershell
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"
```

Throughout, `$IL` is the Isaac Lab checkout and `$WBT` is the patched upstream project:

```powershell
$IL  = "C:\path\to\IsaacLab"
$WBT = "C:\path\to\whole_body_tracking"
$PY  = "$IL\.venv\Scripts\python.exe"
```

---

## 1. Training

```powershell
cd $IL
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"

& $PY -m isaaclab train --rl_library rsl_rl `
  --task Tracking-Flat-G1-v0 `
  --external_callback whole_body_tracking.tasks.register.register `
  --num_envs 64 --max_iterations 30000 `
  --logger tensorboard `
  --visualizer newton_gl `
  physics=isaacsim_physx `
  "env.commands.motion.motion_file=$WBT\motions\g1_stand_synthetic.npz"
```

### Why each argument is required

| Argument | Reason |
|---|---|
| `--external_callback whole_body_tracking.tasks.register.register` | Isaac Lab 3.0's unified entrypoint resolves tasks through the `isaaclab.tasks` entry-point group. A downstream package must register its tasks before that happens. Without it: `--task Tracking-Flat-G1-v0` is not found. |
| `physics=isaacsim_physx` | `SimulationCfg.physics` defaults to `None`, which silently resolves to `PhysxCfg()`. A typed `PresetCfg` with that field name is what makes `physics=` work. |
| `--visualizer newton_gl` | Selects the Newton viewer. **`--headless` does not exist in 3.0** — the parser rejects it. Use `--visualizer none` for no viewer. |
| `env.commands.motion.motion_file=...` | The upstream W&B-registry motion channel (`--registry_name`) was removed with no replacement. Motions load from a local `.npz`, routed through a Hydra dot-override. |
| `--num_envs 64` | Upstream defaults to 4096, which needs ~16 GB of GPU VRAM. |
| `--logger tensorboard` | `wandb` also works but requires credentials. |

### Smoke test first (~10 s)

Confirm the environment before committing to a long run:

```powershell
  --num_envs 8 --max_iterations 3
```

Expected tail:

```
Learning iteration 0/3   Mean reward: ...
Learning iteration 2/3   Mean reward: ...
Training time: ~6 seconds
```

### Logs and checkpoints

`$IL\logs\rsl_rl\g1_flat\<timestamp>/` contains `model_<N>.pt`, `run.json`, `params/{env,agent}.yaml`,
TensorBoard event files, and `git/<repo>.diff`.

### W&B logging

> **Required: `wandb<0.29`.** Isaac Lab 3.0's `announce_new_run` calls `wandb.util.generate_id()`,
> which wandb removed in 0.29.0, so `--logger wandb` aborts with
> `AttributeError: module 'wandb.util' has no attribute 'generate_id'`.
> Fix: `uv pip install --python <IsaacLab>\.venv\Scripts\python.exe "wandb==0.28.0"`.
> Full explanation and two alternatives: README → *Weights & Biases setup*.

```powershell
# entity: use WANDB_USERNAME. WandbLogWriter does not read WANDB_ENTITY, even though the
# resume-hint printer does.
$env:WANDB_API_KEY="..."          # or run: wandb login
$env:WANDB_USERNAME="<entity>"    # account or team
Remove-Item Env:\WANDB_MODE -ErrorAction SilentlyContinue   # for an offline run, set it to "offline"

& $PY -m isaaclab train --rl_library rsl_rl ... `
  --logger wandb --log_project_name <project> --run_name <run>
```

Offline (no credentials, no network):

```powershell
$env:WANDB_MODE="offline"
# same command; artifacts land in <repo>\wandb\offline-run-*/ and can be pushed later with
wandb sync <repo>\wandb\offline-run-*
```

`--run_name` is a log-directory **suffix**; the W&B run name becomes `{timestamp}_{run}`. The printer
emits the generated run id, which is what `--checkpoint wandb:<entity>/<project>/<run_id>` needs later.

`WandbLogWriter` subclasses TensorBoard's writer, so every scalar that reaches TensorBoard also
reaches W&B, and each checkpoint is uploaded automatically by `save_model`.

### Physics collision variant

`main_capsule.urdf` is preferred automatically when present:

```powershell
$env:WBT_G1_URDF="capsule"    # explicit (default when the file exists)
$env:WBT_G1_URDF="cylinder"   # pristine upstream geometry
# unset = auto
```

---

## 2. Playback

```powershell
& $PY -m isaaclab play --rl_library rsl_rl `
  --task Tracking-Flat-G1-v0 --num_envs 2 `
  --checkpoint latest `
  --external_callback whole_body_tracking.tasks.register.register `
  --visualizer newton_gl physics=isaacsim_physx `
  "env.commands.motion.motion_file=$WBT\motions\g1_stand_synthetic.npz"
```

`--checkpoint` also accepts `best`, an explicit `.pt` path, or
`wandb:<entity>/<project>/<run_id>`. The `latest`/`best` selectors read the `run.json` manifest that
training writes, so they only see runs produced by the unified entrypoint — a `logs/` directory made
by the old per-library scripts will not resolve.

To find a checkpoint path manually:

```powershell
$ckpt = (Get-ChildItem "$IL\logs\rsl_rl\g1_flat" -Recurse -Filter "model_*.pt" |
         Sort-Object LastWriteTime | Select-Object -Last 1).FullName
```

`isaaclab play` also writes a plain policy export to `<log_dir>/exported/`:

```
exported/policy.pt     Torch JIT
exported/policy.onnx   plain policy — outputs `actions` only
```

That plain ONNX is **not** the deployment contract for this project. See section 3.

---

## 3. Deployable motion ONNX export

Upstream `MotionOnPolicyRunner.save()` exported a *motion* ONNX on every checkpoint: the policy plus
the reference motion, so a deployment runtime can drive the policy or replay the trajectory. The
unified entrypoint has no such hook, and the upstream exporter is incompatible with rsl-rl ≥ 4.0
(it looks for `policy.actor`, which `MLPModel` does not have).

```powershell
& $PY $WBT\scripts\export_policy.py `
  --task Tracking-Flat-G1-v0 --checkpoint "$ckpt" --num_envs 2 `
  --visualizer none physics=isaacsim_physx `
  "env.commands.motion.motion_file=$WBT\motions\g1_stand_synthetic.npz"
```

Output (defaults to `<checkpoint_dir>/exported/`, override with `--output_dir`):

```
[INFO] Wrote motion ONNX policy: ...\exported\policy.onnx
[INFO] Attached deployment metadata (joints, gains, action scale) to ...\policy.onnx
[INFO] ONNX inputs : {'obs': [1, 160], 'time_step': [1, 1]}
[INFO] ONNX outputs: ['actions','joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']
[INFO] metadata keys: ['action_scale','anchor_body_name','body_names','command_names',
                       'default_joint_pos','joint_damping','joint_names','joint_stiffness',
                       'observation_history_lengths','observation_names','run_path']
```

### ONNX contract

| | Name | Shape | Meaning |
|---|---|---|---|
| in | `obs` | `(1, obs_dim)` | policy observation, already in training order |
| in | `time_step` | `(1, 1)` | reference-motion index, clamped to the last frame |
| out | `actions` | `(1, num_actions)` | deterministic action (distribution mean) |
| out | `joint_pos` | `(1, num_joints)` | reference motion at `time_step` |
| out | `joint_vel` | `(1, num_joints)` | reference motion |
| out | `body_pos_w` | `(1, n_bodies, 3)` | reference motion, **in `body_names` order** |
| out | `body_quat_w` | `(1, n_bodies, 4)` | reference motion, **XYZW** |
| out | `body_lin_vel_w` | `(1, n_bodies, 3)` | reference motion |
| out | `body_ang_vel_w` | `(1, n_bodies, 3)` | reference motion |

`body_*` outputs contain only the configured body subset (14 for the G1 tracking task), in the order
recorded in the `body_names` metadata. `joint_pos`/`joint_vel` cover all joints.

> **Deployment note:** `torch.onnx.export` may store weights in a sibling `policy.onnx.data` file.
> Ship both files together. The W&B upload handles this explicitly.

### Upload to W&B

```powershell
& $PY $WBT\scripts\export_policy.py ... `
  --wandb_project <project> --wandb_run_path <entity>/<project>/<run_id>
```

Offline smoke test, no credentials or network:

```powershell
$env:WANDB_MODE="offline"
```

The artifact contains `policy.onnx` **and** `policy.onnx.data`. To push an offline run later:

```powershell
wandb sync <repo>\wandb\offline-run-*
```

---

## 4. Validate an export (no simulator, seconds)

```powershell
& $PY $WBT\scripts\validate_exported_onnx.py `
  --onnx "$WBT\motions\exported\policy.onnx" `
  --motion "$WBT\motions\g1_stand_synthetic.npz"
```

Expected: `ALL CHECKS PASSED`. It verifies the input/output contract, the metadata keys, that the
embedded motion equals the source `.npz` **compared by body name**, and that `time_step` clamps
correctly past the final frame.

---

## 5. Preparatory and self-check tooling

```powershell
$S = "$WBT\scripts"

# Config drift self-check — run after changing env/agent config. Expect PASS.
& $PY $S\diff_against_upstream.py

# Agent config equivalence — expect IDENTICAL.
& $PY $S\check_agent_cfg_equivalence.py

# Capsule URDF
& $PY $S\make_capsule_urdf.py --check                  # report only
& $PY $S\make_capsule_urdf.py                          # generate
& $PY $S\make_capsule_urdf.py --keep-squat-cylinders   # leave squat cylinders alone

# Synthetic motion for pipeline validation
& $PY $S\make_synthetic_motion.py `
  --output "$WBT\motions\test.npz" --visualizer none
```

---

## 6. Real training data

The bundled motion is a **synthetic stand-still clip**. It exercises the whole pipeline but cannot
produce a useful policy.

1. Obtain retargeted motion in the Unitree CSV convention (e.g. the LAFAN1 retargeting dataset).
2. Convert:

```powershell
& $PY $WBT\scripts\csv_to_npz.py --input_file <motion>.csv --input_fps 30 --output_file <out>.npz
```

> This conversion has **not** been run end to end. The migration fixed its quaternion reordering and
> its hard-coded `/tmp/motion.npz` output path, but the script itself is untested here.

3. Point training at the result via `env.commands.motion.motion_file=<out>.npz`.

Existing `.npz` files produced by the upstream Isaac Sim 4.5 pipeline load correctly without
conversion — `MotionLoader` detects the quaternion order.

---

## 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `UnicodeEncodeError: 'gbk' codec can't encode ...` | Console is not UTF-8. Set `PYTHONIOENCODING=utf-8`. |
| `unrecognized arguments: --headless` | Removed in 3.0. Use `--visualizer none`. |
| `No module named pip` | The venv is uv-managed. Use `uv pip install`, not `python -m pip`. |
| Task not found / `gym.error.NameNotFound` | Missing `--external_callback`. |
| `No compatible manifested run with a checkpoint` | `latest`/`best` see no compatible run. The run must have a `run.json` written by the unified entrypoint, and a custom resolver must pass `normalize_task_name(task)` and the agent entry point as metadata. |
| `AttributeError: module 'wandb.util' has no attribute 'generate_id'` | wandb ≥ 0.29 removed the symbol Isaac Lab 3.0 calls. Pin `wandb==0.28.0`, or pre-set `WANDB_RUN_ID`. |
| `RuntimeError: Can't get __cuda_array_interface__ on Variable that requires grad` | `isaaclab play` stepping in inference mode. Framework/rsl-rl interaction; see KNOWN-ISSUES §5c. Training and export are unaffected. |
| `AttributeError: 'SimulationCfg' object has no attribute 'physx'` | Patch not applied, or you re-added `self.sim.physx`. Use `self.sim.physics`. |
| `TypeError: MLPModel.__init__() got an unexpected keyword argument 'stochastic'` | The agent config was not normalised. Always pass it through `handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())` before building a runner — the raw resolved config carries `stochastic` / `init_noise_std` / `noise_std_type` / `state_dependent_std` as `{}`. |
| `FileNotFoundError: Required asset is missing` | Assets not extracted. The error prints the exact `curl` command. |
| `[ERROR] URDF not found` from `make_capsule_urdf.py` | It could not locate the G1 URDF. Pass `--urdf <path>` or set `WBT_SOURCE`; the error lists every path it tried. |
| `fatal: detected dubious ownership` | `git config --global --add safe.directory <path>`. |
| Empty physics, robot falls through the floor | `sim.physics` unset — see the `physics=` note in section 1. |
| ONNX export loads but inference is nonsense | `policy.onnx.data` was not shipped with `policy.onnx`. |
| A training run with `--logger wandb` never exits | Known: wandb teardown blocks after training completes. Checkpoints and the offline run directory are already written. Bound the run externally. |

---

## 8. Run the whole chain at once

`tools/run_full_chain.ps1` executes every command in this document in sequence and reports pass/fail
per step:

```powershell
powershell -File <this-repo>\tools\run_full_chain.ps1 `
  -IL $IL -WBT $WBT -Repo <this-repo>
```

13 steps, measured **13/13 PASS in 98 s** on the reference machine. Per-step logs land in
`%TEMP%\wbt_fullchain\logs\` and a machine-readable summary in `%TEMP%\wbt_fullchain\results.json`.

Two steps do not exit on their own — training with `--logger wandb` and `isaaclab play` both block in
teardown. The harness stops them as soon as their success markers appear and reports
`PASS (work complete; process stopped after success markers, teardown does not exit)`. A step that
times out *without* its expected output is reported as a failure.

Detailed rationale for the harness design (process-based timeouts, `taskkill` instead of
`Process.Kill($true)`, log-file assertions) is in the README's *How the harness bounds itself*.
