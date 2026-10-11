# Migration notes: Isaac Lab 2.1 → 3.0

What the patch changes and why. Baseline: upstream `whole_body_tracking` at commit `cd65172`.
Target: Isaac Lab 3.0.0-dev (`VERSION` = `3.0.0`), Isaac Sim 6.1, Python 3.12.

Reference: the official guide at `IsaacLab/docs/source/migration/migrating_to_isaaclab_3-0.rst`.

---

## 0. Blocker the official guide does not mention

Found by scanning every framework symbol the project touches (114 symbols) rather than by reading the
guide. These fail at **import** or **construction** time:

| Symbol | Status | Replacement |
|---|---|---|
| `isaaclab.utils.noise.AdditiveUniformNoiseCfg` | **removed** — 0 hits in the 3.0 tree | `UniformNoiseCfg` (same `n_min`/`n_max`; `NoiseCfg.operation` already defaults to `"add"`) |
| `ArticulationData.default_joint_pos_nominal` | **removed** — 0 hits | store the nominal on the term instance |
| `isaaclab.utils.io.dump_pickle` | **removed** | `pickle.dump`, or drop it |
| `isaaclab_tasks.utils.hydra.hydra_task_config` | **removed** | `resolve_task_config` |

The guide covers `Isaac Sim API compatibility`, schema fragments, `ProxyArray`, quaternions, and the
RL entrypoints, but not `AdditiveUniformNoiseCfg` or `default_joint_pos_nominal`.

---

## 1. Python file changes

### `tasks/tracking/tracking_env_cfg.py`

| Change | Reason |
|---|---|
| `AdditiveUniformNoiseCfg` → `UniformNoiseCfg` | removed symbol (above) |
| new `TrackingPhysicsCfg(PresetCfg)` and `self.sim.physics = TrackingPhysicsCfg()` | `SimulationCfg.physics` defaults to `None`, which resolves to `PhysxCfg()`. A typed `PresetCfg` is required for the `physics=` CLI selector to resolve. |
| removed `self.sim.physx.gpu_max_rigid_patch_count = ...` | `SimulationCfg` has no `physx` field; the value moved into `PhysxCfg(gpu_max_rigid_patch_count=...)` |
| `self.viewer.*` → `self.sim.default_visualizer_cfg = KitVisualizerCfg(...)` | `env_cfg.viewer` is deprecated. `origin_type="asset_root"` + `asset_name="robot"` maps to `origin_type="asset"` + `origin_track_path="robot"`. |
| `num_envs` 4096 → 64 | local 4 GB GPU |
| `contact_forces.history_length` 3 → 2, `debug_vis` False, `motion.debug_vis` False | VRAM |

### `tasks/tracking/mdp/commands.py` — the densest change

| Change | Reason |
|---|---|
| `body_quat_relative_w[:, :, 0] = 1.0` → `[..., 3] = 1.0` | **Quaternion convention flipped WXYZ → XYZW.** The old literal sets the *x* component to 1, a 180° rotation about x, not the identity. |
| `MotionLoader` converts reference-motion quaternions | The `.npz` produced by the upstream Isaac Sim 4.5 pipeline is WXYZ. A `quat_order` marker (written by the patched `csv_to_npz.py`) distinguishes new XYZW files; files without it are assumed WXYZ and rolled once. Legacy datasets therefore keep working. |
| 11 × `.data.*` accessors gained `.torch` | `.data.*` now returns `ProxyArray`, not `Tensor` |
| legacy `write_*_to_sim` → `write_*_index` | the index/mask split; explicit writes are the supported path |
| `_resample_command` normalizes `slice(None)` | **Runtime-only discovery.** `CommandManager` passes `slice(None)` to mean "all envs", and `len(slice)` raises `TypeError`. |

### `tasks/tracking/mdp/observations.py`

Fixes a **project-local bug unrelated to the version jump**: `robot_anchor_lin_vel_w` and
`robot_anchor_ang_vel_w` referenced `command.robot_anchor_vel_w`, which `MotionCommand` never
defined. Now they use the two properties that do exist.

### `tasks/tracking/mdp/terminations.py`

`asset.data.GRAVITY_VEC_W` → `.torch`, and `quat_rotate_inverse` → `quat_apply_inverse`.

The `.torch` is required because the `isaaclab.utils.math` quaternion helpers are decorated with
`@torch.jit.script`, and the TorchScript dispatcher **does not honour `ProxyArray.__torch_function__`**.
A scan of all 33 jit-scripted math functions confirmed this was the only affected call site. The rename
matters because the old name warns once per environment per step, flooding the console.

### `tasks/tracking/mdp/events.py`

- `randomize_joint_default_pos` became a `ManagerTermBase` subclass. The upstream implementation
  cached the pre-randomization defaults on the removed `default_joint_pos_nominal` field; the cache
  now lives on the term instance, reachable via
  `env.event_manager.get_term_cfg("add_joint_default_pos").func._default_joint_pos_nominal`.
- The project-local `randomize_rigid_body_com` was deleted in favour of the framework term. It used
  `root_physx_view`, which the Newton backend's `Articulation` does not expose; the framework version
  writes through the neutral `set_coms_index`.

### `tasks/tracking/mdp/rewards.py`

`.torch` on `compute_first_air(...)` and `last_contact_time`.

### `robots/g1.py`

- `effort_limit_sim` → `joint_effort_limit`, `velocity_limit_sim` → `joint_velocity_limit`
  (10 sites), plus the `G1_ACTION_SCALE` read-back.
  **Behaviour change:** explicit actuator groups no longer get their solver effort limit raised to
  `1e9`, so effort is clipped twice. Keep `joint_effort_limit >= actuator_effort_limit`.
- Removed `replace_cylinders_with_capsules=True` — importer 3.0 dropped it. See section 3.
- `asset_path` now resolves through `require_asset()`, so a missing asset fails loudly with the
  download command instead of producing hundreds of `Failed to find rigid body` warnings.
- Added `WBT_G1_URDF` variant selection (auto / `capsule` / `cylinder`).

### `tasks/tracking/config/{g1,humanoid}/agents/rsl_rl_ppo_cfg.py`

Replaced the deprecated `policy=RslRlPpoActorCriticCfg(...)` + `empirical_normalization` pair with
`actor`/`critic = RslRlMLPModelCfg(...)`, and set `obs_groups` explicitly (`MISSING` otherwise
serializes to `{}` and gets auto-filled with a warning). Without this, five deprecation warnings fire
and the config breaks in Isaac Lab 3.1.

### `tasks/tracking/config/humanoid/flat_env_cfg.py`

`velocity_limit` → `joint_velocity_limit`.

### `utils/exporter.py` — rewritten

The upstream exporter subclassed `isaaclab_rl.rsl_rl.exporter._OnnxPolicyExporter`, which copies
`policy.actor`/`policy.student` and raises `ValueError("Policy does not have an actor/student module.")`
for anything else. rsl-rl ≥ 4.0 policies are `MLPModel`, exposing `mlp` / `obs_normalizer` /
`distribution`, so that base class cannot work.

The rewrite builds the graph from the model's public pieces, uses
`distribution.as_deterministic_output_module()` so the export returns the mean action, and reads
`obs_dim` from the model. It also carries the reference motion as non-persistent buffers.

### `assets/__init__.py`

Added `asset_path()` and `require_asset()`.

### `tasks/register.py` — new

`register()` imports the task package and returns `sys.argv[1:]`, for `--external_callback`.

### `scripts/csv_to_npz.py`

- Removed the `[:, [3, 0, 1, 2]]  # convert to wxyz` reorder. In 3.0 everything is XYZW; converting
  to WXYZ desynchronizes the generated file from the math that consumes it.
- Added a `quat_order` marker to the output.
- (Upstream hard-codes `/tmp/motion.npz`; Windows-hostile and still present.)

---

## 2. Verification: no unintended parameter drift

`scripts/diff_against_upstream.py` compares the resolved config against upstream values:

```
ENV CONFIG    fields compared: 32   identical: 28
  EXPECTED: scene.num_envs 4096->64, contact history_length 3->2, two debug_vis True->False
  UNEXPECTED drift: none
AGENT CONFIG  fields compared: 16   identical: 16
  UNEXPECTED drift: none
RESULT: PASS - no unintended drift
```

`scripts/check_agent_cfg_equivalence.py` runs the deprecated config through
`handle_deprecated_rsl_rl_cfg` and diffs it against the modern one: 55 fields, 6 differences, all
benign — four legacy distribution fields that `_update_distribution_cfg` never reads (because
`distribution_cfg` is already set), and `obs_groups`, which the legacy path auto-fills to the same
value.

---

## 3. Capsule collision restoration

Upstream passed `replace_cylinders_with_capsules=True` to the URDF importer. The G1's collision
geometry is **28 cylinders + 1 sphere** (visuals are 35 meshes), and PhysX cylinders produce spurious
rim/edge contacts, so this mattered. Importer 3.0 removed the option.

`scripts/make_capsule_urdf.py` bakes the conversion into a copy of the URDF, edited textually so
comments and formatting inside `<collision>` survive:

```
radius = r
length = max(0, L - 2r)     # capsule length is cap-to-cap, preserving cap positions
```

28/28 converted. The original `main.urdf` is never modified. `<visual>` is untouched (0 cylinders
there anyway). Verified: XML well-formed, 39 joints / 40 links / all mass values identical.

**Six cylinders degenerate to spheres** because `L ≤ 2r`. Those are squat pads — feet and hands:

| original | becomes |
|---|---|
| `r=0.073, L=0.064` | sphere r=0.073 |
| `r=0.07, L=0.056` | sphere r=0.07 |
| `r=0.065, L=0.04` | sphere r=0.065 |
| `r=0.068, L=0.01` | sphere r=0.068 |
| `r=0.05, L=0.05` ×2 | sphere r=0.05 |

This is the mathematically correct capsule mapping and matches what the 2.x importer produced, but it
changes those shapes substantially. `--keep-squat-cylinders` leaves them as cylinders instead.

Both variants were trained end to end (`exit 0`), and variant selection was verified for auto /
`capsule` / `cylinder`.

---

## 4. Behaviour differences summary

What genuinely changes training behaviour, in order of impact:

1. **`num_envs` 4096 → 64.** PPO's batch per update drops from 98304 to 1536 steps. This dominates
   everything else. It is a hardware constraint, not a migration artifact.
2. **Capsule conversion.** Now restored via the URDF, but the squat-cylinder degenerate cases are a
   judgement call.
3. **rsl_rl 2.x → 5.5.1.** *This project's configuration* was proven equivalent; the library's
   internal numerical behaviour was **not** — no 2.x source was available to diff against.
4. **PhysX 5.x → 5.4+** (Isaac Sim 4.5 → 6.1). Solver defaults and contact generation evolve.

What does **not** change:

- Observation and reward term semantics — verified line by line against Isaac Lab 2.3.2's
  implementations of `base_lin_vel`, `base_ang_vel`, `joint_pos_rel`, `joint_vel_rel`,
  `last_action`, `action_rate_l2`. Identical apart from the `.torch` accessor.
- Hydra command-line syntax. 3.0 still uses the same
  `@hydra.main(config_path=None, config_name=task, version_base="1.3")`; only the calling convention
  changed (decorator → `resolve_task_config` return value). Dot-overrides and scalar overrides behave
  as before; verified.
- Quaternion handling, once converted. Legacy and modern `.npz` load identically.
