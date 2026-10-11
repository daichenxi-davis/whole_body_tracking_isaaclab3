"""Equivalence check: deprecated `policy=`+`empirical_normalization` vs modern `actor`/`critic`.

Runs the same migration helper the unified CLI uses (`handle_deprecated_rsl_rl_cfg`) on a legacy
style agent config and diffs the resulting dict against the modern config that
`whole_body_tracking` now ships. Any field difference is printed.
"""

from __future__ import annotations

import json
import sys

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg

ALGO = dict(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.005,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-3,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)


@configclass
class LegacyCfg(RslRlOnPolicyRunnerCfg):
    """Exactly what the repo shipped before the migration (Isaac Lab 2.x style)."""

    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "g1_flat"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(**ALGO)


@configclass
class ModernCfg(RslRlOnPolicyRunnerCfg):
    """What the migrated repo ships now."""

    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "g1_flat"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(**ALGO)


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def main() -> int:
    from isaaclab_rl.rsl_rl.utils import check_rsl_rl_version

    installed = check_rsl_rl_version()
    print(f"rsl_rl version: {'.'.join(str(x) for x in installed)}")

    legacy = LegacyCfg()
    legacy = handle_deprecated_rsl_rl_cfg(legacy, installed)
    modern = ModernCfg()

    ld = flatten(legacy.to_dict())
    md = flatten(modern.to_dict())

    keys = sorted(set(ld) | set(md))
    diffs = []
    for k in keys:
        if k in ("policy", "empirical_normalization"):
            continue  # deprecated carrier fields; intentionally gone in the modern config
        lv, mv = ld.get(k, "<absent>"), md.get(k, "<absent>")
        if lv != mv:
            diffs.append((k, lv, mv))

    print("\n--- fields compared:", len([k for k in keys if k not in ("policy", "empirical_normalization")]))
    if not diffs:
        print("RESULT: IDENTICAL — legacy and modern configs migrate to the same values")
    else:
        print(f"RESULT: {len(diffs)} difference(s)")
        for k, lv, mv in diffs:
            print(f"  {k}\n      legacy : {json.dumps(lv)}\n      modern : {json.dumps(mv)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
