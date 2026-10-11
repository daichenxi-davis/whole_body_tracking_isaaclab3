"""Show what the agent config actually contains at each stage of the play path.

Diagnoses the `TypeError: MLPModel.__init__() got an unexpected keyword argument 'stochastic'`
raised from rsl_rl's `construct_algorithm` (ppo.py:447), which appears only when a checkpoint is
loaded — the training path does not hit it.
"""

import json
import os
import sys

sys.path.insert(0, os.environ.get("WBT_SOURCE", r"C:\Users\chenxi\Desktop\whole_body_tracking\source\whole_body_tracking"))

import whole_body_tracking.tasks  # noqa: F401

from isaaclab_rl.rsl_rl.utils import check_rsl_rl_version, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402


def show(label: str, cfg) -> None:
    d = cfg.to_dict()
    print(f"--- {label} ---")
    for key in ("actor", "critic", "policy", "empirical_normalization"):
        print(f"  {key:24s} {json.dumps(d.get(key, '<absent>'), default=str)[:220]}")
    actor = d.get("actor")
    if isinstance(actor, dict):
        print(f"  actor keys              : {sorted(actor)}")
        print(f"  actor has 'stochastic'  : {'stochastic' in actor}")


env_cfg, agent_cfg = resolve_task_config("Tracking-Flat-G1-v0", "rsl_rl_cfg_entry_point")
show("raw (as resolved from the task registry)", agent_cfg)

agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
show("after handle_deprecated_rsl_rl_cfg", agent_cfg)

print()
print(f"rsl_rl version: {check_rsl_rl_version()}")
