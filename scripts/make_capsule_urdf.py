"""Generate a capsule-collision variant of the G1 URDF.

Upstream Isaac Lab 2.x passed ``replace_cylinders_with_capsules=True`` to the URDF importer, which
rewrote the G1's cylinder collision geoms into capsules. URDF importer 3.0 removed that option
(migration guide, deprecated-settings table), and PhysX cylinders generate spurious rim/edge contacts,
so the conversion has to be baked into the URDF instead.

Geometry mapping
----------------
A URDF ``<cylinder radius="r" length="L"/>`` under ``<collision>`` is a Z-aligned cylinder spanning
``[-L/2, +L/2]``. The capsule that preserves the end-cap **positions** uses

    radius = r
    length = max(0, L - 2r)     # capsule "length" is the cap-to-cap distance

so ``length = 0`` degenerates to a sphere of radius ``r``, which is the correct shape when the
original cylinder was shorter than its diameter. Those degenerate cases are reported.

``<visual>`` geometry is never touched: the G1's visuals are meshes (0 cylinders), and the URDF is
edited textually so that comments, ordering and formatting inside ``<collision>`` are preserved.

Usage::

    python scripts/make_capsule_urdf.py                    # writes main_capsule.urdf next to the source
    python scripts/make_capsule_urdf.py --check            # report only, write nothing
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys

_RELATIVE_URDF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "source",
    "whole_body_tracking",
    "whole_body_tracking",
    "assets",
    "unitree_description",
    "urdf",
    "g1",
    "main.urdf",
)


def _candidate_urdfs() -> list[tuple[str, str]]:
    """Return ``(path, where-it-came-from)`` candidates for the G1 URDF, most reliable first.

    This script lives in the adapter repository, which is separate from the patched upstream
    checkout, so the URDF cannot be found by a path relative to the script alone. Resolution order:

    1. ``WBT_SOURCE`` — the same variable the other tools use, pointing at
       ``<upstream>/source/whole_body_tracking``.
    2. The installed ``whole_body_tracking`` package (an editable install resolves to the checkout).
    3. A path relative to this script, for the layout where the adapter was copied into the upstream
       repository.
    """
    candidates: list[tuple[str, str]] = []
    suffix = os.path.join("whole_body_tracking", "assets", "unitree_description", "urdf", "g1", "main.urdf")

    source = os.environ.get("WBT_SOURCE")
    if source:
        candidates.append((os.path.join(source, suffix), "WBT_SOURCE"))

    try:
        spec = importlib.util.find_spec("whole_body_tracking")
        if spec is not None and spec.origin:
            package_dir = os.path.dirname(os.path.abspath(spec.origin))
            candidates.append((os.path.join(package_dir, "assets", "unitree_description", "urdf", "g1", "main.urdf"), "installed package"))
    except (ImportError, ValueError):
        pass

    candidates.append((_RELATIVE_URDF, "path relative to this script"))
    return candidates


def resolve_urdf(explicit: str | None) -> str:
    """Resolve the source URDF path, raising a helpful error when none of the candidates exist."""
    if explicit:
        path = os.path.abspath(explicit)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"--urdf path does not exist: {path}")
        return path

    tried: list[str] = []
    for path, origin in _candidate_urdfs():
        path = os.path.abspath(path)
        tried.append(f"  [{origin}] {path}")
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "Could not locate the G1 URDF (main.urdf). Tried:\n"
        + "\n".join(tried)
        + "\n\nPass --urdf <path>, or set WBT_SOURCE to the patched upstream source directory, e.g.\n"
        "  set WBT_SOURCE=<...>\\whole_body_tracking\\source\\whole_body_tracking"
    )

# Matches one <collision> ... </collision> block (non-greedy, self-closing not used in this URDF).
_COLLISION_RE = re.compile(r"<collision\b.*?</collision>", re.DOTALL)
# Matches a cylinder geometry tag with its two attributes in either order.
_CYLINDER_RE = re.compile(r'<cylinder\b(?=[^>]*\bradius="([^"]+)")(?=[^>]*\blength="([^"]+)")([^>]*?)/>')


def convert_collision_block(block: str, log: list[str], keep_squat: bool = False) -> tuple[str, int]:
    """Rewrite every cylinder geometry inside one collision block into a capsule.

    Args:
        block: The ``<collision>...</collision>`` text to rewrite.
        log: Accumulator for human-readable conversion notes.
        keep_squat: When True, cylinders with ``length <= 2 * radius`` are left unchanged. Those
            degenerate into a sphere of the same radius (twice the volume of the original squat
            cylinder), which can visibly change a foot/hand pad. Upstream 2.x converted them anyway.
    """
    conversions = 0

    def _swap(match: re.Match) -> str:
        nonlocal conversions
        radius = float(match.group(1))
        length = float(match.group(2))
        new_length = max(0.0, length - 2.0 * radius)
        if new_length <= 0.0:
            if keep_squat:
                log.append(
                    f"    kept cylinder: radius={radius:g} length={length:g} "
                    f"(would degenerate to a sphere of radius {radius:g})"
                )
                return match.group(0)
            log.append(
                f"    degenerate: radius={radius:g} length={length:g} -> capsule length 0 "
                f"(becomes a sphere of radius {radius:g})"
            )
        else:
            log.append(f"    radius={radius:g} length={length:g} -> capsule length={new_length:g}")
        conversions += 1
        attrs = match.group(3)
        # Preserve attribute order/spacing by substituting only the values that change.
        attrs = re.sub(r'(\blength=")[^"]*(")', lambda m: f"{m.group(1)}{new_length:g}{m.group(2)}", attrs)
        return f"<capsule{attrs}/>"

    return _CYLINDER_RE.sub(_swap, block), conversions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urdf", default=None, help="Source URDF. Auto-resolved when omitted (see WBT_SOURCE).")
    ap.add_argument("--output", default=None, help="Output URDF (default: <source>_capsule.urdf).")
    ap.add_argument("--check", action="store_true", help="Report what would change without writing.")
    ap.add_argument(
        "--keep-squat-cylinders",
        action="store_true",
        help="Leave cylinders with length <= 2*radius as cylinders (they would degenerate to spheres).",
    )
    args = ap.parse_args()

    try:
        src = resolve_urdf(args.urdf)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    out = os.path.abspath(args.output) if args.output else src.replace(".urdf", "_capsule.urdf")

    text = open(src, encoding="utf-8").read()

    visual_cylinders = sum(
        len(_CYLINDER_RE.findall(v)) for v in re.findall(r"<visual\b.*?</visual>", text, re.DOTALL)
    )
    if visual_cylinders:
        print(f"[WARN] {visual_cylinders} cylinder(s) found under <visual>; they will NOT be touched.")

    all_log: list[str] = []
    total = 0

    def _convert_block(match: re.Match) -> str:
        nonlocal total
        converted, n = convert_collision_block(match.group(0), all_log, args.keep_squat_cylinders)
        total += n
        return converted

    new_text = _COLLISION_RE.sub(_convert_block, text)

    print(f"source : {src}")
    print(f"cylinders converted in <collision>: {total}")
    for line in all_log:
        print(line)

    if total == 0:
        print("[WARN] no cylinder collision geometry found; nothing to do.")
        return 0

    remaining = len(_CYLINDER_RE.findall("\n".join(_COLLISION_RE.findall(new_text))))
    if remaining:
        print(f"[WARN] {remaining} cylinder(s) remain in <collision>; conversion incomplete.", file=sys.stderr)
        return 1

    if args.check:
        print("[INFO] --check: nothing written.")
        return 0

    with open(out, "w", encoding="utf-8") as fh:
        fh.write(new_text)
    print(f"output : {out}")
    print(f"[INFO] wrote {os.path.getsize(out):,} B (source {os.path.getsize(src):,} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
