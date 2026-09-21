"""
evaluate_ns2d_matched.py -- SUPERSEDED, do not use.

This script targeted `--lto_exp LTO_time_NS2d` (the exp.py-harnessed
fork-side NS2d experiment), but that experiment was NEVER ACTUALLY
TRAINED -- there is no experiments/LTO_time_NS2d/ directory and no
matching slurm log anywhere in this repo (checked both, 2026-09-21). The
paper's ACTUAL reported LTO NS2d numbers come from a completely different
track: the standalone continuous-time model trained by the LTO repo's
examples/train_ns2d_evolution.py and evaluated by
examples/evaluate_ns2d_evolution.py -- a structurally different model,
living in a different repo, that this fork-side script has no way to load.

Running this script as originally written would fail immediately with
FileNotFoundError on --lto_exp LTO_time_NS2d, not silently produce a wrong
number -- but rather than leave a script here that just fails, it's
replaced with this note.

USE INSTEAD: LTO/examples/evaluate_ns2d_all.py (in the standalone LTO
repo, not this fork) -- it compares the LTO evolution-track checkpoint
(the one actually in the paper), both matched-harness LNO retrains
(LNO_NS2d and LNO_NS2d_small, trained in THIS fork by
lno_time_ns2d_job.sh / lno_time_ns2d_small_job.sh), and the new FNO2d
baseline (also in the LTO repo), all under LNO's own start_frame=9
chained-rollout convention -- see that script's module docstring for the
full explanation, and palmetto/evaluate_ns2d_all_job.sh (in the LTO repo)
for how to run it.

This file is intentionally left as a stub (rather than deleted) so a
stale reference to it fails loudly and points here, instead of silently
running against a checkpoint that was never trained.
"""

raise SystemExit(
    "evaluate_ns2d_matched.py is superseded -- see this file's module docstring. "
    "Use LTO/examples/evaluate_ns2d_all.py instead."
)
