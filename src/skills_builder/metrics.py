"""pass@k and pass^k (docs/01-experiment-protocol.md, Metrics).

For a task with c successes out of n trials:
    pass@k = 1 - C(n-c, k) / C(n, k)   at least one of k attempts succeeds
    pass^k = C(c, k) / C(n, k)         all k attempts succeed
Scores are means over tasks. A task with fewer than k trials is left out of
that k (this happens only when infrastructure failures drop trials).
"""

from __future__ import annotations

import random
from math import comb

KS = (1, 2, 4, 8)
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 0


def pass_at_k(n: int, c: int, k: int) -> float:
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    return comb(c, k) / comb(n, k)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _ci(values: list[float], samples: int = BOOTSTRAP_SAMPLES) -> list[float]:
    """Bootstrap 95% CI of the mean, resampling tasks."""
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(values)
    means = sorted(_mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(samples))
    return [round(means[int(0.025 * samples)], 4), round(means[int(0.975 * samples) - 1], 4)]


def score_tasks(per_task: dict[str, tuple[int, int]], ks=KS, with_ci: bool = True) -> dict:
    """per_task: task id -> (successes, trials). Returns pass@k / pass^k per k."""
    out: dict = {"tasks": len(per_task)}
    max_n = max((n for _, n in per_task.values()), default=0)
    for k in ks:
        if k > max_n:
            continue
        rows = [(n, c) for c, n in per_task.values() if n >= k]
        if not rows:
            continue
        at = [pass_at_k(n, c, k) for n, c in rows]
        hat = [pass_hat_k(n, c, k) for n, c in rows]
        out[f"pass@{k}"] = round(_mean(at), 4)
        out[f"pass^{k}"] = round(_mean(hat), 4)
        if with_ci:
            out[f"pass@{k}_ci"] = _ci(at)
            out[f"pass^{k}_ci"] = _ci(hat)
    return out


DEPTH_BUCKETS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 99))


def depth_bucket(depth: int) -> str:
    for lo, hi in DEPTH_BUCKETS:
        if lo <= depth <= hi:
            return f"{lo}-{hi}" if hi < 99 else f"{lo}+"
    raise ValueError(depth)


def summarize(per_task: dict[str, tuple[int, int]], labels: dict) -> dict:
    """Overall scores plus breakdowns by intent label, width and depth bucket."""
    def group(key_fn) -> dict:
        groups: dict[str, dict] = {}
        for tid, sc in per_task.items():
            for key in key_fn(labels[tid]):
                groups.setdefault(key, {})[tid] = sc
        return {k: score_tasks(v, with_ci=False) for k, v in sorted(groups.items())}

    return {
        "overall": score_tasks(per_task),
        "by_intent": group(lambda r: r["labels"] or ["none"]),
        "by_width": group(lambda r: [str(r["width"])]),
        "by_depth": group(lambda r: [depth_bucket(r["depth"])]),
    }
