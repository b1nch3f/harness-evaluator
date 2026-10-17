from math import comb

from test_bench import metrics


def test_pass_k_formulas():
    assert metrics.pass_hat_k(4, 4, 4) == 1.0
    assert metrics.pass_hat_k(4, 3, 4) == 0.0
    assert metrics.pass_hat_k(8, 6, 4) == comb(6, 4) / comb(8, 4)
    assert metrics.pass_at_k(4, 1, 4) == 1.0
    assert metrics.pass_at_k(4, 0, 1) == 0.0
    assert metrics.pass_at_k(4, 2, 1) == 0.5


def test_score_tasks_means_and_skips_large_k():
    s = metrics.score_tasks({"a": (4, 4), "b": (2, 4), "c": (0, 4)})
    assert s["pass^1"] == round((1 + 0.5 + 0) / 3, 4)
    assert s["pass^4"] == round(1 / 3, 4)
    assert "pass^8" not in s
    lo, hi = s["pass^1_ci"]
    assert 0 <= lo <= s["pass^1"] <= hi <= 1


def test_summarize_groups_by_label_width_depth():
    labels = {"a": {"labels": ["cancel"], "width": 1, "depth": 2},
              "b": {"labels": ["cancel", "return"], "width": 2, "depth": 9}}
    s = metrics.summarize({"a": (2, 2), "b": (0, 2)}, labels)
    assert s["by_intent"]["cancel"]["tasks"] == 2 and s["by_intent"]["return"]["pass^1"] == 0
    assert set(s["by_width"]) == {"1", "2"}
    assert set(s["by_depth"]) == {"2-3", "8+"}
