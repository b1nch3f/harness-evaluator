from test_bench.wandb_push import _endpoint_label, wandb_config, wandb_rows

RECORD = {
    "run_id": "20261018-031619_8ca653d1_gpt-oss-120b_train_n4", "status": "complete",
    "started_at": "2026-10-18T03:16:19+00:00",
    "harness": {"name": "a0-bare", "version": "8ca653d17ddf" + "0" * 52, "git_commit": "abc",
                "prompt_sha256": "p", "skills_sha256": None, "uses_policy": False, "skills": []},
    "scaffolding": {"git_sha": "abc", "dirty": False}, "tau2_data_commit": "t",
    "settings": {"split": "train", "trials": 4},
    "models": {"agent": {"model": "openai/gpt-oss-120b", "api_base": "api.runpod.ai/v2/SECRETID/openai/v1",
                         "temperature": 0.0, "reasoning_effort": "medium"}},
    "task_ids": ["0", "1"], "episodes": {"completed": 8}, "seconds": 10.0,
    "scores": {"train": {"overall": {"tasks": 2, "pass@1": 0.5, "pass^1": 0.5, "pass@1_ci": [0.1, 0.9],
                                     "pass^1_ci": [0.1, 0.9], "pass@2": 0.75, "pass^2": 0.25},
                         "by_intent": {"x": {}}}},
    "per_task": {"0": {}, "1": {}},
}


def test_endpoint_is_host_only():
    assert _endpoint_label("api.runpod.ai/v2/SECRETID/openai/v1") == "api.runpod.ai"
    assert _endpoint_label("https://h.example/v1?key=s") == "h.example"
    assert "SECRETID" not in str(wandb_config(RECORD))


def test_rows_are_overall_only_per_k():
    rows, summary = wandb_rows(RECORD)
    assert rows == [{"k": 1, "train/pass^k": 0.5, "train/pass@k": 0.5},
                    {"k": 2, "train/pass^k": 0.25, "train/pass@k": 0.75}]
    assert summary["train/pass^1"] == 0.5 and summary["train/pass^1_ci_low"] == 0.1
    assert summary["train/pass^2_ci_low"] is None
    assert not any("intent" in k or "task_" in k for k in summary)


def test_labelled_charts_name_both_axes():
    import wandb
    from test_bench.wandb_push import X_AXIS, labelled_charts

    rows, _ = wandb_rows(RECORD)
    charts = labelled_charts(rows, wandb)
    assert set(charts) == {"charts/train_pass^k", "charts/train_pass@k"}
    table = charts["charts/train_pass^k"].table
    assert table.columns == [X_AXIS, "pass^k"] and table.data == [[1, 0.5], [2, 0.25]]
