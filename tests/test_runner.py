"""The eval runner, offline: scripted models, temporary run folders."""

import json
import shutil
from dataclasses import replace

import pytest
from scripted import Scripted, call, text

from test_bench import data, metrics
from test_bench.harness import load_harness
from test_bench.retail.episode import EpisodeSpec, InfraError
from test_bench.runner import RunRefused, RunSettings, halves, load_runs, run_eval, select_tasks

NO_JUDGE_TRAIN = [t["id"] for t in data.tasks()
                  if t["id"] in data.splits()["train"] and not t["evaluation_criteria"].get("nl_assertions")][:2]
NO_JUDGE_TEST = next(t["id"] for t in data.tasks()
                     if t["id"] in data.splits()["test"] and not t["evaluation_criteria"].get("nl_assertions"))
QUIET = {"echo": lambda _: None}


def scripted_spec(exp, succeed_on=lambda tid, trial: trial == 0):
    """Trial 0 replays the expected actions (reward 1); later trials do nothing."""
    def make(tid, trial):
        task = data.tasks_by_id()[tid]
        actions = task["evaluation_criteria"]["actions"] if succeed_on(tid, trial) else []
        agent = Scripted([[call(a["name"], **a["arguments"])] for a in actions] + [[text("Done.")]])
        user = Scripted([[text("Hi, I need help.")], [text("Thanks. ###STOP###")]])
        return EpisodeSpec(task_id=tid, harness=exp, trial=trial,
                           agent_model=agent, user_model=user, judge_model=Scripted([]))
    return make


@pytest.fixture
def exp(tmp_path):
    """A committed-looking copy of the domain harness in a temporary harnesses/ root."""
    folder = tmp_path / "domain"
    folder.mkdir()
    shutil.copy(load_harness("domain").dir / "system_prompt.md", folder / "system_prompt.md")
    return replace(load_harness("domain", root=tmp_path), dirty=False)


def test_run_writes_a_self_contained_run_folder(exp):
    record = run_eval(exp, RunSettings(trials=2, tasks=tuple(NO_JUDGE_TRAIN)), make_spec=scripted_spec(exp), **QUIET)
    assert record["status"] == "complete" and record["episodes"]["completed"] == 4
    run_dir = exp.dir / "runs" / record["run_id"]

    # results.json in the run folder holds settings, scores and per-task rewards.
    saved = json.loads((run_dir / "results.json").read_text())
    assert saved == json.loads(json.dumps(record))
    for key in ("harness", "scaffolding", "settings", "models", "task_ids", "tau2_data_commit"):
        assert key in saved
    assert saved["settings"]["trials"] == 2 and saved["settings"]["concurrency"] == 1
    assert saved["harness"]["prompt_sha256"] == exp.prompt_sha256
    for tid in NO_JUDGE_TRAIN:
        assert saved["per_task"][tid]["trials"] == 2 and saved["per_task"][tid]["rewards"][0] == 1.0

    # pass^k recomputed from per-task counts matches the summary.
    rows = {t: (p["successes"], p["trials"]) for t, p in saved["per_task"].items()}
    assert saved["scores"]["train"]["overall"]["pass^2"] == metrics.score_tasks(rows)["pass^2"]
    assert "test" not in saved["scores"]

    # Every train conversation is saved, as JSON and as a readable transcript.
    tid = NO_JUDGE_TRAIN[0]
    log = json.loads((run_dir / "conversations" / f"{tid}_t0.json").read_text())
    assert log["messages"][0]["content"] == "Hi! How can I help you today?"
    md = (run_dir / "conversations" / f"{tid}_t0.md").read_text()
    assert "**customer:**" in md and "**agent calls**" in md and "Expected actions" in md
    assert (run_dir / "summary.md").read_text().startswith(f"# domain / {record['run_id']}")

    # Runs don't change the harness's version; they can be listed back.
    assert load_harness("domain", root=exp.dir.parent).version == exp.version
    assert [r["run_id"] for r in load_runs(exp.dir.parent)] == [record["run_id"]]


def test_test_split_needs_final_and_never_saves_conversations(exp):
    with pytest.raises(RunRefused, match="--final"):
        run_eval(exp, RunSettings(split="test"), make_spec=scripted_spec(exp), **QUIET)
    with pytest.raises(RunRefused):
        select_tasks("train", [NO_JUDGE_TEST], None, final=False)

    record = run_eval(exp, RunSettings(split="all", final=True, trials=1, tasks=(NO_JUDGE_TRAIN[0], NO_JUDGE_TEST)),
                      make_spec=scripted_spec(exp), **QUIET)
    assert set(record["scores"]) == {"train", "test", "all"}
    saved = {p.name for p in (exp.dir / "runs" / record["run_id"] / "conversations").iterdir()}
    assert saved == {f"{NO_JUDGE_TRAIN[0]}_t0.json", f"{NO_JUDGE_TRAIN[0]}_t0.md"}


def test_transport_errors_are_retried_and_recorded(exp):
    inner = scripted_spec(exp)
    calls = {"n": 0}

    def flaky(tid, trial):
        calls["n"] += 1
        if calls["n"] == 1:
            raise InfraError("connection reset")
        return inner(tid, trial)

    record = run_eval(exp, RunSettings(trials=1, tasks=(NO_JUDGE_TRAIN[0],)), make_spec=flaky, **QUIET)
    assert record["episodes"]["retries"] == 1 and record["episodes"]["completed"] == 1
    assert record["per_task"][NO_JUDGE_TRAIN[0]]["rewards"] == [1.0]


def test_endpoint_down_aborts_and_is_left_out_of_listings(exp):
    def down(tid, trial):
        raise InfraError("503")

    record = run_eval(exp, RunSettings(trials=4, tasks=tuple(NO_JUDGE_TRAIN)), make_spec=down, **QUIET)
    assert record["status"] == "aborted" and record["episodes"]["infra_failed"] == 5
    assert load_runs(exp.dir.parent) == []
    assert len(load_runs(exp.dir.parent, complete_only=False)) == 1


def test_parallel_runs_give_the_same_result(exp):
    record = run_eval(exp, RunSettings(trials=2, tasks=tuple(NO_JUDGE_TRAIN), concurrency=3),
                      make_spec=scripted_spec(exp), **QUIET)
    assert record["settings"]["concurrency"] == 3 and record["episodes"]["completed"] == 4


def test_refuses_dirty_harnesses_and_empty_skills(exp):
    with pytest.raises(RunRefused, match="uncommitted"):
        run_eval(replace(exp, dirty=True), make_spec=scripted_spec(exp), **QUIET)
    skills = replace(load_harness("skills"), dirty=False)
    with pytest.raises(RunRefused, match="no skills"):
        run_eval(skills, make_spec=scripted_spec(skills), **QUIET)


def test_settings_defaults_and_validation():
    s = RunSettings()
    assert (s.agent, s.split, s.trials, s.tasks, s.concurrency, s.max_steps) == ("AGENT", "train", 4, "all", 1, 100)
    with pytest.raises(RunRefused, match="split"):
        RunSettings(split="validation")
    with pytest.raises(RunRefused, match="seed"):
        RunSettings(tasks="half:scoring")


def test_halves_split_train_and_depend_on_seed():
    h = halves(17)
    assert set(h["half:authoring"]) | set(h["half:scoring"]) == set(data.splits()["train"])
    assert not set(h["half:authoring"]) & set(h["half:scoring"])
    assert len(h["half:authoring"]) == 37
    assert halves(18) != h and halves(17) == h


def test_resume_finishes_an_aborted_run_without_redoing_episodes(exp):
    inner = scripted_spec(exp, succeed_on=lambda tid, trial: True)
    calls = []
    budget = {"left": 3}  # credits run out after 3 conversations

    def metered(tid, trial):
        if budget["left"] == 0:
            raise InfraError("402 insufficient balance")
        budget["left"] -= 1
        calls.append((tid, trial))
        return inner(tid, trial)

    settings = RunSettings(trials=4, tasks=tuple(NO_JUDGE_TRAIN))
    first = run_eval(exp, settings, make_spec=metered, **QUIET)
    assert first["status"] == "aborted" and first["episodes"]["completed"] == 3
    assert load_runs(exp.dir.parent) == []

    budget["left"] = 100  # topped up
    calls.clear()
    second = run_eval(exp, RunSettings(trials=1), resume="last", make_spec=metered, **QUIET)
    assert second["run_id"] == first["run_id"] and second["status"] == "complete"
    assert len(calls) == 5  # only the 8 - 3 unfinished episodes ran
    assert second["settings"]["trials"] == 4  # the run's own settings, not the new flags
    assert all(p["trials"] == 4 and p["rewards"] == [1.0] * 4 for p in second["per_task"].values())
    assert second["episodes"]["infra_failed"] == 0 and len(second["resumed_at"]) == 1
    assert [r["run_id"] for r in load_runs(exp.dir.parent)] == [first["run_id"]]
    conv = exp.dir / "runs" / first["run_id"] / "conversations"
    assert len(list(conv.glob("*.json"))) == 8

    with pytest.raises(RunRefused, match="already complete"):
        run_eval(exp, resume=first["run_id"], make_spec=metered, **QUIET)


def test_resume_refuses_a_changed_harness(exp):
    def down(tid, trial):
        raise InfraError("503")

    first = run_eval(exp, RunSettings(trials=4, tasks=tuple(NO_JUDGE_TRAIN)), make_spec=down, **QUIET)
    (exp.dir / "system_prompt.md").write_text("Something else.\n")
    changed = replace(load_harness("domain", root=exp.dir.parent), dirty=False)
    with pytest.raises(RunRefused, match="changed since"):
        run_eval(changed, resume=first["run_id"], make_spec=scripted_spec(changed), **QUIET)
    with pytest.raises(RunRefused, match="no run"):
        run_eval(exp, resume="nope", make_spec=down, **QUIET)
