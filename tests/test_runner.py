"""The eval runner, offline: scripted models, temporary results.json and runs/."""

import json
from dataclasses import replace

import pytest
from scripted import Scripted, call, text

from skills_builder import data, metrics
from skills_builder.experiment import RunConfig, load_experiment
from skills_builder.retail.episode import EpisodeSpec, InfraError
from skills_builder.runner import EvalSettings, RunRefused, halves, load_results, run_eval, select_tasks

NO_JUDGE_TRAIN = [t["id"] for t in data.tasks()
                  if t["id"] in data.splits()["train"] and not t["evaluation_criteria"].get("nl_assertions")][:2]
NO_JUDGE_TEST = next(t["id"] for t in data.tasks()
                     if t["id"] in data.splits()["test"] and not t["evaluation_criteria"].get("nl_assertions"))


def scripted_spec(exp, succeed_on=lambda tid, trial: trial == 0):
    """Trial 0 replays the expected actions (reward 1); later trials do nothing (reward 0 or 1)."""
    def make(tid, trial):
        task = data.tasks_by_id()[tid]
        actions = task["evaluation_criteria"]["actions"] if succeed_on(tid, trial) else []
        agent = Scripted([[call(a["name"], **a["arguments"])] for a in actions] + [[text("Done.")]])
        user = Scripted([[text("Hi, I need help.")], [text("Thanks. ###STOP###")]])
        return EpisodeSpec(task_id=tid, experiment=exp, trial=trial,
                           agent_model=agent, user_model=user, judge_model=Scripted([]))
    return make


@pytest.fixture
def paths(tmp_path):
    return {"results_path": tmp_path / "results.json", "runs_dir": tmp_path / "runs"}


def _domain(**run):
    exp = load_experiment("domain")
    return replace(exp, dirty=False, config=replace(exp.config, run=replace(exp.config.run, **run)))


def test_run_writes_scores_conversations_and_results_json(paths):
    exp = _domain()
    entry = run_eval(exp, EvalSettings(trials=2, tasks=NO_JUDGE_TRAIN), make_spec=scripted_spec(exp),
                     echo=lambda _: None, **paths)
    assert entry["status"] == "complete" and entry["episodes"]["completed"] == 4

    # Trial 0 succeeds; trial 1 does nothing, which still matches when the task expects no writes.
    for tid in NO_JUDGE_TRAIN:
        p = entry["per_task"][tid]
        assert p["trials"] == 2 and p["rewards"][0] == 1.0

    # pass^k recomputed from per-task counts matches the summary.
    rows = {t: (p["successes"], p["trials"]) for t, p in entry["per_task"].items()}
    assert entry["scores"]["train"]["overall"]["pass^2"] == metrics.score_tasks(rows)["pass^2"]
    assert "test" not in entry["scores"]

    # results.json gets exactly this run.
    runs = load_results(paths["results_path"])
    assert len(runs) == 1 and runs[0]["run_id"] == entry["run_id"]
    cfg = runs[0]
    for key in ("experiment", "scaffolding", "models", "split", "trials", "max_steps", "tau2_data_commit"):
        assert key in cfg
    assert cfg["experiment"]["prompt_sha256"] == exp.prompt_sha256

    # Every train conversation is saved, as JSON and as a readable transcript.
    run_dir = paths["runs_dir"] / entry["run_id"]
    tid = NO_JUDGE_TRAIN[0]
    log = json.loads((run_dir / "conversations" / f"{tid}_t0.json").read_text())
    assert log["messages"][0]["content"] == "Hi! How can I help you today?"
    md = (run_dir / "conversations" / f"{tid}_t0.md").read_text()
    assert "**customer:**" in md and "**agent calls**" in md and "Expected actions" in md
    assert (run_dir / "summary.md").read_text().startswith(f"# {entry['run_id']}")


def test_test_split_needs_final_and_never_saves_conversations(paths):
    exp = _domain()
    with pytest.raises(RunRefused, match="--final"):
        run_eval(exp, EvalSettings(split="test"), make_spec=scripted_spec(exp), echo=lambda _: None, **paths)
    with pytest.raises(RunRefused):
        select_tasks("train", [NO_JUDGE_TEST], None, final=False)

    entry = run_eval(exp, EvalSettings(split="all", final=True, trials=1, tasks=[NO_JUDGE_TRAIN[0], NO_JUDGE_TEST]),
                     make_spec=scripted_spec(exp), echo=lambda _: None, **paths)
    assert set(entry["scores"]) == {"train", "test", "all"}
    saved = {p.name for p in (paths["runs_dir"] / entry["run_id"] / "conversations").iterdir()}
    assert saved == {f"{NO_JUDGE_TRAIN[0]}_t0.json", f"{NO_JUDGE_TRAIN[0]}_t0.md"}


def test_transport_errors_are_retried_and_recorded(paths):
    exp = _domain()
    inner = scripted_spec(exp)
    calls = {"n": 0}

    def flaky(tid, trial):
        calls["n"] += 1
        if calls["n"] == 1:
            raise InfraError("connection reset")
        return inner(tid, trial)

    entry = run_eval(exp, EvalSettings(trials=1, tasks=[NO_JUDGE_TRAIN[0]]), make_spec=flaky,
                     echo=lambda _: None, **paths)
    assert entry["episodes"]["retries"] == 1 and entry["episodes"]["completed"] == 1
    assert entry["per_task"][NO_JUDGE_TRAIN[0]]["rewards"] == [1.0]


def test_endpoint_down_aborts_and_skips_results_json(paths):
    exp = _domain()

    def down(tid, trial):
        raise InfraError("503")

    entry = run_eval(exp, EvalSettings(trials=4, tasks=NO_JUDGE_TRAIN), make_spec=down, echo=lambda _: None, **paths)
    assert entry["status"] == "aborted" and entry["episodes"]["infra_failed"] == 5
    assert load_results(paths["results_path"]) == []


def test_parallel_runs_give_the_same_result(paths):
    exp = _domain(concurrency=3)
    entry = run_eval(exp, EvalSettings(trials=2, tasks=NO_JUDGE_TRAIN), make_spec=scripted_spec(exp),
                     echo=lambda _: None, **paths)
    assert entry["concurrency"] == 3 and entry["episodes"]["completed"] == 4


def test_refuses_dirty_experiments_and_empty_skills(paths):
    exp = replace(load_experiment("domain"), dirty=True)
    with pytest.raises(RunRefused, match="uncommitted"):
        run_eval(exp, make_spec=scripted_spec(exp), echo=lambda _: None, **paths)
    skills = replace(load_experiment("skills"), dirty=False)
    with pytest.raises(RunRefused, match="no skills"):
        run_eval(skills, make_spec=scripted_spec(skills), echo=lambda _: None, **paths)


def test_halves_split_train_and_depend_on_seed():
    h = halves(17)
    train = set(data.splits()["train"])
    assert set(h["half:authoring"]) | set(h["half:scoring"]) == train
    assert not set(h["half:authoring"]) & set(h["half:scoring"])
    assert len(h["half:authoring"]) == 37
    assert halves(18) != h and halves(17) == h
    assert RunConfig().concurrency == 1
