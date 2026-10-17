"""Experiment folders and their config.yaml."""

import pytest
import yaml

from skills_builder import data
from skills_builder.experiment import list_experiments, load_experiment, parse_config


def test_the_four_experiments_exist():
    assert list_experiments() == ["a0-bare", "domain", "domain-skills", "skills"]


def test_domain_prompt_is_tau2s_agent_prompt():
    from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT

    tau2_prompt = SYSTEM_PROMPT.format(domain_policy=data.policy(), agent_instruction=AGENT_INSTRUCTION)
    assert load_experiment("domain").prompt_text == tau2_prompt.strip()


def test_bare_prompt_is_domain_prompt_without_the_policy_block():
    bare = load_experiment("a0-bare").prompt_text
    domain = load_experiment("domain").prompt_text
    assert domain.startswith(bare)
    assert domain[len(bare):] == f"\n<policy>\n{data.policy()}\n</policy>"
    assert not load_experiment("a0-bare").uses_policy and load_experiment("domain").uses_policy


def test_skill_experiments_share_their_prompts_and_start_empty():
    assert load_experiment("skills").prompt_text == load_experiment("a0-bare").prompt_text
    assert load_experiment("domain-skills").prompt_text == load_experiment("domain").prompt_text
    for name in ("skills", "domain-skills"):
        e = load_experiment(name)
        assert e.skills_dir is not None and e.skill_names == ()
    assert load_experiment("a0-bare").skills_dir is None


def test_committed_configs_run_one_at_a_time_on_train():
    for name in list_experiments():
        c = load_experiment(name).config
        assert c.run.concurrency == 1 and c.run.split == "train" and c.agent.profile == "AGENT"


def test_version_covers_the_harness_not_the_config(tmp_path):
    folder = tmp_path / "x"
    folder.mkdir()
    (folder / "system_prompt.md").write_text("Be helpful.\n")
    v1 = load_experiment("x", root=tmp_path).version
    (folder / "config.yaml").write_text(yaml.safe_dump({"run": {"trials": 8}}))
    e = load_experiment("x", root=tmp_path)
    assert e.version == v1 and e.config.run.trials == 8
    (folder / "system_prompt.md").write_text("Be very helpful.\n")
    assert load_experiment("x", root=tmp_path).version != v1


def test_config_rejects_unknown_keys_and_bad_values():
    with pytest.raises(ValueError, match="unknown keys"):
        parse_config({"run": {"trails": 4}})
    with pytest.raises(ValueError, match="split"):
        parse_config({"run": {"split": "validation"}})
    with pytest.raises(ValueError, match="seed"):
        parse_config({"run": {"tasks": "half:scoring"}})
    assert parse_config({"run": {"tasks": [3, 5]}}).run.tasks == ["3", "5"]
