"""Our database check must agree with tau2's EnvironmentEvaluator (tau2 code is used only here)."""

import copy
import random

import pytest

from harness_builder import data
from harness_builder.retail.models import RetailDB
from harness_builder.retail.scorer import JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT, db_check
from harness_builder.retail.tools import WRITE_TOOLS, RetailTools


def _tau2_db_match(task: dict, actions: list[dict]) -> bool:
    """tau2's verdict for a trajectory that made exactly these tool calls."""
    from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
    from tau2.data_model.tasks import Task
    from tau2.domains.retail.environment import get_environment
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    env = get_environment()
    trajectory = [UserMessage(role="user", content="hi")]
    for i, a in enumerate(actions):
        call = ToolCall(id=f"c{i}", name=a["name"], arguments=a["arguments"], requestor="assistant")
        trajectory.append(AssistantMessage(role="assistant", content=None, tool_calls=[call]))
        trajectory.append(env.get_response(call))
    info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment, task=Task.model_validate(task), full_trajectory=trajectory
    )
    return bool(info.db_check.db_match)


def _our_db_match(task: dict, actions: list[dict]) -> bool:
    tools = RetailTools(RetailDB.fresh())
    for a in actions:
        try:
            tools.call(a["name"], dict(a["arguments"]))
        except Exception:
            pass
    return db_check(task, tools.db)[0]


def _variants(task: dict) -> list[tuple[str, list[dict]]]:
    actions = task["evaluation_criteria"]["actions"]
    out = [("expected", actions), ("nothing", [])]
    writes = [i for i, a in enumerate(actions) if a["name"] in WRITE_TOOLS]
    if writes:
        i = writes[-1]
        out.append(("drop last write", actions[:i] + actions[i + 1:]))
        out.append(("repeat last write", actions + [actions[i]]))
        changed = copy.deepcopy(actions)
        args = changed[i]["arguments"]
        if "reason" in args:
            args["reason"] = "ordered by mistake" if args["reason"] == "no longer needed" else "no longer needed"
        elif "item_ids" in args and len(args["item_ids"]) > 1:
            args["item_ids"] = args["item_ids"][:-1]
            if "new_item_ids" in args:
                args["new_item_ids"] = args["new_item_ids"][:-1]
        elif "address1" in args:
            args["address1"] = args["address1"] + " X"
        out.append(("change last write", changed))
    return out


@pytest.mark.parametrize("task_id", [t["id"] for t in data.tasks()])
def test_db_check_agrees_with_tau2(task_id):
    task = data.tasks_by_id()[task_id]
    for label, actions in _variants(task):
        assert _our_db_match(task, actions) == _tau2_db_match(task, actions), label


def test_expected_actions_always_match():
    for task in data.tasks():
        assert _our_db_match(task, task["evaluation_criteria"]["actions"]), task["id"]


def test_judge_prompts_are_tau2s():
    """Compare with the string literals in tau2's judge source."""
    import ast
    import inspect

    from tau2.evaluator import evaluator_nl_assertions as m

    tree = ast.parse(inspect.getsource(m))
    consts = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert JUDGE_SYSTEM_PROMPT in consts
    fstrings = [n for n in ast.walk(tree) if isinstance(n, ast.JoinedStr)]
    rendered = []
    for f in fstrings:
        s = ""
        for v in f.values:
            s += v.value if isinstance(v, ast.Constant) else "{" + ast.unparse(v.value) + "}"
        rendered.append(s)
    assert JUDGE_USER_PROMPT in rendered
