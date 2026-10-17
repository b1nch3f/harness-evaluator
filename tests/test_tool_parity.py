"""Our retail tools must behave exactly like tau2's (tau2 code is used only here)."""

import json

import pytest

from harness_builder import data
from harness_builder.retail.models import RetailDB
from harness_builder.retail.tools import TOOL_NAMES, RetailTools
from harness_builder.retail.runtime import result_text


def _tau2_env():
    from tau2.domains.retail.environment import get_environment

    return get_environment()


def _tau2_call(env, name, args):
    from tau2.data_model.message import ToolCall

    msg = env.get_response(ToolCall(id="x", name=name, arguments=args, requestor="assistant"))
    return msg.content, msg.error


def _ours_call(tools, name, args):
    try:
        return result_text(tools.call(name, dict(args))), False
    except Exception as e:
        return f"Error: {e}", True


def _tau2_db(env) -> dict:
    return json.loads(json.dumps(env.tools.db.model_dump(), default=str))


def test_same_tool_names_and_order():
    assert tuple(t.name for t in _tau2_env().get_tools()) == TOOL_NAMES


def test_same_fresh_database():
    ours = RetailDB.fresh().model_dump(mode="json")
    assert ours == _tau2_db(_tau2_env())


@pytest.mark.parametrize("task_id", [t["id"] for t in data.tasks()])
def test_expected_actions_match_tau2(task_id):
    task = data.tasks_by_id()[task_id]
    env = _tau2_env()
    tools = RetailTools(RetailDB.fresh())
    for action in task["evaluation_criteria"]["actions"]:
        theirs = _tau2_call(env, action["name"], action["arguments"])
        ours = _ours_call(tools, action["name"], action["arguments"])
        assert ours == theirs, f"{action['name']}({action['arguments']})"
    assert tools.db.model_dump(mode="json") == _tau2_db(env)


ERROR_CASES = [
    # (tool, arguments) that fail or hit edge cases
    ("cancel_pending_order", {"order_id": "#W0000000", "reason": "no longer needed"}),
    ("get_user_details", {"user_id": "nobody_123"}),
    ("calculate", {"expression": "2 ** 8"}),
    ("calculate", {"expression": "(10 + 5) / 3"}),
    ("find_user_id_by_email", {"email": "missing@example.com"}),
    ("get_item_details", {"item_id": "0"}),
]


@pytest.mark.parametrize("name,args", ERROR_CASES)
def test_error_and_edge_cases_match_tau2(name, args):
    assert _ours_call(RetailTools(RetailDB.fresh()), name, args) == _tau2_call(_tau2_env(), name, args)


def _first(pred):
    db = RetailDB.fresh()
    return next(o for o in db.orders.values() if pred(o, db))


def test_rule_violations_match_tau2():
    """Calls the policy forbids must fail the same way in both."""
    from harness_builder.retail.models import GiftCard

    delivered = _first(lambda o, db: o.status == "delivered")
    pending = _first(lambda o, db: o.status == "pending")
    db = RetailDB.fresh()
    user = db.users[delivered.user_id]
    other_pm = next((pid for pid, pm in user.payment_methods.items()
                     if pid != delivered.payment_history[0].payment_method_id and not isinstance(pm, GiftCard)), None)
    cases = [
        ("cancel_pending_order", {"order_id": delivered.order_id, "reason": "no longer needed"}),
        ("cancel_pending_order", {"order_id": pending.order_id, "reason": "too expensive"}),
        ("modify_pending_order_address", {"order_id": delivered.order_id, "address1": "1 A St", "address2": "",
                                          "city": "X", "state": "CA", "country": "USA", "zip": "00000"}),
        ("exchange_delivered_order_items", {"order_id": pending.order_id, "item_ids": [pending.items[0].item_id],
                                            "new_item_ids": [pending.items[0].item_id],
                                            "payment_method_id": pending.payment_history[0].payment_method_id}),
        ("modify_pending_order_items", {"order_id": pending.order_id, "item_ids": [pending.items[0].item_id],
                                        "new_item_ids": [pending.items[0].item_id],
                                        "payment_method_id": pending.payment_history[0].payment_method_id}),
        ("modify_pending_order_payment", {"order_id": pending.order_id,
                                          "payment_method_id": pending.payment_history[0].payment_method_id}),
    ]
    if other_pm:
        cases.append(("return_delivered_order_items", {"order_id": delivered.order_id,
                                                       "item_ids": [delivered.items[0].item_id],
                                                       "payment_method_id": other_pm}))
    for name, args in cases:
        ours = _ours_call(RetailTools(RetailDB.fresh()), name, args)
        theirs = _tau2_call(_tau2_env(), name, args)
        assert ours == theirs, name
        assert ours[1] is True, f"{name} should have failed"
