"""Readable Markdown transcript of one episode log, for reading failures."""

from __future__ import annotations

import json


def _block(text: str) -> str:
    return "\n".join("> " + line if line else ">" for line in str(text).splitlines()) or ">"


def render(log: dict, task: dict | None = None) -> str:
    score = log.get("score") or {}
    out = [
        f"# Task {log['task_id']}, trial {log['trial']}: reward {log['reward']}",
        "",
        f"- experiment: `{log['experiment']}`",
        f"- ended: `{log['termination_reason']}` after {log['steps']} steps, {log['seconds']}s",
        f"- database match: {score.get('db_match')}; assertion check: {score.get('nl_reward')}",
        f"- tool calls: {len(log['tool_calls'])} ({log['tool_errors']} errors); "
        f"skill-tool calls: {len(log['skill_tool_calls'])}",
    ]
    if log.get("error"):
        out.append(f"- error: {log['error']}")
    for a in score.get("nl_assertions") or []:
        out.append(f"- assertion {'met' if a['met'] else 'NOT met'}: {a['nl_assertion']} ({a['justification']})")
    if task is not None:
        ins = task["user_scenario"]["instructions"]
        reason = ins.get("reason_for_call") if isinstance(ins, dict) else ins
        out += ["", "## Customer's goal", "", _block(reason)]
        actions = task["evaluation_criteria"].get("actions") or []
        out += ["", "## Expected actions", ""]
        out += [f"{i}. `{a['name']}({json.dumps(a['arguments'])})`" for i, a in enumerate(actions, 1)] or ["(none)"]
    out += ["", "## Conversation", ""]

    skills = {}
    for c in log.get("skill_tool_calls") or []:
        skills.setdefault(c.get("at_message", 0), []).append(c)

    def skill_lines(i):
        return [f"*skill tool* `{c['name']}({json.dumps(c['arguments'])})`\n" for c in skills.get(i, [])]

    for i, m in enumerate(log["messages"]):
        out += skill_lines(i)
        if m["role"] == "tool":
            mark = " (error)" if m.get("error") else ""
            out += [f"**tool result{mark}:**", "```", str(m["content"]), "```", ""]
        elif m.get("tool_calls"):
            for c in m["tool_calls"]:
                out += [f"**agent calls** `{c['name']}({json.dumps(c['arguments'])})`", ""]
        else:
            who = "agent" if m["role"] == "assistant" else "customer"
            out += [f"**{who}:**", "", _block(m["content"]), ""]
    out += skill_lines(len(log["messages"]))
    return "\n".join(out).rstrip() + "\n"
