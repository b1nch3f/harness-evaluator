"""The simulated user's system prompt (tau2's layout). Agent prompts live in experiments/."""

from __future__ import annotations

from harness_builder import data


def user_scenario_text(task: dict) -> str:
    """The user scenario, formatted as tau2's UserScenario.__str__ does."""
    import textwrap

    scenario = task["user_scenario"]
    ins = scenario["instructions"]
    tab = "\t"
    if isinstance(ins, dict):
        lines = [f"Domain: {ins['domain']}", f"Reason for call:\n{textwrap.indent(ins['reason_for_call'], tab)}"]
        if ins.get("known_info") is not None:
            lines.append(f"Known info:\n{textwrap.indent(ins['known_info'], tab)}")
        if ins.get("unknown_info") is not None:
            lines.append(f"Unknown info:\n{textwrap.indent(ins['unknown_info'], tab)}")
        lines.append(f"Task instructions:\n{textwrap.indent(ins['task_instructions'], tab)}")
        ins_text = "\n".join(lines)
    else:
        ins_text = str(ins)
    out = []
    if scenario.get("persona") is not None:
        out += ["Persona:", textwrap.indent(scenario["persona"], tab)]
    out += ["Instructions:", textwrap.indent(ins_text, tab)]
    return "\n".join(out)


def user_prompt(task: dict) -> str:
    """tau2's user-simulator system prompt layout: guidelines, then <scenario>."""
    guidelines = data.user_guidelines().replace("<PERSONA_GUIDELINES>", "")
    return f"{guidelines}\n\n<scenario>\n{user_scenario_text(task)}\n</scenario>"
