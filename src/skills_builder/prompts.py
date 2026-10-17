"""System prompts: one file per condition in prompts/ (CLAUDE.md rule 4)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from skills_builder import data
from skills_builder.config import REPO_ROOT

PROMPTS_DIR = REPO_ROOT / "prompts"

# condition -> (prompt file, uses skills)
CONDITIONS: dict[str, tuple[str, bool]] = {
    "A0_bare": ("A0_bare.md", False),
    "A_baseline": ("A_baseline.md", False),
    "B_skills": ("A_baseline.md", True),
    "C_skills_only": ("C_skills_only.md", True),
}


@dataclass(frozen=True)
class AgentPrompt:
    condition: str
    file: Path
    text: str
    uses_skills: bool

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


def agent_prompt(condition: str) -> AgentPrompt:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; one of {list(CONDITIONS)}")
    name, uses_skills = CONDITIONS[condition]
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist yet (condition {condition})")
    # The file ends with one newline; tau2's prompt template is stripped.
    text = path.read_text().rstrip("\n").replace("{policy}", data.policy())
    return AgentPrompt(condition=condition, file=path, text=text, uses_skills=uses_skills)


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
