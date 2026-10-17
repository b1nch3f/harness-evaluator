"""Harnesses: one folder each under harnesses/ (CLAUDE.md rule 3).

    harnesses/<name>/system_prompt.md   the agent's system prompt, used as written;
                                        `{policy}` is replaced with tau2's retail policy
    harnesses/<name>/skills/            optional; each subfolder with a SKILL.md is a
                                        skill, given to the agent through ADK's SkillToolset
    harnesses/<name>/runs/              its runs (git-ignored); see runner.py

How to run it (model, split, trials, ...) is given per run as `sb eval` flags.
A harness's version is the content hash of its prompt file and skills
folder, plus the last commit that touched the folder.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from test_bench import data
from test_bench.config import REPO_ROOT

HARNESSES_DIR = REPO_ROOT / "harnesses"
PROMPT_FILE = "system_prompt.md"
SKILLS_DIR = "skills"
POLICY_PLACEHOLDER = "{policy}"


@dataclass(frozen=True)
class Harness:
    name: str
    dir: Path
    prompt_text: str
    uses_policy: bool
    skills_dir: Path | None  # None when the harness has no skills folder
    skill_names: tuple[str, ...]
    version: str  # sha256 of the harness: prompt file + skills folder (paths + bytes)
    prompt_sha256: str  # sha256 of the prompt as sent (policy filled in)
    skills_sha256: str | None
    git_commit: str | None  # last commit that touched the folder
    dirty: bool  # harness or folder has uncommitted changes (runs/ is ignored)

    def public(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "git_commit": self.git_commit,
            "dirty": self.dirty,
            "prompt_sha256": self.prompt_sha256,
            "uses_policy": self.uses_policy,
            "skills_sha256": self.skills_sha256,
            "skills": list(self.skill_names),
        }


def _files_hash(root: Path, paths: list[Path] | None = None) -> str:
    """sha256 over (relative path, bytes) of the given files, or every file under root."""
    if paths is None:
        paths = [q for q in root.rglob("*") if q.is_file()]
    h = hashlib.sha256()
    for p in sorted(q for q in paths if "__pycache__" not in q.parts):
        h.update(p.relative_to(root).as_posix().encode() + b"\0")
        h.update(p.read_bytes() + b"\0")
    return h.hexdigest()


def _git(*args: str, cwd: Path) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def list_harnesses(root: Path = HARNESSES_DIR) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / PROMPT_FILE).is_file())


def load_harness(name: str, root: Path = HARNESSES_DIR) -> Harness:
    folder = root / name
    prompt_path = folder / PROMPT_FILE
    if not prompt_path.is_file():
        known = ", ".join(list_harnesses(root)) or "none"
        raise FileNotFoundError(f"no harness {name!r} (needs {prompt_path}); harnesses: {known}")

    raw = prompt_path.read_text().rstrip("\n")  # tau2 strips its prompt template
    uses_policy = POLICY_PLACEHOLDER in raw
    text = raw.replace(POLICY_PLACEHOLDER, data.policy())

    skills_dir = folder / SKILLS_DIR
    skill_names: tuple[str, ...] = ()
    if skills_dir.is_dir():
        skill_names = tuple(sorted(
            p.name for p in skills_dir.iterdir()
            if p.is_dir() and ((p / "SKILL.md").exists() or (p / "skill.md").exists())
        ))
    else:
        skills_dir = None

    harness = [prompt_path, *([q for q in skills_dir.rglob("*") if q.is_file()] if skills_dir else [])]

    rel = folder.relative_to(REPO_ROOT).as_posix() if folder.is_relative_to(REPO_ROOT) else str(folder)
    commit = _git("log", "-1", "--format=%H", "--", rel, cwd=REPO_ROOT) or None
    status = _git("status", "--porcelain", "--", rel, cwd=REPO_ROOT)
    return Harness(
        name=name,
        dir=folder,
        prompt_text=text,
        uses_policy=uses_policy,
        skills_dir=skills_dir,
        skill_names=skill_names,
        version=_files_hash(folder, harness),
        prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
        skills_sha256=_files_hash(skills_dir) if skills_dir else None,
        git_commit=commit,
        dirty=bool(status),
    )
