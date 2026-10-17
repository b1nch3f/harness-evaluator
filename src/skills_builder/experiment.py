"""Experiments: one folder each under experiments/ (CLAUDE.md rule 4).

    experiments/<name>/system_prompt.md   the agent's system prompt, used as written;
                                          `{policy}` is replaced with tau2's retail policy
    experiments/<name>/skills/            optional; each subfolder with a SKILL.md is a
                                          skill, given to the agent through ADK's SkillToolset
    experiments/<name>/config.yaml        how to run it: agent model settings, split,
                                          trials, tasks, concurrency, max steps

An experiment's version is the content hash of its harness (the prompt file
and the skills folder; not config.yaml), plus the last commit that touched
the folder.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from skills_builder import data
from skills_builder.config import REPO_ROOT

EXPERIMENTS_DIR = REPO_ROOT / "experiments"
PROMPT_FILE = "system_prompt.md"
SKILLS_DIR = "skills"
CONFIG_FILE = "config.yaml"
POLICY_PLACEHOLDER = "{policy}"
SPLITS = ("train", "test", "all")


@dataclass(frozen=True)
class AgentConfig:
    profile: str = "AGENT"  # .env prefix: <PROFILE>_MODEL, <PROFILE>_API_BASE, <PROFILE>_API_KEY
    model: str | None = None  # overrides <PROFILE>_MODEL
    temperature: float | None = None  # overrides <PROFILE>_TEMPERATURE
    reasoning_effort: str | None = None  # low | medium | high | none; overrides <PROFILE>_REASONING_EFFORT


@dataclass(frozen=True)
class RunConfig:
    split: str = "train"  # train | test | all; test and all also need --final
    trials: int = 4
    tasks: str | list[str] = "all"  # all | half:authoring | half:scoring | list of task ids
    seed: int | None = None  # required for half:*
    concurrency: int = 1  # conversations at a time
    max_steps: int = 100


@dataclass(frozen=True)
class ExperimentConfig:
    agent: AgentConfig = field(default_factory=AgentConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_config(raw: dict | None) -> ExperimentConfig:
    raw = raw or {}
    unknown = set(raw) - {"agent", "run"}
    if unknown:
        raise ValueError(f"config.yaml: unknown sections {sorted(unknown)}")

    def build(cls, section: str):
        values = raw.get(section) or {}
        allowed = set(cls.__dataclass_fields__)
        bad = set(values) - allowed
        if bad:
            raise ValueError(f"config.yaml [{section}]: unknown keys {sorted(bad)}; allowed {sorted(allowed)}")
        return cls(**values)

    agent, run = build(AgentConfig, "agent"), build(RunConfig, "run")
    if run.split not in SPLITS:
        raise ValueError(f"config.yaml: run.split must be one of {SPLITS}")
    if run.trials < 1 or run.concurrency < 1 or run.max_steps < 1:
        raise ValueError("config.yaml: trials, concurrency and max_steps must be >= 1")
    if isinstance(run.tasks, list):
        run = RunConfig(**{**asdict(run), "tasks": [str(t) for t in run.tasks]})
    elif run.tasks not in ("all", "half:authoring", "half:scoring"):
        raise ValueError("config.yaml: run.tasks must be all, half:authoring, half:scoring or a list of ids")
    if isinstance(run.tasks, str) and run.tasks.startswith("half:") and run.seed is None:
        raise ValueError("config.yaml: run.seed is required for half:authoring / half:scoring")
    return ExperimentConfig(agent=agent, run=run)


@dataclass(frozen=True)
class Experiment:
    name: str
    dir: Path
    prompt_text: str
    uses_policy: bool
    skills_dir: Path | None  # None when the experiment has no skills folder
    skill_names: tuple[str, ...]
    version: str  # sha256 of the harness: prompt file + skills folder (paths + bytes)
    prompt_sha256: str  # sha256 of the prompt as sent (policy filled in)
    skills_sha256: str | None
    git_commit: str | None  # last commit that touched the folder
    dirty: bool  # folder has uncommitted changes
    config: ExperimentConfig = field(default_factory=ExperimentConfig)

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
            "config": self.config.to_dict(),
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


def list_experiments(root: Path = EXPERIMENTS_DIR) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / PROMPT_FILE).is_file())


def load_experiment(name: str, root: Path = EXPERIMENTS_DIR) -> Experiment:
    folder = root / name
    prompt_path = folder / PROMPT_FILE
    if not prompt_path.is_file():
        known = ", ".join(list_experiments(root)) or "none"
        raise FileNotFoundError(f"no experiment {name!r} (needs {prompt_path}); experiments: {known}")

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

    import yaml

    config_path = folder / CONFIG_FILE
    config = parse_config(yaml.safe_load(config_path.read_text()) if config_path.is_file() else None)
    harness = [prompt_path, *([q for q in skills_dir.rglob("*") if q.is_file()] if skills_dir else [])]

    rel = folder.relative_to(REPO_ROOT).as_posix() if folder.is_relative_to(REPO_ROOT) else str(folder)
    commit = _git("log", "-1", "--format=%H", "--", rel, cwd=REPO_ROOT) or None
    status = _git("status", "--porcelain", "--", rel, cwd=REPO_ROOT)
    return Experiment(
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
        config=config,
    )
