# Instructions for Claude Code in this repo

Build and run the experiment described in `docs/`. Follow `docs/02-implementation-plan.md` milestone by milestone; do not skip an acceptance check.

## Non-negotiable rules

1. **Never run the test split unless the user explicitly asks for the final run.** The eval runner must refuse `--split test` without `--final`. Never write test trajectories into the repo.
2. **Everything that runs in an eval is our own ADK-native code.** The retail environment (database and tools), the simulated user and the scorer live in `src/skills_builder/`. τ²-bench is a reference and a data source only: its task, split, policy, database and user-guideline files are read from the pinned submodule, but no τ² Python code runs during an eval. τ² code may be imported only in tests that check our tools and scorer agree with τ²'s.
3. **Do not edit files under `skills/`.** Skills are written by the skill author (see `docs/03-skill-authoring-loop.md`). If a skill looks broken, report it; do not fix it.
4. **A condition is a prompt file plus an optional skills folder.** System prompts live in `prompts/`, one file per condition, and are used as written (ADK's automatic "You are an agent…" identity line is removed). Skills are added only through ADK's `SkillToolset`, never pasted into a prompt. Changing a prompt file starts a new experiment for that condition.
5. **Pin versions.** τ²-bench data by submodule commit, google-adk and litellm by exact version, in `pyproject.toml` and `uv.lock`.
6. **Secrets stay in `.env`** (git-ignored). Never commit keys or endpoint URLs that contain credentials.
7. **Every result records its configuration:** git SHA, condition, prompt file hash, skills version hash, agent model, endpoint label, user-simulator model, judge model, temperatures, reasoning efforts, max steps, trials, split, and τ² data commit.

## Tooling

- Python 3.12 (τ²-bench, used in tests, requires `>=3.12,<3.14` and imports `audioop`, removed in 3.13), managed with `uv`.
- Run everything with `uv run ...`.
- Tests: `uv run pytest`.
