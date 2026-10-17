# Instructions for Claude Code in this repo

Build and run the experiment described in `docs/`. Follow `docs/02-implementation-plan.md` milestone by milestone; do not skip an acceptance check.

## Non-negotiable rules

1. **Never run the test split unless the user explicitly asks for the final run.** The eval runner must refuse `--split test` without `--final`. Never write test trajectories into the repo.
2. **Every τ² domain tool call must go through the τ² environment.** If the agent executes a tool outside τ²'s orchestrator, τ²'s evaluator cannot see the database change and the score is wrong.
3. **Do not edit files under `skills/`.** Skills are written by the skill author (see `docs/03-skill-authoring-loop.md`). If a skill looks broken, report it; do not fix it.
4. **Keep the system prompt identical to τ²'s `LLMAgent` prompt** (instruction + `<policy>`), so baseline scores are comparable with published τ² results. Skills are added only through ADK's `SkillToolset`, never pasted into the prompt.
5. **Pin versions.** τ²-bench by commit hash, google-adk and litellm by exact version, in `pyproject.toml` and `uv.lock`.
6. **Secrets stay in `.env`** (git-ignored). Never commit keys or endpoint URLs that contain credentials.
7. **Every result records its configuration:** git SHA, skills version hash, model name, endpoint label, user-simulator model, temperature, reasoning effort, trials, split, and τ² commit.

## Tooling

- Python 3.12 or 3.13 (τ²-bench requires `>=3.12,<3.14`), managed with `uv`.
- Run everything with `uv run ...`.
- Tests: `uv run pytest`.
