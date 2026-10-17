# Instructions for Claude Code in this repo

Build and run the experiment described in `docs/`. Follow `docs/02-implementation-plan.md` milestone by milestone; do not skip an acceptance check.

## Non-negotiable rules

1. **Test tasks run only when the owner explicitly asks.** The runner refuses `split: test` or `all` without `--final`. Test runs happen twice in the whole project: once for `a0-bare` (the dummy baseline) and once for the finally selected experiment. Test conversations are never saved, and no edit is ever based on a test result.
2. **Everything that runs in an eval is our own ADK-native code.** The retail environment (database and tools), the simulated user and the scorer live in `src/skills_builder/`. τ²-bench is a reference and a data source only: its task, split, policy, database and user-guideline files are read from the pinned submodule, but no τ² Python code runs during an eval. τ² code may be imported only in tests that check our tools, scorer and prompts agree with τ²'s.
3. **An experiment is a folder `experiments/<name>/`:** `system_prompt.md` (used as written; `{policy}` is replaced with τ²'s retail policy; ADK's automatic "You are an agent…" identity line is removed), an optional `skills/` folder (given to the agent only through ADK's `SkillToolset`, never pasted into a prompt), and its git-ignored `runs/`. How to run it (model, split, trials, concurrency) is given per run as `sb eval` flags. The experiment's version is the hash of its prompt and skills.
4. **Do not edit an experiment's harness (`system_prompt.md`, `skills/`) unless the owner asks.** Skills are written by the skill author (`docs/03-skill-authoring-loop.md`); if one looks broken, report it. Commit every harness change before running it; the runner refuses uncommitted changes.
5. **Paid runs need the owner's go-ahead.** Before any run that calls the model, state the number of conversations and wait. Run one conversation at a time (`--concurrency 1`, the default) unless the owner asks otherwise.
6. **Pin versions.** τ²-bench data by submodule commit, google-adk and litellm by exact version, in `pyproject.toml` and `uv.lock`.
7. **Secrets stay in `.env`** (git-ignored). Never print or commit keys or endpoint URLs that contain credentials. `--agent` names a `.env` profile, never a key.
8. **Every result records its configuration:** scaffolding git SHA, experiment name, version, commit, prompt hash, skills hash, agent model, endpoint label, user-simulator model, judge model, temperatures, reasoning efforts, max steps, trials, split, task selection and τ² data commit. Each run is self-contained in `experiments/<name>/runs/<run_id>/` (`results.json`, `summary.md`, `conversations/`), git-ignored and local.

## Tooling

- Python 3.12 (τ²-bench, used in tests, requires `>=3.12,<3.14` and imports `audioop`, removed in 3.13), managed with `uv`.
- Run everything with `uv run ...`: `sb experiments`, `sb eval <experiment>`, `sb results`, `sb episode <experiment> --task-id N`.
- Tests: `uv run pytest`.
