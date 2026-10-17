# automatic-agent-skills-builder

An experiment to measure how much **agent skills** (SKILL.md files) and, later, **model tuning** improve an open-weights agent on the τ²-bench **retail** domain, measured with pass@k and pass^k.

- **Agent:** Google ADK, model `gpt-oss-120b` served behind an OpenAI-compatible endpoint (via LiteLLM).
- **Environment and scoring:** τ²-bench retail tasks, policy and database, with the environment, simulated user and scorer reimplemented natively in ADK (τ² code runs only in tests that check agreement).
- **Experiments:** one folder each in `experiments/` (system prompt, optional skills): `a0-bare` (dummy baseline, no policy), `domain` (policy in the prompt), `skills`, `domain-skills`.
- **Skills:** Agent Skills format, loaded by ADK's `SkillToolset`.
- **Results:** each run is saved in its experiment's folder, `runs/<run_id>/`: `results.json`, `summary.md` and full conversation transcripts (local, git-ignored); Weights & Biases later.

```
uv run sb experiments              # list experiments and versions
uv run sb eval a0-bare --trials 4  # run one (flags: --agent, --split, --tasks, --concurrency, ...)
uv run sb results                  # every run so far
```

## How the work is split

| Who | Does what |
|---|---|
| Claude Code, run locally by the repo owner | Builds the project from `docs/02-implementation-plan.md`: uv project, ADK harness, eval runner, W&B logging, the `tau-eval` Claude skill |
| Repo owner | Provides model endpoints and API keys locally; runs experiments; decides what to try next |
| Claude (skill author) | Reads train transcripts, writes and revises `experiments/*/skills/`, commits harness versions |

## Read in this order

1. [`docs/01-experiment-protocol.md`](docs/01-experiment-protocol.md): what is measured, splits, conditions, rules that keep the test set clean
2. [`docs/02-implementation-plan.md`](docs/02-implementation-plan.md): what to build, in milestone order, with acceptance checks
3. [`docs/03-skill-authoring-loop.md`](docs/03-skill-authoring-loop.md): how harness versions are written, evaluated and accepted
4. [`docs/04-model-endpoint.md`](docs/04-model-endpoint.md): where to get a gpt-oss-120b endpoint and how to configure it
5. [`CLAUDE.md`](CLAUDE.md): standing instructions for Claude Code working in this repo
