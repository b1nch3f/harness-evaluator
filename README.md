# automatic-agent-skills-builder

An experiment to measure how much **agent skills** (SKILL.md files) and, later, **model tuning** improve an open-weights agent on the τ²-bench **retail** domain, measured with pass@k and pass^k.

- **Agent:** Google ADK, model `gpt-oss-120b` served behind an OpenAI-compatible endpoint (via LiteLLM).
- **Environment and scoring:** τ²-bench retail (simulated user, mock database, official evaluator), driven through τ²'s Gymnasium interface.
- **Skills:** Agent Skills format, loaded by ADK's `SkillToolset`.
- **Tracking:** Weights & Biases, plus result files committed to this repo.

## How the work is split

| Who | Does what |
|---|---|
| Claude Code, run locally by the repo owner | Builds the project from `docs/02-implementation-plan.md`: uv project, harness, τ² bridge, eval runner, W&B logging, the `tau-eval` Claude skill |
| Repo owner | Provides the model endpoint and API keys locally; runs evaluations through the `tau-eval` skill; commits results |
| Claude (cloud session, skill author) | Reads committed train results, writes and revises `skills/retail/*`, pushes skill versions |

## Read in this order

1. [`docs/01-experiment-protocol.md`](docs/01-experiment-protocol.md): what is measured, splits, conditions, rules that keep the test set clean
2. [`docs/02-implementation-plan.md`](docs/02-implementation-plan.md): what to build, in milestone order, with acceptance checks
3. [`docs/03-skill-authoring-loop.md`](docs/03-skill-authoring-loop.md): how skill versions are written, evaluated and accepted
4. [`docs/04-model-endpoint.md`](docs/04-model-endpoint.md): where to get a gpt-oss-120b endpoint and how to configure it
5. [`CLAUDE.md`](CLAUDE.md): standing instructions for Claude Code working in this repo
