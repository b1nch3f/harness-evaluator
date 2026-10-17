# Implementation plan

Build in milestone order. Each milestone ends with an acceptance check; do not start the next one until it passes.

The harness is **ADK-native**: the retail environment, the simulated user and the scorer are our own code. τ²-bench (`sierra-research/tau2-bench` at commit `96a3ef92329baf908ffe9dfe188cea34e8701d2a`, v1.0.1) is the reference for their behavior and the source of the data files; its code runs only in tests. ADK facts below were checked against google-adk 2.12.0.

## M1. Project skeleton (done)

- `uv` package `skills_builder`, `src/` layout, Python 3.12 (τ² imports `audioop`, removed in 3.13).
- Pinned: `google-adk==2.12.0`, `litellm==1.82.6`, `wandb==0.28.1`, `python-dotenv`, `pydantic`, `typer`, `pytest` (dev). `tau2` (git, pinned commit) is a **dev** dependency, used only by tests. Notes: litellm is held below 1.82.7 by tau2; newer wandb conflicts with ADK's opentelemetry pin.
- τ²-bench is a git submodule at `third_party/tau2-bench`, pinned to the commit above, for its data files.
- `.env` loaded at startup; CLI entry point `uv run sb ...`.

**Accept:** `uv sync` succeeds; `import google.adk; from google.adk.tools.skill_toolset import SkillToolset` succeeds.

## M2. Data and task labels (done)

- Read from `third_party/tau2-bench/data/tau2/domains/retail/`: `tasks.json` (114), `split_tasks.json` (`train` 74, `test` 40), `policy.md`, `db.json`; and `data/tau2/user_simulator/simulation_guidelines.md`. Do not copy them into this repo.
- `sb labels` writes `data/labels/retail.json`: per task, split, intent labels, width (distinct write intents) and depth (expected actions). Committed.

**Accept:** train label counts: `modify_items` 20, `return` 23, `exchange` 19, `cancel` 13, `modify_order_address` 13, `modify_user_address` 6, `transfer` 2, `modify_payment` 0, `answer` 25.

## M3. Model connection (done)

- `google.adk.models.lite_llm.LiteLlm(model=AGENT_MODEL, api_base=..., api_key=..., temperature=..., extra_body={"reasoning_effort": ...})`. LiteLLM rejects `reasoning_effort` as a top-level argument for generic OpenAI-compatible models; vLLM accepts it in the body.
- The RunPod vLLM endpoint needs `ENABLE_AUTO_TOOL_CHOICE=true` and `TOOL_CALL_PARSER=openai`, or tool calls are silently dropped.
- `sb ping` sends one chat and one tool-call request and prints the parsed tool call.

**Accept:** `sb ping` returns a valid, parsed tool call from gpt-oss-120b.

## M4. ADK-native retail harness (done)

### Environment (`skills_builder/retail/`)

- **Database:** pydantic models for products, variants, users, addresses, payment methods (credit card, PayPal, gift card), orders, order items, payments, fulfillments, matching τ²'s `tau2/domains/retail/data_model.py`. Loaded fresh from `db.json` for every episode.
- **Tools:** the 16 retail tools (`calculate`, `cancel_pending_order`, `exchange_delivered_order_items`, `find_user_id_by_name_zip`, `find_user_id_by_email`, `get_order_details`, `get_product_details`, `get_item_details`, `get_user_details`, `list_all_product_types`, `modify_pending_order_address`, `modify_pending_order_items`, `modify_pending_order_payment`, `modify_user_address`, `return_delivered_order_items`, `transfer_to_human_agents`) as ADK `FunctionTool`s bound to the episode's database. Same names, arguments, descriptions, checks and state changes as τ²'s `tau2/domains/retail/tools.py`, **including its quirks** (for example, `modify_pending_order_items` gives every modified item the price and options of the last new item), because expected actions are replayed with the same logic.
- **Tool results:** a returned record goes back as its JSON; a failed check returns `{"error": "<message>"}` and counts as a tool error.

### Conversation loop

1. Fresh database. The agent session is seeded with the greeting "Hi! How can I help you today?".
2. **Simulated user:** an ADK `LlmAgent` with no tools. System prompt = τ²'s user-simulator layout: `simulation_guidelines.md`, then `<scenario>` with the task's user scenario formatted as τ² does. It sees the agent's messages as user messages.
3. Turn: the user's message goes to the agent's `runner.run_async(...)`; ADK runs tool calls natively against the episode database; the agent's final text goes to the user simulator. Repeat.
4. **End:** user message contains `###STOP###`, `###TRANSFER###` or `###OUT-OF-SCOPE###`; or steps reach `MAX_STEPS` (a text message is 1 step, a tool call with its result 2, as in τ²'s orchestrator); or 10 tool errors; or the agent sends an empty message (agent error).
5. Agent: ADK `LlmAgent` with the experiment's `system_prompt.md` as `instruction` (a callable, so ADK does no `{placeholder}` injection), the 16 retail tools, plus `SkillToolset(skills=load_skills_from_dir(<experiment>/skills))` when the experiment has skills (default lazy discovery: the model calls `list_skills`, then `load_skill`). A `before_model_callback` removes ADK's identity line so the prompt is exactly the file.
6. Fresh `InMemoryRunner` session per episode for the agent and for the user.
7. The episode log records every message, every tool call with arguments and result, every skill-tool call, model-call counts and token usage, termination reason, and the score breakdown.

### Scorer

- **Database check:** fresh database, apply the task's expected actions with the same tool code (skip any that raise), compare with the agent's final database (canonical JSON of the whole database).
- **Assertion check:** for tasks with `nl_assertions`, one judge call with τ²'s judge prompt (`tau2/evaluator/evaluator_nl_assertions.py`), conversation rendered as `role: content` lines; every returned `metExpectation` must be true. Parse failures retry up to 3 times, then count as an infrastructure error (not a model failure).
- Reward = database check × assertion check.

### Experiments

- `experiments/<name>/system_prompt.md` and optional `skills/` (see CLAUDE.md rule 3).
- `a0-bare`: `<instructions>` + τ²'s agent instruction. `domain`: τ²'s layout, `<instructions>` + `<policy>`, with `{policy}` filled from `policy.md` at load time (a test checks it equals τ²'s own agent prompt). `skills` and `domain-skills` reuse those prompts and have empty `skills/` folders.

**Accept (met):** one train task ran end to end with the policy prompt (reward 1.0) and with a fixture skill (ran; reward 0, a model error); the log shows every tool call and its result.

## M5. Eval runner (done)

```
uv run sb eval <experiment> [--agent AGENT] [--model M] [--temperature T] [--reasoning-effort E]
                            [--split train|test|all] [--trials 4] [--tasks all|half:authoring|half:scoring|0,5,12]
                            [--seed S] [--concurrency 1] [--max-steps 100] [--final] [--allow-dirty]
uv run sb eval <experiment> --resume <run_id>|last
```

- Every setting is a flag with a default; an experiment folder holds only its harness, so the same experiment runs with any model or settings.
- One conversation at a time unless `--concurrency` > 1 (then a thread pool, one event loop and database per episode).
- Refuses: test tasks without `--final`; an experiment folder with uncommitted changes (unless `--allow-dirty`); a skills folder with no skills.
- Retries an episode on exceptions (transport errors, unparseable judge replies), up to 3 attempts, and records them; a low reward is never retried. Five consecutive failed episodes abort the run (for example when endpoint credits run out).
- **Resume:** every finished episode is appended to the run's `episodes.jsonl` (task, trial, reward, ending, usage; no conversation text). `--resume <run_id>` (or `last`, the newest unfinished run) reopens the run with its own settings, skips finished episodes, reruns the rest, and rescores the whole run. It refuses if the experiment's prompt or skills, or the model settings, changed since the run started. Ctrl-C'd runs resume the same way.
- Each run is self-contained in `experiments/<name>/runs/<run_id>/` (git-ignored):
  - `results.json`: settings, experiment version and hashes, models (CLAUDE.md rule 8), status, episode counts, token usage, scores per split (pass@k and pass^k for k = 1, 2, 4, 8 with bootstrap CIs; per intent, width and depth), per-task rewards and endings. Written at the start (`status: running`) and completed at the end.
  - `summary.md`: the same, readable.
  - `episodes.jsonl`: progress, one line per finished episode.
  - `conversations/<task>_t<trial>.json` + `.md` for every **train** episode. Test conversations are not written.
- `run_id` = `<UTC time>_<version[:8]>_<agent model>_<split>_n<trials>`.
- `sb results` lists completed runs of every experiment (from each run's `results.json`); `sb experiments` lists experiments and versions.
- Other LLMs: add `<PROFILE>_MODEL`, `<PROFILE>_API_BASE`, `<PROFILE>_API_KEY` to `.env` and pass `--agent PROFILE`. The simulated user and judge stay fixed.

**Accept (met, offline):** with scripted models, a run writes all files; pass^k recomputed from per-task counts matches the summary; test tasks are refused without `--final` and never saved; retries and aborts behave as above; parallel and sequential runs agree.

## M6. Pre-flight checks

1. **Tool parity (test, uses τ² code):** done; all 114 tasks' expected actions give the same results and databases as τ²'s.
2. **Scorer agreement (test, uses τ² code):** done; our database check agrees with τ²'s `EnvironmentEvaluator` on expected, dropped, repeated and changed action variants for all 114 tasks.
3. **Tool-call health:** `sb eval a0-bare --tasks <10 train ids> --trials 1`: zero tool-call parse errors (10 conversations).
4. **Noise floor:** `a0-bare` on train, `n = 4`, twice (2 × 296 conversations); record the difference in pass^1 and pass^4.

**Accept:** 3 and 4 recorded as runs in `experiments/a0-bare/runs/`, with the noise floor written into `docs/01-experiment-protocol.md`.

## M7. Weights & Biases

- One W&B run per eval run; project `agent-harness-builder`; `config` = the run's `config.json`.
- Log pass@k and pass^k for each k, overall and per intent, plus a per-task table.
- Log the experiment's harness as an artifact versioned by its hash. Group runs by experiment; tag with the agent model.
- W&B is the off-machine copy of the runs' `results.json` files.

**Accept:** two runs appear in W&B with comparable pass^k charts.

## M8. Claude skill: `tau-eval`

Create `.claude/skills/tau-eval/SKILL.md` so the owner can say "run experiment X" and have Claude Code:

1. Check `.env` has the profile's keys and run `sb ping`.
2. Check the experiment folder is committed, state the number of conversations, and wait for the go-ahead.
3. Run `sb eval <experiment>`.
4. Show `summary.md`, the comparison with earlier runs from `sb results`, and the most common failure patterns from the saved transcripts.

Also a section for the final test run, which requires the owner to say so explicitly.

**Accept:** invoking the skill runs an eval and reports it.

## M9. Baselines and hand-off

- Run `a0-bare` on train (`n = 8`) and, on the owner's explicit request, on test (`--split test --final`).
- Run `domain` on train (`n = 8`).
- Tell the skill author which runs to read.

## Out of scope here

Model tuning (RFT, DPO, agentic RL), serving fine-tuned adapters, and other τ² domains.
