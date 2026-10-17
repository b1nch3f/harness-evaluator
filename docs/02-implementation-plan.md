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

## M4. ADK-native retail harness

### Environment (`skills_builder/retail/`)

- **Database:** pydantic models for products, variants, users, addresses, payment methods (credit card, PayPal, gift card), orders, order items, payments, fulfillments, matching τ²'s `tau2/domains/retail/data_model.py`. Loaded fresh from `db.json` for every episode.
- **Tools:** the 15 retail tools (`calculate`, `cancel_pending_order`, `exchange_delivered_order_items`, `find_user_id_by_name_zip`, `find_user_id_by_email`, `get_order_details`, `get_product_details`, `get_item_details`, `get_user_details`, `list_all_product_types`, `modify_pending_order_address`, `modify_pending_order_items`, `modify_pending_order_payment`, `modify_user_address`, `return_delivered_order_items`, `transfer_to_human_agents`) as ADK `FunctionTool`s bound to the episode's database. Same names, arguments, descriptions, checks and state changes as τ²'s `tau2/domains/retail/tools.py`, **including its quirks** (for example, `modify_pending_order_items` gives every modified item the price and options of the last new item), because expected actions are replayed with the same logic.
- **Tool results:** a returned record goes back as its JSON; a failed check returns `{"error": "<message>"}` and counts as a tool error.

### Conversation loop

1. Fresh database. The agent session is seeded with the greeting "Hi! How can I help you today?".
2. **Simulated user:** an ADK `LlmAgent` with no tools. System prompt = τ²'s user-simulator layout: `simulation_guidelines.md`, then `<scenario>` with the task's user scenario formatted as τ² does. It sees the agent's messages as user messages.
3. Turn: the user's message goes to the agent's `runner.run_async(...)`; ADK runs tool calls natively against the episode database; the agent's final text goes to the user simulator. Repeat.
4. **End:** user message contains `###STOP###`, `###TRANSFER###` or `###OUT-OF-SCOPE###`; or steps reach `MAX_STEPS` (a text message is 1 step, a tool call with its result 2, as in τ²'s orchestrator); or 10 tool errors; or the agent sends an empty message (agent error).
5. Agent: ADK `LlmAgent` with the condition's prompt file as `instruction` (a callable, so ADK does no `{placeholder}` injection), the 15 retail tools, plus `SkillToolset(skills=load_skills_from_dir(<skills dir>))` for `B_skills` and `C_skills_only` (default lazy discovery: the model calls `list_skills`, then `load_skill`). A `before_model_callback` removes ADK's identity line so the prompt is exactly the file.
6. Fresh `InMemoryRunner` session per episode for the agent and for the user.
7. The episode log records every message, every tool call with arguments and result, every skill-tool call, model-call counts and token usage, termination reason, and the score breakdown.

### Scorer

- **Database check:** fresh database, apply the task's expected actions with the same tool code (skip any that raise), compare with the agent's final database (canonical JSON of the whole database).
- **Assertion check:** for tasks with `nl_assertions`, one judge call with τ²'s judge prompt (`tau2/evaluator/evaluator_nl_assertions.py`), conversation rendered as `role: content` lines; every returned `metExpectation` must be true. Parse failures retry up to 3 times, then count as an infrastructure error (not a model failure).
- Reward = database check × assertion check.

### Prompts

- `prompts/A0_bare.md`: `<instructions>` + τ²'s agent instruction.
- `prompts/A_baseline.md`: τ²'s layout, `<instructions>` + `<policy>`, with `{policy}` filled from `policy.md` at load time. `B_skills` uses the same file.
- `prompts/C_skills_only.md`: written by the skill author when C starts.

**Accept:** one train task runs end to end in `A_baseline` and in `B_skills` (with a test fixture skill, since `skills/retail/` is empty until M9); the reward and its breakdown are reported; the log shows every tool call and its result.

## M5. Pre-flight checks

1. **Tool parity (test, uses τ² code):** for every expected action in all 114 tasks, run it on our database and on τ²'s, starting from fresh copies; results and resulting databases must match. Plus targeted error cases (non-pending order, wrong payment method, insufficient gift-card balance).
2. **Scorer agreement (test, uses τ² code):** for a set of trajectories (the expected actions themselves, expected actions with one action dropped or changed, and real logged episodes), our database check must give the same result as τ²'s `EnvEvaluator`.
3. **Tool-call health:** in 10 `A_baseline` train episodes, zero tool-call parse errors.
4. **Noise floor:** run `A_baseline` on all 74 train tasks twice with `n = 4`; record the difference in pass^1 and pass^4.

**Accept:** 1 and 2 pass in `uv run pytest`; 3 and 4 recorded in `results/preflight/`.

## M6. Eval runner

`uv run sb eval --condition {A0_bare,A_baseline,B_skills,C_skills_only} --split train --trials 8 [--skills-dir skills/retail] [--tasks half:scoring --seed 17] [--concurrency 16] [--final]`

- Runs every selected task `n` times with bounded concurrency (one episode per worker thread, each with its own event loop and database).
- Retries an episode on transport errors (not on model failures), up to 3 times, and records retries.
- `--split test` is refused without `--final`.
- `--tasks half:authoring|half:scoring --seed S` selects the halves defined in the protocol.
- Writes `results/<run_id>/`:
  - `config.json`: everything listed in CLAUDE.md rule 7
  - `summary.json`: pass@k and pass^k (k = 1, 2, 4, 8), overall, per intent, per width, per depth, with bootstrap CIs
  - `per_task.csv`: task id, labels, successes, trials, pass@k, pass^k
  - `failures/` (**train only**): one JSON per failed episode (full message list, tool calls, tool results, skills loaded, score breakdown)
  - `summary.md`: a short human-readable table
- `run_id` format: `<date>_<condition>_<skills-version-or-none>_<split>_n<trials>`.

**Accept:** a train run of `A_baseline` with `n = 2` produces all files; recomputing pass^k from `per_task.csv` matches `summary.json`.

## M7. Weights & Biases

- One W&B run per eval run; project `automatic-agent-skills-builder`.
- `config`: the contents of `config.json`.
- Log pass@k and pass^k for each k, overall and per intent, as summary metrics, plus a per-task table.
- Log the skills folder as a W&B artifact named `skills-retail`, versioned by its content hash.
- Group runs by `condition`; tag with skills version.
- W&B is for charts across iterations. **The committed `results/` files remain the source of truth**, because the skill author cannot reach W&B.

**Accept:** two runs appear in W&B with comparable pass^k charts.

## M8. Claude skill: `tau-eval`

Create `.claude/skills/tau-eval/SKILL.md` (Agent Skills format: YAML frontmatter with `name` and `description`, then instructions). The skill lets the repo owner say "run the eval for the current skills" and have Claude Code:

1. Check `.env` has the required keys and run `sb ping`.
2. Record the current skills version (git SHA of `skills/`).
3. Run the requested eval (default: `B_skills`, train, scoring half for the current iteration seed, `n = 4`).
4. Show `summary.md` and the comparison with the previous accepted version.
5. Commit `results/<run_id>/` (never test trajectories) and push.

Also add a short section for the final test run, which requires the user to say so explicitly.

**Accept:** invoking the skill runs an eval and pushes its results.

## M9. Hand-off to the skill author

- Run `A0_bare` and `A_baseline` on all of train with `n = 8`; commit the results.
- Create an empty `skills/retail/` with a `README.md` pointing to `docs/03-skill-authoring-loop.md`.
- Push, then tell the skill author the run ids.

## Out of scope here

Model tuning (RFT, DPO, agentic RL), serving fine-tuned adapters, and other τ² domains.
