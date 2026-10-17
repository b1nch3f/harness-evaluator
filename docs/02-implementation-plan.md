# Implementation plan

Build in milestone order. Each milestone ends with an acceptance check; do not start the next one until it passes.

Facts below were checked against the source of `sierra-research/tau2-bench` at commit `96a3ef92329baf908ffe9dfe188cea34e8701d2a` (v1.0.1, 2026-10-15) and `google/adk-python` main as of 2026-10-17 (version 2.12.0). If the installed versions differ, re-check the APIs named here before relying on them.

## M1. Project skeleton

- `uv init` a package named `skills_builder` with a `src/` layout. Python `>=3.12,<3.14`.
- Dependencies, pinned exactly in `pyproject.toml`:
  - `google-adk`: latest release that includes `google.adk.tools.skill_toolset.SkillToolset` and `google.adk.skills.load_skills_from_dir`.
  - `litellm` (used by ADK's `LiteLlm` wrapper).
  - `tau2` from git: `tau2 @ git+https://github.com/sierra-research/tau2-bench@96a3ef92329baf908ffe9dfe188cea34e8701d2a`. If installing from git fails, vendor it as a git submodule under `third_party/tau2-bench` and add it as a path dependency.
  - `wandb`, `python-dotenv`, `pydantic`, `typer` (CLI), `pytest` (dev).
- `.env` loaded at startup (see `.env.example`).
- CLI entry point: `uv run sb ...`.

**Accept:** `uv sync` succeeds; `uv run python -c "import tau2, google.adk; from google.adk.tools.skill_toolset import SkillToolset"` succeeds.

## M2. τ² data and task labels

- τ² retail data ships inside the τ² package/repo: `data/tau2/domains/retail/` contains `tasks.json` (114 tasks), `split_tasks.json` (`train`: 74, `test`: 40, `base`: 114), `policy.md`, and `db.json`. Locate it through τ²'s own data-dir resolution; do not copy it into this repo.
- Write `sb labels` to produce `data/labels/retail.json`: for each task, its split, intent labels, width and depth as defined in `docs/01-experiment-protocol.md`. Commit this file.

**Accept:** label counts on train match: `modify_items` 20, `return` 23, `exchange` 19, `cancel` 13, `modify_order_address` 13, `modify_user_address` 6, `transfer` 2, `modify_payment` 0, `answer` 25.

## M3. Model connection

- Build the model with ADK's LiteLLM wrapper: `google.adk.models.lite_llm.LiteLlm(model=AGENT_MODEL, api_base=AGENT_API_BASE, api_key=AGENT_API_KEY, temperature=..., **extra)`. `LiteLlm.__init__(model, **kwargs)` forwards kwargs to `litellm.completion`.
- Reasoning effort for gpt-oss is passed through as a completion argument; confirm the exact parameter name your endpoint accepts (often `reasoning_effort`, sometimes inside `extra_body`).
- `sb ping` sends one chat request and one tool-call request and prints the parsed tool call.

**Accept:** `sb ping` returns a valid, parsed tool call from gpt-oss-120b.

## M4. Harness: ADK agent driven by the τ² gym

### Why the gym

τ²'s evaluator scores the conversation's tool calls and final database state, so every domain tool call must be executed by τ²'s orchestrator. τ²'s Gymnasium interface (`tau2.gym.gym_agent.AgentGymEnv`, registered by `tau2.gym.register_gym_agent()`) lets external code act as the agent while τ² runs the user simulator, executes tools and evaluates.

### Gym facts (from the source)

- `AgentGymEnv(domain=..., task_id=..., max_steps=100, solo_mode=False, user_llm=..., user_llm_args=...)`.
- `reset()` returns `(observation, info)`; `info` includes `tools`, `policy`, and `simulation_run` when available.
- `step(action: str)` returns `(observation, reward, terminated, truncated, info)`. `action` is either:
  - a JSON tool call: `{"name": "get_order_details", "arguments": {"order_id": "#W123"}}` (functional syntax also parses), executed against the environment; or
  - plain text, sent to the simulated user.
- The observation is the **whole conversation as one string**, one `role: content` line per message. Tool calls are rendered in functional form.
- The reward is τ²'s standard evaluation of the finished conversation; it is 0.0 until the simulation ends.
- The gym agent adds a `done` tool; the conversation also ends when the user simulator stops.

### Bridge design

1. For each episode, create the gym env for one task and `reset()`.
2. Build an ADK `LlmAgent`:
   - `instruction` = τ²'s prompt, reproduced exactly: `tau2.agent.llm_agent.SYSTEM_PROMPT.format(agent_instruction=AGENT_INSTRUCTION, domain_policy=<retail policy>)`.
   - `model` = the LiteLlm from M3.
   - `tools` = one proxy tool per τ² retail tool (from `info["tools"]`), plus `SkillToolset` in condition `B_skills`.
3. **Proxy tools:** subclass `google.adk.tools.base_tool.BaseTool`. `_get_declaration()` returns a `google.genai.types.FunctionDeclaration` built from τ²'s own tool name, description and JSON schema (do not regenerate schemas from Python signatures). `run_async(args, tool_context)` calls `env.step(json.dumps({"name": ..., "arguments": args}))` and returns the tool result from the new observation. Prefer reading the newest tool message as a structured object if the gym exposes the message list; otherwise parse the last observation line.
4. **One user turn = one ADK run:** send the user's latest message to `runner.run_async(...)`, let ADK execute proxy tool calls (each one an `env.step`) and skill tools (local), take ADK's final text reply, and send it with `env.step(text)`. The returned observation holds the user's next message. Repeat until `terminated`.
5. ADK keeps its own session history, so pass ADK **only the new user message** each turn, never the gym's full observation string.
6. Skill tools (`list_skills`, `load_skill`, `load_skill_resource`, etc.) never go to `env.step`. Log every skill-tool call per episode; it feeds the routing diagnostics.
7. Use `google.adk.runners.InMemoryRunner` with a fresh session per episode.
8. If ADK returns both text and a tool call in one turn, follow τ²'s rule (one or the other per turn): execute the tool call and drop the text.

### Skills in condition `B_skills`

```python
from google.adk.skills import load_skills_from_dir
from google.adk.tools.skill_toolset import SkillToolset
toolset = SkillToolset(skills=load_skills_from_dir("skills/retail"))
```

`load_skills_from_dir` loads every subfolder that contains `SKILL.md`. Keep the default lazy discovery (the model calls `list_skills`). Scripts under a skill's `scripts/` may only compute values; they must not touch τ² state.

**Accept:** one train task runs end to end with each condition; the τ² reward is reported; the episode log shows every domain tool call executed through `env.step`.

## M5. Pre-flight checks

1. **Adapter parity:** run 10 train tasks, 1 trial, with the ADK harness in `A_baseline`, and the same 10 with τ²'s own CLI (`tau2 run --domain retail --agent-llm <model> --agent-llm-args '{"api_base": ..., "api_key": ...}' --user-llm <sim> --task-split-name train --task-ids ...`). Scores should match within noise. A large gap means the bridge is wrong.
2. **Tool-call health:** in those runs, zero tool-call parse errors.
3. **Noise floor:** run `A_baseline` on all 74 train tasks twice with `n = 4`; record the difference in pass^1 and pass^4.

**Accept:** all three recorded in `results/preflight/`.

## M6. Eval runner

`uv run sb eval --condition {A_baseline,B_skills} --split train --trials 8 [--skills-dir skills/retail] [--tasks half:scoring --seed 17] [--concurrency 16] [--final]`

- Runs every selected task `n` times with bounded concurrency (threads or processes; one gym env per episode).
- Retries an episode on transport errors (not on model failures), up to 3 times, and records retries.
- `--split test` is refused without `--final`.
- `--tasks half:authoring|half:scoring --seed S` selects the halves defined in the protocol.
- Writes `results/<run_id>/`:
  - `config.json`: everything listed in CLAUDE.md rule 7
  - `summary.json`: pass@k and pass^k (k = 1, 2, 4, 8), overall, per intent, per width, per depth, with bootstrap CIs
  - `per_task.csv`: task id, labels, successes, trials, pass@k, pass^k
  - `failures/` (**train only**): one JSON per failed episode (full message list, tool calls, tool results, skills loaded, reward breakdown)
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

- Run `A_baseline` on all of train with `n = 8`; commit the results.
- Create an empty `skills/retail/` with a `README.md` pointing to `docs/03-skill-authoring-loop.md`.
- Push, then tell the skill author the run id.

## Out of scope here

Model tuning (RFT, DPO, agentic RL), serving fine-tuned adapters, and other τ² domains.
