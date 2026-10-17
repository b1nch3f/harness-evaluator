# Experiment protocol

## Question

How much do agent skills improve an open-weights agent's reliability on τ²-bench retail, and (in later phases) how do skills and model tuning (RFT, DPO, agentic RL) combine?

This repo covers **phase 1: skills on a fixed base model.** Model tuning happens later and elsewhere; this repo's results become its baseline.

## Data

- τ²-bench **retail** domain only: 114 human-written tasks.
- Official splits from `data/tau2/domains/retail/split_tasks.json` in the τ²-bench repo: **train = 74 tasks, test = 40 tasks.** Use them unchanged.
- All tasks share one database, so test tasks may involve the same users or orders as train tasks. Skills must therefore never contain user, order, item, or amount specifics (see `docs/03-skill-authoring-loop.md`).

### Retail intents (labels for per-intent reporting)

Derived from each task's expected write actions (`evaluation_criteria.actions`) and required facts (`communicate_info`):

| Label | Source |
|---|---|
| `cancel` | `cancel_pending_order` |
| `modify_items` | `modify_pending_order_items` |
| `modify_order_address` | `modify_pending_order_address` |
| `modify_payment` | `modify_pending_order_payment` (appears in 1 test task and 0 train tasks) |
| `return` | `return_delivered_order_items` |
| `exchange` | `exchange_delivered_order_items` |
| `modify_user_address` | `modify_user_address` |
| `answer` | task has non-empty `communicate_info` |
| `transfer` | `transfer_to_human_agents` |

A task can carry several labels. **Width** = number of distinct write intents in the task; **depth** = number of expected actions.

## Conditions in phase 1

A condition is a system prompt file in `prompts/` plus an optional skills folder.

| Condition | System prompt | Skills | Question it answers |
|---|---|---|---|
| `A0_bare` | agent instruction only, no policy | none | floor: the model with no domain knowledge |
| `A_baseline` | agent instruction + full retail policy | none | `A_baseline` vs `A0_bare`: does domain knowledge help? |
| `B_skills` | same as `A_baseline` | `skills/retail/` at a given version | `B_skills` vs `A_baseline`: do skills add to the policy? |
| `C_skills_only` | agent instruction + a few cross-cutting rules | skills that carry the per-intent rules | `C_skills_only` vs `B_skills`: can skills replace the policy in the prompt? |

The agent instruction is τ²'s `LLMAgent` instruction; `A_baseline` uses τ²'s prompt layout (`<instructions>` + `<policy>`). `A0_bare` is the same prompt with the `<policy>` block removed. The `C_skills_only` prompt and its skills are written by the skill author when that condition starts.

All conditions use the same model, endpoint, temperature, reasoning effort, user simulator and judge.

## Roles in a conversation

| Role | What it does | Model |
|---|---|---|
| Agent | answers the customer, calls the 16 retail tools against a fresh copy of the database | gpt-oss-120b (ADK `LlmAgent`) |
| Simulated user | plays the customer from the task's user scenario, using τ²'s user-simulator guidelines; ends with `###STOP###`, `###TRANSFER###` or `###OUT-OF-SCOPE###` | gpt-oss-120b (ADK `LlmAgent`, no tools) |
| Judge | checks the task's natural-language assertions against the conversation, using τ²'s judge prompt | gpt-oss-120b, temperature 0 |

The conversation opens with the agent's fixed greeting ("Hi! How can I help you today?"), as in τ². It ends when the user sends a stop token, after `max_steps` (100) steps, or after 10 tool errors.

## Scoring

Success means reward 1, which requires both checks below (τ²'s rules for retail, reimplemented):

1. **Database check (no LLM, every task).** Replay the task's expected actions on a fresh database (an action that errors is skipped), then compare it with the agent's final database. They must be identical.
2. **Assertion check (LLM judge, 40 of 114 tasks).** Tasks with `nl_assertions` need the judge to mark every assertion as met. Tasks without them pass this check.

`communicate_info` is not scored (it is used only for the `answer` label).

Because the agent, the user simulator and the judge are all gpt-oss-120b on our own harness, scores are **not comparable with published τ² results**. Comparisons between conditions are valid.
## Metrics

Run each task `n` times (`n = 8` for scored runs, `n = 4` allowed during skill iteration). For a task with `c` successes out of `n`:

- **pass@k** = 1 − C(n−c, k) / C(n, k): at least one of k attempts succeeds.
- **pass^k** = C(c, k) / C(n, k): all k attempts succeed.

Report the mean over tasks for k = 1, 2, 4 (and 8 when n = 8), overall, per intent label, per width, and per depth bucket, with a bootstrap 95% confidence interval over tasks. **pass^4 is the primary metric.** Success = τ²'s reward equals 1.

## Decision rules

- **All decisions use the train split.** In each skill iteration, train is split at random into an *authoring half* (failures the author reads) and a *scoring half* (where the edit is judged). Reshuffle every iteration with a recorded seed.
- **Accept a skill version** only if pass^4 on the scoring half improves beyond noise and no previously passing scoring-half task regresses.
- **Stop iterating** when two consecutive iterations give no accepted improvement, or after 10 iterations.
- **Noise floor:** before the first iteration, run the same configuration twice; the difference sets the minimum meaningful improvement.

## Test-set rules

- The test split runs **once per condition at the end**: `A0_bare`, `A_baseline`, and the final `B_skills` and `C_skills_only` versions.
- Test trajectories are never committed and never shown to the skill author.
- No decision of any kind is based on test results.

## Fixed for the whole experiment

τ² data commit, model and endpoint, temperature, reasoning effort, maximum steps per episode, user-simulator and judge models and their settings, and each condition's prompt file. Changing any of these starts a new experiment with a new baseline.
