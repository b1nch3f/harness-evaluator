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

| Condition | System prompt | Skills |
|---|---|---|
| `A_baseline` | τ² `LLMAgent` prompt (instruction + retail policy) | none |
| `B_skills` | same prompt | `SkillToolset` loaded with `skills/retail/` at a given version |

Both use the same model, endpoint, temperature, reasoning effort, and user simulator.

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

- The test split runs **once per condition at the end**: `A_baseline` and the final `B_skills` version.
- Test trajectories are never committed and never shown to the skill author.
- No decision of any kind is based on test results.

## Fixed for the whole experiment

τ² commit, model and endpoint, temperature, reasoning effort, maximum steps per episode, user-simulator model and its settings. Changing any of these starts a new experiment with a new baseline.
