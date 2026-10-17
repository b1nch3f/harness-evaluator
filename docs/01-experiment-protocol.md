# Experiment protocol

## Question

How much do domain knowledge and agent skills improve an LLM agent's reliability on τ²-bench retail, and (in later phases) how do they combine with model tuning (RFT, DPO, agentic RL)?

This repo covers **phase 1: tuning the agent's harness (system prompt and skills) on a fixed model.** The same harnesses can be evaluated with other LLMs.

## Framing

Treat it like a classical data-science project:

| Step | Like | Runs on |
|---|---|---|
| 1. `a0-bare` | dummy classifier | train **and** test, reported separately. Nothing is tuned, so its test score leaks nothing. |
| 2. Candidates: `domain`, `skills`, `domain-skills` | model selection | train |
| 3. Tune the selected candidate (prompt and/or skills) | hyperparameter tuning | train: an authoring half (failures are read) and a scoring half (edits are judged), like train/validation |
| 4. Final | test-set evaluation | test, **once**, compared with `a0-bare` on test |

The domain policy is a harness component like any other: it can be added, removed or edited. Every edit of a harness is a new version of it.

## Data

- τ²-bench **retail** domain only: 114 human-written tasks.
- Official splits from `data/tau2/domains/retail/split_tasks.json`: **train = 74 tasks, test = 40 tasks.** Used unchanged.
- All tasks share one database, so test tasks may involve the same users or orders as train tasks. Harnesses must never contain user, order, item, or amount specifics (see `docs/03-skill-authoring-loop.md`).

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

A task can carry several labels. **Width** = number of distinct write intents in the task; **depth** = number of expected actions (bucketed 0-1, 2-3, 4-5, 6-7, 8+).

## Harnesses

Each is a folder in `harnesses/` with `system_prompt.md` and an optional `skills/`. Each run of it is saved in the same folder, self-contained, in `runs/<run_id>/`: `results.json` (settings, scores, per-task rewards), `summary.md` and `conversations/` (git-ignored). Run settings (model, split, trials) are `sb eval` flags, so one harness can be run with several models.

| Harness | System prompt | Skills | Question |
|---|---|---|---|
| `a0-bare` | τ²'s agent instruction only | none | the floor (dummy baseline) |
| `domain` | instruction + τ²'s retail policy (τ²'s own prompt, unchanged) | none | does domain knowledge help? |
| `skills` | instruction only | yes | do skills alone help? |
| `domain-skills` | instruction + policy | yes | do skills add to the policy? |

`a0-bare` is τ²'s prompt with the `<policy>` block removed; its instruction still says "according to the `<policy>` provided below", so the policy block is the only difference from `domain`. The skills folders start empty; the runner refuses them until a skill exists.

All harnesses evaluated with one model use the same simulated user and judge, so they are comparable.

## Roles in a conversation

| Role | What it does | Model |
|---|---|---|
| Agent | answers the customer, calls the 16 retail tools against a fresh copy of the database | chosen per run (`sb eval --agent PROFILE`, a `.env` profile); gpt-oss-120b by default |
| Simulated user | plays the customer from the task's user scenario, using τ²'s user-simulator guidelines; ends with `###STOP###`, `###TRANSFER###` or `###OUT-OF-SCOPE###` | gpt-oss-120b, fixed in `.env` |
| Judge | checks the task's natural-language assertions against the conversation, using τ²'s judge prompt | gpt-oss-120b, temperature 0, fixed in `.env` |

The conversation opens with the agent's fixed greeting ("Hi! How can I help you today?"), as in τ². It ends when the user sends a stop token, after `max_steps` (100) steps, after 10 tool errors, or on an empty agent message.

## Scoring

Success means reward 1, which requires both checks below (τ²'s rules for retail, reimplemented):

1. **Database check (no LLM, every task).** Replay the task's expected actions on a fresh database (an action that errors is skipped), then compare with the agent's final database. They must be identical.
2. **Assertion check (LLM judge, 40 of 114 tasks).** Tasks with `nl_assertions` need the judge to mark every assertion as met. Tasks without them pass this check.

`communicate_info` is not scored (it is used only for the `answer` label). Because all roles run on our own test bench with gpt-oss-120b as user and judge, scores are **not comparable with published τ² results**. Comparisons between harnesses are valid.

## Metrics

Run each task `n` times (`n = 4` by default; `n = 8` for selection and final runs). For a task with `c` successes out of `n`:

- **pass@k** = 1 − C(n−c, k) / C(n, k): at least one of k attempts succeeds.
- **pass^k** = C(c, k) / C(n, k): all k attempts succeed.

Report the mean over tasks for k = 1, 2, 4 (and 8 when n = 8), overall, per intent, per width and per depth, with a bootstrap 95% confidence interval over tasks. **pass^4 is the primary metric.**

## What is captured

Every train conversation is saved in full: the customer's messages, the agent's replies, every tool call with its arguments and result, every skill-tool call, the score breakdown (database match, each assertion with the judge's reasoning), steps, token usage and termination reason. Each is stored as JSON and as a readable Markdown transcript, so the next harness can be planned from what actually happened.

## Decision rules

- **All decisions use train.** While tuning, train is split at random into an authoring half and a scoring half, reshuffled every iteration with a recorded seed.
- **Accept a harness version** only if pass^4 on the scoring half improves beyond the noise floor and no previously passing scoring-half task regresses.
- **Stop tuning** after two consecutive iterations with no accepted improvement, or after 10 iterations.
- **Noise floor:** run the same harness version twice; the difference sets the minimum meaningful improvement.

## Test-set rules

- Test runs twice in total: `a0-bare` at the start, and the finally selected harness version at the end.
- Test conversations are never saved and never shown to anyone tuning a harness.
- No decision of any kind is based on test results.

## Fixed for the whole experiment (per model)

τ² data commit, the agent model, endpoint, temperature and reasoning effort, maximum steps, and the user-simulator and judge models and settings. Changing any of these starts a new series with a new `a0-bare` baseline.
