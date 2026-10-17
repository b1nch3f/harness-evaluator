# Harness tuning loop

A harness is an experiment folder's `system_prompt.md` plus its `skills/`. Tuning means editing them on train and keeping only edits that help. Skills are written by a separate author (a Claude session); prompt edits follow the same loop.

## Where things are

- Harnesses: `experiments/<name>/` (committed).
- Results: `experiments/<name>/runs/<run_id>/` (git-ignored, on the owner's machine). Each run has `results.json`, `summary.md` and, for train tasks, `conversations/<task>_t<trial>.md`: the full conversation, tool calls and results, skills loaded, the customer's goal, the expected actions and the score breakdown.
- Because results are local, an author without access to the owner's machine gets the failure transcripts from the owner.

## One iteration

1. **Owner:** runs `sb eval <experiment> --tasks half:authoring` and `--tasks half:scoring` with the iteration's `--seed`.
2. **Author:** reads `summary.md` and the failed transcripts from the **authoring half only**, picks one intent to work on (normally the weakest by pass^4) and states a hypothesis.
3. **Author:** edits one skill (or a small related set, or the prompt) and commits with a message naming the target intent and the hypothesis.
4. **Owner:** runs the scoring half for the new version.
5. **Accept or reject** (rule in the protocol). An accepted version is tagged `<experiment>-vN`. A rejected one is reverted; its run stays in the experiment's `runs/`.

## Skill layout

```
experiments/<skills|domain-skills>/skills/
  authenticate-user/SKILL.md
  cancel-pending-order/SKILL.md
  modify-pending-order-items/SKILL.md
  modify-pending-order-address/SKILL.md
  modify-pending-order-payment/SKILL.md
  return-delivered-items/SKILL.md
  exchange-delivered-items/SKILL.md
  modify-user-address/SKILL.md
  answer-account-questions/SKILL.md
  <skill>/references/   detailed rules, loaded on demand
  <skill>/scripts/      calculations only; never touch the retail database
```

Each `SKILL.md` has YAML frontmatter (`name`, `description` saying when to use it **and when not to**), then:

- **Rules:** policy preconditions, required confirmations, when to refuse or transfer.
- **Steps:** the workflow in order.
- **Hints:** advice that corrects the current model's habits, kept in a separate, labelled section so its size can be tracked across models.

Keep each `SKILL.md` under 300 lines; move detail into `references/`.

## The two skill experiments

- **`domain-skills`:** the full policy stays in the system prompt; skills add procedure, ordering, checks and edge cases the policy leaves implicit. Don't copy the policy into skills.
- **`skills`:** no policy in the prompt, so each skill carries the policy rules for its own intent; the agent sees them only after loading the skill. The prompt may gain a few cross-cutting rules (authenticate first, explicit confirmation before any change, one user per conversation, no invented information, when to transfer).

## Rules for the author

1. **No record-specific content.** No user IDs, names, emails, order or item IDs, product option values, or amounts from any task. Harnesses hold rules and procedures only. Every edit is checked for this before commit.
2. **Never read test results.** Test conversations are never saved.
3. **One target per iteration,** so each accepted gain can be attributed.
4. **Record the author** in each commit: skills written by a strong external model matter when comparing with self-generated skills later.
