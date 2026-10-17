# Skill authoring loop

Skills are written by a separate author (a Claude cloud session) that cannot reach the model endpoint or W&B. The repo is the channel: the author pushes skills, the repo owner runs evals locally with the `tau-eval` Claude skill, and the results come back as committed files.

## One iteration

1. **Owner:** runs the eval for the current skills version on the train split (see the protocol for halves and seeds) and pushes `results/<run_id>/`.
2. **Author:** pulls, reads `summary.json`, `per_task.csv` and the train failures in `failures/` (authoring half only), and decides which intent to work on, normally the weakest by pass^4.
3. **Author:** edits one skill (or a small related set), commits with a message naming the target intent and the hypothesis, and pushes.
4. **Owner:** runs the eval on the scoring half.
5. **Accept or reject** (rule in the protocol). An accepted version is tagged `skills-vN`. A rejected one is reverted, with the result kept for the record.

## Skill layout

```
skills/retail/
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
  <skill>/scripts/      calculations only; never touch τ² state
```

Each `SKILL.md` has YAML frontmatter (`name`, `description` saying when to use it **and when not to**), then:

- **Rules:** policy preconditions, required confirmations, when to refuse or transfer. These come from the τ² retail policy.
- **Steps:** the workflow in order.
- **Hints:** advice that corrects the current model's habits. Kept in a separate, labelled section so its size can be tracked across model versions.

Keep each `SKILL.md` under 300 lines; move detail into `references/`.

## Two skill conditions

- **`B_skills`:** the full policy stays in the system prompt; skills add procedure, ordering, checks and edge cases.
- **`C_skills_only`:** the system prompt holds only the agent instruction and a few cross-cutting rules (authenticate first, explicit confirmation before any change, one user per conversation, no invented information, when to transfer), in `prompts/C_skills_only.md`. Each skill carries the policy rules for its intent, because the agent sees them only after loading the skill. Written by the author once `B_skills` iteration has started; same iteration and acceptance rules.

## Rules for the author

1. **No record-specific content.** No user IDs, names, emails, order or item IDs, product option values, or amounts from any task. Skills hold rules and procedures only. Every edit is checked for this before commit.
2. **Never read test results or test trajectories.**
3. **One target per iteration,** so each accepted gain can be attributed.
4. **Don't copy the policy wholesale (B).** In `B_skills` the policy is already in the system prompt; skills add procedure, ordering, checks and edge cases that the policy leaves implicit. In `C_skills_only`, each skill must include the rules for its own intent.
5. **Record the author** in each commit: these are skills written by a strong external model, which matters when comparing with self-generated skills later.
