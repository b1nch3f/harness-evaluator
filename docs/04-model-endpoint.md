# Model endpoint for gpt-oss-120b

Phase 1 uses only the base model, so any hosted endpoint works. Pick one and keep it fixed for the whole experiment: different providers serve the same weights with different tool-call parsing and reasoning settings, which changes scores.

## Hosted, pay per token

Prices per million tokens as listed by [LLM Reference](https://www.llmreference.com/model/gpt-oss-120b/providers) (refreshed 2026-10-13); confirm on the provider's own page before committing.

| Provider | Input | Output |
|---|---|---|
| OpenRouter | $0.037 | $0.17 |
| Novita AI | $0.05 | $0.25 |
| Google Cloud Vertex AI (Model Garden, `gpt-oss-120b-maas`) | $0.09 | $0.36 |
| Fireworks AI, GroqCloud, Together AI | $0.15 | $0.60 |

Notes:

- **OpenRouter** routes to several underlying providers; pin one provider in the request so the serving stack does not change between runs.
- **Vertex AI** needs a GCP project, the model enabled in Model Garden, and quota for the request rate you plan to run.
- Check each provider's rate limits against your planned concurrency.

## Self-hosted (needed later for fine-tuned adapters)

vLLM on a single 80 GB GPU (H100 or H200) serving `openai/gpt-oss-120b`, OpenAI-compatible. See the design discussion for RunPod setup and costs. Not needed in phase 1.

## Configuration

Secrets and endpoints go in `.env` (see `.env.example`); each run picks a model with `sb eval` flags.

- **Agent, by profile.** `--agent PROFILE` (default `AGENT`) makes the runner read `<PROFILE>_MODEL` (a LiteLLM model string, for example `openai/openai/gpt-oss-120b` for the RunPod vLLM route), `<PROFILE>_API_BASE`, `<PROFILE>_API_KEY`, and optionally `<PROFILE>_TEMPERATURE` and `<PROFILE>_REASONING_EFFORT`. `--model`, `--temperature` and `--reasoning-effort` override them. To try another LLM, add another prefix (for example `QWEN_MODEL`, `QWEN_API_BASE`, `QWEN_API_KEY`) and run `sb eval <experiment> --agent QWEN`. Use `--reasoning-effort none` for models that don't take it.
- `USER_MODEL`, `USER_API_BASE`, `USER_API_KEY`, `USER_TEMPERATURE`, `USER_REASONING_EFFORT`: the simulated user. Fixed for every experiment and every agent model, so scores stay comparable.
- `JUDGE_MODEL`, `JUDGE_API_BASE`, `JUDGE_API_KEY`, `JUDGE_REASONING_EFFORT`: the assertion judge (temperature fixed at 0). Defaults to the user-simulator settings.
- The RunPod vLLM worker needs `ENABLE_AUTO_TOOL_CHOICE=true` and `TOOL_CALL_PARSER=openai`. Running workers keep old settings after a change; terminate them so new ones start.
- `--concurrency` should not exceed what the endpoint can serve; with one RunPod worker, keep it at 1.
- Scores are not comparable with published τ² results (they use gpt-4.1 as user and judge, and τ²'s own harness).

## Rough cost for phase 1

About 5,000 conversations (baseline, ~10 skill iterations at n = 4, final evaluations at n = 8). With gpt-oss-120b on a hosted endpoint, the agent costs well under $100; the user simulator is the main cost (about $230–345 with gpt-4.1 at its launch pricing).
