# GPT-5.4 weighted evaluator

Select `weighted_alpaca_eval_gpt5_4` to evaluate with the pinned
`gpt-5.4-2026-03-05` snapshot. This opt-in configuration uses the existing GPT-4
Turbo comparison prompt and weighted `m`/`M` logprob parser. It does not change
the default evaluator.

Install this checkout with `pip install -e .` and upgrade the OpenAI SDK with
`pip install --upgrade openai` so it accepts `max_completion_tokens` and
`reasoning_effort`. Configure your OpenAI API credentials as described in
[client_configs/README.md](../client_configs/README.md).
For a small initial run with local candidate and reference outputs:

```sh
alpaca_eval evaluate \
  --model_outputs /path/to/model_outputs.json \
  --reference_outputs /path/to/reference_outputs.json \
  --annotators_config weighted_alpaca_eval_gpt5_4 \
  --output_path /path/to/gpt54-evaluation \
  --max_instances 10
```

This command makes paid API requests. The model must be available to your API
project. Use the same reference outputs when comparing evaluators and separate
output directories for their results.

GPT-5.4 supports `logprobs` only with `reasoning_effort="none"`. The configuration
sets `max_completion_tokens=1`; the decoder omits the legacy `max_tokens` field
when this modern limit is provided. Keep reasoning disabled for this evaluator.
The prompt, label probabilities, and parser should be checked with real responses
before a full evaluation. In particular, the parser treats a missing label in
`top_logprobs` as zero probability and returns NaN if neither label is present.

The reported cost uses the existing decoder's approximate `total_tokens *
price_per_token` calculation at $2.50 per million tokens. It is not an invoice
estimate: output tokens, cached inputs, long-context surcharges, and other pricing
adjustments are not accounted for separately.

Results use a different judge from the original AlpacaEval leaderboard and should
be reported with the evaluator name and model snapshot. Human agreement and
position bias have not been validated for this configuration. Offline tests cover
request construction and parsing; they do not establish live API compatibility
or judging quality.

Official references: [GPT-5.4 model and pricing](https://developers.openai.com/api/docs/models/gpt-5.4),
[parameter compatibility](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.4).
