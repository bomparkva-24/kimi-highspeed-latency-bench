# Kimi Highspeed Latency Bench

Is Kimi K3.1 Pro Highspeed worth the higher token price? Compare it with the standard variant using your own prompts.

This benchmark measures:

- Time to first visible answer text
- Total streamed response time
- Estimated token cost when usage is available
- Median latency across repeated runs

[View Kimi K3.1 Pro on CometAPI](https://www.cometapi.com/models/moonshotai/kimi-k3-1-pro/?utm_source=medium&utm_medium=distribution&utm_campaign=bom&utm_content=kimi-k3-1-pro)

## Background

The benchmark compares these model IDs:

- `kimi-k3.1-pro`
- `kimi-k3.1-pro-highspeed`

Highspeed is advertised as offering faster output at a higher token price. This script measures end-to-end latency rather than assuming the advertised speedup applies to every request.

The script uses these editable rates for cost estimates:

| Model | Input / 1M tokens | Output / 1M tokens |
|---|---:|---:|
| Standard | $1.60 | $8.00 |
| Highspeed | $3.20 | $16.00 |

Confirm current prices, model availability, and streaming support before running.

> This script has not been tested against the live CometAPI endpoint. Cost estimates exclude special caching rates, separate fees, and any billing rules not represented by the returned token counts.

## Setup

Requires Python 3.10 or later.

Install the dependency:

```bash
python -m pip install --upgrade openai
```

Set your API key:

```bash
export COMETAPI_KEY="your_key_here"
```

Keep your API key out of source control.

Create `prompts.txt` with one prompt per line:

```text
Explain what an API is in three sentences.
Write a Python function that merges overlapping intervals.
Compare queues and event streams for background processing.
```

## Run

Save the script below as `bench.py`, then run:

```bash
python bench.py prompts.txt --repeats 3
```

The script runs requests sequentially and randomizes model order for each prompt pair. Three prompts with three repetitions produce 18 requests.

Each request may incur charges. The SDK's automatic retries are disabled so they do not silently affect latency measurements.

## Script

```python
import argparse
import random
import statistics
import time
from pathlib import Path
import os

from openai import OpenAI

PRICES = {
    "kimi-k3.1-pro": (1.60, 8.00),
    "kimi-k3.1-pro-highspeed": (3.20, 16.00),
}


def run(client, model, prompt):
    start = time.perf_counter()
    first_text = None
    usage = None
    finish_reason = None

    with client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True},
    ) as stream:
        for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage

            if not chunk.choices:
                continue

            choice = chunk.choices[0]

            # Measures visible answer text, not reasoning or metadata.
            if first_text is None and choice.delta.content:
                first_text = time.perf_counter() - start

            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason

    total = time.perf_counter() - start

    if first_text is None:
        raise RuntimeError("No visible answer text returned.")

    if finish_reason != "stop":
        raise RuntimeError(
            f"Response did not finish normally: {finish_reason}"
        )

    cost = None
    if (
        usage is not None
        and usage.prompt_tokens is not None
        and usage.completion_tokens is not None
    ):
        input_price, output_price = PRICES[model]
        cost = (
            usage.prompt_tokens * input_price
            + usage.completion_tokens * output_price
        ) / 1_000_000

    return {
        "first_text": first_text,
        "total": total,
        "cost": cost,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare standard and Highspeed latency."
    )
    parser.add_argument("prompts", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1.")

    api_key = os.environ.get("COMETAPI_KEY")
    if not api_key:
        parser.error("Set COMETAPI_KEY first.")

    prompts = [
        line.strip()
        for line in args.prompts.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if not prompts:
        parser.error("The prompt file is empty.")

    results = {model: [] for model in PRICES}
    failures = {model: 0 for model in PRICES}

    # Fixed seed makes the scheduling order reproducible.
    rng = random.Random(42)
    jobs = [
        (repeat, index, prompt)
        for repeat in range(args.repeats)
        for index, prompt in enumerate(prompts, start=1)
    ]
    rng.shuffle(jobs)

    with OpenAI(
        base_url="https://api.cometapi.com/v1",
        api_key=api_key,
        timeout=300.0,
        max_retries=0,
    ) as client:
        for repeat, index, prompt in jobs:
            models = list(PRICES)
            rng.shuffle(models)

            for model in models:
                print(
                    f"Run {repeat + 1}, prompt {index}: {model}",
                    flush=True,
                )
                try:
                    results[model].append(
                        run(client, model, prompt)
                    )
                except Exception as exc:
                    failures[model] += 1
                    print(
                        f"  Failed ({type(exc).__name__}). "
                        "Check endpoint support, access, and limits."
                    )

    print("\nSummary — successful requests only")

    for model, rows in results.items():
        print(f"\n{model}")
        print(
            f"  Successful: {len(rows)} | "
            f"Failed: {failures[model]}"
        )

        if not rows:
            continue

        first_text = statistics.median(
            row["first_text"] for row in rows
        )
        total = statistics.median(
            row["total"] for row in rows
        )

        print(f"  Median first visible text: {first_text:.2f}s")
        print(f"  Median total time:         {total:.2f}s")

        known_costs = [
            row["cost"]
            for row in rows
            if row["cost"] is not None
        ]

        if known_costs:
            print(
                f"  Estimated cost: ${sum(known_costs):.6f} "
                f"for {len(known_costs)}/{len(rows)} "
                "successful requests"
            )
        else:
            print("  Estimated cost: unavailable")

        if len(known_costs) != len(rows):
            print(
                "  Usage missing for some requests; "
                "check provider billing."
            )

    if any(failures.values()):
        print(
            "\nFailed requests may still incur charges. "
            "Their costs are not included above."
        )


if __name__ == "__main__":
    main()
```

## Understanding the results

### First visible text

This measures the time until the first non-empty answer-text chunk.

It does not measure the first network byte or first reasoning token. A model may spend time reasoning before returning visible text.

### Total time

This includes request setup, waiting, reasoning, answer generation, and stream completion.

A faster output-generation rate may produce a smaller improvement in total response time.

### Estimated cost

When the stream returns usage, the script applies the configured input and output rates.

Missing usage is reported as unavailable—not as a zero-dollar request. Actual billing may differ because of caching, reasoning-token rules, or other charges.

## Benchmark limitations

- Results depend on network conditions, provider load, and rate limits.
- The script does not control or identify cache hits.
- Default reasoning settings may differ between endpoints.
- Different answer lengths can affect total time and cost.
- Failed requests are excluded from latency medians.
- There are no separate warm-up requests.
- The script measures speed and estimated cost, not answer quality.

For a fairer comparison, use representative prompts, repeat the test at different times, and review answer quality separately.

If you configure reasoning effort or output limits, apply equivalent supported settings to both variants.

## How to decide

For interactive applications, compare both the delay before visible text and the time to a complete answer.

For background jobs, consider whether faster completion improves throughput enough to justify the price difference.

Choose based on successful task completion, quality, latency, and actual billing—not speed alone.

## Links

- [Kimi K3.1 Pro model page and pricing](https://www.cometapi.com/models/moonshotai/kimi-k3-1-pro/?utm_source=medium&utm_medium=distribution&utm_campaign=bom&utm_content=kimi-k3-1-pro)
- [CometAPI documentation](https://apidoc.cometapi.com)
````
