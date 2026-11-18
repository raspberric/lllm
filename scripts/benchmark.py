#!/usr/bin/env python3
"""Single-request streaming benchmark; writes responses as well as timing data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request

PROMPTS = {
    "python": "Write a Python standard-library implementation of an LRU cache with get/put, "
              "O(1) operations, capacity validation, and five unittest tests. Explain the invariants.",
    "sql": "A PostgreSQL jobs table has id, status, run_at, attempts and payload. Design a "
           "concurrency-safe worker claiming query using SKIP LOCKED, retries with backoff, "
           "and the necessary indexes. Include SQL and explain transaction boundaries.",
    "debug": "Find and fix every bug in this function, explain the failures, and provide tests:\n"
             "def merge(intervals):\n    intervals.sort()\n    out = [intervals[0]]\n"
             "    for start, end in intervals[1:]:\n"
             "        if start < out[-1][1]:\n            out[-1][1] = end\n"
             "        else:\n            out.append([start, end])\n    return out\n"
             "Inputs may be empty, unsorted, nested, touching, or immutable tuples. Do not mutate inputs.",
}


def request_json(base, path, body=None):
    request = urllib.request.Request(
        base.rstrip("/") + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('API_KEY', 'local')}"})
    try:
        return urllib.request.urlopen(request, timeout=600)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode()}") from exc


def generate(base, body):
    start = time.perf_counter()
    first = None
    content, reasoning, usage, timings = [], [], {}, {}
    finish = None
    done = False
    with request_json(base, "/v1/chat/completions", body) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            chunk = json.loads(data)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            usage = chunk.get("usage") or usage
            timings = chunk.get("timings") or timings
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                text = delta.get("content") or ""
                thought = delta.get("reasoning_content") or ""
                if first is None and (text or thought or delta.get("tool_calls")):
                    first = time.perf_counter() - start
                content.append(text)
                reasoning.append(thought)
                finish = choice.get("finish_reason") or finish
    elapsed = time.perf_counter() - start
    if not done or finish is None or first is None:
        raise RuntimeError("Incomplete or empty stream")
    tokens = usage.get("completion_tokens")
    if not isinstance(tokens, int) or tokens < 1:
        raise RuntimeError("Missing completion-token usage; cannot calculate honest throughput")
    return {"ttft_s": first, "elapsed_s": elapsed, "usage": usage,
            "end_to_end_tok_s": tokens / elapsed, "timings": timings,
            "finish_reason": finish, "content": "".join(content), "reasoning": "".join(reasoning)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--base-url", default=f"http://127.0.0.1:{os.environ.get('PORT', '8080')}")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--reasoning", choices=["none", "low", "medium", "xhigh"], default="low")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.max_tokens < 1:
        parser.error("repeats and max-tokens must be positive")
    out = args.out or Path(os.environ.get("DATA_DIR", "/workspace")) / "results" / args.label
    out.mkdir(parents=True, exist_ok=True)
    with request_json(args.base_url, "/v1/models") as response:
        (out / "models.json").write_bytes(response.read())
    rows = []
    with (out / "requests.jsonl").open("w") as results:
        for case, prompt in PROMPTS.items():
            for repeat in range(-1, args.repeats):
                # Distinct leading system text prevents warmup/repetitions from becoming
                # full-prefix cache hits. Kernel/model warmup remains useful.
                marker = hashlib.sha256(f"{case}:{repeat}".encode()).hexdigest()[:32]
                body = {"model": os.environ.get("MODEL_ID", "qwen3.8-27b"),
                        "messages": [{"role": "system", "content": f"Request {marker}. You are a precise coding assistant."},
                                     {"role": "user", "content": prompt}],
                        "stream": True, "stream_options": {"include_usage": True},
                        "max_tokens": args.max_tokens, "seed": 42, "temperature": 0.7,
                        "top_p": 0.8, "top_k": 20, "min_p": 0, "presence_penalty": 0,
                        "reasoning_effort": args.reasoning, "thinking_budget_tokens": 512}
                result = generate(args.base_url, body)
                row = {"label": args.label, "case": case, "repeat": repeat,
                       "warmup": repeat == -1, "request": body, **result}
                results.write(json.dumps(row) + "\n")
                results.flush()
                print(f"{args.label} {case} {'warmup' if repeat == -1 else repeat}: "
                      f"TTFT {result['ttft_s']:.2f}s, "
                      f"end-to-end {result['end_to_end_tok_s']:.2f} tok/s, "
                      f"decode {result['timings'].get('predicted_per_second', 'unreported')}", flush=True)
                if repeat >= 0:
                    rows.append(row)
    summary = {}
    for case in PROMPTS:
        subset = [r for r in rows if r["case"] == case]
        decode = [r["timings"]["predicted_per_second"] for r in subset
                  if "predicted_per_second" in r["timings"]]
        summary[case] = {
            "median_ttft_s": statistics.median(r["ttft_s"] for r in subset),
            "median_end_to_end_tok_s": statistics.median(r["end_to_end_tok_s"] for r in subset),
            "median_server_decode_tok_s": statistics.median(decode) if decode else None,
            "length_limited_responses": sum(r["finish_reason"] == "length" for r in subset)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
