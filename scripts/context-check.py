#!/usr/bin/env python3
"""Fill almost the entire context and sample GPU headroom during cold/repeated requests.

Uses the native Anthropic token-count/generation routes implemented by both engines.
This is a capacity test, not a throughput or answer-quality benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.error
import urllib.request


class APIError(RuntimeError):
    def __init__(self, status, payload):
        super().__init__(f"HTTP {status}: {payload}")
        self.status, self.payload = status, payload


def post(base, path, body):
    request = urllib.request.Request(base.rstrip("/") + path, json.dumps(body).encode(),
                                    {"Content-Type": "application/json",
                                     "x-api-key": os.environ.get("API_KEY", "local"),
                                     "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise APIError(exc.code, exc.read().decode()) from exc


def gpu_memory(gpu):
    result = subprocess.run(["nvidia-smi", "-i", gpu,
                             "--query-gpu=memory.free,memory.used",
                             "--format=csv,noheader,nounits"],
                            text=True, capture_output=True, check=True, timeout=10)
    free, used = (float(value.strip()) for value in result.stdout.strip().split(","))
    return {"time": time.time(), "free_mib": free, "used_mib": used}


def make_body(text, output):
    return {"model": os.environ.get("MODEL_ID", "qwen3.8-27b"),
            "messages": [{"role": "user", "content": text}],
            "thinking": {"type": "disabled"}, "max_tokens": output, "temperature": 0}


def fill_prompt(base, context, output):
    # Non-repetitive synthetic records avoid a trivial single repeated-token test.
    records = "\n".join(f"Record {i}: {hashlib.sha256(str(i).encode()).hexdigest()}"
                        for i in range(context // 8))
    prefix = "Read the following reference records.\n"
    suffix = ("\nWrite a very long numbered explanation of hash tables, collision resolution, "
              "memory management, transactions and concurrency. Include extensive Python and SQL "
              "examples. Continue until the output token limit; do not give a short summary.")
    target = context - output - 256  # reserve template/rounding space, in addition to output
    low, high = 0, len(records)
    best = None
    while low <= high:
        middle = (low + high) // 2
        body = make_body(prefix + records[:middle] + suffix, output)
        try:
            count = post(base, "/v1/messages/count_tokens", body)["input_tokens"]
        except APIError as exc:
            # NInfer's counting route rejects over-context candidates itself.
            # Its Anthropic adapter exposes the diagnostic in error.message.
            if exc.status == 400 and "context" in exc.payload.lower() and "exceed" in exc.payload.lower():
                high = middle - 1
                continue
            raise
        if count <= target:
            if best is None or count > best[1]:
                best = body, count
            low = middle + 1
        else:
            high = middle - 1
    if best is None or best[1] < target - 128:
        raise RuntimeError("Could not construct a prompt near the requested context size")
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, default=int(os.environ.get("CONTEXT_SIZE", "131072")))
    parser.add_argument("--output-tokens", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--min-free-mib", type=int, default=1024)
    parser.add_argument("--gpu", default="0", help="nvidia-smi GPU index or UUID")
    parser.add_argument("--base-url", default=f"http://127.0.0.1:{os.environ.get('PORT', '8080')}")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if (args.context > 262144 or args.output_tokens < 1 or args.repeats < 1 or
            args.context <= args.output_tokens + 512 or args.min_free_mib < 0):
        parser.error("require output_tokens + 512 < context <= 262144, positive repeats/output, nonnegative headroom")
    out = args.out or Path(os.environ.get("DATA_DIR", "/workspace")) / "results" / (
        "context-" + time.strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    samples = [gpu_memory(args.gpu)]
    errors, runs = [], []
    done = threading.Event()

    def monitor():
        while not done.wait(0.5):
            try:
                samples.append(gpu_memory(args.gpu))
            except Exception as exc:
                errors.append(str(exc))
                return

    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    report = {"settings": {**vars(args), "out": str(out)}, "runs": runs,
              "gpu_samples": samples, "monitor_errors": errors}
    try:
        body, counted = fill_prompt(args.base_url, args.context, args.output_tokens)
        (out / "request.json").write_text(json.dumps(body) + "\n")
        report["counted_prompt_tokens"] = counted
        for repeat in range(args.repeats):
            print(f"Pass {repeat + 1}: {counted} prompt + up to {args.output_tokens} output tokens", flush=True)
            start = time.monotonic()
            answer = post(args.base_url, "/v1/messages", body)
            (out / f"response-{repeat + 1}.json").write_text(json.dumps(answer) + "\n")
            runs.append({"elapsed_s": time.monotonic() - start, "usage": answer["usage"],
                         "stop_reason": answer.get("stop_reason")})
            # Repeat the same near-full prompt to exercise retained cache/restore pressure.
        samples.append(gpu_memory(args.gpu))
        done.set()
        worker.join(timeout=15)
        minimum = min(sample["free_mib"] for sample in samples)
        report["minimum_sampled_free_mib"] = minimum
        report["passed"] = (not errors and minimum >= args.min_free_mib and all(
            run["usage"].get("output_tokens", 0) >= args.output_tokens and
            run["stop_reason"] == "max_tokens" for run in runs))
        print(f"Minimum sampled free VRAM: {minimum:.0f} MiB; results: {out}", flush=True)
        if not report["passed"]:
            raise RuntimeError("Capacity check not passed: insufficient headroom, monitoring failure, "
                               "or early EOS (full output growth not exercised). See report.json")
    except Exception as exc:
        report["passed"] = False
        report["error"] = str(exc)
        raise
    finally:
        done.set()
        worker.join(timeout=15)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
