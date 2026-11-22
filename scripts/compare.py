#!/usr/bin/env python3
"""Launch backends sequentially, wait for readiness, check tools, and benchmark."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

SCRIPTS = Path(__file__).resolve().parent


def stop(server):
    if server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()


def interrupted(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backends", nargs="*", default=["llama", "turbo"],
                        choices=["llama", "llama-mtp", "turbo", "turbo-nospec", "turbo-dflash2"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--startup-timeout", type=int, default=900)
    args = parser.parse_args()
    data = Path(os.environ.get("DATA_DIR", "/workspace"))
    campaign = data / "results" / time.strftime("%Y%m%d-%H%M%S")
    campaign.mkdir(parents=True, exist_ok=True)
    port = os.environ.get("PORT", "8080")
    # Refuse an occupied port before spawning; never benchmark an unrelated server.
    import socket
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", int(port))) == 0:
            sys.exit(f"Port {port} is occupied. Stop the existing server first.")
    gpu = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
    (campaign / "nvidia-smi.txt").write_text(gpu.stdout)
    root = SCRIPTS.parent
    metadata = {"arguments": vars(args), "models": json.loads((root / "models.json").read_text()),
                "versions": {p.name: p.read_text() for p in root.glob("*-version.txt")},
                "environment": {k: os.environ[k] for k in
                                ["CONTEXT_SIZE", "DRAFT_TOKENS", "THINKING_BUDGET", "HOST", "PORT",
                                 "KV_CAPACITY", "PREFILL_CHUNK", "DEVICE_STATE_SLOTS",
                                 "HOST_STATE_SLOTS", "HOST_KV_MIB", "BATCH_SIZE", "UBATCH_SIZE",
                                 "CTX_CHECKPOINTS", "CACHE_RAM_MIB"]
                                if k in os.environ}}
    (campaign / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    signal.signal(signal.SIGTERM, interrupted)
    for backend in args.backends:
        out = campaign / backend
        out.mkdir()
        with (out / "server.log").open("w") as log:
            server = subprocess.Popen(["bash", str(SCRIPTS / "serve.sh"), backend],
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + args.startup_timeout
                while True:
                    if server.poll() is not None:
                        sys.exit(f"{backend} exited ({server.returncode}); inspect {out / 'server.log'}")
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                            if response.status == 200:
                                break
                    except (urllib.error.URLError, TimeoutError):
                        pass
                    if time.monotonic() > deadline:
                        sys.exit(f"{backend} startup timed out; inspect {out / 'server.log'}")
                    time.sleep(2)
                with (out / "tools.log").open("w") as tool_log:
                    result = subprocess.run([sys.executable, str(SCRIPTS / "tool-smoke.py")],
                                            stdout=tool_log, stderr=subprocess.STDOUT)
                print(f"{backend}: tool smoke {'PASS' if result.returncode == 0 else 'FAIL (see tools.log)'}", flush=True)
                subprocess.run([sys.executable, str(SCRIPTS / "benchmark.py"), "--label", backend,
                                "--out", str(out), "--repeats", str(args.repeats),
                                "--max-tokens", str(args.max_tokens)], check=True)
            finally:
                stop(server)
    print(f"Results: {campaign}")


if __name__ == "__main__":
    main()
