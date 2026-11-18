"""CPU-only contract tests for process isolation and measurement plumbing."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_json(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def do_GET(self):
        if self.path == "/health":
            self.send_json({"status": "ok"})
        else:
            self.send_json({"data": [{"id": "qwen3.8-27b"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not body.get("stream"):
            self.send_json({"choices": [{"message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "test-call", "type": "function", "function": {
                    "name": "lookup_test_value", "arguments": '{"key":"v100"}'}}]}}]})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunks = [
            {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
            {"choices": [{"delta": {"reasoning_content": "checking"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "VOLTA_TEST_7391"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 20, "completion_tokens": 10},
             "timings": {"predicted_per_second": 25.0}},
        ]
        for chunk in chunks:
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        if not getattr(self.server, "truncated", False):
            self.wfile.write(b"data: [DONE]\n\n")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class HarnessTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.env = {**os.environ, "DATA_DIR": self.temp.name, "PORT": str(free_port()),
                    "PATH": f"{self.directory}:{os.environ['PATH']}"}
        source = Path(__file__).resolve()
        fake = (f"#!{sys.executable}\nimport runpy, sys\n"
                f"ns=runpy.run_path({str(source)!r})\n"
                "port=int(sys.argv[sys.argv.index('--port')+1])\n"
                "ns['ThreadingHTTPServer'](('127.0.0.1',port),ns['Handler']).serve_forever()\n")
        for binary in ["llama-server", "ninfer-serve"]:
            executable = self.directory / binary
            executable.write_text(fake)
            executable.chmod(0o755)
        gpu = self.directory / "nvidia-smi"
        gpu.write_text(f"#!{sys.executable}\nprint('CPU-only test double')\n")
        gpu.chmod(0o755)
        for name in ["llama/Qwen3.8-27B-UD-Q4_K_M.gguf", "turbo/qwen3_8_27b_nvfp4.ninfer"]:
            model = self.directory / "models" / name
            model.parent.mkdir(parents=True, exist_ok=True)
            model.touch()

    def test_sequential_comparison_and_tool_replay(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/compare.py"),
                                 "llama", "turbo", "--repeats", "1"], env=self.env,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summaries = list((self.directory / "results").glob("*/*/summary.json"))
        self.assertEqual(len(summaries), 2)
        for path in summaries:
            summary = json.loads(path.read_text())
            self.assertEqual(summary["python"]["median_server_decode_tok_s"], 25)
            self.assertIn("PASS", (path.parent / "tools.log").read_text())
            rows = [json.loads(line) for line in (path.parent / "requests.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertEqual(sum(row["warmup"] for row in rows), 3)
            self.assertEqual(len({row["request"]["messages"][0]["content"] for row in rows}), 6)
        with socket.socket() as sock:
            self.assertNotEqual(sock.connect_ex(("127.0.0.1", int(self.env["PORT"]))), 0)

    def test_truncated_stream_is_not_a_successful_measurement(self):
        spec = importlib.util.spec_from_file_location("benchmark", ROOT / "scripts/benchmark.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.truncated = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                module.generate(f"http://127.0.0.1:{server.server_port}", {"stream": True})
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_second_launcher_cannot_acquire_gpu_lock(self):
        with (self.directory / "inference.lock").open("w") as lock:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = subprocess.run(["bash", str(ROOT / "scripts/serve.sh"), "turbo"],
                                    env=self.env, capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Another inference server", result.stderr)


if __name__ == "__main__":
    unittest.main()
