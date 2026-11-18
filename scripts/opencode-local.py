#!/usr/bin/env python3
"""Launch the installed OpenCode with a local provider and the selected context size."""
import json
import os
from pathlib import Path
import sys

config = json.loads((Path(__file__).resolve().parent.parent / "opencode.json").read_text())
config["providers"]["v100"]["models"]["qwen3.8-27b"]["limit"]["context"] = int(
    os.environ.get("CONTEXT_SIZE", "32768"))
os.environ.setdefault("QWEN_BASE_URL", f"http://127.0.0.1:{os.environ.get('PORT', '8080')}/v1")
os.environ.setdefault("API_KEY", "local")
os.environ["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
os.execvp("opencode", ["opencode", "--standalone", *sys.argv[1:]])
