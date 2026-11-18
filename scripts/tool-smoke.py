#!/usr/bin/env python3
"""Check parsed tool calls, tool-result replay, and a subsequent streamed answer."""
import json
import os
import sys
from benchmark import generate, request_json

base = f"http://127.0.0.1:{os.environ.get('PORT', '8080')}"
messages = [{"role": "user", "content":
             "Call lookup_test_value with key='v100'. You must obtain the value from the tool, "
             "then repeat the exact value it returns. Do not invent it."}]
tools = [{"type": "function", "function": {
    "name": "lookup_test_value", "description": "Return a test value by key.",
    "parameters": {"type": "object", "properties": {"key": {"type": "string"}},
                   "required": ["key"]}}}]
body = {"model": os.environ.get("MODEL_ID", "qwen3.8-27b"), "messages": messages,
        "tools": tools, "tool_choice": "auto", "max_tokens": 512,
        "temperature": 0, "reasoning_effort": "none"}
with request_json(base, "/v1/chat/completions", body) as response:
    answer = json.load(response)["choices"][0]["message"]
calls = answer.get("tool_calls", [])
if len(calls) != 1 or calls[0]["function"]["name"] != "lookup_test_value":
    sys.exit(f"FAIL: expected one parsed tool call, received {answer}")
if json.loads(calls[0]["function"]["arguments"]) != {"key": "v100"}:
    sys.exit(f"FAIL: incorrect tool arguments: {calls}")
messages.append({k: v for k, v in answer.items()
                 if k in {"role", "content", "tool_calls", "reasoning_content"}})
messages.append({"role": "tool", "tool_call_id": calls[0]["id"], "content": "VOLTA_TEST_7391"})
body.update(stream=True, stream_options={"include_usage": True}, tool_choice="none")
result = generate(base, body)
if "VOLTA_TEST_7391" not in result["content"]:
    sys.exit(f"FAIL: tool result was not returned: {result}")
print("PASS: parsed function call, JSON arguments, tool-result replay, streaming answer")
