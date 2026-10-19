"""Anthropic Messages content blocks → OpenAI chat/completions messages.

Convention (per sglang PR #29083 and axolotl/LLaMA-Factory defaults):

- Assistant turn with [thinking?, text?, tool_use, tool_use, ...] becomes ONE
  assistant message with `content: "<think>...</think>\n\n<text>"` and
  `tool_calls: [...]`.
- Each `tool_result` content block becomes its OWN
  `{"role":"tool","tool_call_id":"...","content":"..."}` message. Do not merge
  into the next user turn — that attributes the result to the user and breaks
  the tool_call_id linkage trainers rely on.

Thinking handling (`thinking_mode`):
- "keep" (default) — wrap thinking.text in `<think>...</think>` inside the
  assistant content string, strip the Claude-specific `signature`. Matches
  Qwen 3 Thinking + Hermes-function-calling-v1 conventions.
- "drop" — omit thinking entirely. Matches OpenAI's historical SFT guidance.

Tool definitions: Anthropic `{"name","description","input_schema"}` →
OpenAI `{"type":"function","function":{"name","description","parameters"}}`.

Tool call arguments: Anthropic `input` is a JSON object; OpenAI `arguments`
is a JSON-encoded string. We JSON-serialize.
"""
from __future__ import annotations

import hashlib
import json
import secrets
from typing import Any, Literal

ThinkingMode = Literal["keep", "drop"]
DEFAULT_THINKING_MODE: ThinkingMode = "keep"

from .store import Record


class NormalizeError(RuntimeError):
    """Raised when a record cannot be normalized to OpenAI chat/completions shape."""


# --- Public API ---------------------------------------------------------------


def anthropic_tools_to_openai(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not tools:
        return []
    out: list[dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        name = t.get("name")
        if not isinstance(name, str):
            continue
        fn: dict[str, Any] = {"name": name}
        if isinstance(t.get("description"), str):
            fn["description"] = t["description"]
        schema = t.get("input_schema")
        if isinstance(schema, dict):
            fn["parameters"] = schema
        out.append({"type": "function", "function": fn})
    return out


def anthropic_system_to_openai(system: Any) -> str:
    """Flatten Anthropic's `system` (str or list[block]) into one string."""
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts: list[str] = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n\n".join(parts)
    return ""


def flatten_messages(
    messages: list[dict[str, Any]],
    *,
    assistant_response: list[dict[str, Any]] | None = None,
    thinking_mode: ThinkingMode = DEFAULT_THINKING_MODE,
) -> list[dict[str, Any]]:
    """Flatten Anthropic Messages array into OpenAI chat/completions messages.

    `messages` is the request's conversation history. `assistant_response`, if
    passed, is the response.content array that we append as the final assistant
    turn.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "user":
            out.extend(_flatten_user(content))
        elif role == "assistant":
            out.append(_flatten_assistant(content, thinking_mode=thinking_mode))
        else:
            raise NormalizeError(f"Unexpected message role: {role!r}")
    if assistant_response is not None:
        out.append(_flatten_assistant(assistant_response, thinking_mode=thinking_mode))
    return out


# --- Internals ---------------------------------------------------------------


def _flatten_user(content: Any) -> list[dict[str, Any]]:
    """A user-role Anthropic message carrying tool_result blocks is actually
    one-or-more OpenAI `tool` role messages plus an optional user text message.
    """
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    if not isinstance(content, list):
        raise NormalizeError(f"Unsupported user content shape: {type(content).__name__}")

    tool_messages: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        t = block.get("type")
        if t == "tool_result":
            tool_messages.append(_tool_result_message(block))
        elif t == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                text_parts.append(text)
        # Image / document / other block types are not emitted in v1.
    out = list(tool_messages)
    if text_parts:
        out.append({"role": "user", "content": "\n\n".join(text_parts)})
    return out


def _tool_result_message(block: dict[str, Any]) -> dict[str, Any]:
    tool_use_id = block.get("tool_use_id")
    if not isinstance(tool_use_id, str):
        raise NormalizeError("tool_result block missing tool_use_id")
    content = _tool_result_content_to_string(block.get("content"))
    msg: dict[str, Any] = {
        "role": "tool",
        "tool_call_id": tool_use_id,
        "content": content,
    }
    return msg


def _tool_result_content_to_string(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                text = b.get("text")
                if isinstance(text, str):
                    parts.append(text)
            # Non-text tool_result content (images, etc.) is skipped in v1.
        return "\n\n".join(parts)
    # Fallback: json-serialize anything exotic so we don't silently drop it.
    return json.dumps(content, ensure_ascii=False)


def _flatten_assistant(
    content: Any,
    *,
    thinking_mode: ThinkingMode = DEFAULT_THINKING_MODE,
) -> dict[str, Any]:
    """Collapse an assistant turn's content blocks into one OpenAI assistant message."""
    if isinstance(content, str):
        return {"role": "assistant", "content": content}
    if not isinstance(content, list):
        raise NormalizeError(f"Unsupported assistant content shape: {type(content).__name__}")

    thinking_parts: list[str] = []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        t = block.get("type")
        if t == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                text_parts.append(text)
        elif t == "tool_use":
            tool_calls.append(_tool_use_to_tool_call(block))
        elif t == "thinking":
            if thinking_mode == "keep":
                text = block.get("thinking")
                # Note: Claude's `signature` field is Claude-specific crypto and
                # has no training value for a non-Anthropic target — drop it.
                if isinstance(text, str) and text:
                    thinking_parts.append(text)
        # Other block types ignored.
    body_parts: list[str] = []
    if thinking_parts:
        body_parts.append("<think>" + "\n\n".join(thinking_parts) + "</think>")
    if text_parts:
        body_parts.append("\n\n".join(text_parts))
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": "\n\n".join(body_parts) if body_parts else "",
    }
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _tool_use_to_tool_call(block: dict[str, Any]) -> dict[str, Any]:
    call_id = block.get("id")
    name = block.get("name")
    if not isinstance(call_id, str) or not isinstance(name, str):
        raise NormalizeError("tool_use block missing id/name")
    input_obj = block.get("input")
    arguments = json.dumps(input_obj if input_obj is not None else {}, ensure_ascii=False)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


# --- Record-to-conversation helper -------------------------------------------


def record_to_openai_conversation(
    record: Record,
    *,
    thinking_mode: ThinkingMode = DEFAULT_THINKING_MODE,
) -> dict[str, Any]:
    """Convert a single claude-tap record into a complete OpenAI SFT example.

    Because the Anthropic Messages API resends full conversation state in each
    request, a single record carries the complete prior conversation in its
    request body. We take the request's messages + prepend system + append the
    response content as the final assistant turn.
    """
    req_body = record.payload.get("request", {}).get("body")
    resp_body = record.payload.get("response", {}).get("body")
    if not isinstance(req_body, dict) or not isinstance(resp_body, dict):
        raise NormalizeError("record missing request.body or response.body")

    system = anthropic_system_to_openai(req_body.get("system"))
    tools = anthropic_tools_to_openai(req_body.get("tools"))
    messages_in = req_body.get("messages")
    if not isinstance(messages_in, list):
        raise NormalizeError("request.body.messages is not a list")
    response_content = resp_body.get("content")
    if not isinstance(response_content, list):
        raise NormalizeError("response.body.content is not a list")

    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(
        flatten_messages(
            messages_in,
            assistant_response=response_content,
            thinking_mode=thinking_mode,
        )
    )

    example: dict[str, Any] = {"messages": messages}
    if tools:
        example["tools"] = tools
    return example


def example_fingerprint(example: dict[str, Any]) -> str:
    """Stable content hash — lets the exporter dedupe or detect reprocessing."""
    canonical = json.dumps(example, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def new_export_run_id() -> str:
    return secrets.token_hex(8)
