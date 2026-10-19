"""Thinking-mode handling in the Anthropic → OpenAI normalizer."""
from __future__ import annotations

from trace_forge.normalize import _flatten_assistant


def _blocks() -> list[dict]:
    return [
        {
            "type": "thinking",
            "thinking": "Let me think about which tool to call first.",
            "signature": "sig_abc123_claude_specific_crypto_token",
        },
        {"type": "text", "text": "I'll search for it."},
        {
            "type": "tool_use",
            "id": "toolu_01abc",
            "name": "Grep",
            "input": {"path": "/tmp", "pattern": "foo"},
        },
    ]


def test_keep_wraps_thinking_and_strips_signature() -> None:
    msg = _flatten_assistant(_blocks(), thinking_mode="keep")
    assert msg["role"] == "assistant"
    assert msg["content"].startswith("<think>")
    assert "</think>" in msg["content"]
    assert "Let me think about which tool to call first." in msg["content"]
    assert "I'll search for it." in msg["content"]
    # Signature must not leak into training content.
    assert "sig_abc123" not in msg["content"]
    # tool_calls must survive untouched.
    assert len(msg["tool_calls"]) == 1
    assert msg["tool_calls"][0]["function"]["name"] == "Grep"


def test_drop_omits_thinking_entirely() -> None:
    msg = _flatten_assistant(_blocks(), thinking_mode="drop")
    assert "<think>" not in msg["content"]
    assert "Let me think" not in msg["content"]
    assert msg["content"] == "I'll search for it."
    assert len(msg["tool_calls"]) == 1


def test_empty_thinking_block_is_not_wrapped() -> None:
    """Matches our live capture: thinking blocks with text_len=0 should not
    produce a vacuous <think></think>."""
    blocks = [
        {"type": "thinking", "thinking": "", "signature": ""},
        {
            "type": "tool_use",
            "id": "toolu_02xyz",
            "name": "Read",
            "input": {"file_path": "/etc/os-release"},
        },
    ]
    msg = _flatten_assistant(blocks, thinking_mode="keep")
    assert "<think>" not in msg["content"]
    assert msg["content"] == ""
    assert msg["tool_calls"][0]["function"]["name"] == "Read"


def test_multiple_thinking_blocks_concatenated() -> None:
    blocks = [
        {"type": "thinking", "thinking": "First thought.", "signature": ""},
        {"type": "thinking", "thinking": "Second thought.", "signature": ""},
        {"type": "text", "text": "Done thinking."},
    ]
    msg = _flatten_assistant(blocks, thinking_mode="keep")
    assert msg["content"].count("<think>") == 1
    assert msg["content"].count("</think>") == 1
    assert "First thought." in msg["content"]
    assert "Second thought." in msg["content"]
    assert "Done thinking." in msg["content"]
    # Think block must precede text.
    assert msg["content"].index("</think>") < msg["content"].index("Done thinking.")
