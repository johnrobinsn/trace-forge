# trace-forge

Turn your coding-agent LLM traces into fine-tune-ready JSONL.

**trace-forge** is the complement to [`claude-tap`](https://github.com/liaohch3/claude-tap). `claude-tap` captures and archives the wire-level request/response traffic from your coding agents (Claude Code, Codex, Gemini, Grok, Kimi, OpenCode, Hermes, Cursor, Pi, Qoder, Antigravity, CodeBuddy, DeepSeek). `trace-forge` reads that archive, filters out auxiliary chatter (session-naming/classifier calls), normalizes Anthropic's content-block format to OpenAI `messages` + `tools` + `tool_calls`, and writes JSONL you can feed directly to Unsloth, axolotl, LLaMA-Factory, or TRL.

Companion blog post: [Own Your Own Agent Traces](https://liminal-bits.com/posts/own_your_agent_traces/).

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/johnrobinsn/trace-forge
cd trace-forge
uv sync
```

## Quickstart

```sh
# 1. Launch a tapped Claude Code session (claude-tap handles the proxy; auth is OAuth-transparent)
uv run trace-forge run claude

# 2. List what you've captured
uv run trace-forge sessions

# 3. Export one session (or all) to OpenAI SFT JSONL
uv run trace-forge export --session-id <id> -o out.jsonl
uv run trace-forge export --since 2026-10-01 -o week.jsonl
```

Each JSONL line is one complete conversation in OpenAI chat/completions shape:
`{"messages": [{"role": "system", ...}, ...], "tools": [{"type": "function", ...}, ...]}`.

Feed it straight to Unsloth / axolotl / LLaMA-Factory — the trainer's `apply_chat_template(..., tools=[...])` handles model-specific rendering.

## Supported harnesses

Picked via `trace-forge run <harness>`:

| Harness | Mode | Notes |
|---|---|---|
| `claude` | reverse | Claude Code — OAuth passes through transparently |
| `codex` | reverse | Codex CLI |
| `codexapp` | forward | Codex desktop app (CA trust auto-injected) |
| `gemini` | forward | Gemini CLI |
| `grok` | reverse | Grok Build CLI |
| `kimi`, `kimi-code` | forward | Kimi CLI, Kimi Code CLI |
| `opencode`, `openclaw`, `mimo` | forward | OpenCode & forks |
| `hermes` | forward | Hermes Agent |
| `cursor` | transcript-only | Watches Cursor's agent transcripts; doesn't MITM |
| `pi` | forward | Pi coding agent |
| `qoder` | forward | Qoder Agent |
| `agy` | forward | Antigravity CLI |
| `codebuddy` | forward | CodeBuddy CLI |
| `dsh` | forward | DeepSeek Harness |
| `vllm` | reverse | Local vLLM via `--target http://localhost:8000` |

## Export options

```sh
trace-forge export \
  --session-id <id>              # single session (omit to export all)
  --client claude                # filter by harness
  --since 2026-10-01             # YYYY-MM-DD or full ISO
  --until 2026-10-15
  --thinking keep                # 'keep' wraps in <think>...</think> (Qwen 3 / Hermes targets);
                                 # 'drop' omits thinking (OpenAI SFT targets)
  --format openai-messages       # only format in v1
  --output out.jsonl             # default: ~/.trace-forge/exports/export-<timestamp>.jsonl
```

Auxiliary calls (Claude Code's background haiku classifier / title-generator) are dropped by default via Claude Code's `X-Claude-Code-Request-Class` header. Only `main` and `subagent` request classes are kept.

## Where things live

| Path | Purpose |
|---|---|
| `$CLOUDTAP_DB` (or `~/.local/share/claude-tap/traces.sqlite3`) | claude-tap's capture database |
| `$TRACE_FORGE_HOME` (or `~/.trace-forge/`) | trace-forge state (default export directory) |
| `~/.trace-forge/exports/` | JSONL exports |

## How it fits together

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ claude /    │───▶│  claude-tap  │───▶│ SQLite       │───▶│ trace-forge  │───▶ out.jsonl
│ codex / ... │    │  (proxy)     │    │ (your disk)  │    │ (export)     │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
     your             captures            archive              normalizes +
   coding agent      wire traffic        (lossless)            exports JSONL
```

`claude-tap` does the hard work. trace-forge is the opinionated launcher + filter + Anthropic→OpenAI normalizer + JSONL exporter on top.

## Status

v0.1 — opinionated glue around claude-tap. Covers the capture→export loop end-to-end, with Claude Code as the primary target.

Not yet in v1: tested only on Linux and macOS (Windows is best-effort — upstream claude-tap has open subprocess issues on Windows), scheduled rotation/compression, richer privacy layer beyond auth-header redaction.

## License

Apache-2.0. See [LICENSE](LICENSE).

Built on top of [`liaohch3/claude-tap`](https://github.com/liaohch3/claude-tap) — if you build anything in this space, start from their code.
