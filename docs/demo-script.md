# agentloop demo video — narration script

Runtime 43.8s · 1920×1080 · 30fps · voice: macOS `say` (Samantha, 172 wpm)
Render pipeline: `/tmp/agentloop-video/` (cdp record → frame pacing → cards → voice → overlay), not part of the repo.

| # | Beat on screen | Narration |
|---|---|---|
| L1 | title card + landing | agentloop — a minimal agent runtime, built from scratch in about seven hundred lines of Python. No LangChain. No framework. |
| L2 | pricing task runs, result lands | Give it a task, and it plans its own tool calls: first, a search of the knowledge base for the pricing doc. |
| L3 | timeline walk (thought → action → observation) | Then it computes the annual cost — reason, act, observe, one step at a time. |
| L4 | final answer + copy button | Every run ends in a grounded answer, with the full trace right beside it. |
| L5 | error-recovery run (red ERROR observation) | If the agent reaches for a tool that doesn't exist, the error comes back as an observation — and it self-corrects instead of crashing. |
| L6 | max-steps guard (red pill, amber warning) | A max-steps guard keeps runaway loops in check. |
| L7 | end card (history + schema recap) | Runs are timestamped and replayable, and every tool documents its own schema. agentloop — the mechanics of AI agents, made visible. |

Swap-in note: the voice track is generated, so re-recording with a human narrator
only needs the lines above — timing anchors live in `assemble.py` (marks.json),
and each line maps to an on-screen beat.
