"""Advisor logic: settings, transcript capture, the advisor call, the Jev gate."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PLUGIN_ID = "advisor"
DEFAULTS: Dict[str, Any] = {
    "provider": "",            # empty = main provider
    "model": "claude-opus-5-5",
    "reasoning_effort": "high",
    "max_tokens": 1500,
    "timeout": 120,
    "transcript_chars": 60000,  # tail of the conversation the advisor sees
    "mode": "auto",             # off | nudge | auto  (Jev gate)
    "threshold": 0.7,           # Jev p(needs advisor) to escalate
    "min_message_chars": 25,    # skip "thanks", "ok", ...
    "platforms": [],            # empty = every platform
    "allowed_users": [],        # sender ids allowed to trigger the advisor; empty = anyone the gateway admits
    "max_per_hour": 20,         # escalations per sender per hour (0 = unlimited)
}

ADVISOR_SYSTEM = (
    "You are the senior advisor to another AI agent (the executor). The executor is a cheaper model "
    "running the conversation below; it consults you only at hard or high-stakes points. Read the "
    "transcript and the executor's question, then give guidance the executor can act on now:\n"
    "1. The answer or plan, in concrete steps.\n"
    "2. Risks, wrong assumptions, or mistakes you see in what was done or proposed.\n"
    "3. Anything that needs the user's confirmation before acting.\n"
    "Be terse: under 400 words, no preamble, no restating the transcript. Do not do the work yourself "
    "and do not address the user; you are talking to the executor."
)

JEV_QUESTION = {
    "needs_advisor": {
        "type": "noul",
        "instructions": (
            "Would a fast, cheap AI assistant likely get this request wrong or handle it poorly without "
            "help from a stronger reasoning model? Yes if it needs multi-step reasoning, architecture or "
            "debugging judgment, strategy, careful tradeoffs, or touches money, production, credentials, "
            "deletion, or messages sent to other people. No for chit-chat, lookups, simple edits, "
            "status questions, acknowledgements."
        ),
    }
}

_ctx = None
_lock = threading.Lock()
_history: "OrderedDict[str, List[Any]]" = OrderedDict()
_last_session: Optional[str] = None
_MAX_SESSIONS = 64
_senders: Dict[str, str] = {}
_usage: Dict[str, List[float]] = {}


def bind(ctx) -> None:
    global _ctx
    _ctx = ctx


def settings() -> Dict[str, Any]:
    """plugins.entries.advisor.* over DEFAULTS, read per call so config edits apply live."""
    cfg: Dict[str, Any] = {}
    try:
        from hermes_cli.config import load_config_readonly
        node: Any = load_config_readonly() or {}
        for key in ("plugins", "entries", PLUGIN_ID):
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, dict):
            cfg = {k: v for k, v in node.items() if k in DEFAULTS}
    except Exception:  # config IO failure -> defaults
        logger.debug("advisor: config read failed", exc_info=True)
    return {**DEFAULTS, **cfg}


# ---------------------------------------------------------------- transcript
def remember(session_id: str, messages: List[Any], sender_id: str = "") -> None:
    global _last_session
    if not session_id:
        return
    with _lock:
        if sender_id:
            _senders[session_id] = str(sender_id)
        _history[session_id] = messages
        _history.move_to_end(session_id)
        _last_session = session_id
        while len(_history) > _MAX_SESSIONS:
            _history.popitem(last=False)


def recall(session_id: Optional[str]) -> List[Any]:
    with _lock:
        sid = session_id if session_id in _history else _last_session
        return list(_history.get(sid, [])) if sid else []


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") in ("text", "input_text", "output_text"):
                    parts.append(str(block.get("text", "")))
                elif block.get("type") in ("image", "image_url", "input_image"):
                    parts.append("[image]")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def render_transcript(messages: List[Any], limit: int) -> str:
    lines: List[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "?")
        if role == "system":
            continue
        body = _text(m.get("content"))
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                calls.append(f"{fn.get('name', '?')}({str(fn.get('arguments', ''))[:400]})")
            body = (body + "\n" if body else "") + "[tool calls] " + "; ".join(calls)
        if role == "tool" and len(body) > 2000:
            body = body[:1000] + "\n...[truncated]...\n" + body[-800:]
        if body.strip():
            lines.append(f"### {role.upper()}\n{body.strip()}")
    text = "\n\n".join(lines)
    if len(text) > limit:
        text = "...[earlier conversation omitted]...\n" + text[-limit:]
    return text


def permit(session_id: Optional[str], s: Dict[str, Any]) -> Optional[str]:
    """None if this session's sender may consult now, else the reason. Local CLI/desktop
    sessions carry no sender id and are always the operator."""
    with _lock:
        sender = _senders.get(session_id or "", "")
        allowed = [str(u) for u in (s.get("allowed_users") or [])]
        if sender and allowed and sender not in allowed:
            return "advisor is restricted to allowed users"
        cap = int(s.get("max_per_hour") or 0)
        if cap > 0:
            key = sender or "local"
            now = time.time()
            hits = [t for t in _usage.get(key, []) if now - t < 3600]
            if len(hits) >= cap:
                _usage[key] = hits
                return f"advisor hourly limit reached ({cap}/h)"
            hits.append(now)
            _usage[key] = hits
    return None


# ---------------------------------------------------------------- advisor call
def consult(question: str, messages: List[Any], *, purpose: str) -> Dict[str, Any]:
    s = settings()
    transcript = render_transcript(messages, int(s["transcript_chars"]))
    user = (f"<transcript>\n{transcript or '(empty)'}\n</transcript>\n\n"
            f"<executor_question>\n{question.strip() or 'What should I do next?'}\n</executor_question>")
    t0 = time.monotonic()
    text, model = _call_model(s, [{"role": "system", "content": ADVISOR_SYSTEM},
                                  {"role": "user", "content": user}], purpose)
    ms = int((time.monotonic() - t0) * 1000)
    logger.info("advisor: %s answered in %dms (%s)", model, ms, purpose)
    return {"advice": text.strip(), "advisor_model": model, "latency_ms": ms}


def _call_model(s: Dict[str, Any], messages: List[Dict[str, Any]], purpose: str):
    """Host-owned call (Hermes auth, fallback, accounting). Effort rides as reasoning_config."""
    from agent.auxiliary_client import call_llm
    effort = str(s.get("reasoning_effort") or "").strip().lower()
    reasoning = {"enabled": True, "effort": effort} if effort and effort != "none" else None
    resp = call_llm(
        provider=(s.get("provider") or None), model=(s.get("model") or None), messages=messages,
        max_tokens=int(s["max_tokens"]), timeout=float(s["timeout"]), reasoning_config=reasoning,
    )
    choice = resp.choices[0].message
    return (choice.content or ""), getattr(resp, "model", None) or s.get("model") or "main"


# ---------------------------------------------------------------- entry points
def handle_tool(args: Dict[str, Any], session_id: Optional[str] = None, **_: Any) -> str:
    question = str((args or {}).get("question", ""))
    denied = permit(session_id, settings())
    if denied:
        return json.dumps({"error": f"{denied}. Continue on your own judgment."})
    try:
        return json.dumps(consult(question, recall(session_id), purpose="tool"), ensure_ascii=False)
    except Exception as exc:
        logger.warning("advisor: tool call failed: %s", exc)
        return json.dumps({"error": f"advisor unavailable: {exc}. Continue on your own judgment."})


def handle_slash(raw_args: str) -> str:
    question = (raw_args or "").strip()
    if question in ("status", "config"):
        s = settings()
        return (f"advisor: {s['model']} (effort {s['reasoning_effort']}), gate mode {s['mode']}, "
                f"threshold {s['threshold']}, Jev key {'set' if os.environ.get('TYPESAFE_API_KEY') else 'missing'}")
    denied = permit(_last_session, settings())
    if denied:
        return f"advisor: {denied}"
    try:
        r = consult(question, recall(None), purpose="slash")
        return f"🧭 Advisor ({r['advisor_model']}, {r['latency_ms']} ms)\n\n{r['advice']}"
    except Exception as exc:
        return f"advisor unavailable: {exc}"


def jev_score(text: str, timeout: float = 4.0) -> Optional[float]:
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not key:
        return None
    import httpx
    try:
        r = httpx.post("https://api.typesafe.ai/v1/systemone", timeout=timeout,
                       headers={"Authorization": f"Bearer {key}"},
                       json={"model": "jev-latest", "state": text[-6000:], "questions": JEV_QUESTION})
        r.raise_for_status()
        return float(r.json()["answers"]["needs_advisor"]["noul"])
    except Exception as exc:  # cost gate: fail open (no escalation)
        logger.info("advisor: jev gate skipped: %s", exc)
        return None


def on_pre_llm_call(session_id: str = "", user_message: Any = None, conversation_history=None,
                    platform: str = "", sender_id: str = "", **_: Any):
    messages = list(conversation_history or [])
    remember(session_id, messages, sender_id)
    s = settings()
    mode = str(s["mode"]).lower()
    if mode not in ("nudge", "auto"):
        return None
    if s["platforms"] and platform not in s["platforms"]:
        return None
    text = _text(user_message).strip()
    if len(text) < int(s["min_message_chars"]):
        return None
    p = jev_score(text)
    if p is None or p < float(s["threshold"]):
        if p is not None:
            logger.info("advisor: gate p=%.2f below %.2f", p, float(s["threshold"]))
        return None
    denied = permit(session_id, s)
    if denied:
        logger.info("advisor: gate p=%.2f but %s", p, denied)
        return None
    logger.info("advisor: gate p=%.2f -> %s", p, mode)
    if mode == "nudge":
        return {"context": (f"[advisor] This request looks hard or high-stakes (p={p:.2f}). Call the "
                            "`advisor` tool with your question and draft plan before answering or acting.")}
    try:
        r = consult(f"The user just said:\n{text}\n\nHow should I handle this?", messages, purpose="auto")
    except Exception as exc:
        logger.warning("advisor: auto consult failed: %s", exc)
        return None
    return {"context": (f"[advisor guidance from {r['advisor_model']} — escalated automatically, p={p:.2f}. "
                        "Use it; the user did not write this.]\n" + r["advice"])}
