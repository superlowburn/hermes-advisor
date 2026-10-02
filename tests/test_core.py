import importlib.util, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("advisor_pkg", ROOT / "__init__.py",
                                              submodule_search_locations=[str(ROOT)])
pkg = importlib.util.module_from_spec(spec); sys.modules["advisor_pkg"] = pkg; spec.loader.exec_module(pkg)
core = sys.modules["advisor_pkg.core"]


def test_transcript_keeps_tail_and_tool_calls():
    msgs = [{"role": "system", "content": "SYS"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "terminal", "arguments": "{\"command\":\"ls\"}"}}]},
            {"role": "tool", "content": "x" * 5000}]
    t = core.render_transcript(msgs, 100000)
    assert "SYS" not in t and "terminal(" in t and "[truncated]" in t
    assert len(core.render_transcript(msgs, 300)) < 400


def test_gate_off_and_short_messages_never_call_jev(monkeypatch):
    calls = []
    monkeypatch.setattr(core, "jev_score", lambda text: calls.append(text) or 0.99)
    monkeypatch.setattr(core, "settings", lambda: {**core.DEFAULTS, "mode": "off"})
    assert core.on_pre_llm_call("s1", "please redesign the whole billing system", []) is None
    monkeypatch.setattr(core, "settings", lambda: {**core.DEFAULTS, "mode": "nudge"})
    assert core.on_pre_llm_call("s1", "ok thanks", []) is None
    assert calls == []


def test_nudge_and_auto(monkeypatch):
    monkeypatch.setattr(core, "jev_score", lambda text: 0.9)
    monkeypatch.setattr(core, "settings", lambda: {**core.DEFAULTS, "mode": "nudge"})
    out = core.on_pre_llm_call("s2", "should we migrate prod database to postgres tonight?", [{"role": "user", "content": "hi"}])
    assert "advisor" in out["context"]
    monkeypatch.setattr(core, "settings", lambda: {**core.DEFAULTS, "mode": "auto"})
    monkeypatch.setattr(core, "_call_model", lambda s, m, p: ("Do a dry run first.", "opus"))
    out = core.on_pre_llm_call("s2", "should we migrate prod database to postgres tonight?", [])
    assert "Do a dry run first." in out["context"]


def test_gate_fails_open(monkeypatch):
    monkeypatch.setattr(core, "jev_score", lambda text: None)
    monkeypatch.setattr(core, "settings", lambda: {**core.DEFAULTS, "mode": "auto"})
    assert core.on_pre_llm_call("s3", "a long and quite complicated request here", []) is None


def test_tool_uses_session_history_and_reports_errors(monkeypatch):
    seen = {}
    core.remember("s4", [{"role": "user", "content": "context-marker"}])
    def fake(s, m, p):
        seen["user"] = m[1]["content"]; return ("advice", "opus")
    monkeypatch.setattr(core, "_call_model", fake)
    import json
    r = json.loads(core.handle_tool({"question": "q"}, session_id="s4"))
    assert r["advice"] == "advice" and "context-marker" in seen["user"]
    monkeypatch.setattr(core, "_call_model", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    assert "error" in json.loads(core.handle_tool({"question": "q"}, session_id="s4"))
