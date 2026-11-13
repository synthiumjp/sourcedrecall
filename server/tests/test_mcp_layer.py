"""The MCP wiring: the nine memory_* tools and history_search registered, the diagnostics only
when asked for."""

import asyncio


# 2026-10-12 (review 3.11): the diagnostics are hidden by default
EXTRA_TOOLS = {"memory_dynamics", "memory_quarantine", "memory_status", "memory_export"}
MEMORY_TOOLS = {"memory_conflicts", "memory_recall", "memory_context", "memory_correct",
                 "memory_read", "memory_store", "memory_forget", "memory_confirm",
                 "memory_check", "history_search"}


def _server(monkeypatch, tmp_path, all_tools=False):
    import importlib
    monkeypatch.setenv("SOURCEDRECALL_STATE", str(tmp_path / "state"))
    monkeypatch.setenv("SOURCEDRECALL_BROWSER_PORT", "0")
    if all_tools:
        monkeypatch.setenv("SOURCEDRECALL_ALL_TOOLS", "1")
    else:
        monkeypatch.delenv("SOURCEDRECALL_ALL_TOOLS", raising=False)
    import sourcedrecall.mcp_server as srv
    return importlib.reload(srv)


def test_a_new_user_sees_only_the_memory_tools(tmp_path, monkeypatch):
    srv = _server(monkeypatch, tmp_path)
    tools = asyncio.run(srv.mcp.list_tools())
    assert {t.name for t in tools} == MEMORY_TOOLS


def test_the_diagnostics_only_when_asked_for(tmp_path, monkeypatch):
    srv = _server(monkeypatch, tmp_path, all_tools=True)
    tools = asyncio.run(srv.mcp.list_tools())
    assert {t.name for t in tools} == MEMORY_TOOLS | EXTRA_TOOLS
    assert all(t.description for t in tools)


def test_tool_descriptions_are_short(tmp_path, monkeypatch):
    srv = _server(monkeypatch, tmp_path)
    for t in asyncio.run(srv.mcp.list_tools()):
        assert len(t.description) < 450, t.name


def test_the_client_is_told_when_to_use_the_memory(tmp_path, monkeypatch):
    # 2026-11-11 (review): server instructions, so Claude reaches for the
    # history without the user typing /recall
    srv = _server(monkeypatch, tmp_path)
    text = srv.mcp._mcp_server.instructions or ""
    assert "history_search" in text and "memory_recall" in text
