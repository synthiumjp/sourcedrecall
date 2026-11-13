"""Review 2026-10-12 (3.2): the host is told which tools only read and which
change or delete the memory, so it can ask the user first."""
import asyncio

import pytest


def test_tools_say_whether_they_change_the_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("RG_MEMORY_DIR", str(tmp_path))
    monkeypatch.delenv("SOURCEDRECALL_ALL_TOOLS", raising=False)
    pytest.importorskip("mcp.types").ToolAnnotations
    import sourcedrecall.mcp_server as m
    tools = {t.name: t.annotations for t in asyncio.run(m.mcp.list_tools())}
    for name in ("memory_recall", "memory_check", "memory_context", "memory_read"):
        assert tools[name].readOnlyHint is True
    # 2026-10-28 (review H5): ingest and confirm too
    for name in ("memory_forget", "memory_correct", "memory_store", "memory_confirm"):
        assert tools[name].readOnlyHint is False and tools[name].destructiveHint is True
