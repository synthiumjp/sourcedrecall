import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SOURCEDRECALL_BROWSER_PORT", "0")


import pytest

# Set for their whole process by the hooks, CLI, importers and MCP server, as
# they should be: session_start turns the conflict model off (RG_NLI=0) and
# main() marks chats with an assistant (RGX_ASSISTANT_CHAT=1).
_PROCESS_FLAGS = ("RG_NLI", "RGX_ASSISTANT_CHAT")


@pytest.fixture(autouse=True)
def _process_flags_restored():
    """2026-11-13 (launch export): a test that ran a session start left
    RG_NLI=0 set, and in the product repo, which runs server/ before tools/,
    the dogfood test of conflict asks failed. Each server test now leaves
    these as it found them. (monkeypatch.delenv of an unset variable records
    nothing to restore, so it is done by hand.)"""
    saved = {k: os.environ.get(k) for k in _PROCESS_FLAGS}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
