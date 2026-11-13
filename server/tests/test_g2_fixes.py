"""2026-11-13, found by JP's G2 check (his own Claude Code sessions): text he
pasted was stored as his own words, the conversation a search was made from
came first in its results, and a first search after an import embedded every
passage (minutes, and the peak memory stayed with the server)."""
import json
import os

import numpy as np
import pytest

import sourcedrecall.profile_memory as pm_mod  # noqa: F401  (sets the import path)
from sourcedrecall import claude_hooks as H
from sourcedrecall import coding as C

PASTE = ('\n\n<pasted_content id="af6a">\nThe other chat is trying the wrong address. '
         'The Mac is up: this session has been reaching it all night. We decided to '
         'use the LAN address from now on.\n</pasted_content id="af6a">\n')


# -- pasted text ------------------------------------------------------------

def test_a_tagged_paste_is_someone_elses_text():
    from prose import _paste_span, _drop_pasted
    sp = _paste_span(PASTE, "Dana Cole")
    assert isinstance(sp, tuple)
    assert _drop_pasted(PASTE, "Dana Cole").strip() == ""
    mixed = "can you check this?" + PASTE + "I think it's wrong."
    left = _drop_pasted(mixed, "Dana Cole")
    assert "can you check this?" in left and "I think it's wrong." in left
    assert "Mac is up" not in left


def test_a_tagged_paste_the_user_says_is_theirs_stays_theirs():
    from prose import _paste_span
    assert _paste_span("here's my draft, I'm sending it tonight:" + PASTE, "Dana Cole") == "own"


def test_the_quote_marks_the_paste_and_drops_the_tags():
    from memory_api import _quote_own
    q = _quote_own("can you check this?" + PASTE, "Dana Cole", query="Where is the Mac?")
    assert "[pasted in by Dana Cole" in q and "<pasted_content" not in q
    assert '"can you check this?"' in q


def _say(role, text, ts="2026-10-01T10:00:02Z"):
    return json.dumps(dict(type=role, timestamp=ts, cwd="/w/proj",
                           message={"role": role, "content": text if role == "user"
                                    else [{"type": "text", "text": text}]})) + "\n"


def test_a_paste_is_not_the_users_decision_in_the_coding_records(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(_say("user", PASTE) + _say("assistant", "I'll try the LAN address.")
                 + _say("user", "We decided to keep the old tests for now."))
    recs = C.session_records(str(p), H._classify)
    decisions = [r["text"] for r in recs if r["kind"] == "decision"]
    assert not any("LAN address" in d for d in decisions if d.startswith("the user said"))
    assert any("keep the old tests" in d for d in decisions)
    text = "\n".join(r["text"] for r in recs if r["kind"] == "excerpt")
    assert "PASTED BY THE USER (not their words): The other chat" in text
    assert "USER: The other chat" not in text and "<pasted_content" not in text
    sess = next(r["text"] for r in recs if r["kind"] == "session")
    assert "Mac is up" not in sess


# -- the conversation the search comes from ---------------------------------

def _hit(conv, i):
    return {"id": f"c{conv}{i}", "kind": "excerpt", "text": f"{conv} {i}", "date": "2026-11-01",
            "scope": "/w", "conv": f"claude-code:{conv}"}


def test_this_sessions_passages_come_after_others_marked(monkeypatch):
    pm = pm_mod
    ranked = [_hit("now", 0), _hit("now", 1), _hit("now", 2)] + [_hit("old", i) for i in range(20)]
    monkeypatch.setattr(pm, "_episodes_for", lambda q, k=6, scope=None, kinds=None: ranked[:k])
    monkeypatch.setattr(pm, "_load_episodes", lambda: [{"conv": h["conv"], "scope": "/w"} for h in ranked])
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "now")
    out = pm.profile_history("anything", all_projects=True)
    res = out["results"]
    assert len(res) == 10
    assert [r["session"] for r in res[:8]] == ["claude-code:old"] * 8
    assert [r.get("this_session") for r in res[8:]] == [True, True]
    assert "this_session" in out["note"]
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID")
    plain = pm.profile_history("anything", all_projects=True)["results"]
    assert [r["session"] for r in plain[:3]] == ["claude-code:now"] * 3
    assert not any(r.get("this_session") for r in plain)


# -- passages embedded when stored -------------------------------------------

class _Bi:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kw):
        texts = list(texts)
        self.calls.append(texts)
        return np.stack([np.full(384, (len(t) % 97) / 97, np.float32) for t in texts])


def test_passages_are_embedded_when_stored(tmp_path, monkeypatch):
    import retrieve_v3 as RV
    monkeypatch.setenv("SOURCEDRECALL_STATE", str(tmp_path))
    monkeypatch.setenv("RG_MEMORY_DIR", str(tmp_path / "conversations"))
    os.makedirs(tmp_path / "conversations")
    bi = _Bi()
    monkeypatch.setattr(RV, "_models", lambda: (bi, None))
    monkeypatch.setattr(RV, "_EMB_CACHE", {})
    pm_mod.save_episodes("claude-code:s1", "2026-11-01", "/w",
                         [{"kind": "excerpt", "text": "USER: why does the build fail"},
                          {"kind": "decision", "text": "the user said: let's pin it"}])
    assert bi.calls == [[": USER: why does the build fail"]]
    stored = RV._load_emb(str(tmp_path / "conversations" / "embeddings.bin"))
    assert RV._emb_key(": USER: why does the build fail") in stored


def test_the_server_reads_vectors_a_worker_appended(tmp_path, monkeypatch):
    import retrieve_v3 as RV
    monkeypatch.setenv("RG_MEMORY_DIR", str(tmp_path))
    monkeypatch.setattr(RV, "_EMB_CACHE", {})
    server = _Bi()
    RV._encode_cached(server, ["first passage"])
    server_cache = RV._EMB_CACHE
    monkeypatch.setattr(RV, "_EMB_CACHE", {})          # a worker: its own process
    RV._encode_cached(_Bi(), ["second passage"])
    monkeypatch.setattr(RV, "_EMB_CACHE", server_cache)
    v = RV._encode_cached(server, ["second passage", "first passage"])
    assert v.shape == (2, 384)
    assert server.calls == [["first passage"]]         # nothing embedded again


def test_the_hook_keeps_the_paste_tags():
    line = {"type": "user", "message": {"role": "user", "content": "check this" + PASTE}}
    role, text = H._classify(line)
    assert role == "user" and '<pasted_content id="af6a">' in text


def test_a_paste_cut_short_is_still_pasted():
    from prose import _drop_pasted
    cut = PASTE[:PASTE.index("</pasted_content")]
    assert "Mac is up" not in _drop_pasted("check this" + cut, "Dana Cole")
    own, pasted = C.split_pasted("check this" + cut)
    assert own == "check this" and "Mac is up" in pasted[0]


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("RG_MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("RG_NLI", "0")
    monkeypatch.delenv("SOURCEDRECALL_OWNER", raising=False)
    pm = pm_mod

    def reset():
        pm._state.update({"mem": None, "audit_pass": None, "needs_reload": False,
                          "uncached_turns": None, "transcripts": None})
        pm._ingest_state.update({"extractor": None, "owner": None})
    reset()
    yield pm
    reset()


def test_a_pasted_claim_is_not_stored_as_the_users(fresh):
    pytest.importorskip("stanza", reason="ingest needs the rgx parser (stanza)")
    pm = fresh
    pasted = '<pasted_content id="b1">\nI live in Leeds and I work as a nurse.\n</pasted_content id="b1">'
    pm.profile_ingest([{"role": "user", "content": pasted},
                       {"role": "user", "content": "I live in Fitzroy."}],
                      conversation_id="p1", owner_name="Dana Cole")
    texts = " ".join(f.get("text") or "" for f in pm.profile_recall("Where do I live?").get("ranked") or [])
    assert "Fitzroy" in texts and "Leeds" not in texts.split("[pasted in")[0]
    nurse = pm.profile_recall("What is my job?").get("ranked") or []
    assert not any("nurse" in (f.get("text") or "") and "pasted in" not in (f.get("text") or "")
                   for f in nurse)


def test_a_paste_stored_untagged_before_is_replaced_by_the_tagged_message(fresh):
    pytest.importorskip("stanza", reason="ingest needs the rgx parser (stanza)")
    import json as _json
    pm = fresh
    tagged = '<pasted_content id="c2">\nI live in Leeds.\n</pasted_content id="c2">'
    old = "I live in Leeds."                     # how the hooks stored it before
    assert pm._untagged_copy(tagged) == old
    pm.profile_ingest([{"role": "user", "content": old}], conversation_id="r1",
                      owner_name="Dana Cole")
    before = " ".join(f.get("text") or "" for f in pm.profile_recall("Where do I live?").get("ranked") or [])
    assert "Leeds" in before
    pm.profile_ingest([{"role": "user", "content": tagged}], conversation_id="r1",
                      owner_name="Dana Cole")
    conv = next(c for c in _json.load(open(pm._conversations_path())) if c["uuid"] == "r1")
    assert [m["text"] for m in conv["chat_messages"]] == [tagged]
    pm._state.update({"mem": None, "needs_reload": True})
    after = pm.profile_recall("Where do I live?").get("ranked") or []
    assert not any("Leeds" in (f.get("text") or "") and "pasted in" not in (f.get("text") or "")
                   for f in after)


def test_an_untagged_copy_stored_beside_the_tagged_message_is_dropped(fresh):
    import json as _json
    pm = fresh
    tagged = '<pasted_content id="c3">\nThe Mac is up again.\n</pasted_content id="c3">'
    pm.profile_ingest([{"role": "user", "content": "The Mac is up again."},
                       {"role": "user", "content": tagged},
                       {"role": "user", "content": "thanks"}], conversation_id="r2")
    pm.profile_ingest([{"role": "user", "content": tagged}], conversation_id="r2")
    conv = next(c for c in _json.load(open(pm._conversations_path())) if c["uuid"] == "r2")
    assert [m["text"] for m in conv["chat_messages"]] == [tagged, "thanks"]
