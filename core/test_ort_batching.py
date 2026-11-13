"""2026-11-13: one MCP server reached 14.8 GB embedding long coding passages
256 at a time (each batch padded to 512 tokens) and took a 16 GB WSL box
down with it. A batch now holds at most ort_models.BATCH_TOKENS padded
tokens, cut in order, so short texts batch exactly as before."""
import os
import subprocess
import sys

import numpy as np
import pytest

import ort_models as OM

needs_model = pytest.mark.skipif(not OM.available(OM.EMBEDDER),
                                 reason="the embedder is not installed")

LONG = " ".join(f"step{i} failed with exit code {i % 7} in module_{i}.py" for i in range(80))


class _Recorder:
    def __init__(self, sess):
        self.sess, self.shapes = sess, []

    def run(self, out, feed):
        self.shapes.append(feed["input_ids"].shape)
        return self.sess.run(out, feed)

    def get_inputs(self):
        return self.sess.get_inputs()


@needs_model
def test_long_texts_are_batched_under_the_token_cap():
    enc = OM.Encoder(OM.EMBEDDER)
    enc.sess = rec = _Recorder(enc.sess)
    texts = [f"{i} {LONG}" for i in range(70)]
    v = enc.encode(texts, batch_size=256)
    assert v.shape == (70, 384)
    assert all(b * n <= OM.BATCH_TOKENS for b, n in rec.shapes)
    assert sum(b for b, _ in rec.shapes) == 70


@needs_model
def test_short_texts_batch_as_before_and_vectors_do_not_change():
    enc = OM.Encoder(OM.EMBEDDER)
    enc.sess = rec = _Recorder(enc.sess)
    texts = [f"I moved to Brunswick in {1990 + i}." for i in range(256)]
    v = enc.encode(texts, batch_size=256)
    assert [b for b, _ in rec.shapes] == [256]
    one = np.stack([enc.encode([t])[0] for t in texts[:5]])
    assert np.allclose(v[:5], one, atol=1e-5)


@needs_model
def test_mixed_lengths_give_the_same_vectors_as_one_at_a_time():
    enc = OM.Encoder(OM.EMBEDDER)
    texts = [LONG if i % 3 == 0 else f"short text {i}" for i in range(40)]
    v = enc.encode(texts, batch_size=256)
    one = np.stack([enc.encode([t])[0] for t in texts])
    assert np.allclose(v, one, atol=1e-5)


@needs_model
def test_embedding_many_long_passages_stays_small():
    code = (
        "import resource, sys, ort_models as OM\n"
        f"LONG = {LONG!r}\n"
        "OM.Encoder(OM.EMBEDDER).encode([f'{i} ' + LONG for i in range(600)], batch_size=256)\n"
        "r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        "print(r if sys.platform == 'darwin' else r * 1024)\n")
    here = os.path.dirname(os.path.abspath(__file__))
    out = subprocess.run([sys.executable, "-c", code], cwd=here, capture_output=True,
                         text=True, env={**os.environ, "PYTHONPATH": here}, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    peak = int(out.stdout.split()[-1])
    assert peak < 1.5 * 1024 ** 3, f"peak {peak / 1e9:.1f} GB"
