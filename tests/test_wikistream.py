"""Tests on synthetic events with known answers (no network, no captured data)."""
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wikistream import lakehouse as L  # noqa: E402
from wikistream.content import chunk_article, wikitext_sections  # noqa: E402
from wikistream.index import ChunkIndex, Revision  # noqa: E402
from wikistream.landing import LandingWriter  # noqa: E402
from wikistream.stream import StreamMessage, _parse_sse  # noqa: E402

T0 = datetime(2026, 10, 18, 16, 0, tzinfo=timezone.utc)


def event(i: int, title="Alpha", wiki="enwiki", minutes=0, comment="copyedit", rev=None, user="Ann"):
    ts = T0 + timedelta(minutes=minutes, seconds=i)
    rev = rev or 1000 + i
    return {"meta": {"id": f"id-{i}", "dt": ts.isoformat().replace("+00:00", "Z"), "domain": "en.wikipedia.org"},
            "type": "edit", "wiki": wiki, "namespace": 0, "title": title, "user": user, "bot": False, "minor": False,
            "id": i, "revision": {"old": rev - 1, "new": rev}, "length": {"old": 100, "new": 120}, "comment": comment}


def land(root: Path, events: list[dict], per_file: int = 5) -> None:
    writer = LandingWriter(root, flush_seconds=1e9, flush_events=per_file)
    for e in events:
        sent = datetime.fromisoformat(e["meta"]["dt"].replace("Z", "+00:00")).timestamp()
        writer.add(StreamMessage(e["meta"]["id"], e, sent + 1.0))
    writer.flush()


def hash_embed(texts):  # deterministic stand-in for WordLlama: bag of hashed words
    out = np.zeros((len(texts), 256), dtype=np.float32)
    for i, text in enumerate(texts):
        for w in text.lower().split():
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


# ------------------------------------------------------------------ stream and landing
def test_sse_parser_keeps_ids_and_skips_comments():
    lines = [":ok", "", "id: [1]", "event: message", 'data: {"a": 1}', "", "id: [2]", 'data: {"a": 2}', ""]
    assert list(_parse_sse(iter(lines))) == [("[1]", '{"a": 1}'), ("[2]", '{"a": 2}')]


def test_landing_files_and_checkpoint(tmp_path):
    land(tmp_path, [event(i) for i in range(12)], per_file=5)
    files = L.landing_files(tmp_path)
    assert len(files) == 3
    assert json.loads((tmp_path / "_checkpoint.json").read_text())["last_event_id"] == "id-11"
    assert sum(len(p.read_text().splitlines()) for p in files) == 12


# ------------------------------------------------------------------ exactly-once bronze
@pytest.mark.parametrize("offset_store,expected_dupes", [("snapshot", 0), ("file", 4)])
def test_crash_after_commit(tmp_path, offset_store, expected_dupes):
    """A crash after the commit duplicates a batch only when the offset lives outside the commit."""
    land(tmp_path / "landing", [event(i) for i in range(20)], per_file=4)        # 5 files
    cat = L.catalog(tmp_path / "lake")
    L.ingest_bronze(cat, tmp_path / "landing", files_per_commit=1, max_commits=2, offset_store=offset_store)
    with pytest.raises(L.SimulatedCrash):
        L.ingest_bronze(cat, tmp_path / "landing", files_per_commit=1, offset_store=offset_store, crash="after_commit")
    L.ingest_bronze(cat, tmp_path / "landing", files_per_commit=1, offset_store=offset_store)
    ids = L.tables(cat)["bronze"].scan().to_arrow().column("meta_id").to_pylist()
    assert len(set(ids)) == 20 and len(ids) - 20 == expected_dupes


def test_crash_before_commit_loses_nothing(tmp_path):
    land(tmp_path / "landing", [event(i) for i in range(20)], per_file=4)
    cat = L.catalog(tmp_path / "lake")
    with pytest.raises(L.SimulatedCrash):
        L.ingest_bronze(cat, tmp_path / "landing", files_per_commit=2, crash="before_commit")
    assert L.tables(cat)["bronze"].scan().to_arrow().num_rows == 0
    L.ingest_bronze(cat, tmp_path / "landing", files_per_commit=2)
    assert L.tables(cat)["bronze"].scan().to_arrow().num_rows == 20


# ------------------------------------------------------------------ silver and gold
def test_silver_dedups_stream_replays_and_flags_reverts(tmp_path):
    events = [event(0), event(1, comment="Reverted edits by X (talk) to last revision by Y"),
              event(2, comment="Undid revision 123 by Z"), event(3, comment="Adding a reverse-chronological list")]
    land(tmp_path / "landing", events + events[:2], per_file=3)          # the reconnect replayed two events
    cat = L.catalog(tmp_path / "lake")
    L.ingest_bronze(cat, tmp_path / "landing")
    out = L.build_silver(cat)
    assert out["duplicates"] == 2
    silver = L.tables(cat)["silver"].scan().to_arrow().sort_by("meta_id").to_pydict()
    assert silver["is_revert"] == [False, True, True, False]
    assert L.build_silver(cat)["rows_in"] == 0                           # watermark: nothing new on re-run


def test_scd2_intervals_are_contiguous_with_out_of_order_events():
    ts = [T0 + timedelta(minutes=m) for m in (0, 10, 5)]                   # the third revision arrives late
    history = pa.table({"rev_id": [1, 3, 2], "title": ["A"] * 3, "parent_rev_id": [0, 2, 1], "valid_from": ts,
                        "valid_to": [None] * 3, "is_current": [True] * 3, "user": ["u"] * 3, "is_bot": [False] * 3,
                        "is_revert": [False] * 3, "len_new": [1] * 3, "byte_delta": [1] * 3},
                       schema=L.REVISIONS)
    rows = L.scd2_rows(history).sort_by("valid_from").to_pydict()
    assert rows["rev_id"] == [1, 2, 3]
    assert rows["valid_to"][:2] == rows["valid_from"][1:] and rows["valid_to"][2] is None
    assert rows["is_current"] == [False, False, True]


def test_gold_revisions_incremental(tmp_path):
    land(tmp_path / "landing", [event(0, rev=10), event(1, title="Beta", rev=20)], per_file=2)
    cat = L.catalog(tmp_path / "lake")
    L.run_once(cat, tmp_path / "landing")
    land(tmp_path / "landing", [event(5, rev=11, minutes=30)], per_file=1)
    L.run_once(cat, tmp_path / "landing")
    rows = L.tables(cat)["revisions"].scan().to_arrow().sort_by("rev_id").to_pydict()
    assert rows["rev_id"] == [10, 11, 20] and rows["is_current"] == [False, True, True]
    hourly = L.tables(cat)["hourly"].scan().to_arrow().to_pydict()
    assert sum(hourly["edits"]) == 3


# ------------------------------------------------------------------ chunking and the CDC index
WIKITEXT = """'''Alpha''' is a town in [[Example County]].<ref>Census 2020</ref> It has a river, a market and a school for children.

It was founded in 1820 by settlers who built a mill on the river and a road to the coast.

== History ==
The town grew quickly after the railway arrived in 1870, bringing trade and new residents to the area.

== References ==
{{reflist}} Some reference text that should never be indexed because it is a list of sources.
"""


def test_wikitext_to_chunks():
    sections = dict(wikitext_sections(WIKITEXT))
    assert "References" not in sections and "Census" not in sections["Introduction"]
    chunks = chunk_article("Alpha", WIKITEXT)
    assert [c.section for c in chunks] == ["Introduction", "Introduction", "History"]
    edited = chunk_article("Alpha", WIKITEXT.replace("in 1870", "in 1871"))
    assert len({c.key for c in edited} - {c.key for c in chunks}) == 1     # one paragraph changed, one chunk


def test_index_cdc_and_as_of_queries(tmp_path):
    idx = ChunkIndex(tmp_path / "index", embed=hash_embed)
    v1 = chunk_article("Alpha", WIKITEXT)
    v2 = chunk_article("Alpha", WIKITEXT.replace("railway arrived in 1870", "railway and canal arrived in 1871"))
    first = idx.apply([Revision("Alpha", 1, 100.0, v1, "baseline")])
    second = idx.apply([Revision("Alpha", 2, 200.0, v2)])
    assert first["embedded"] == 3 and second["embedded"] == 1 and second["closed"] == 1
    assert "1871" in idx.search("when did the railway arrive", k=1)[0]["text"]
    assert "1870" in idx.search("when did the railway arrive", k=1, as_of=150.0)[0]["text"]
    replay = idx.apply([Revision("Alpha", 1, 100.0, v1, "baseline"), Revision("Alpha", 2, 200.0, v2)])
    assert replay["revisions"] == 0 and replay["already_applied"] == 2
    current = idx.current(["Alpha"]).column("chunk_key").to_pylist()
    assert sorted(current) == sorted(c.key for c in v2)
