"""Feed new article edits from the silver table into the chunk index.

For every article seen for the first time, the revision *before* the first
captured edit is indexed as a baseline, so each captured edit is applied as a
real diff. The driver's watermark (the last bronze batch it consumed) is saved
after the index commit; replaying a batch is harmless because the index skips
revisions it already holds.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pyarrow.compute as pc
from pyiceberg.catalog import Catalog
from pyiceberg.expressions import And, EqualTo, GreaterThan

from . import config as C
from .content import chunk_article, fetch_revisions
from .index import ChunkIndex, Revision
from .lakehouse import tables


def _state_path(index_root: Path) -> Path:
    return Path(index_root) / "_cdc_state.json"


def load_state(index_root: Path) -> dict:
    path = _state_path(index_root)
    return json.loads(path.read_text()) if path.exists() else {"last_batch": 0, "tracked": []}


def save_state(index_root: Path, state: dict) -> None:
    path = _state_path(index_root)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, path)


def _epoch(api_timestamp: str) -> float:
    return datetime.fromisoformat(api_timestamp.replace("Z", "+00:00")).timestamp()


def sync_index(cat: Catalog, index: ChunkIndex, index_root: Path, cache: Path = C.CACHE,
               max_titles: int | None = C.RAG_MAX_TITLES, revisions_per_apply: int = 400) -> dict:
    state = load_state(index_root)
    silver = tables(cat)["silver"]
    edits = silver.scan(
        row_filter=And(GreaterThan("bronze_batch", state["last_batch"]), EqualTo("wiki", C.RAG_WIKI),
                       EqualTo("namespace", C.RAG_NAMESPACE)),
        selected_fields=["title", "type", "rev_old", "rev_new", "event_ts", "received_ts", "bronze_batch"],
    ).to_arrow()
    if edits.num_rows == 0:
        return {"edits": 0}
    edits = edits.sort_by([("event_ts", "ascending"), ("rev_new", "ascending")])
    top_batch = pc.max(edits.column("bronze_batch")).as_py()

    tracked = dict.fromkeys(state["tracked"])
    rows = edits.to_pylist()
    for r in rows:  # first-come tracking keeps the replay's API use bounded
        if r["title"] not in tracked and (max_titles is None or len(tracked) < max_titles):
            tracked[r["title"]] = None
    rows = [r for r in rows if r["title"] in tracked]
    known = index.applied(sorted({r["title"] for r in rows}))
    first_edit = {}
    for r in rows:
        first_edit.setdefault(r["title"], r)
    baselines = [r["rev_old"] for t, r in first_edit.items()
                 if t not in known and r["type"] == "edit" and r["rev_old"]]
    texts = fetch_revisions(baselines + [r["rev_new"] for r in rows], cache=cache)

    revisions, unavailable = [], 0
    for rev_id in baselines:
        rec = texts.get(rev_id)
        if rec:
            revisions.append(Revision(rec["title"], rev_id, _epoch(rec["timestamp"]), chunk_article(rec["title"], rec["wikitext"]), "baseline"))
    for r in rows:
        rec = texts.get(r["rev_new"])
        if not rec:
            unavailable += 1  # deleted or hidden since the edit
            continue
        revisions.append(Revision(r["title"], r["rev_new"], r["event_ts"].timestamp(), chunk_article(r["title"], rec["wikitext"]),
                                  "new_page" if r["type"] == "new" else "edit", r["received_ts"].timestamp()))

    totals = {"edits": len(rows), "baselines": len(baselines), "unavailable": unavailable}
    revisions.sort(key=lambda r: r.ts)
    for i in range(0, len(revisions), revisions_per_apply):
        for k, v in index.apply(revisions[i:i + revisions_per_apply]).items():
            totals[k] = totals.get(k, 0) + v
    save_state(index_root, {"last_batch": top_batch, "tracked": list(tracked)})
    totals["tracked_titles"] = len(tracked)
    return totals
