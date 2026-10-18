"""A RAG chunk index kept in step with Wikipedia by change data capture.

Each revision is chunked and compared with the chunks currently live for that
article. Unchanged chunks are left alone, chunks that disappeared are closed
(`valid_to` set), and only new chunks are embedded and inserted. Because rows
are closed rather than deleted, the same table answers two questions:

* "what does the article say now?"      -> valid_to IS NULL
* "what did it say at time T?" (as-of)  -> valid_from <= T < valid_to

All changes for a micro-batch land in one LanceDB `merge_insert`, keyed on a
row id, and a revision already in the log is skipped, so replaying a batch
after a crash changes nothing.
"""
from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa

from . import config as C
from .content import Chunk

Embedder = Callable[[list[str]], np.ndarray]

CHUNKS = pa.schema([
    ("row_id", pa.string()), ("chunk_key", pa.string()), ("title", pa.string()), ("section", pa.string()),
    ("text", pa.string()), ("vector", pa.list_(pa.float32(), C.EMBED_DIM)), ("rev_from", pa.int64()),
    ("rev_to", pa.int64()), ("valid_from", pa.float64()), ("valid_to", pa.float64()), ("indexed_at", pa.float64()),
])
LOG = pa.schema([
    ("rev_id", pa.int64()), ("title", pa.string()), ("kind", pa.string()), ("ts", pa.float64()),
    ("chunks", pa.int64()), ("added", pa.int64()), ("removed", pa.int64()), ("kept", pa.int64()),
    ("indexed_at", pa.float64()), ("received_ts", pa.float64()),
])


def wordllama_embedder() -> Embedder:
    from wordllama import WordLlama

    model = WordLlama.load(dim=C.EMBED_DIM)
    return lambda texts: np.asarray(model.embed(texts, norm=True), dtype=np.float32)


@dataclass
class Revision:
    title: str
    rev_id: int
    ts: float                    # when the revision was saved (epoch seconds)
    chunks: list[Chunk]
    kind: str = "edit"           # "baseline" = the version before the first edit we saw
    received_ts: float | None = None


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class ChunkIndex:
    def __init__(self, path: Path, embed: Embedder | None = None):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(path))
        self.embed = embed or wordllama_embedder()
        self.chunks = self.db.create_table("chunks", schema=CHUNKS, exist_ok=True)
        self.log = self.db.create_table("revision_log", schema=LOG, exist_ok=True)

    # ---------------------------------------------------------------- reads
    def _rows(self, where: str, columns: list[str] | None = None) -> pa.Table:
        query = self.chunks.search().where(where).limit(10_000_000)
        return (query.select(columns) if columns else query).to_arrow()

    def current(self, titles: list[str]) -> pa.Table:
        parts = [self._rows(f"valid_to IS NULL AND title IN ({', '.join(_quote(t) for t in titles[i:i + 200])})")
                 for i in range(0, len(titles), 200)]
        return pa.concat_tables(parts) if parts else CHUNKS.empty_table()

    def applied(self, titles: list[str]) -> dict[str, tuple[set[int], float]]:
        """Per title: revision ids already indexed and the newest revision time."""
        out: dict[str, tuple[set[int], float]] = {}
        for i in range(0, len(titles), 200):
            where = f"title IN ({', '.join(_quote(t) for t in titles[i:i + 200])})"
            rows = self.log.search().where(where).select(["title", "rev_id", "ts"]).limit(10_000_000).to_arrow().to_pylist()
            for r in rows:
                revs, newest = out.get(r["title"], (set(), 0.0))
                revs.add(r["rev_id"])
                out[r["title"]] = (revs, max(newest, r["ts"]))
            # the chunk rows also carry revision times; they matter if the log write was lost in a crash
            for r in self._rows(where, ["title", "valid_from", "valid_to"]).to_pylist():
                revs, newest = out.get(r["title"], (set(), 0.0))
                out[r["title"]] = (revs, max(newest, r["valid_from"], r["valid_to"] or 0.0))
        return out

    def search(self, query: str, k: int = 5, as_of: float | None = None) -> list[dict]:
        where = "valid_to IS NULL" if as_of is None else f"valid_from <= {as_of} AND (valid_to IS NULL OR valid_to > {as_of})"
        vector = self.embed([query])[0]
        return (self.chunks.search(vector).where(where, prefilter=True).limit(k)
                .select(["chunk_key", "title", "section", "text", "rev_from", "valid_from", "valid_to"]).to_list())

    # ---------------------------------------------------------------- writes
    def apply(self, revisions: list[Revision]) -> dict:
        """Apply revisions (any order, any titles) as one atomic merge into the index."""
        titles = sorted({r.title for r in revisions})
        done = self.applied(titles)
        live: dict[str, dict[str, dict]] = defaultdict(dict)       # title -> chunk_key -> row
        for row in self.current(titles).to_pylist():
            live[row["title"]][row["chunk_key"]] = row
        changed: dict[str, dict] = {}                              # row_id -> row to upsert
        log_rows, skipped = [], {"already_applied": 0, "out_of_order": 0}
        now = time.time()
        for rev in sorted(revisions, key=lambda r: (r.ts, r.rev_id)):
            seen, newest = done.get(rev.title, (set(), 0.0))
            if rev.rev_id in seen:
                skipped["already_applied"] += 1
                continue
            if rev.ts < newest:          # a late event older than what the index already shows
                skipped["out_of_order"] += 1
                continue
            state = live[rev.title]
            new = {c.key: c for c in rev.chunks}
            removed, added = state.keys() - new.keys(), new.keys() - state.keys()
            for key in removed:
                row = {**state.pop(key), "valid_to": rev.ts, "rev_to": rev.rev_id}
                changed[row["row_id"]] = row
            for key in added:
                c = new[key]
                row = {"row_id": f"{rev.title}|{key}|{rev.rev_id}", "chunk_key": key, "title": rev.title,
                       "section": c.section, "text": c.text, "vector": None, "rev_from": rev.rev_id, "rev_to": None,
                       "valid_from": rev.ts, "valid_to": None, "indexed_at": now}
                state[key] = row
                changed[row["row_id"]] = row
            done[rev.title] = (seen | {rev.rev_id}, max(newest, rev.ts))
            log_rows.append({"rev_id": rev.rev_id, "title": rev.title, "kind": rev.kind, "ts": rev.ts,
                             "chunks": len(new), "added": len(added), "removed": len(removed),
                             "kept": len(new) - len(added), "indexed_at": now, "received_ts": rev.received_ts})
        to_embed = [row for row in changed.values() if row["vector"] is None]
        if to_embed:
            vectors = self.embed([f"{r['title']} - {r['section']}\n{r['text']}" for r in to_embed])
            for row, vector in zip(to_embed, vectors):
                row["vector"] = vector.tolist()
        if changed:
            batch = pa.Table.from_pylist(list(changed.values()), schema=CHUNKS)
            self.chunks.merge_insert("row_id").when_matched_update_all().when_not_matched_insert_all().execute(batch)
        if log_rows:  # after the chunks; a crash in between replays the batch, and the newest-time check makes that a no-op
            self.log.add(pa.Table.from_pylist(log_rows, schema=LOG))
        return {"revisions": len(log_rows), "embedded": len(to_embed), "closed": len(changed) - len(to_embed), **skipped}
