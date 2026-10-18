"""Runs that produce the numbers in the README: replay, CDC sync, crash test, live freshness."""
from __future__ import annotations

import random
import shutil
import tempfile
import threading
import time
from pathlib import Path

import pandas as pd

from . import config as C
from . import lakehouse as L
from .cdc import sync_index
from .index import ChunkIndex
from .landing import LandingWriter
from .stream import stream_events


def replay(landing: Path = C.LANDING, lake: Path = C.LAKE) -> pd.DataFrame:
    """Build the lakehouse from the captured landing zone one micro-batch at a time, timing each step."""
    cat = L.catalog(lake)
    rows = []
    while True:
        t0 = time.perf_counter()
        bronze = L.ingest_bronze(cat, landing, max_commits=1)
        if bronze["commits"] == 0:
            break
        t1 = time.perf_counter()
        silver = L.build_silver(cat)
        t2 = time.perf_counter()
        hourly = L.build_gold_hourly(cat)
        t3 = time.perf_counter()
        revisions = L.build_gold_revisions(cat)
        t4 = time.perf_counter()
        rows.append({"batch": bronze["batch_id"], "events": bronze["rows"], "edits": silver["rows_out"],
                     "duplicates_removed": silver["duplicates"], "article_revisions": revisions["revisions_in"],
                     "scd2_rows_closed": revisions["rows_updated"], "bronze_s": t1 - t0, "silver_s": t2 - t1,
                     "gold_hourly_s": t3 - t2, "gold_scd2_s": t4 - t3})
        print(f"batch {bronze['batch_id']}: {bronze['rows']:,} events in {t4 - t0:.1f}s", flush=True)
    return pd.DataFrame(rows)


def build_index(lake: Path = C.LAKE, index_root: Path = C.INDEX, max_titles: int | None = C.RAG_MAX_TITLES) -> dict:
    cat = L.catalog(lake)
    t0 = time.perf_counter()
    stats = sync_index(cat, ChunkIndex(index_root), index_root, max_titles=max_titles)
    stats["seconds"] = time.perf_counter() - t0
    return stats


def crash_test(landing: Path = C.LANDING, trials: int = 3, files: int = 20, seed: int = C.SEED) -> pd.DataFrame:
    """Kill bronze ingestion at random points and compare where the offset is stored.

    Each trial ingests the same `files` landing files with 1 file per commit, crashing once
    before a commit and once after a commit at random batches, then resumes to the end.
    """
    rng = random.Random(seed)
    sample = L.landing_files(landing)[:files]
    expected = sum(1 for p in sample for line in p.read_text(encoding="utf-8").splitlines() if line)
    rows = []
    for trial in range(trials):
        crash_points = sorted(rng.sample(range(2, files - 2), 2))
        for store in ("file", "snapshot"):
            tmp = Path(tempfile.mkdtemp())
            try:
                land = tmp / "landing"
                land.mkdir()
                for p in sample:
                    shutil.copy(p, land / p.name)
                cat = L.catalog(tmp / "lake")
                L.ingest_bronze(cat, land, files_per_commit=1, max_commits=crash_points[0], offset_store=store)
                for kind in ("before_commit", "after_commit"):
                    try:
                        L.ingest_bronze(cat, land, files_per_commit=1, offset_store=store, crash=kind)
                    except L.SimulatedCrash:
                        pass
                    L.ingest_bronze(cat, land, files_per_commit=1, max_commits=crash_points[1] - crash_points[0], offset_store=store)
                L.ingest_bronze(cat, land, files_per_commit=1, offset_store=store)
                ids = L.tables(cat)["bronze"].scan(selected_fields=["meta_id", "landing_file"]).to_arrow().to_pandas()
                rows.append({"trial": trial, "offset_store": store, "expected_rows": expected, "rows": len(ids),
                             "duplicate_rows": len(ids) - len(ids.drop_duplicates()),
                             "missing_files": len({p.name for p in sample} - set(ids.landing_file))})
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    return pd.DataFrame(rows)


def live(minutes: float, root: Path = C.LIVE, flush_seconds: float = 10.0, loop_seconds: float = 15.0,
         max_titles: int = 300) -> pd.DataFrame:
    """Capture, lakehouse and index running together against the live stream.

    A background thread lands events every `flush_seconds`; the main loop runs one
    pipeline cycle every `loop_seconds`. The index's revision log then gives, for
    every edit, the time it was saved on Wikipedia and the time it became searchable.
    """
    landing, lake, index_root = root / "landing", root / "lake", root / "index"
    writer = LandingWriter(landing, flush_seconds=flush_seconds, flush_events=10_000)
    stop = threading.Event()

    def capture() -> None:
        for message in stream_events(writer.checkpoint()["last_event_id"]):
            writer.add(message)
            if stop.is_set():
                break
        writer.flush()

    thread = threading.Thread(target=capture, daemon=True)
    thread.start()
    cat, index = L.catalog(lake), ChunkIndex(index_root)
    cycles, deadline = [], time.time() + minutes * 60
    while time.time() < deadline:
        start = time.time()
        lake_stats = L.run_once(cat, landing)
        t_lake = time.time()
        cdc = sync_index(cat, index, index_root, max_titles=max_titles)
        t_end = time.time()
        cycles.append({"cycle_start": start, "events": lake_stats["bronze"]["rows"], "lake_s": t_lake - start,
                       "index_s": t_end - t_lake, "revisions_indexed": cdc.get("revisions", 0),
                       "chunks_embedded": cdc.get("embedded", 0)})
        print(f"{time.strftime('%H:%M:%S')} cycle: {lake_stats['bronze']['rows']:,} events, "
              f"{cdc.get('revisions', 0)} revisions indexed in {t_end - start:.1f}s", flush=True)
        time.sleep(max(0.0, loop_seconds - (time.time() - start)))
    stop.set()
    return pd.DataFrame(cycles)
