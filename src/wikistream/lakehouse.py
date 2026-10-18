r"""Bronze, silver and gold Apache Iceberg tables, built incrementally.

    landing/*.jsonl --(exactly-once)--> bronze.recentchange --> silver.edits --> gold.wiki_hourly
                                                                            \--> gold.article_revisions (SCD2)

Every step reads only what is new since its last commit, and stores its
watermark *inside* the Iceberg commit (the snapshot summary) together with the
data it wrote. A crash before the commit leaves the table untouched; a crash
after it leaves the watermark already moved. Either way a re-run neither loses
nor repeats rows. That is the same idea Spark and Flink use for exactly-once
Iceberg sinks, done here with plain PyIceberg.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pyarrow as pa
from pyiceberg.catalog import Catalog
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.exceptions import NoSuchTableError
from pyiceberg.expressions import GreaterThan, GreaterThanOrEqual, In
from pyiceberg.table import Table
from pyiceberg.transforms import HourTransform

from . import config as C

TS = pa.timestamp("us", tz="UTC")

BRONZE = pa.schema([
    pa.field("meta_id", pa.string(), nullable=False), ("event_ts", TS), ("received_ts", TS), ("wiki", pa.string()),
    ("type", pa.string()), ("namespace", pa.int32()), ("title", pa.string()), ("user", pa.string()), ("bot", pa.bool_()),
    ("minor", pa.bool_()), ("rc_id", pa.int64()), ("rev_old", pa.int64()), ("rev_new", pa.int64()), ("len_old", pa.int64()),
    ("len_new", pa.int64()), ("comment", pa.string()), ("log_type", pa.string()), ("raw", pa.string()),
    ("landing_file", pa.string()), ("batch_id", pa.int64()),
])
SILVER = pa.schema([
    pa.field("meta_id", pa.string(), nullable=False), ("event_ts", TS), ("received_ts", TS), ("wiki", pa.string()),
    ("namespace", pa.int32()), ("title", pa.string()), ("type", pa.string()), ("user", pa.string()), ("is_bot", pa.bool_()),
    ("is_anon", pa.bool_()), ("is_minor", pa.bool_()), ("is_revert", pa.bool_()), ("rev_old", pa.int64()),
    ("rev_new", pa.int64()), ("len_old", pa.int64()), ("len_new", pa.int64()), ("byte_delta", pa.int64()),
    ("comment", pa.string()), ("lag_seconds", pa.float64()), ("bronze_batch", pa.int64()),
])
HOURLY = pa.schema([
    ("hour", TS), ("wiki", pa.string()), ("edits", pa.int64()), ("new_pages", pa.int64()), ("bot_edits", pa.int64()),
    ("anon_edits", pa.int64()), ("reverts", pa.int64()), ("pages", pa.int64()), ("editors", pa.int64()),
    ("bytes_added", pa.int64()), ("bytes_removed", pa.int64()),
])
REVISIONS = pa.schema([
    pa.field("rev_id", pa.int64(), nullable=False), ("title", pa.string()), ("parent_rev_id", pa.int64()),
    ("valid_from", TS), ("valid_to", TS), ("is_current", pa.bool_()), ("user", pa.string()), ("is_bot", pa.bool_()),
    ("is_revert", pa.bool_()), ("len_new", pa.int64()), ("byte_delta", pa.int64()),
])

# Edit summaries MediaWiki writes for rollback ("Reverted edits by ...") and undo ("Undid revision ..."),
# plus the abbreviations editors type by hand. English only, so revert analytics use enwiki.
REVERT_PATTERN = r"(?i)(\brevert(ed|ing)?\b|\bundid revision\b|\bundo\b|^\s*rvv?\b|\brollback\b|\brestored revision\b)"
ANON_PATTERN = r"^(~\d{4}-|\d{1,3}(\.\d{1,3}){3}$|[0-9a-fA-F:]+:[0-9a-fA-F:]*$)"   # temporary accounts and IPs


# ------------------------------------------------------------------ catalog and tables
def catalog(root: Path = C.LAKE) -> Catalog:
    root.mkdir(parents=True, exist_ok=True)
    cat = SqlCatalog("wikistream", uri=f"sqlite:///{root.as_posix()}/catalog.db", warehouse=root.joinpath("warehouse").as_posix())
    if (C.NAMESPACE,) not in cat.list_namespaces():
        cat.create_namespace(C.NAMESPACE)
    return cat


def _table(cat: Catalog, name: str, schema: pa.Schema, hour_field: str | None = None) -> Table:
    ident = f"{C.NAMESPACE}.{name}"
    try:
        return cat.load_table(ident)
    except NoSuchTableError:
        table = cat.create_table(ident, schema=schema, properties={"write.parquet.compression-codec": "zstd"})
        if hour_field:  # hidden partitioning: readers filter on the timestamp, Iceberg prunes hours
            with table.update_spec() as spec:
                spec.add_field(hour_field, HourTransform(), f"{hour_field}_hour")
        return table


def tables(cat: Catalog) -> dict[str, Table]:
    return {"bronze": _table(cat, "bronze_recentchange", BRONZE, "event_ts"),
            "silver": _table(cat, "silver_edits", SILVER, "event_ts"),
            "hourly": _table(cat, "gold_wiki_hourly", HOURLY),
            "revisions": _table(cat, "gold_article_revisions", REVISIONS)}


def watermark(table: Table, key: str) -> str | None:
    """The newest value of `key` in the snapshot summaries of the table's current lineage."""
    snapshot = table.current_snapshot()
    by_id = {s.snapshot_id: s for s in table.metadata.snapshots}
    while snapshot is not None:
        value = (snapshot.summary.additional_properties if snapshot.summary else {}).get(key)
        if value is not None:
            return value
        snapshot = by_id.get(snapshot.parent_snapshot_id)
    return None


# ------------------------------------------------------------------ bronze: landing -> Iceberg, exactly once
def landing_files(root: Path) -> list[Path]:
    # file names are part-<UTC timestamp>.jsonl from a single writer, so name order is arrival order
    return sorted(root.rglob("part-*.jsonl"), key=lambda p: p.name)


def _ts(value: str | float | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_landing_file(path: Path, batch_id: int) -> pa.Table:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        e = record["event"]
        rev, length = e.get("revision") or {}, e.get("length") or {}
        rows.append({
            "meta_id": e["meta"]["id"], "event_ts": _ts(e["meta"].get("dt") or e.get("timestamp")),
            "received_ts": _ts(record["received_at"]), "wiki": e.get("wiki"), "type": e.get("type"),
            "namespace": e.get("namespace"), "title": e.get("title"), "user": e.get("user"), "bot": e.get("bot"),
            "minor": e.get("minor"), "rc_id": e.get("id"), "rev_old": rev.get("old"), "rev_new": rev.get("new"),
            "len_old": length.get("old"), "len_new": length.get("new"), "comment": e.get("comment"),
            "log_type": e.get("log_type"), "raw": json.dumps(e, ensure_ascii=False), "landing_file": path.name,
            "batch_id": batch_id,
        })
    return pa.Table.from_pylist(rows, schema=BRONZE)


class SimulatedCrash(RuntimeError):
    pass


def ingest_bronze(cat: Catalog, landing_root: Path = C.LANDING, files_per_commit: int = C.FILES_PER_COMMIT,
                  max_commits: int | None = None, offset_store: str = "snapshot", crash: str | None = None) -> dict:
    """Append new landing files to bronze, `files_per_commit` files per Iceberg commit.

    offset_store="snapshot": the last file name is written into the same commit as the rows (exactly-once).
    offset_store="file": the offset lives in a JSON file saved after the commit (the common naive design,
    kept to measure what it costs). `crash` injects a failure "before_commit" or "after_commit".
    """
    bronze = tables(cat)["bronze"]
    offset_file = Path(str(cat.properties["warehouse"])) / "_bronze_offset.json"
    if offset_store == "snapshot":
        last_file = watermark(bronze, C.OFFSET_KEY) or ""
    else:
        last_file = json.loads(offset_file.read_text())["last_file"] if offset_file.exists() else ""
    batch_id = int(watermark(bronze, "wikistream.bronze.batch-id") or 0)
    pending = [p for p in landing_files(landing_root) if p.name > last_file]
    commits = rows = 0
    for start in range(0, len(pending), files_per_commit):
        if max_commits is not None and commits >= max_commits:
            break
        group = pending[start:start + files_per_commit]
        batch_id += 1
        data = pa.concat_tables([parse_landing_file(p, batch_id) for p in group])
        props = {"wikistream.bronze.batch-id": str(batch_id)}
        if offset_store == "snapshot":
            props[C.OFFSET_KEY] = group[-1].name
        with bronze.transaction() as tx:
            tx.append(data, snapshot_properties=props)
            if crash == "before_commit":
                raise SimulatedCrash("died after writing data files, before the commit")
        if crash == "after_commit":
            raise SimulatedCrash("died after the commit, before saving the offset elsewhere")
        if offset_store == "file":
            offset_file.write_text(json.dumps({"last_file": group[-1].name}))
        commits += 1
        rows += data.num_rows
    return {"commits": commits, "rows": rows, "files": min(len(pending), commits * files_per_commit), "batch_id": batch_id}


# ------------------------------------------------------------------ silver: typed, deduplicated edits
def build_silver(cat: Catalog) -> dict:
    t = tables(cat)
    last = int(watermark(t["silver"], C.BRONZE_BATCH_KEY) or 0)
    new = t["bronze"].scan(row_filter=GreaterThan("batch_id", last),
                           selected_fields=[f.name for f in BRONZE if f.name != "raw"]).to_arrow()
    if new.num_rows == 0:
        return {"rows_in": 0, "rows_out": 0, "duplicates": 0}
    con = duckdb.connect()
    con.register("new_rows", new)
    min_ts = con.sql("select min(event_ts) from new_rows").fetchone()[0]
    since = min_ts - timedelta(hours=C.DEDUP_LOOKBACK_HOURS)
    existing = t["silver"].scan(row_filter=GreaterThanOrEqual("event_ts", since.isoformat()), selected_fields=["meta_id"]).to_arrow()
    con.register("existing", existing)
    silver = con.sql(f"""
        with typed as (
            select *, row_number() over (partition by meta_id order by received_ts) as copy
            from new_rows where type in ('edit', 'new')
        )
        select meta_id, event_ts, received_ts, wiki, namespace, title, type, "user",
               coalesce(bot, false) as is_bot,
               regexp_matches(coalesce("user", ''), '{ANON_PATTERN}') as is_anon,
               coalesce(minor, false) as is_minor,
               regexp_matches(coalesce(comment, ''), '{REVERT_PATTERN}') as is_revert,
               rev_old, rev_new, len_old, len_new,
               coalesce(len_new, 0) - coalesce(len_old, 0) as byte_delta, comment,
               epoch(received_ts) - epoch(event_ts) as lag_seconds, batch_id as bronze_batch
        from typed
        where copy = 1 and meta_id not in (select meta_id from existing)
    """).to_arrow_table().cast(SILVER)
    edits_in = con.sql("select count(*) from new_rows where type in ('edit', 'new')").fetchone()[0]
    top_batch = con.sql("select max(batch_id) from new_rows").fetchone()[0]
    t["silver"].append(silver, snapshot_properties={C.BRONZE_BATCH_KEY: str(top_batch)})
    return {"rows_in": new.num_rows, "edits_in": edits_in, "rows_out": silver.num_rows, "duplicates": edits_in - silver.num_rows}


# ------------------------------------------------------------------ gold: hourly activity and SCD2 revisions
def _new_silver(t: dict[str, Table], table: str) -> tuple[pa.Table, int]:
    last = int(watermark(t[table], "wikistream.silver.last-batch") or 0)
    rows = t["silver"].scan(row_filter=GreaterThan("bronze_batch", last)).to_arrow()
    top = max(rows.column("bronze_batch").to_pylist()) if rows.num_rows else last
    return rows, top


def build_gold_hourly(cat: Catalog) -> dict:
    """Recompute only the hours the new silver rows touch, then overwrite those hours."""
    t = tables(cat)
    new, top = _new_silver(t, "hourly")
    if new.num_rows == 0:
        return {"hours": 0}
    con = duckdb.connect()
    con.register("new_rows", new)
    first_hour = con.sql("select date_trunc('hour', min(event_ts)) from new_rows").fetchone()[0]
    con.register("silver", t["silver"].scan(row_filter=GreaterThanOrEqual("event_ts", first_hour.isoformat())).to_arrow())
    hourly = con.sql("""
        select date_trunc('hour', event_ts) as hour, wiki, count(*) as edits,
               count(*) filter (where type = 'new') as new_pages, count(*) filter (where is_bot) as bot_edits,
               count(*) filter (where is_anon) as anon_edits, count(*) filter (where is_revert) as reverts,
               count(distinct title) as pages, count(distinct "user") as editors,
               sum(greatest(byte_delta, 0)) as bytes_added, sum(greatest(-byte_delta, 0)) as bytes_removed
        from silver group by all
    """).to_arrow_table().cast(HOURLY)
    t["hourly"].overwrite(hourly, overwrite_filter=GreaterThanOrEqual("hour", first_hour.isoformat()),
                          snapshot_properties={"wikistream.silver.last-batch": str(top)})
    return {"hours": len(set(hourly.column("hour").to_pylist())), "rows": hourly.num_rows}


def scd2_rows(history: pa.Table) -> pa.Table:
    """Rebuild validity intervals for every title in `history` (old rows plus new revisions).

    Each revision is valid from its own timestamp until the next revision of the same title. Sorting the
    whole history of the affected titles, instead of only closing the previous current row, makes the
    result correct when events arrive out of order and idempotent when a batch is replayed.
    """
    con = duckdb.connect()
    con.register("h", history)
    return con.sql("""
        with d as (select * from h qualify row_number() over (partition by rev_id order by valid_from) = 1)
        select rev_id, title, parent_rev_id, valid_from,
               lead(valid_from) over w as valid_to, lead(valid_from) over w is null as is_current,
               "user", is_bot, is_revert, len_new, byte_delta
        from d window w as (partition by title order by valid_from, rev_id)
    """).to_arrow_table().cast(REVISIONS)


def build_gold_revisions(cat: Catalog) -> dict:
    t = tables(cat)
    new, top = _new_silver(t, "revisions")
    con = duckdb.connect()
    con.register("new_rows", new)
    fresh = con.sql(f"""
        select rev_new as rev_id, title, rev_old as parent_rev_id, event_ts as valid_from,
               null::timestamptz as valid_to, true as is_current, "user", is_bot, is_revert, len_new, byte_delta
        from new_rows where wiki = '{C.RAG_WIKI}' and namespace = {C.RAG_NAMESPACE} and rev_new is not null
    """).to_arrow_table().cast(REVISIONS)
    if fresh.num_rows == 0:
        return {"revisions_in": 0, "rows_updated": 0, "rows_inserted": 0}
    titles = sorted(set(fresh.column("title").to_pylist()))
    old = pa.concat_tables([t["revisions"].scan(row_filter=In("title", titles[i:i + 500])).to_arrow().cast(REVISIONS)
                            for i in range(0, len(titles), 500)])
    rebuilt = scd2_rows(pa.concat_tables([old, fresh]))
    result = t["revisions"].upsert(rebuilt, join_cols=["rev_id"], snapshot_properties={"wikistream.silver.last-batch": str(top)})
    return {"revisions_in": fresh.num_rows, "titles": len(titles), "rows_updated": result.rows_updated,
            "rows_inserted": result.rows_inserted}


def run_once(cat: Catalog, landing_root: Path = C.LANDING, max_commits: int | None = None) -> dict:
    return {"bronze": ingest_bronze(cat, landing_root, max_commits=max_commits), "silver": build_silver(cat),
            "hourly": build_gold_hourly(cat), "revisions": build_gold_revisions(cat)}
