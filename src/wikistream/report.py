"""Analytics, data-quality checks, freshness evaluation, figures and RESULTS.md.

Article titles and wiki names are published; editor user names are not.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from . import lakehouse as L
from . import plotstyle
from .content import _cache_path, chunk_article
from .index import ChunkIndex


def _con(lake: Path = C.LAKE) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for name, table in L.tables(L.catalog(lake)).items():
        if name == "bronze":
            table.scan(selected_fields=[f.name for f in L.BRONZE if f.name != "raw"]).to_duckdb(name, connection=con)
        else:
            table.scan().to_duckdb(name, connection=con)
    return con


def q(con, sql: str) -> pd.DataFrame:
    return con.sql(sql).df()


# ------------------------------------------------------------------ checks
def lake_checks(con) -> dict:
    landed = sum(1 for p in L.landing_files(C.LANDING) for line in p.read_text(encoding="utf-8").splitlines() if line)
    c = q(con, """select (select count(*) from bronze) as bronze_rows, (select count(distinct meta_id) from bronze) as bronze_ids,
                         (select count(*) from silver) as silver_rows, (select count(distinct meta_id) from silver) as silver_ids,
                         (select sum(edits) from hourly) as hourly_edits""").iloc[0]
    scd = q(con, """with o as (select title, valid_to, lead(valid_from) over (partition by title order by valid_from, rev_id) as nxt,
                                      is_current from revisions)
                    select count(distinct title) as titles, sum(is_current::int) as current_rows,
                           count(*) filter (where not is_current and valid_to <> nxt) as interval_gaps,
                           (select count(*) from (select title from revisions where is_current group by 1 having count(*) > 1)) as multi_current
                    from o""").iloc[0]
    return {"landed_events": landed, "bronze_rows": int(c.bronze_rows), "bronze_matches_landing": int(c.bronze_rows) == landed,
            "stream_replays_in_bronze": int(c.bronze_rows - c.bronze_ids), "silver_rows": int(c.silver_rows),
            "silver_unique_ids": int(c.silver_rows) == int(c.silver_ids), "hourly_matches_silver": int(c.hourly_edits) == int(c.silver_rows),
            "scd2_titles": int(scd.titles), "scd2_current_rows": int(scd.current_rows), "scd2_interval_gaps": int(scd.interval_gaps),
            "scd2_titles_with_two_current": int(scd.multi_current)}


def index_consistency(index: ChunkIndex) -> dict:
    """For every tracked article, the live chunks must equal a fresh chunking of its newest indexed revision."""
    log = index.log.to_pandas()
    newest = log.sort_values(["ts", "rev_id"]).groupby("title").tail(1)
    current = index.chunks.search().where("valid_to IS NULL").select(["title", "chunk_key"]).limit(10_000_000).to_pandas()
    live = current.groupby("title").chunk_key.apply(set).to_dict()
    match = mismatch = 0
    for row in newest.itertuples():
        record = json.loads(_cache_path(int(row.rev_id), C.CACHE).read_text(encoding="utf-8"))
        expected = {c.key for c in chunk_article(row.title, record["wikitext"])}
        if live.get(row.title, set()) == expected:
            match += 1
        else:
            mismatch += 1
    return {"articles_checked": match + mismatch, "articles_matching_source": match, "mismatches": mismatch}


def freshness_probes(index: ChunkIndex, window_start: float, n: int = 300) -> pd.DataFrame:
    """Query with the text of paragraphs rewritten during the window, against the live index and against
    the same index as of the window start (what a nightly rebuild would still be serving)."""
    rows = index.chunks.search().where(f"valid_from >= {window_start}").select(
        ["row_id", "chunk_key", "title", "text", "rev_from", "valid_to"]).limit(10_000_000).to_pandas()
    closed = index.chunks.search().where(f"valid_to >= {window_start}").select(
        ["chunk_key", "title", "rev_to"]).limit(10_000_000).to_pandas()
    replaced = closed.groupby(["title", "rev_to"]).chunk_key.apply(set).to_dict()
    candidates = rows[rows.valid_to.isna() & rows.apply(lambda r: (r.title, r.rev_from) in replaced, axis=1)
                      & (rows.text.str.split().str.len() >= 15)]
    picks = candidates.sample(min(n, len(candidates)), random_state=C.SEED) if len(candidates) else candidates
    out = []
    for r in picks.itertuples():
        old_keys = replaced[(r.title, r.rev_from)]
        fresh = index.search(r.text, k=1)[0]
        stale = index.search(r.text, k=1, as_of=window_start)[0]
        out.append({"title": r.title, "fresh_returns_current": fresh["chunk_key"] == r.chunk_key,
                    "stale_returns_outdated": stale["title"] == r.title and stale["chunk_key"] in old_keys,
                    "stale_returns_same_article": stale["title"] == r.title})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ report
def build_report() -> dict:
    plotstyle.apply()
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    C.TABLES.mkdir(parents=True, exist_ok=True)
    con = _con()
    m: dict = {}

    overview = q(con, """select min(event_ts) as first_event, max(event_ts) as last_event, count(*) as events,
                                count(distinct wiki) as wikis from bronze""").iloc[0]
    span = (overview.last_event - overview.first_event).total_seconds()
    m["capture"] = {"first_event_utc": str(overview.first_event), "last_event_utc": str(overview.last_event),
                    "hours": round(span / 3600, 2), "events": int(overview.events), "events_per_second": round(overview.events / span, 1),
                    "wikis": int(overview.wikis)}
    types = q(con, "select type, count(*) as events from bronze group by 1 order by 2 desc")
    types.to_csv(C.TABLES / "event_types.csv", index=False)
    m["event_types"] = dict(zip(types.type, types.events.astype(int)))

    wikis = q(con, """select wiki, count(*) as edits, avg(is_bot::int) as bot_share, avg(is_anon::int) as anon_share,
                             avg(is_revert::int) as revert_share, count(distinct title) as pages, count(distinct "user") as editors,
                             sum(greatest(byte_delta, 0)) as bytes_added
                      from silver group by 1 order by edits desc""")
    wikis.head(30).to_csv(C.TABLES / "top_wikis.csv", index=False)
    lag = q(con, "select lag_seconds from silver").lag_seconds
    offset = max(0.0, float(-lag.min()))      # negative lag = this machine's clock behind Wikimedia's
    lag = lag + offset
    m["stream_lag_seconds"] = {"clock_offset_corrected": round(offset, 2), "p50": round(float(lag.median()), 2),
                               "share_within_5s": float((lag <= 5).mean()),
                               "edits_delivered_late_after_restart": int((lag > 60).sum())}
    m["wikidata_share_of_edits"] = float(wikis.loc[wikis.wiki == "wikidatawiki", "edits"].sum() / wikis.edits.sum())
    m["bot_share_of_edits"] = float((wikis.edits * wikis.bot_share).sum() / wikis.edits.sum())

    en = q(con, f"""select count(*) as edits, avg(is_revert::int) as revert_share, avg(is_bot::int) as bot_share,
                          avg(is_anon::int) as anon_share, count(distinct title) as articles
                   from silver where wiki = '{C.RAG_WIKI}' and namespace = 0""").iloc[0]
    m["enwiki_articles"] = {k: (float(v) if "share" in k else int(v)) for k, v in en.items()}
    wars = q(con, f"""select title, count(*) as edits, count(*) filter (where is_revert) as reverts,
                             count(distinct "user") as editors
                      from silver where wiki = '{C.RAG_WIKI}' and namespace = 0 group by 1
                      having count(*) filter (where is_revert) >= 3 order by reverts desc, edits desc""")
    wars.to_csv(C.TABLES / "enwiki_contested_articles.csv", index=False)
    m["enwiki_articles_with_3plus_reverts"] = len(wars)
    m["lake_checks"] = lake_checks(con)

    batches = pd.read_csv(C.TABLES / "microbatch_timing.csv")
    stage_cols = ["bronze_s", "silver_s", "gold_hourly_s", "gold_scd2_s"]
    batches["total_s"] = batches[stage_cols].sum(axis=1)
    m["replay"] = {"micro_batches": len(batches), "events_per_batch": float(batches.events.mean()),
                   "seconds_per_batch_p50": float(batches.total_s.median()), "seconds_per_batch_p95": float(batches.total_s.quantile(0.95)),
                   "events_per_second": float(batches.events.sum() / batches.total_s.sum()),
                   "replay_duplicates_removed": int(batches.duplicates_removed.sum())}

    # ---------------- CDC index
    index = ChunkIndex(C.INDEX)
    log = index.log.to_pandas()
    edits = log[log.kind == "edit"]
    m["cdc"] = {**json.loads((C.TABLES / "index_build.json").read_text()),
                "edits_applied": len(edits), "baselines": int((log.kind == "baseline").sum()),
                "chunks_full_reembed": int(edits.chunks.sum()), "chunks_cdc_embedded": int(edits.added.sum()),
                "chunks_closed": int(edits.removed.sum()),
                "embedding_savings": float(1 - edits.added.sum() / max(1, edits.chunks.sum())),
                "edits_changing_no_prose": float((edits.added + edits.removed == 0).mean()),
                "index_rows": int(index.chunks.count_rows()),
                "index_live_rows": int(index.chunks.count_rows("valid_to IS NULL"))}
    m["index_consistency"] = index_consistency(index)
    window_start = float(edits.ts.min())
    probes = freshness_probes(index, window_start)
    probes.to_csv(C.TABLES / "freshness_probes.csv", index=False)
    m["freshness_probes"] = {"probes": len(probes), "fresh_index_returns_current": float(probes.fresh_returns_current.mean()),
                             "stale_index_returns_outdated": float(probes.stale_returns_outdated.mean()),
                             "stale_index_same_article": float(probes.stale_returns_same_article.mean())}

    crash = pd.read_csv(C.TABLES / "crash_test.csv")
    m["crash_test"] = crash.groupby("offset_store")[["duplicate_rows", "missing_files"]].sum().astype(int).to_dict("index")
    m["crash_test"]["trials"] = int(crash.trial.nunique())

    live_log = None
    if (C.LIVE / "index").exists() and (C.TABLES / "live_cycles.csv").exists():
        live_index = ChunkIndex(C.LIVE / "index", embed=lambda texts: np.zeros((len(texts), C.EMBED_DIM), np.float32))
        live_log = live_index.log.to_pandas()
        live_log = live_log[live_log.kind != "baseline"].copy()
        # An event cannot arrive before Wikimedia stamped it, so a negative lag is this machine's clock
        # running behind. Add the smallest observed lag back to every latency to correct for it.
        offset = max(0.0, float(-(live_log.received_ts - live_log.ts).min()))
        live_log["latency_s"] = live_log.indexed_at - live_log.ts + offset
        live_log["lag_s"] = live_log.received_ts - live_log.ts + offset
        cycles = pd.read_csv(C.TABLES / "live_cycles.csv")
        lat = live_log.latency_s
        m["live"] = {"minutes": round((cycles.cycle_start.max() - cycles.cycle_start.min()) / 60, 1), "cycles": len(cycles),
                     "revisions_indexed": len(live_log), "events": int(cycles.events.sum()),
                     "latency_p50_s": float(lat.median()), "latency_p95_s": float(lat.quantile(0.95)),
                     "latency_max_s": float(lat.max()), "stream_lag_p50_s": float(live_log.lag_s.median()),
                     "clock_offset_corrected_s": offset, "tracked_titles_cap": 300,
                     "cycle_seconds_p50": float((cycles.lake_s + cycles.index_s).median()),
                     "cycle_seconds_p95": float((cycles.lake_s + cycles.index_s).quantile(0.95))}
    (C.REPORTS / "metrics.json").write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")

    _figures(con, wikis, batches, edits, probes, crash, live_log, m)
    _results_md(m, wikis, wars, batches)
    return m


def _figures(con, wikis, batches, edits, probes, crash, live_log, m) -> None:
    per_min = q(con, """select date_trunc('minute', event_ts) as minute, type, count(*) as n from bronze
                        where type in ('edit', 'new', 'log', 'categorize') group by all order by 1""")
    pivot = per_min.pivot(index="minute", columns="type", values="n").fillna(0)[["edit", "categorize", "log", "new"]]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.stackplot(pivot.index, pivot.T.values, labels=pivot.columns, colors=plotstyle.SERIES[:4], alpha=0.9)
    ax.set_ylabel("Events per minute")
    ax.set_title(f"{m['capture']['events']:,} events from {m['capture']['wikis']} wikis in {m['capture']['hours']:.1f} hours")
    ax.legend(loc="upper left", ncols=4)
    plotstyle.save(fig, C.FIGURES / "01_events_per_minute.png")

    top = wikis.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    human = top.edits * (1 - top.bot_share)
    ax.barh(top.wiki, human, color=plotstyle.SERIES[0], label="human", height=0.65)
    ax.barh(top.wiki, top.edits - human, left=human, color=plotstyle.NEUTRAL, label="bot", height=0.65)
    ax.set_xlabel("Edits in the capture window")
    ax.set_title("Busiest wikis, split by bot flag")
    ax.legend(loc="lower right")
    plotstyle.save(fig, C.FIGURES / "02_top_wikis.png")

    share = (edits.added / edits.chunks.where(edits.chunks > 0)).dropna()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(100 * share, bins=np.arange(0, 102.5, 2.5), color=plotstyle.SERIES[2])
    ax.set_xlabel("Share of the article's chunks re-embedded by the edit (%)")
    ax.set_ylabel("Edits")
    ax.set_title(f"CDC re-embeds {100 * (1 - m['cdc']['embedding_savings']):.1f}% of what a full re-index would")
    plotstyle.save(fig, C.FIGURES / "03_cdc_reembed_share.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    stages = {"bronze_s": "bronze append", "silver_s": "silver dedup", "gold_hourly_s": "gold hourly", "gold_scd2_s": "gold SCD2 upsert"}
    bottom = np.zeros(len(batches))
    for i, (col, label) in enumerate(stages.items()):
        ax.bar(batches.batch, batches[col], bottom=bottom, color=plotstyle.SERIES[i], label=label, width=0.9)
        bottom += batches[col].to_numpy()
    ax.set_xlabel("Micro-batch")
    ax.set_ylabel("Seconds")
    ax.set_title(f"Each micro-batch of ~{m['replay']['events_per_batch']:,.0f} events takes {m['replay']['seconds_per_batch_p50']:.1f}s (median)")
    ax.legend(ncols=4, loc="upper left")
    plotstyle.save(fig, C.FIGURES / "04_microbatch_timing.png")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    agg = crash.groupby("offset_store").duplicate_rows.sum().reindex(["file", "snapshot"])
    bars = ax.bar(["offset in a separate file", "offset in the Iceberg commit"], agg.values,
                  color=[plotstyle.CRITICAL, plotstyle.GOOD], width=0.55)
    ax.bar_label(bars, labels=[f"{v:,}" for v in agg.values], padding=3)
    ax.set_ylabel("Duplicate rows after recovery")
    ax.set_title(f"{m['crash_test']['trials']} crash-and-resume trials on captured data")
    plotstyle.save(fig, C.FIGURES / "05_crash_test.png")

    if len(probes):
        fig, ax = plt.subplots(figsize=(7, 3.6))
        vals = [100 * probes.fresh_returns_current.mean(), 100 * probes.stale_returns_outdated.mean()]
        bars = ax.bar(["Live CDC index:\nreturns the current paragraph", "Index frozen at window start:\nreturns the outdated paragraph"], vals, color=[plotstyle.SERIES[0], plotstyle.NEUTRAL], width=0.5)
        ax.bar_label(bars, labels=[f"{v:.0f}%" for v in vals], padding=3)
        ax.set_ylim(0, 110)
        ax.set_ylabel("% of probes")
        ax.set_title(f"Querying {len(probes)} paragraphs rewritten during the window")
        plotstyle.save(fig, C.FIGURES / "06_freshness_probes.png")

    if live_log is not None and len(live_log):
        fig, ax = plt.subplots(figsize=(7.5, 4))
        for col, label, color in [("lag_s", "edit saved → event received", plotstyle.NEUTRAL),
                                  ("latency_s", "edit saved → searchable in index", plotstyle.SERIES[0])]:
            x = np.sort(live_log[col].dropna().to_numpy())
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=color, label=label)
        ax.axhline(0.95, color=plotstyle.AXIS, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("Seconds (log scale)")
        ax.set_ylabel("Share of edits")
        ax.set_title(f"Live run: median {m['live']['latency_p50_s']:.0f}s, p95 {m['live']['latency_p95_s']:.0f}s from edit to searchable")
        ax.legend(loc="lower right")
        plotstyle.save(fig, C.FIGURES / "07_live_freshness.png")


def _results_md(m, wikis, wars, batches) -> None:
    md = lambda f: f.to_markdown(index=False, floatfmt=",.3f")  # noqa: E731
    lines = ["# Results (auto-generated by scripts/run_pipeline.py)\n",
             "## Busiest wikis\n", md(wikis.head(15)),
             "\n## enwiki articles with 3+ reverts in the window\n", md(wars.head(15)),
             "\n## Micro-batch timing (first 10)\n", md(batches.head(10)),
             f"\n## All metrics\n\n```json\n{json.dumps(m, indent=2, default=str)}\n```\n"]
    (C.REPORTS / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
