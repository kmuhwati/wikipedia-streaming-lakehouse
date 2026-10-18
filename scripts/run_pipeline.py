"""Replay the captured stream through the lakehouse, build the CDC index, run the crash test, write the report.

    python scripts/capture_landing.py --minutes 180   # first, capture the live stream
    python scripts/run_pipeline.py                    # then this (re-runs reuse the revision cache)
"""
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
logging.getLogger("httpx").setLevel(logging.WARNING)

from wikistream import config as C  # noqa: E402
from wikistream import experiments  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[], choices=["replay", "index", "crash", "report"])
    args = ap.parse_args()
    C.TABLES.mkdir(parents=True, exist_ok=True)
    if "replay" not in args.skip:
        shutil.rmtree(C.LAKE, ignore_errors=True)
        experiments.replay().to_csv(C.TABLES / "microbatch_timing.csv", index=False)
    if "index" not in args.skip:
        shutil.rmtree(C.INDEX, ignore_errors=True)
        stats = experiments.build_index()
        print(stats)
        (C.TABLES / "index_build.json").write_text(json.dumps(stats, indent=2))
    if "crash" not in args.skip:
        experiments.crash_test().to_csv(C.TABLES / "crash_test.csv", index=False)
    if "report" not in args.skip:
        from wikistream.report import build_report

        print(build_report())
