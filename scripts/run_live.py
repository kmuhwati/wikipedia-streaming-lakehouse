"""Run capture, lakehouse and CDC index together on the live stream and measure freshness.

    python scripts/run_live.py --minutes 15
"""
import argparse
import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
logging.getLogger("httpx").setLevel(logging.WARNING)

from wikistream import config as C  # noqa: E402
from wikistream.experiments import live  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--fresh", action="store_true", help="delete the previous live run first")
    args = ap.parse_args()
    if args.fresh:
        shutil.rmtree(C.LIVE, ignore_errors=True)
    C.TABLES.mkdir(parents=True, exist_ok=True)
    live(args.minutes).to_csv(C.TABLES / "live_cycles.csv", index=False)
