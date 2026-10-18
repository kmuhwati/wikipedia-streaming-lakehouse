"""Capture the live Wikimedia recentchange stream into data/landing/ (resumable).

    python scripts/capture_landing.py --minutes 120
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wikistream.landing import capture  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--root", default=str(ROOT / "data" / "landing"))
    args = ap.parse_args()
    print(capture(Path(args.root), args.minutes, args.max_events))
