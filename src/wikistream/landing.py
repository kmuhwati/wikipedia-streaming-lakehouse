"""Landing zone: immutable, append-only JSONL files of raw events, plus a checkpoint.

Delivery is at-least-once: a file is renamed into place *before* the checkpoint
moves, so a crash between the two replays (duplicates) rather than loses
events. Duplicates are removed by event id when the bronze table is built.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .stream import StreamMessage, stream_events


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class LandingWriter:
    def __init__(self, root: Path, flush_seconds: float = 30.0, flush_events: int = 5000):
        self.root = Path(root)
        self.checkpoint_path = self.root / "_checkpoint.json"
        self.flush_seconds, self.flush_events = flush_seconds, flush_events
        self.buffer: list[StreamMessage] = []
        self.last_flush = time.time()
        self.root.mkdir(parents=True, exist_ok=True)

    def checkpoint(self) -> dict:
        if self.checkpoint_path.exists():
            return json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        return {"last_event_id": None, "events_written": 0, "files_written": 0}

    def add(self, message: StreamMessage) -> None:
        self.buffer.append(message)
        if len(self.buffer) >= self.flush_events or time.time() - self.last_flush >= self.flush_seconds:
            self.flush()

    def flush(self) -> Path | None:
        self.last_flush = time.time()
        if not self.buffer:
            return None
        now = datetime.now(timezone.utc)
        folder = self.root / f"dt={now:%Y-%m-%d}" / f"hour={now:%H}"
        folder.mkdir(parents=True, exist_ok=True)
        state = self.checkpoint()
        path = folder / f"part-{now:%Y%m%dT%H%M%S%f}-{state['files_written']:06d}.jsonl"  # sequence: unique even within one microsecond
        lines = [json.dumps({"received_at": m.received_at, "event": m.data}, ensure_ascii=False) for m in self.buffer]
        _atomic_write(path, "\n".join(lines) + "\n")
        state.update(
            last_event_id=self.buffer[-1].event_id,
            events_written=state["events_written"] + len(self.buffer),
            files_written=state["files_written"] + 1,
            last_flush_utc=now.isoformat(),
        )
        _atomic_write(self.checkpoint_path, json.dumps(state, indent=2))
        self.buffer = []
        return path


def capture(root: Path, minutes: float, max_events: int | None = None) -> dict:
    """Capture the live stream into the landing zone, resuming from the checkpoint."""
    writer = LandingWriter(root)
    state = writer.checkpoint()
    deadline = time.time() + minutes * 60
    seen = 0
    print(f"resuming from {state['last_event_id'] or 'now'}; {state['events_written']:,} events already landed", flush=True)
    for message in stream_events(state["last_event_id"]):
        writer.add(message)
        seen += 1
        if seen % 10_000 == 0:
            print(f"{datetime.now():%H:%M:%S} {seen:,} events this run", flush=True)
        if time.time() >= deadline or (max_events and seen >= max_events):
            break
    writer.flush()
    return writer.checkpoint()
