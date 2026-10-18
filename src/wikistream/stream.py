"""Resumable consumer for the Wikimedia EventStreams `recentchange` feed.

EventStreams is server-sent events (SSE) backed by Kafka. Every message carries
an `id` holding the Kafka partition offsets; sending it back as
`Last-Event-ID` on reconnect resumes the stream exactly after that message.
That id is the consumer's checkpoint.
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass

import httpx

STREAM_URL = "https://stream.wikimedia.org/v2/stream/recentchange"
USER_AGENT = "kmuhwati-wikistream/1.0 (https://github.com/kmuhwati; portfolio project)"


@dataclass
class StreamMessage:
    event_id: str          # SSE id: Kafka offsets, used to resume
    data: dict             # the recentchange event
    received_at: float     # wall-clock seconds when this consumer read it


def _parse_sse(lines: Iterator[str]) -> Iterator[tuple[str | None, str]]:
    """Yield (id, data) for each SSE message; comments and keep-alives are skipped."""
    event_id, data = None, []
    for line in lines:
        if line == "":
            if data:
                yield event_id, "\n".join(data)
            event_id, data = None, []
        elif line.startswith("id:"):
            event_id = line[3:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())


def stream_events(last_event_id: str | None = None, timeout: float = 60.0) -> Iterator[StreamMessage]:
    """Yield events forever, reconnecting with Last-Event-ID after any drop."""
    backoff = 1.0
    while True:
        headers = {"User-Agent": USER_AGENT, "Accept": "text/event-stream"}
        if last_event_id:
            headers["Last-Event-ID"] = last_event_id
        try:
            with httpx.stream("GET", STREAM_URL, headers=headers, timeout=timeout) as response:
                response.raise_for_status()
                backoff = 1.0
                for event_id, payload in _parse_sse(response.iter_lines()):
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if data.get("meta", {}).get("domain") == "canary":
                        continue  # Wikimedia's synthetic health-check events
                    if event_id:
                        last_event_id = event_id
                    yield StreamMessage(event_id or "", data, time.time())
        except (httpx.HTTPError, httpx.StreamError) as exc:
            print(f"stream dropped ({exc.__class__.__name__}); reconnecting in {backoff:.0f}s", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
