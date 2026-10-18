"""Fetch revision text from the MediaWiki API and turn wikitext into chunks.

Revisions are fetched by id, so a replay months later gets exactly the text the
editor saved. Responses are cached on disk (one JSON file per revision) so
re-running the pipeline makes no further API calls.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import mwparserfromhell
import xxhash

from . import config as C
from .stream import USER_AGENT

DROP_SECTIONS = {"references", "external links", "see also", "further reading", "notes", "bibliography",
                 "sources", "citations", "footnotes", "notes and references"}
_REF = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.S | re.I)
_FILE_LINK = re.compile(r"\[\[(File|Image|Category):[^\[\]]*(\[\[[^\]]*\]\][^\[\]]*)*\]\]", re.I)
_SPACE = re.compile(r"[ \t]+")


# ------------------------------------------------------------------ API with a disk cache
def _cache_path(rev_id: int, cache: Path) -> Path:
    return cache / f"{rev_id % 1000:03d}" / f"{rev_id}.json"


def fetch_revisions(rev_ids: list[int], cache: Path = C.CACHE, client: httpx.Client | None = None) -> dict[int, dict]:
    """Return {rev_id: {"title", "timestamp", "wikitext"}}; deleted or hidden revisions are left out."""
    out, missing = {}, []
    for rev in dict.fromkeys(rev_ids):
        path = _cache_path(rev, cache)
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("wikitext") is not None:
                out[rev] = record
        else:
            missing.append(rev)
    own = client is None
    client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60)
    try:
        while missing:
            batch, missing = missing[:C.API_BATCH], missing[C.API_BATCH:]
            params = {"action": "query", "prop": "revisions", "revids": "|".join(map(str, batch)),
                      "rvprop": "ids|timestamp|content", "rvslots": "main", "format": "json",
                      "formatversion": "2", "maxlag": "5"}
            data = _get(client, params)
            found = {}
            for page in data.get("query", {}).get("pages", []):
                for r in page.get("revisions", []):
                    text = r.get("slots", {}).get("main", {}).get("content")
                    found[r["revid"]] = {"title": page.get("title"), "timestamp": r.get("timestamp"), "wikitext": text}
            bad = {int(r) for r in data.get("query", {}).get("badrevids", {})}
            if "continue" in data and found:  # the response hit its size limit: ask again for what was cut off
                missing = [rev for rev in batch if rev not in found and rev not in bad] + missing
                batch = [rev for rev in batch if rev in found or rev in bad]
            for rev in batch:  # cache misses too, so a deleted revision is asked for only once
                record = found.get(rev, {"title": None, "timestamp": None, "wikitext": None})
                path = _cache_path(rev, cache)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
                if record["wikitext"] is not None:
                    out[rev] = record
            time.sleep(C.API_PAUSE)
    finally:
        if own:
            client.close()
    return out


def _get(client: httpx.Client, params: dict) -> dict:
    for attempt in range(6):
        try:
            response = client.get(C.API_URL, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get("error", {}).get("code") != "maxlag":
                    return data
            wait = float(response.headers.get("Retry-After", 2 ** attempt))
        except httpx.HTTPError:
            wait = 2 ** attempt
        time.sleep(min(wait, 60))
    raise RuntimeError("MediaWiki API kept failing")


# ------------------------------------------------------------------ wikitext -> plain-text chunks
@dataclass(frozen=True)
class Chunk:
    key: str          # stable id: content hash plus occurrence number within the article
    section: str
    text: str


def wikitext_sections(wikitext: str) -> list[tuple[str, str]]:
    """Split into (heading, plain text) pairs, dropping reference-style sections, refs, files and templates."""
    wikitext = _FILE_LINK.sub("", _REF.sub("", wikitext))
    code = mwparserfromhell.parse(wikitext)
    sections = []
    for section in code.get_sections(levels=[2], include_lead=True, flat=False):
        headings = section.filter_headings()
        title = headings[0].title.strip_code().strip() if headings else "Introduction"
        if title.lower() in DROP_SECTIONS:
            continue
        for h in section.filter_headings():
            section.replace(h, "\n\n" + h.title.strip_code().strip() + "\n\n")
        sections.append((title, section.strip_code(normalize=True, collapse=True)))
    return sections


def chunk_article(title: str, wikitext: str) -> list[Chunk]:
    """One chunk per paragraph (long ones split by sentences), so an edit only touches nearby chunks.

    Fixed-size windows would shift every later chunk when a sentence is inserted; paragraph
    boundaries are where editors' changes naturally stop, which is what makes change-data-capture
    re-embedding small.
    """
    chunks, seen = [], {}
    for section, text in wikitext_sections(wikitext):
        for para in re.split(r"\n\s*\n", text):
            para = _SPACE.sub(" ", para).strip()
            words = para.split()
            if len(words) < C.MIN_CHUNK_WORDS or para.startswith(("{|", "|", "!")):
                continue
            for piece in _split_long(para):
                digest = xxhash.xxh64(f"{section}\n{piece}".encode()).hexdigest()
                seen[digest] = seen.get(digest, -1) + 1
                chunks.append(Chunk(f"{digest}-{seen[digest]}", section, piece))
    return chunks


def _split_long(para: str) -> list[str]:
    if len(para.split()) <= C.MAX_CHUNK_WORDS:
        return [para]
    sentences, pieces, current = re.split(r"(?<=[.!?])\s+", para), [], []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence]).split()) > C.MAX_CHUNK_WORDS:
            pieces.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces
