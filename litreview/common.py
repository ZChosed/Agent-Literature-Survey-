"""Shared helpers: polite HTTP, CSV/JSON/JSONL I/O, provenance, title matching."""

from __future__ import annotations

import csv
import json
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher

import requests

VERSION = "2.0.0"
OPENALEX = "https://api.openalex.org"
UNPAYWALL = "https://api.unpaywall.org/v2"
S2 = "https://api.semanticscholar.org/graph/v1/paper"

csv.field_size_limit(10_000_000)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- HTTP -------------------------------------------------------------------

class Http:
    """One polite client for every API: rate limit, retries on 429/5xx, request count."""

    def __init__(self, mailto: str, rate: float = 0.12, retries: int = 4, timeout: float = 60):
        self.mailto, self.rate, self.retries, self.timeout = mailto, rate, retries, timeout
        self.s = requests.Session()
        self.s.headers["User-Agent"] = f"litreview/{VERSION} (mailto:{mailto})"
        self.n = 0
        self._last = 0.0

    def get(self, url: str, **kw):
        """Response, or None after repeated network failure. Never raises."""
        backoff = 1.0
        for attempt in range(self.retries):
            gap = self.rate - (time.monotonic() - self._last)
            if gap > 0:
                time.sleep(gap)
            self._last = time.monotonic()
            try:
                r = self.s.get(url, timeout=self.timeout, **kw)
                self.n += 1
            except requests.RequestException:
                time.sleep(backoff)
                backoff *= 2
                continue
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.retries - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            return r
        return None

    def json(self, url: str, **kw):
        r = self.get(url, **kw)
        if r is None or r.status_code != 200:
            return None
        try:
            return r.json()
        except ValueError:
            return None

    def openalex(self, path: str, **params) -> dict:
        d = self.json(OPENALEX + path, params={**params, "mailto": self.mailto})
        if d is None:
            raise RuntimeError(f"OpenAlex request failed: {path} {params}")
        return d


# --- files ------------------------------------------------------------------

def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: list[dict], fields: list[str] | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else [])
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def read_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


class JsonlLog:
    """Append-only call log that doubles as a response cache."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.f = open(path, "a", encoding="utf-8")

    def write(self, rec: dict) -> None:
        self.f.write(json.dumps({**rec, "ts": now()}, ensure_ascii=False, default=str) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def provenance(cfg, block: str, data: dict) -> None:
    """Replace ONE block of data/provenance.json; every other block is kept."""
    prov = read_json(cfg.provenance, {})
    prov[block] = {"litreview_version": VERSION, "written_utc": now(), **data}
    write_json(cfg.provenance, prov)


def latest_run(runs_dir: str, prefix: str) -> str | None:
    if not os.path.isdir(runs_dir):
        return None
    runs = sorted(d for d in os.listdir(runs_dir) if d.startswith(prefix))
    return os.path.join(runs_dir, runs[-1]) if runs else None


def run_id(prefix: str) -> str:
    return prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# --- text -------------------------------------------------------------------

def norm_title(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, norm_title(a), norm_title(b)).ratio()


def clean_doi(doi: str | None) -> str:
    return (doi or "").replace("https://doi.org/", "").strip().lower()


def short_id(openalex_id: str | None) -> str:
    return (openalex_id or "").rsplit("/", 1)[-1]


def truthy(v, default: bool = True) -> bool:
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip().lower() in ("yes", "true", "1", "y")
