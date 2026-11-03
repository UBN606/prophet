#!/usr/bin/env python3
"""Prophet Pattern Lab: make the call, lock it in, get graded.

The lab writes a prediction down (a "call") before the outcome is known, locks it with a seal,
then grades it against what really happened and against the obvious guess (the "lazy guess").
It keeps a record, wins and misses together.

  Commands that never touch the network (they read and write the lab folder only):
  prophet lab call ...                lock in a call on any question by hand (any domain)
  prophet lab resolve <id> --outcome  grade a hand-made call
  prophet lab list                    list the calls
  prophet lab verify                  check every seal; a broken seal is reported as TAMPERED
  prophet lab record                  write RECORD.md and record.html
  prophet lab domains                 what each domain is, and whether it has a free data source

  Commands that download public data (each site's robots.txt is checked first):
  prophet lab backtest <domain|all>   practice run: replay history, using only data from before each date
  prophet lab predict  <domain|all>   lock in live calls for things that have not happened yet
  prophet lab score                   grade locked calls whose outcome is now in the data

  Command that sends one hash to the internet:
  prophet lab anchor                  send the chain head (32 bytes, nothing else) to two OpenTimestamps
                                      servers, so the date can be proved later. --dry-run sends nothing.

Pattern research only. No bets, no trades, no investment advice. Free public data, no keys.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import http.client
import importlib.util
import io
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import prophet_net as net      # the robots.txt rule and the official-API allowlist, shared with the source harvester

import prophet_config as pcfg

SELF = Path(__file__).resolve()
HERE = SELF.parent
DEFAULT_LAB = pcfg.NOTEBOOKS / "pattern-lab"
UA = "ProphetLab/1.0 research"
ROBOTS_TOKEN = "prophetlab"          # the name robots.txt rules are matched against (the product name in UA)
UTC = dt.timezone.utc
MIN_CALLS = 30                  # no skill claim below this many scored calls
BASELINE_EPS = 1e-9

# --------------------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------------------


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def canon(obj) -> str:
    """Canonical JSON: sorted keys, no spaces, ASCII only. The seal is the sha256 of this."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def utcnow() -> dt.datetime:
    return dt.datetime.now(UTC).replace(microsecond=0)


def iso(t: dt.datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def day_end_iso(d: dt.date) -> str:
    return d.strftime("%Y-%m-%dT23:59:59Z")


def slug(s: str, n: int = 48) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")[:n].strip("-") or "x"


def num(x, digits=3, plus=False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    s = f"{x:+.{digits}f}" if plus else f"{x:.{digits}f}"
    return s


_CODE_SHA = None


def code_sha() -> str:
    global _CODE_SHA
    if _CODE_SHA is None:
        _CODE_SHA = sha256_bytes(SELF.read_bytes())
    return _CODE_SHA


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_line(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (obj if isinstance(obj, str) else canon(obj)) + "\n"
    with open(path, "ab") as f:
        f.write(line.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


class LabError(Exception):
    pass


class FetchError(Exception):
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect on its own: the lab checks the new address (robots.txt or the API allowlist) before it asks."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect())


class file_lock:
    """Exclusive lock file so two runs cannot append to the same log at once."""

    def __init__(self, path: Path, wait: float = 20.0):
        self.lock = Path(str(path) + ".lock")
        self.wait = wait

    def __enter__(self):
        self.lock.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        while True:
            try:
                fd = os.open(str(self.lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    age = time.time() - self.lock.stat().st_mtime
                except OSError:
                    age = 0
                if age > 180:
                    try:
                        self.lock.unlink()
                    except OSError:
                        pass
                    continue
                if time.time() - t0 > self.wait:
                    raise LabError(f"another run holds {self.lock.name}; try again in a minute")
                time.sleep(0.25)

    def __exit__(self, *exc):
        try:
            self.lock.unlink()
        except OSError:
            pass


# --------------------------------------------------------------------------------------
# numerics: ridge logistic regression and friends
# --------------------------------------------------------------------------------------


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class Logit:
    """Ridge logistic regression on standardised features. The intercept is not penalised."""

    def __init__(self, l2: float = 1.0):
        self.l2 = l2
        self.w = None

    def fit(self, X, y, iters: int = 40):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-9] = 1.0
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        d = Z.shape[1]
        pen = np.full(d, self.l2)
        pen[0] = 0.0
        w = np.zeros(d)
        for _ in range(iters):
            p = sigmoid(Z @ w)
            g = Z.T @ (p - y) + pen * w
            H = (Z.T * (p * (1 - p) + 1e-9)) @ Z + np.diag(pen) + 1e-9 * np.eye(d)
            step = np.linalg.solve(H, g)
            w -= step
            if np.max(np.abs(step)) < 1e-8:
                break
        self.w = w
        return self

    def prob(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        return sigmoid(Z @ self.w)


def score_pair(kind: str, pred: float, outcome: float) -> float:
    """Brier score for yes/no calls, absolute error for numbers. Lower is better."""
    if kind == "binary":
        return float((pred - outcome) ** 2)
    return float(abs(pred - outcome))


SCORE_NAME = {"binary": "Brier", "number": "MAE"}

# --------------------------------------------------------------------------------------
# the lab: paths, config, polite cached downloads
# --------------------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "schema": "prophet-lab-config/1",
    "note": "Edit freely. Prophet fetches only what is listed here: free public data, no keys.",
    "weather": {
        "places": [
            {"name": "New York", "lat": 40.7128, "lon": -74.0060},
            {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
            {"name": "Houston", "lat": 29.7604, "lon": -95.3698},
            {"name": "Denver", "lat": 39.7392, "lon": -104.9903},
            {"name": "Seattle", "lat": 47.6062, "lon": -122.3321},
            {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
        ],
        "history_start": "2016-01-01",
        "test_start": "2019-01-01",
    },
    "sports": {"leagues": ["E0", "D1", "SP1", "I1", "F1"], "seasons_back": 5},
    "markets": {
        "series": [
            {"id": "NASDAQCOM", "name": "Nasdaq Composite"},
            {"id": "DJIA", "name": "Dow Jones Industrial Average"},
            {"id": "SP500", "name": "S&P 500"},
        ],
        "since": "2014-01-01",
    },
}


class Fetched(SimpleNamespace):
    def text(self, encoding="utf-8-sig"):
        return self.data.decode(encoding, errors="replace")


class Lab:
    def __init__(self, root=None, calls=None, scores=None, ttl_hours: float = 6.0, opts=None):
        self.root = Path(root or os.environ.get("PROPHET_LAB_DIR") or DEFAULT_LAB)
        self.opts = dict(opts or {})       # per-run switches from the command line that a domain may read (geopolitics: days, gdelt_doc_api)
        self.calls_path = Path(calls) if calls else self.root / "calls.jsonl"
        self.scores_path = Path(scores) if scores else self.root / "scores.jsonl"
        self.raw_dir = self.root / "raw"
        self.backtest_dir = self.root / "backtest"
        self.snap_dir = self.root / "code_snapshots"
        self.ttl = ttl_hours * 3600.0
        self._manifest = None
        self._last_hit = {}
        self._robots = {}          # origin -> parsed robots.txt, read once per site per run
        self.wait_429 = 10.0       # least wait before the first retry of a 429 answer (10 s, then 20 s); tests make it short
        self.cfg = self._load_config()
        self.fetched = []          # every download or cache hit this run (url, sha256)

    # ---- config ---------------------------------------------------------------------
    def _load_config(self):
        path = self.root / "config.json"
        if path.exists():
            try:
                cfg = json.loads(path.read_text(encoding="utf-8"))
            except ValueError as e:
                raise LabError(f"config.json is not valid JSON: {e}")
        else:
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            try:
                atomic_write(path, json.dumps(cfg, indent=2) + "\n")
            except OSError:
                pass
        for k, v in DEFAULT_CONFIG.items():
            cfg.setdefault(k, v)
        added = [n for n, a in REGISTRY.items() if a.default_config and n not in cfg]
        for n in added:                       # drop-in domains bring their own defaults; add them once so they can be edited
            cfg[n] = REGISTRY[n].default_config
        if added:
            try:
                atomic_write(path, json.dumps(cfg, indent=2) + "\n")
            except OSError:
                pass
        return cfg

    # ---- polite cached download -----------------------------------------------------
    def _manifest_index(self):
        if self._manifest is None:
            self._manifest = {}
            p = self.raw_dir / "manifest.jsonl"
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    try:
                        e = json.loads(line)
                        self._manifest[e["url"]] = e
                    except (ValueError, KeyError):
                        continue
        return self._manifest

    def fetch(self, url: str, ttl: float | None = None, validate=None, min_gap: float = 1.0,
              timeout: float = 60.0, max_bytes: int = 80_000_000, suffix: str | None = None, retry_429: int = 2) -> Fetched:
        """Plain HTTPS GET with a polite User-Agent. The raw bytes are cached under raw/ and the URL, fetch time, sha256 and
        the reason the download was allowed are appended to raw/manifest.jsonl.

        Before anything is downloaded the address is checked (project decision): it must be an official API endpoint
        on prophet_net.API_ALLOWLIST, or the site's robots.txt must allow it; otherwise FetchError and no download. The same
        check is made again for every redirect. min_gap is the least number of seconds between two requests to one site (a
        robots.txt read counts as a request). retry_429 is how many times a "too many requests" answer is retried, each after
        a wait of at least 10 s, 20 s, ... before the download is given up."""
        ttl = self.ttl if ttl is None else ttl
        if self.ttl <= 0:               # --refresh: always download again
            ttl = 0
        idx = self._manifest_index()
        ent = idx.get(url)
        if ent and ttl > 0:
            age = time.time() - parse_iso(ent["fetched_at"]).timestamp()
            p = self.root / ent["file"]
            if age <= ttl and p.exists():
                data = p.read_bytes()
                if sha256_bytes(data) == ent["sha256"]:
                    self.fetched.append({"url": url, "sha256": ent["sha256"]})
                    return Fetched(url=url, data=data, sha256=ent["sha256"], path=p,
                                   fetched_at=ent["fetched_at"], from_cache=True)
        data, status, ctype, basis = self._http_get(url, min_gap, timeout, max_bytes, retry_429)
        if validate is not None:
            try:
                validate(data)
            except ValueError as e:
                raise FetchError(f"{url} returned unusable content: {e}")
        sha = sha256_bytes(data)
        if ent and ent["sha256"] == sha and (self.root / ent["file"]).exists():
            rel = ent["file"]
        else:
            host = urllib.parse.urlsplit(url).netloc
            tail = urllib.parse.urlsplit(url)
            name = slug(tail.path + "-" + tail.query, 60) + "-" + sha256_text(url)[:8]
            ext = suffix or (Path(tail.path).suffix if Path(tail.path).suffix in (".csv", ".json", ".txt", ".xml") else ".dat")
            rel = f"raw/{slug(host, 40)}/{name}.{sha[:12]}{ext}"
            target = self.root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        now = iso(utcnow())
        entry = {"url": url, "fetched_at": now, "sha256": sha, "bytes": len(data), "file": rel,
                 "status": status, "content_type": ctype, "allowed_by": basis}
        append_line(self.raw_dir / "manifest.jsonl", entry)
        idx[url] = entry
        self.fetched.append({"url": url, "sha256": sha})
        return Fetched(url=url, data=data, sha256=sha, path=self.root / rel, fetched_at=now, from_cache=False)

    def _pause(self, host, gap):
        """Wait until `gap` seconds have passed since the last request to this site."""
        wait = min(gap, 60.0) - (time.time() - self._last_hit.get(host, 0.0))
        if wait > 0:
            time.sleep(min(wait, 60.0))

    def _robots_basis(self, url, min_gap):
        """Why this address may be requested, and the Crawl-delay to honour: (basis text, delay in seconds).
        An official API endpoint on prophet_net.API_ALLOWLIST needs no robots.txt; for anything else the site's robots.txt must
        allow it. robots.txt is read once per site per run, and that read waits for the same gap as any request. A robots.txt that
        cannot be read, or that disallows the address, raises FetchError: nothing is downloaded."""
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise FetchError(f"{url} is not a web address; the lab downloads only http(s) addresses")
        if net.api_entry(parts) is not None:
            return "official API (allowlist)", 0.0
        origin = f"{parts.scheme}://{parts.netloc}"
        robots = self._robots.get(origin)
        if robots is None:
            self._pause(parts.netloc, min_gap)
            try:
                status, text = net.fetch_robots(origin, UA)
            except net.RobotsUnreachable as e:
                raise FetchError(f"robots.txt of {origin} could not be read ({e}); not downloading {url}")
            finally:
                self._last_hit[parts.netloc] = time.time()
            append_line(self.raw_dir / "robots.jsonl", {"origin": origin, "read_at": iso(utcnow()), "status": status,
                                                         "sha256": sha256_text(text), "bytes": len(text.encode("utf-8"))})    # a receipt for the decision
            try:
                robots = net.robots_from_answer(status, text, ROBOTS_TOKEN)       # 404/410: no restrictions; 429, 5xx, a refusal: blocked
            except net.RobotsUnreadable:
                raise FetchError(f"robots.txt of {origin} answered HTTP {status}, so its rules cannot be known; not downloading {url}")
            self._robots[origin] = robots
        allowed, rule = robots.rule_for(parts)
        if not allowed:
            raise FetchError(f"robots.txt of {origin} disallows {url} for {ROBOTS_TOKEN} ({rule}); not downloading it")
        return "robots.txt allows", robots.delay or 0.0

    def _get_once(self, url, min_gap, timeout, max_bytes):
        """One request chain. Redirects are followed by hand so that every address is checked before it is requested.
        Returns (data, status, content type, basis); an error answer is raised as urllib.error.HTTPError, like urlopen."""
        current = url
        for _ in range(6):
            basis, crawl_delay = self._robots_basis(current, min_gap)
            host = urllib.parse.urlsplit(current).netloc
            self._pause(host, max(min_gap, min(crawl_delay, 30.0)))
            req = urllib.request.Request(current, headers={"User-Agent": UA, "Accept": "*/*"})
            try:
                with _OPENER.open(req, timeout=timeout) as r:
                    data = r.read(max_bytes + 1)
                    status = r.status
                    ctype = r.headers.get("Content-Type", "")
                return data, status, ctype, basis
            except urllib.error.HTTPError as e:
                where = e.headers.get("Location") if e.headers else None
                if e.code in (301, 302, 303, 307, 308) and where:
                    e.close()
                    current = urllib.parse.urljoin(current, where)
                    continue
                raise
            finally:
                self._last_hit[host] = time.time()
        raise FetchError(f"too many redirects from {url}")

    def _http_get(self, url, min_gap, timeout, max_bytes, retry_429=2):
        last_err = "unknown error"
        for attempt in range(3):
            try:
                data, status, ctype, basis = self._get_once(url, min_gap, timeout, max_bytes)
                if len(data) > max_bytes:
                    raise FetchError(f"{url} is larger than {max_bytes} bytes; not downloaded")
                return data, status, ctype, basis
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read(300).decode("utf-8", "replace").strip().replace("\n", " ")
                    body = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "[e-mail address removed]", body)     # a server's error text can carry someone's address; it never goes into a record
                except Exception:
                    pass
                if e.code == 429 and attempt < retry_429:       # "slow down": wait what the server asks for (never less than 10 s, then 20 s) and ask again, politely
                    try:
                        wait = float(e.headers.get("Retry-After") or 0)
                    except (TypeError, ValueError):
                        wait = 0.0
                    time.sleep(min(60.0, max(wait, self.wait_429 * (attempt + 1))))
                    continue
                if e.code == 429:
                    raise FetchError(f"HTTP 429 (too many requests) from {url}: {body}. The lab waited and asked {attempt} more time(s); "
                                     "giving up cleanly, nothing was stored. Try again later.", code=429)
                if e.code in (400, 401, 403, 404, 410, 451):
                    raise FetchError(f"HTTP {e.code} from {url}: {body}", code=e.code)
                last_err = f"HTTP {e.code} {body}"
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, http.client.HTTPException) as e:       # HTTPException: a body cut short (IncompleteRead)
                last_err = str(getattr(e, "reason", e))
            time.sleep(2.0 * (attempt + 1))
        raise FetchError(f"could not fetch {url}: {last_err}")


# --------------------------------------------------------------------------------------
# sealed calls: append-only log, canonical seal, hash chain, verification
# --------------------------------------------------------------------------------------

CALL_REQUIRED = ["id", "domain", "target", "kind", "question", "made_at", "resolves_at", "prediction",
                 "baseline_prediction", "method", "inputs", "code_sha256", "prev_seal", "seal"]


def seal_of(rec: dict) -> str:
    return sha256_text(canon({k: v for k, v in rec.items() if k != "seal"}))


class CallLog:
    def __init__(self, path):
        self.path = Path(path)
        self.entries = []          # dicts: line, rec, ok, problems

    @property
    def ok_calls(self):
        return [e["rec"] for e in self.entries if e["ok"]]

    @property
    def bad(self):
        return [e for e in self.entries if not e["ok"]]

    @property
    def head(self):
        for e in reversed(self.entries):
            if e["rec"] and e["rec"].get("seal"):
                return e["rec"]["seal"]
        return ""

    def by_id(self):
        return {e["rec"]["id"]: e for e in self.entries if e["rec"] and "id" in e["rec"]}


def read_calls(path) -> CallLog:
    """Read calls.jsonl and verify every seal and the hash chain."""
    log = CallLog(path)
    path = Path(path)
    if not path.exists():
        return log
    prev = ""
    seen = set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        e = {"line": n, "rec": None, "ok": False, "problems": []}
        log.entries.append(e)
        try:
            rec = json.loads(line)
        except ValueError:
            e["problems"].append("line is not valid JSON")
            continue
        if not isinstance(rec, dict):
            e["problems"].append("line is not a call record")
            continue
        e["rec"] = rec
        missing = [k for k in CALL_REQUIRED if k not in rec]
        if missing:
            e["problems"].append("missing fields: " + ", ".join(missing))
        else:
            calc = seal_of(rec)
            if calc != rec["seal"]:
                e["problems"].append(f"seal mismatch (stored {str(rec['seal'])[:12]}, recomputed {calc[:12]})")
        if rec.get("prev_seal", None) != prev:
            e["problems"].append("chain break: a record before this one was changed, removed or reordered")
        cid = rec.get("id")
        if cid in seen:
            e["problems"].append("duplicate id")
        seen.add(cid)
        e["ok"] = not e["problems"]
        if rec.get("seal"):
            prev = rec["seal"]
    return log


def read_scores(path):
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError
        except ValueError:
            rows.append({"_line": n, "_bad": "line is not valid JSON"})
            continue
        row["_line"] = n
        rows.append(row)
    return rows


class Verification:
    def __init__(self, log: CallLog, scores: list):
        self.log = log
        self.scores = scores
        self.score_problems = []       # (line, id, problem)
        by_id = log.by_id()
        seen = set()
        for row in scores:
            if row.get("_bad"):
                self.score_problems.append((row["_line"], None, row["_bad"]))
                continue
            cid = row.get("id")
            call = by_id.get(cid)
            if call is None:
                self.score_problems.append((row["_line"], cid, "score refers to a call that is not in the log"))
                continue
            if not call["ok"]:
                self.score_problems.append((row["_line"], cid, "score refers to a TAMPERED call"))
                continue
            rec = call["rec"]
            if row.get("seal") != rec["seal"]:
                self.score_problems.append((row["_line"], cid, "score seal differs from the call's seal"))
                continue
            if cid in seen:
                self.score_problems.append((row["_line"], cid, "call scored twice"))
                continue
            seen.add(cid)
            try:
                want = score_pair(rec["kind"], rec["prediction"], float(row["outcome"]))
                if abs(want - float(row["score"])) > 1e-9:
                    self.score_problems.append((row["_line"], cid, f"score {row['score']} does not match outcome (expected {want:.9f})"))
                    continue
                wantb = score_pair(rec["kind"], rec["baseline_prediction"], float(row["outcome"]))
                if abs(wantb - float(row["baseline_score"])) > 1e-9:
                    self.score_problems.append((row["_line"], cid, "baseline_score does not match outcome"))
                for name, info in (row.get("late_baselines") or {}).items():       # yardsticks added at grading: only ones the call named, and they must add up
                    p = float(info["p"])
                    if name not in (rec.get("late_baselines") or []):
                        self.score_problems.append((row["_line"], cid, f"score row carries the baseline '{name}', which the call never named"))
                    elif abs(score_pair(rec["kind"], p, float(row["outcome"])) - float(row["baseline_scores"][name])) > 1e-9:
                        self.score_problems.append((row["_line"], cid, f"baseline_scores['{name}'] does not match its stored chance and the outcome"))
                    elif info.get("odds") and abs(devig3(info["odds"])[0] - p) > 2e-6:
                        self.score_problems.append((row["_line"], cid, f"late baseline '{name}': the stored chance is not what its odds imply"))
            except (KeyError, TypeError, ValueError, AttributeError, ZeroDivisionError):
                self.score_problems.append((row["_line"], cid, "score row is incomplete"))

    @property
    def ok(self):
        return not self.log.bad and not self.score_problems


def verify_lab(lab: Lab) -> Verification:
    return Verification(read_calls(lab.calls_path), read_scores(lab.scores_path))


def print_verification(v: Verification, out=print, per_call_limit: int = 80):
    n = len(v.log.entries)
    out(f"Prophet seal check of {v.log.path.name}: {n} sealed calls")
    if n <= per_call_limit:
        for e in v.log.entries:
            rec = e["rec"] or {}
            if e["ok"]:
                out(f"  OK        {rec.get('id')}  seal {str(rec.get('seal'))[:12]}")
            else:
                out(f"  TAMPERED  {rec.get('id', '(line ' + str(e['line']) + ')')}: " + "; ".join(e["problems"]))
    else:
        for e in v.log.bad:
            rec = e["rec"] or {}
            out(f"  TAMPERED  {rec.get('id', '(line ' + str(e['line']) + ')')}: " + "; ".join(e["problems"]))
    for line, cid, problem in v.score_problems:
        out(f"  TAMPERED  scores line {line} ({cid}): {problem}")
    n_ok = sum(1 for e in v.log.entries if e["ok"])
    out(f"Calls: {n_ok} OK, {len(v.log.bad)} TAMPERED. Score rows: {len(v.scores)}, problems: {len(v.score_problems)}.")
    out("Chain head (seal of the newest call): " + (v.log.head or "(empty log)"))
    out("RESULT: ALL SEALS OK" if v.ok else "RESULT: TAMPERED - do not trust this record until the cause is understood")


def save_snapshot(lab: Lab, path: Path) -> str:
    """Keep a copy of the exact code a call was made with, named by its hash."""
    data = path.read_bytes()
    sha = sha256_bytes(data)
    lab.snap_dir.mkdir(parents=True, exist_ok=True)
    target = lab.snap_dir / f"{path.stem}.{sha[:12]}.py"
    if not target.exists():
        target.write_bytes(data)
    return sha


def append_call(lab: Lab, rec: dict) -> dict:
    """Seal and append one call. Refuses duplicates and refuses to extend a tampered log."""
    with file_lock(lab.calls_path):
        log = read_calls(lab.calls_path)
        if log.bad:
            raise LabError(f"{lab.calls_path.name} fails verification; run `verify` before adding calls")
        if rec["id"] in log.by_id():
            raise LabError(f"call {rec['id']} is already sealed (a seal is never rewritten)")
        rec = dict(rec)
        rec["prev_seal"] = log.head
        rec.pop("seal", None)
        rec["seal"] = seal_of(rec)
        append_line(lab.calls_path, rec)
    return rec


def append_score(lab: Lab, row: dict) -> None:
    with file_lock(lab.scores_path):
        append_line(lab.scores_path, row)


def make_score_row(call: dict, outcome: float, scored_at: dt.datetime, how: str, evidence=None, note: str = "", late=None) -> dict:
    """late: {name: {"p": chance, "odds": [...]}}, yardsticks the call named in late_baselines that exist only now (a closing line).
    Each is scored like any baseline, and when there is one the first becomes this call's primary baseline in the record."""
    kind = call["kind"]
    sc = score_pair(kind, call["prediction"], outcome)
    bs = {k: score_pair(kind, v, outcome) for k, v in (call.get("baselines") or {}).items()}
    row = {
        "id": call["id"], "seal": call["seal"], "outcome": outcome, "score": round(sc, 9),
        "baseline_score": round(score_pair(kind, call["baseline_prediction"], outcome), 9),
        "baseline_scores": {k: round(v, 9) for k, v in bs.items()},
        "scored_at": iso(scored_at), "how": how,
    }
    if late:
        row["late_baselines"] = {}
        for name, info in late.items():
            entry = {"p": round(float(info["p"]), 6)}
            if info.get("odds"):
                entry["odds"] = [float(x) for x in info["odds"]]
            row["late_baselines"][name] = entry
            row["baseline_scores"][name] = round(score_pair(kind, entry["p"], outcome), 9)
        row["primary_baseline"] = next(iter(row["late_baselines"]))
    if evidence:
        row["evidence"] = evidence
    if note:
        row["note"] = note
    return row


# --------------------------------------------------------------------------------------
# aggregation and the honest verdict
# --------------------------------------------------------------------------------------


def aggregate(rows: list, min_n: int = MIN_CALLS) -> dict:
    """rows: dicts with score, bases {name: score}, primary (name). Skill = baseline - score."""
    n = len(rows)
    out = {"n": n, "score": None, "primary": None, "baselines": {}, "verdict": "no scored calls yet", "verdict_code": "none"}
    if n == 0:
        return out
    sc = np.array([r["score"] for r in rows], float)
    out["score"] = float(sc.mean())
    out["primary"] = rows[0]["primary"]
    names = []
    for r in rows:
        for k in r["bases"]:
            if k not in names:
                names.append(k)
    if out["primary"] in names:
        names.remove(out["primary"])
        names.insert(0, out["primary"])
    for k in names:
        pairs = [(r["score"], r["bases"][k]) for r in rows if k in r["bases"]]
        s = np.array([p[0] for p in pairs], float)
        b = np.array([p[1] for p in pairs], float)
        d = b - s
        m = float(d.mean())
        se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
        bm = float(b.mean())
        out["baselines"][k] = {
            "n": int(len(d)), "score": bm, "skill": m, "skill_pct": (m / bm * 100.0) if bm > 0 else None,
            "lo": m - 1.96 * se, "hi": m + 1.96 * se,
            "wins": int((d > BASELINE_EPS).sum()), "misses": int((d < -BASELINE_EPS).sum()),
            "ties": int((np.abs(d) <= BASELINE_EPS).sum()),
        }
    if n < min_n:
        out["verdict_code"] = "too_few"
        out["verdict"] = f"too few scored calls (n={n}, need {min_n}): no skill claim"
        return out
    ahead = [k for k, v in out["baselines"].items() if v["skill"] > 0]
    behind = [k for k, v in out["baselines"].items() if v["skill"] <= 0]
    clear = all(v["lo"] > 0 for v in out["baselines"].values())
    if not behind and clear:
        out["verdict_code"], out["verdict"] = "beats", "beats baseline"
    elif not behind:
        out["verdict_code"], out["verdict"] = "ahead_unclear", "ahead of baseline, but the 95% range includes zero: not a clear edge"
    elif ahead:
        out["verdict_code"], out["verdict"] = "mixed", "mixed: ahead of " + ", ".join(ahead) + "; not ahead of " + ", ".join(behind)
    else:
        out["verdict_code"], out["verdict"] = "behind", "does not beat baseline"
    return out


# --------------------------------------------------------------------------------------
# the plugin interface: one small adapter per domain
# --------------------------------------------------------------------------------------


def make_subject(key, target, kind, question, resolves_at, meta, **private):
    """A subject is one thing to call: one question with one resolution time. Underscore keys stay private."""
    s = {"key": key, "target": target, "kind": kind, "question": question, "resolves_at": resolves_at, "meta": meta}
    s.update(private)
    return s


def call_subject(call: dict) -> dict:
    """Rebuild the subject of a sealed call so an adapter can resolve it later."""
    return {"key": call["id"], "target": call["target"], "kind": call["kind"], "question": call["question"],
            "resolves_at": call["resolves_at"], "meta": call.get("meta") or {}}


class Adapter:
    """Base class for a Prophet domain. Subclass it, fill in the calls below, register it.

    Required calls (all use only data strictly before `as_of`):
      fetch_history(lab)                   download (cached, hashed) and parse; returns an object with .inputs
      features(hist, subject, as_of)       features for one subject, or None when data is missing
      predict(hist, subject, as_of, feats) {"value": probability or number, "method": text}
      baseline(hist, subject, as_of, feats){"primary": name, "values": {name: value}}
      resolve(hist, subject)               outcome (1 or 0 for yes/no, else a number) or None while unknown
    Work lists:
      backtest_subjects(hist)              subjects oldest first, each with an "as_of"
      live_subjects(hist, now)             subjects to call right now (things that have not happened yet)
    """

    name = ""
    title = ""
    status = "ready"                # ready | needs_data_source | blocked
    status_note = ""
    can_predict = True
    sources: list = []              # [{"name", "url", "note"}]
    default_config: dict = {}
    plugin_file = None              # set by the loader for drop-in files
    grades_manual_calls = False     # True: `score` also hands this domain the hand-made calls (origin manual, made with `call`) that carry its name;
                                    # False (the default): `score` leaves them for `resolve`, exactly as before

    def config(self, lab: Lab) -> dict:
        return lab.cfg.get(self.name, self.default_config)

    def fetch_history(self, lab: Lab, fresh: bool = False):
        raise NotImplementedError

    def features(self, hist, subject, as_of):
        raise NotImplementedError

    def predict(self, hist, subject, as_of, feats):
        raise NotImplementedError

    def baseline(self, hist, subject, as_of, feats):
        raise NotImplementedError

    def resolve(self, hist, subject):
        raise NotImplementedError

    def backtest_subjects(self, hist):
        return []

    def live_subjects(self, hist, now):
        return []

    # ---- generic drivers (adapters may override for speed) -------------------------------
    def backtest(self, lab: Lab) -> dict:
        hist = self.fetch_history(lab)
        rows = []
        for subj in self.backtest_subjects(hist):
            as_of = subj["as_of"]
            feats = self.features(hist, subj, as_of)
            if feats is None:
                continue
            fc = self.predict(hist, subj, as_of, feats)
            if fc is None:
                continue
            bl = self.baseline(hist, subj, as_of, feats)
            out = self.resolve(hist, subj)
            if out is None:
                continue
            rows.append({"target": subj["target"], "kind": subj["kind"], "key": subj["key"], "as_of": str(as_of),
                         "pred": float(fc["value"]), "bases": {k: float(v) for k, v in bl["values"].items()},
                         "primary": bl["primary"], "outcome": float(out)})
        return {"rows": rows, "inputs": list(getattr(hist, "inputs", [])), "notes": list(getattr(hist, "notes", []))}

    def resolve_calls(self, lab: Lab, calls: list) -> dict:
        """Outcomes for sealed calls: {id: (outcome or None, evidence inputs)}. An adapter with late_baselines (yardsticks that only exist
        after the event, such as a closing price, named in the call when it is sealed) may add a third item for the calls whose data has
        them: {name: {"p": chance, "odds": [...]}}. A call that did not name a yardstick never gets one."""
        hist = self.fetch_history(lab, fresh=True)
        return {c["id"]: (self.resolve(hist, call_subject(c)), self.evidence(hist, call_subject(c))) for c in calls}

    def evidence(self, hist, subject) -> list:
        """The input files that decided an outcome; recorded in the score row."""
        return list(getattr(hist, "inputs", []))


REGISTRY: dict = {}
PLUGIN_ERRORS: dict = {}


def register(adapter: Adapter) -> Adapter:
    if not adapter.name:
        raise LabError("an adapter needs a name")
    REGISTRY[adapter.name] = adapter
    return adapter


def load_plugins() -> None:
    """Drop-in domains: any prophet_lab_<name>.py next to this file that defines ADAPTER (an Adapter)."""
    me = sys.modules.get("__main__") if __name__ == "__main__" else sys.modules[__name__]
    sys.modules.setdefault("prophet_lab", me)
    for p in sorted(HERE.glob("prophet_lab_*.py")):
        if "." in p.stem:               # prophet_lab_x.before-fix-DATE.py is a backup copy, never a second domain
            continue
        try:
            spec = importlib.util.spec_from_file_location(p.stem, p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[p.stem] = mod
            spec.loader.exec_module(mod)
            found = getattr(mod, "ADAPTERS", None) or ([mod.ADAPTER] if hasattr(mod, "ADAPTER") else [])
            if not found:
                raise LabError("file defines no ADAPTER")
            for a in found:
                a.plugin_file = str(p)
                register(a)
        except Exception as e:           # a broken plugin must be visible, not silent
            PLUGIN_ERRORS[p.stem] = f"{type(e).__name__}: {e}"


# ======================================================================================
# domain: weather  (Open-Meteo, free, no key)
# ======================================================================================

OM_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OM_FORECAST = "https://api.open-meteo.com/v1/forecast"
WET_INCH = 0.01                  # "rain" means measurable precipitation, 0.01 inch or more, on the local day
WEATHER_LAG_DAYS = 8             # outcome is read once the archive value has settled


def doy_clim(doys, values, half_window: int):
    """Mean of `values` for each day of year, smoothed over +/- half_window days (circular)."""
    m = ~np.isnan(values)
    s = np.bincount(doys[m], weights=values[m], minlength=367)[1:367]
    c = np.bincount(doys[m], minlength=367)[1:367].astype(float)
    S = np.zeros(366)
    C = np.zeros(366)
    for k in range(-half_window, half_window + 1):
        S += np.roll(s, k)
        C += np.roll(c, k)
    return np.where(C > 0, S / np.maximum(C, 1.0), np.nan)


def om_url(base, place, start, end, extra=None):
    q = {"latitude": f"{place['lat']:.4f}", "longitude": f"{place['lon']:.4f}",
         "daily": "temperature_2m_max,precipitation_sum", "temperature_unit": "fahrenheit",
         "precipitation_unit": "inch", "timezone": "auto"}
    if start:
        q["start_date"] = start
    if end:
        q["end_date"] = end
    if extra:
        q.update(extra)
    return base + "?" + urllib.parse.urlencode(q, safe=",")


def _validate_om(data: bytes):
    j = json.loads(data.decode("utf-8"))
    if not isinstance(j, dict) or "daily" not in j:
        raise ValueError("not an Open-Meteo daily response")


class Series:
    """Contiguous daily series (local dates) for one place."""

    def __init__(self, name, lat, lon, offset, ords, tmax, precip, inputs):
        self.name, self.lat, self.lon, self.offset, self.inputs = name, lat, lon, offset, inputs
        self.ord = np.asarray(ords, dtype=np.int64)
        self.tmax = np.asarray(tmax, float)
        self.precip = np.asarray(precip, float)
        self.n = len(self.ord)
        self.doy = np.array([dt.date.fromordinal(int(o)).timetuple().tm_yday for o in self.ord], dtype=np.int64)
        self.wet = np.where(np.isnan(self.precip), np.nan, (self.precip >= WET_INCH).astype(float))

    def idx(self, d: dt.date):
        i = d.toordinal() - int(self.ord[0])
        return i if 0 <= i < self.n else None

    def date(self, i: int) -> dt.date:
        return dt.date.fromordinal(int(self.ord[i]))


class WeatherFit:
    """Models trained only on days before index `end`: anomaly regression for the high, ridge logistic for rain."""

    def __init__(self, ser: Series, end: int):
        self.ser, self.end, self.ok = ser, end, False
        d = ser.doy[:end]
        self.clim_t = doy_clim(d, ser.tmax[:end], 3)
        self.clim_r = doy_clim(d, ser.wet[:end], 7)
        self.an = ser.tmax - self.clim_t[ser.doy - 1]
        tt = np.arange(3, end)
        D = tt - 1
        Xt = np.column_stack([np.ones(len(tt)), self.an[D], self.an[D - 1], self.an[D - 2], ser.wet[D]])
        yt = self.an[tt]
        m = ~np.isnan(Xt).any(axis=1) & ~np.isnan(yt)
        if m.sum() < 400:
            return
        self.w_t = np.linalg.lstsq(Xt[m], yt[m], rcond=None)[0]
        Xr = self._rain_x(D, ser.doy[tt])
        yr = ser.wet[tt]
        m2 = ~np.isnan(Xr).any(axis=1) & ~np.isnan(yr)
        self.lr = Logit(l2=2.0).fit(Xr[m2], yr[m2])
        self.n_train = int(m.sum())
        self.first = ser.date(0)
        self.last = ser.date(end - 1)
        self.ok = True

    def _rain_x(self, D, doy_next):
        ser = self.ser
        cr = np.clip(self.clim_r[doy_next - 1], 0.02, 0.98)
        return np.column_stack([ser.wet[D], ser.wet[D - 1], np.log1p(10.0 * ser.precip[D]), np.log(cr / (1 - cr)),
                                self.an[D], ser.tmax[D] - ser.tmax[D - 1]])

    def row(self, D: int):
        """Everything known on local day D about day D+1."""
        ser = self.ser
        if D < 2 or D >= ser.n:
            return None
        dn = dt.date.fromordinal(int(ser.ord[D]) + 1).timetuple().tm_yday
        c_next, r_next = self.clim_t[dn - 1], self.clim_r[dn - 1]
        xt = np.array([1.0, self.an[D], self.an[D - 1], self.an[D - 2], ser.wet[D]])
        if np.isnan(c_next) or np.isnan(r_next) or np.isnan(xt).any():
            return None
        xr = self._rain_x(np.array([D]), np.array([dn]))
        if np.isnan(xr).any():
            return None
        return {"tmax": float(c_next + xt @ self.w_t), "rain": float(self.lr.prob(xr)[0]),
                "persist_t": float(ser.tmax[D]), "clim_t": float(c_next),
                "persist_r": float(ser.wet[D]), "clim_r": float(r_next)}


class WeatherAdapter(Adapter):
    name = "weather"
    title = "Weather: tomorrow's high temperature and rain yes/no"
    sources = [
        {"name": "Open-Meteo Historical Weather API", "url": "https://archive-api.open-meteo.com/v1/archive",
         "note": "free, no key; CC BY 4.0 data, attribution to Open-Meteo.com"},
        {"name": "Open-Meteo Forecast API (past days only, to fill days the archive lacks)",
         "url": "https://api.open-meteo.com/v1/forecast", "note": "free, no key"},
    ]
    default_config = DEFAULT_CONFIG["weather"]

    def __init__(self):
        self._fits = {}

    def fetch_history(self, lab, fresh=False):
        cfg = self.config(lab)
        ttl = 1800.0 if fresh else None
        today = utcnow().date()
        hist = SimpleNamespace(places={}, inputs=[], notes=[])
        for pl in cfg["places"]:
            url = om_url(OM_ARCHIVE, pl, cfg["history_start"], today.isoformat())
            f = lab.fetch(url, ttl=ttl, validate=_validate_om)
            inputs = [{"url": url, "sha256": f.sha256}]
            j = json.loads(f.text("utf-8"))
            d = j["daily"]
            vals = {}
            for t, a, b in zip(d["time"], d["temperature_2m_max"], d["precipitation_sum"]):
                vals[dt.date.fromisoformat(t).toordinal()] = (np.nan if a is None else a, np.nan if b is None else b)
            if not vals:
                raise FetchError(f"no daily data for {pl['name']}")
            lo, hi = min(vals), max(vals)
            ords = list(range(lo, hi + 1))
            tmax = [vals.get(o, (np.nan, np.nan))[0] for o in ords]
            prec = [vals.get(o, (np.nan, np.nan))[1] for o in ords]
            offset = int(j.get("utc_offset_seconds") or 0)
            # the archive can lag by a few days at the very end; fill only the gaps from the forecast API's past days
            if any(np.isnan(x) for x in tmax[-3:]):
                furl = om_url(OM_FORECAST, pl, None, None, {"past_days": 7, "forecast_days": 1})
                ff = lab.fetch(furl, ttl=ttl, validate=_validate_om)
                inputs.append({"url": furl, "sha256": ff.sha256})
                fj = json.loads(ff.text("utf-8"))["daily"]
                for t, a, b in zip(fj["time"], fj["temperature_2m_max"], fj["precipitation_sum"]):
                    o = dt.date.fromisoformat(t).toordinal()
                    if lo <= o <= hi and np.isnan(tmax[o - lo]) and a is not None:
                        tmax[o - lo] = a
                        prec[o - lo] = np.nan if b is None else b
                hist.notes.append(f"{pl['name']}: last days filled from the forecast API's past days")
            hist.places[pl["name"]] = Series(pl["name"], pl["lat"], pl["lon"], offset, ords, tmax, prec, inputs)
            hist.inputs.extend(inputs)
        return hist

    # ---- walk-forward backtest: refit every month on days before that month -------------
    def backtest(self, lab):
        cfg = self.config(lab)
        hist = self.fetch_history(lab)
        t0 = dt.date.fromisoformat(cfg["test_start"])
        rows = []
        for name, ser in hist.places.items():
            i0 = ser.idx(t0)
            if i0 is None or i0 < 500:
                raise LabError(f"{name}: not enough history before test_start {t0}")
            starts = [i for i in range(i0, ser.n) if i == i0 or ser.date(i).day == 1]
            for a, b in zip(starts, starts[1:] + [ser.n]):
                fit = WeatherFit(ser, a)
                if not fit.ok:
                    continue
                for D in range(a, min(b, ser.n - 1)):
                    r = fit.row(D)
                    if r is None:
                        continue
                    key = f"weather:{slug(name)}:{ser.date(D + 1).isoformat()}"
                    t_out, w_out = ser.tmax[D + 1], ser.wet[D + 1]
                    if not np.isnan(t_out):
                        rows.append({"target": "tmax", "kind": "number", "key": key + ":tmax", "as_of": ser.date(D).isoformat(),
                                     "pred": r["tmax"], "bases": {"persistence": r["persist_t"], "climatology": r["clim_t"]},
                                     "primary": "persistence", "outcome": float(t_out)})
                    if not np.isnan(w_out):
                        rows.append({"target": "rain", "kind": "binary", "key": key + ":rain", "as_of": ser.date(D).isoformat(),
                                     "pred": r["rain"], "bases": {"climatology": r["clim_r"], "persistence": r["persist_r"]},
                                     "primary": "climatology", "outcome": float(w_out)})
        notes = [f"{len(hist.places)} places ({', '.join(hist.places)}); each month is predicted by models fitted only on days before that month.",
                 "tmax: anomaly regression on today, yesterday and the day before plus a rain flag. Baselines: persistence (same as today) and climatology (same date, +/-3 days, earlier years).",
                 "rain (0.01 inch or more): ridge logistic on rain today and yesterday, amount, climatology, temperature anomaly and change. Baselines: climatology (+/-7 days) and persistence (rain today means rain tomorrow).",
                 "Daily errors are correlated from day to day, so the 95% ranges are tighter than they should be; read them as a guide."]
        return {"rows": rows, "inputs": hist.inputs, "notes": notes + hist.notes}

    # ---- the five calls, for live use ---------------------------------------------------
    def _fit(self, ser, D):
        k = (ser.name, D, ser.n)
        if k not in self._fits:
            self._fits[k] = WeatherFit(ser, D)
        return self._fits[k]

    def features(self, hist, subject, as_of):
        ser, D = subject["_ser"], subject["_D"]
        fit = self._fit(ser, D)
        if not fit.ok:
            return None
        row = fit.row(D)
        return None if row is None else dict(row, _fit=fit)

    def predict(self, hist, subject, as_of, feats):
        fit, tg = feats["_fit"], subject["target"]
        window = f"{fit.first} to {fit.last}"
        if tg == "tmax":
            return {"value": feats["tmax"], "method": f"anomaly regression (today, yesterday, day before, rain flag) trained on {fit.n_train} days of {fit.ser.name} archive, {window}"}
        return {"value": feats["rain"], "method": f"ridge logistic (rain today and yesterday, amount, climatology, temperature anomaly and change) trained on {fit.n_train} days of {fit.ser.name} archive, {window}"}

    def baseline(self, hist, subject, as_of, feats):
        if subject["target"] == "tmax":
            return {"primary": "persistence", "values": {"persistence": feats["persist_t"], "climatology": feats["clim_t"]}}
        return {"primary": "climatology", "values": {"climatology": feats["clim_r"], "persistence": feats["persist_r"]}}

    def resolve(self, hist, subject):
        meta = subject["meta"]
        ser = hist.places.get(meta["place"])
        if ser is None:
            return None
        i = ser.idx(dt.date.fromisoformat(meta["date"]))
        if i is None:
            return None
        v = ser.tmax[i] if subject["target"] == "tmax" else ser.wet[i]
        return None if np.isnan(v) else float(v)

    def evidence(self, hist, subject):
        ser = hist.places.get(subject["meta"].get("place"))
        return list(ser.inputs) if ser else []

    def live_subjects(self, hist, now):
        subs = []
        for name, ser in hist.places.items():
            L = (now + dt.timedelta(seconds=ser.offset)).date()
            D = ser.idx(L)
            if D is None or np.isnan(ser.tmax[D]):
                hist.notes.append(f"{name}: no value for today ({L}) yet; skipped")
                continue
            target = L + dt.timedelta(days=1)
            resolves = f"{(target + dt.timedelta(days=WEATHER_LAG_DAYS)).isoformat()}T00:00:00Z"
            meta = {"place": name, "date": target.isoformat(), "today": L.isoformat()}
            subs.append(make_subject(
                f"weather:{slug(name)}:tmax:{target.isoformat()}", "tmax", "number",
                f"What will the highest temperature in {name} be on {target} (local day, degrees F, Open-Meteo archive value)?",
                resolves, dict(meta), _ser=ser, _D=D, as_of=L.isoformat(), _inputs=ser.inputs))
            subs.append(make_subject(
                f"weather:{slug(name)}:rain:{target.isoformat()}", "rain", "binary",
                f"Will measurable rain (0.01 inch or more) fall in {name} on {target} (local day, Open-Meteo archive value)?",
                resolves, dict(meta), _ser=ser, _D=D, as_of=L.isoformat(), _inputs=ser.inputs))
        return subs


register(WeatherAdapter())

# ======================================================================================
# domain: sports  (football-data.co.uk results CSVs; Elo + recent form)
# ======================================================================================

FD_RESULTS = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
FD_FIXTURES = "https://www.football-data.co.uk/fixtures.csv"
LEAGUE_NAMES = {"E0": "English Premier League", "E1": "English Championship", "D1": "German Bundesliga",
                "SP1": "Spanish La Liga", "I1": "Italian Serie A", "F1": "French Ligue 1",
                "N1": "Dutch Eredivisie", "P1": "Portuguese Primeira Liga", "B1": "Belgian Pro League",
                "SC0": "Scottish Premiership", "T1": "Turkish Super Lig", "G1": "Greek Super League"}


def _validate_fd(data: bytes):
    head = data[:600].decode("utf-8-sig", "replace")
    if "HomeTeam" not in head:
        raise ValueError("not a football-data results CSV")


def fd_date(s: str):
    s = (s or "").strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


# The market's own chance of a home win, used only to GRADE a call, never to make one. football-data.co.uk lists each bookmaker's
# closing odds (the price just before kick-off) as three columns: home, draw, away. Pinnacle is the sharpest public price.
# In the files read on 2026-10-25 the seasons 2021-22 to 2024-25 carry Pinnacle's closing odds for every match, 2025-26 for about
# half of them, and 2026-27 for none: a match without them has no closing line and is never given a made-up one.
CLOSE_LINES = {"pinnacle_close": ("PSCH", "PSCD", "PSCA")}
MAX_BOOK = 1.25                  # 1/home + 1/draw + 1/away; below 1 or above this is not a real price list, so the row gets no closing line


def devig3(odds) -> list:
    """The chances three decimal odds imply: 1/odds, scaled so the three add up to 1. That takes the bookmaker's margin out."""
    inv = [1.0 / float(x) for x in odds]
    total = sum(inv)
    return [v / total for v in inv]


def odds3(row: dict, cols):
    """(home, draw, away) decimal odds from one CSV row, or None when one is blank, not a number or not above 1, or the book is not plausible."""
    try:
        odds = tuple(float(row.get(c)) for c in cols)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(x) and x > 1.0 for x in odds):
        return None
    return odds if 1.0 <= sum(1.0 / x for x in odds) <= MAX_BOOK else None


def closing_prob(m: dict, name: str = "pinnacle_close"):
    """The market's chance that the home side wins, for one parsed match: the closing odds with the margin taken out. None when the file has none for it."""
    odds = (m.get("close") or {}).get(name)
    return None if odds is None else devig3(odds)[0]


def parse_fd(text: str, season=None) -> list:
    out = []
    for r in csv.DictReader(io.StringIO(text)):
        home, away = (r.get("HomeTeam") or "").strip(), (r.get("AwayTeam") or "").strip()
        d = fd_date(r.get("Date"))
        if not home or not away or d is None:
            continue
        ftr = (r.get("FTR") or "").strip().upper()
        try:
            hg, ag = int(r.get("FTHG")), int(r.get("FTAG"))
        except (TypeError, ValueError):
            hg = ag = None
        close = {}
        for name, cols in CLOSE_LINES.items():
            odds = odds3(r, cols)
            if odds is not None:
                close[name] = odds
        out.append({"div": (r.get("Div") or "").strip(), "date": d, "time": (r.get("Time") or "").strip(),
                    "home": home, "away": away, "ftr": ftr if ftr in ("H", "D", "A") else "",
                    "hg": hg, "ag": ag, "season": season, "close": close})
    out.sort(key=lambda m: (m["date"], m["time"], m["home"]))
    return out


def season_list(today: dt.date, back: int) -> list:
    y = today.year if today.month >= 7 else today.year - 1
    return [f"{(y - k) % 100:02d}{(y - k + 1) % 100:02d}" for k in range(back, -1, -1)]


def _last_sunday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 31)
    return d - dt.timedelta(days=(d.weekday() + 1) % 7)


def uk_to_utc(d: dt.date, hhmm: str):
    """football-data kick-off times are UK time (GMT, or BST from the last Sunday of March to the last Sunday of October)."""
    try:
        h, m = (int(x) for x in (hhmm or "00:00").split(":")[:2])
    except ValueError:
        h, m = 0, 0
    local = dt.datetime(d.year, d.month, d.day, h, m, tzinfo=UTC)
    start = dt.datetime.combine(_last_sunday(d.year, 3), dt.time(1, 0), UTC)
    end = dt.datetime.combine(_last_sunday(d.year, 10), dt.time(1, 0), UTC)
    as_bst = local - dt.timedelta(hours=1)
    return as_bst if start <= as_bst < end else local


class LeagueModel:
    """Elo ratings plus last-5 form for one league, learned strictly in match order."""
    K, HFA, BASE, NEW, SHRINK, AVG_PPG = 20.0, 60.0, 1500.0, 1450.0, 0.75, 1.35
    MIN_TRAIN, REFIT = 300, 100

    def __init__(self):
        self.ratings, self.recent = {}, {}
        self.season, self.season_index = None, -1
        self.X, self.y = [], []
        self.model, self.fit_n = None, 0
        self.home_wins = self.played = 0

    def start_match(self, season):
        if season != self.season:
            if self.season is not None:
                for t in self.ratings:
                    self.ratings[t] = self.BASE + self.SHRINK * (self.ratings[t] - self.BASE)
            self.season = season
            self.season_index += 1

    def rating(self, team):
        return self.ratings.get(team, self.BASE if self.season_index <= 0 else self.NEW)

    def ppg(self, team):
        r = self.recent.get(team, [])[-5:]
        return (sum(r) + self.AVG_PPG * (5 - len(r))) / 5.0

    def feats(self, home, away):
        return [(self.rating(home) - self.rating(away)) / 100.0, self.ppg(home) - self.ppg(away)]

    def update(self, m):
        home, away, ftr = m["home"], m["away"], m["ftr"]
        rh, ra = self.rating(home), self.rating(away)
        e_h = 1.0 / (1.0 + 10.0 ** (-(rh + self.HFA - ra) / 400.0))
        s_h = {"H": 1.0, "D": 0.5, "A": 0.0}[ftr]
        gd = abs((m["hg"] or 0) - (m["ag"] or 0))
        g = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11.0 + gd) / 8.0)
        delta = self.K * g * (s_h - e_h)
        self.ratings[home], self.ratings[away] = rh + delta, ra - delta
        pts = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}[ftr]
        self.recent.setdefault(home, []).append(pts[0])
        self.recent.setdefault(away, []).append(pts[1])
        self.recent[home], self.recent[away] = self.recent[home][-5:], self.recent[away][-5:]
        self.played += 1
        self.home_wins += 1 if ftr == "H" else 0

    def fit_if_needed(self, force=False):
        n = len(self.X)
        if n >= self.MIN_TRAIN and (force or self.model is None or n - self.fit_n >= self.REFIT):
            self.model = Logit(l2=1.0).fit(np.array(self.X), np.array(self.y))
            self.fit_n = n

    def home_rate(self):
        return (self.home_wins + 1.0) / (self.played + 2.0)


def run_league(matches, collect: bool):
    lm, rows = LeagueModel(), []
    for m in matches:
        lm.start_match(m["season"])
        if not m["ftr"]:
            continue
        x = lm.feats(m["home"], m["away"])
        if collect and lm.season_index >= 1:
            lm.fit_if_needed()
            if lm.model is not None:
                rows.append((m, float(lm.model.prob(x)[0]), lm.home_rate()))
        lm.X.append(x)
        lm.y.append(1.0 if m["ftr"] == "H" else 0.0)
        lm.update(m)
    return lm, rows


class SportsAdapter(Adapter):
    name = "sports"
    title = "Sports: football (soccer) home win yes/no"
    sources = [{"name": "football-data.co.uk results and fixtures CSVs", "url": "https://www.football-data.co.uk/data.php",
                "note": "free historical results; credit football-data.co.uk. Prophet makes its calls from results only. Pinnacle's closing odds "
                        "(PSCH, PSCD, PSCA) are read after the match, to grade a call against the market"}]
    default_config = DEFAULT_CONFIG["sports"]
    late_baselines = ("pinnacle_close",)       # written into every NEW call; the closing odds do not exist yet when a call is sealed, so they are added when it is graded

    def fetch_history(self, lab, fresh=False):
        cfg = self.config(lab)
        ttl = 1800.0 if fresh else None
        seasons = season_list(utcnow().date(), int(cfg["seasons_back"]))
        hist = SimpleNamespace(matches={}, fixtures=[], inputs=[], notes=[], by_key={}, season_inputs={}, fixture_input=None, seasons=seasons)
        for div in cfg["leagues"]:
            allm = []
            for s in seasons:
                url = FD_RESULTS.format(season=s, div=div)
                try:
                    f = lab.fetch(url, ttl=ttl, validate=_validate_fd)
                except FetchError as e:
                    if e.code == 404:
                        hist.notes.append(f"{div} {s}: no results file ({e})")
                        continue
                    raise
                allm.extend(parse_fd(f.text(), s))
                inp = {"url": url, "sha256": f.sha256}
                hist.season_inputs.setdefault(div, []).append(inp)
                hist.inputs.append(inp)
            allm.sort(key=lambda m: (m["date"], m["time"], m["home"]))
            hist.matches[div] = allm
            hist.by_key[div] = {(m["date"].isoformat(), m["home"], m["away"]): m["ftr"] for m in allm if m["ftr"]}
        try:
            f = lab.fetch(FD_FIXTURES, ttl=ttl, validate=_validate_fd)
            hist.fixtures = [m for m in parse_fd(f.text()) if m["div"] in cfg["leagues"]]
            hist.fixture_input = {"url": FD_FIXTURES, "sha256": f.sha256}
            hist.inputs.append(hist.fixture_input)
        except FetchError as e:
            hist.notes.append(f"fixtures file unavailable: {e}")
        return hist

    def backtest(self, lab):
        hist = self.fetch_history(lab)
        rows, per_league, unpriced = [], [], 0
        for div, ms in hist.matches.items():
            _, res = run_league(ms, True)
            scored = 0
            for m, p, base in res:
                q = closing_prob(m)
                if q is None:               # no Pinnacle closing odds in the file for this match: there is nothing to grade it against
                    unpriced += 1
                    continue
                rows.append({"target": "home_win", "kind": "binary", "key": f"{div}:{m['date'].isoformat()}:{m['home']}-{m['away']}",
                             "as_of": m["date"].isoformat(), "pred": p, "bases": {"pinnacle_close": q, "league_home_rate": base},
                             "primary": "pinnacle_close", "outcome": 1.0 if m["ftr"] == "H" else 0.0})
                scored += 1
            per_league.append(f"{div} {scored}")
        if not rows:
            raise LabError("sports: no match in the downloaded files has Pinnacle closing odds (PSCH, PSCD, PSCA), so the practice run has nothing to be graded against")
        notes = [f"Leagues and calls: {', '.join(per_league)}. Seasons {hist.seasons[0]} to {hist.seasons[-1]}; the first season of each league only warms the ratings up."
                 + (f" {unpriced} matches have no Pinnacle closing odds in their file (the newest files carry none) and are left out, not given a made-up price." if unpriced else ""),
                 "Pattern: Elo rating gap (K=20, goal-difference weighting, ratings pulled 25% toward average each new season) plus points in the last five matches, turned into a home-win probability by a ridge logistic refitted every 100 matches. Draws count as not a home win.",
                 "Baselines: pinnacle_close (primary), Pinnacle's closing odds for the match with the bookmaker's margin taken out (1/odds, scaled so home, draw and away add up to 1); "
                 "league_home_rate, the league's home-win rate so far. The call itself never uses odds: the closing odds are only the yardstick, and the closing price is the market's final answer, a hard one to be ahead of."] + hist.notes
        return {"rows": rows, "inputs": hist.inputs, "notes": notes}

    # ---- live: upcoming fixtures ----------------------------------------------------
    def live_subjects(self, hist, now):
        models = {}
        for div, ms in hist.matches.items():
            lm, _ = run_league(ms, False)
            lm.fit_if_needed(force=True)
            models[div] = lm
        subs = []
        for fx in hist.fixtures:
            ko = uk_to_utc(fx["date"], fx["time"])
            lm = models.get(fx["div"])
            if lm is None or lm.model is None or ko <= now + dt.timedelta(minutes=30):
                continue
            if (fx["date"].isoformat(), fx["home"], fx["away"]) in hist.by_key.get(fx["div"], {}):
                continue
            date_iso = fx["date"].isoformat()
            league = LEAGUE_NAMES.get(fx["div"], fx["div"])
            meta = {"div": fx["div"], "date": date_iso, "home": fx["home"], "away": fx["away"],
                    "season": season_code_for(fx["date"]), "kickoff_utc": iso(ko)}
            unseen = [t for t in (fx["home"], fx["away"]) if t not in lm.ratings]
            subs.append(make_subject(
                f"sports:{fx['div']}:{date_iso}:{slug(fx['home'], 24)}-{slug(fx['away'], 24)}:home_win", "home_win", "binary",
                f"Will {fx['home']} beat {fx['away']} at home in the {league} match on {date_iso} (home win in normal time)?",
                iso(ko + dt.timedelta(hours=3)), meta, _lm=lm, _x=lm.feats(fx["home"], fx["away"]), _unseen=unseen,
                as_of=iso(now), _inputs=list(hist.season_inputs.get(fx["div"], [])) + ([hist.fixture_input] if hist.fixture_input else [])))
        return subs

    def features(self, hist, subject, as_of):
        return {"x": subject["_x"], "unseen": subject["_unseen"]}

    def predict(self, hist, subject, as_of, feats):
        lm = subject["_lm"]
        p = float(lm.model.prob(feats["x"])[0])
        note = f"; no history for {', '.join(feats['unseen'])} (started at the default rating)" if feats["unseen"] else ""
        return {"value": p, "method": f"Elo gap + last-5 form, ridge logistic on {len(lm.X)} {subject['meta']['div']} matches before {as_of[:10]}{note}"}

    def baseline(self, hist, subject, as_of, feats):
        return {"primary": "league_home_rate", "values": {"league_home_rate": subject["_lm"].home_rate()}}

    def resolve(self, hist, subject):
        m = subject["meta"]
        r = hist.by_key.get(m["div"], {}).get((m["date"], m["home"], m["away"]))
        return None if not r else (1.0 if r == "H" else 0.0)

    def resolve_calls(self, lab, calls):
        need = {}
        for c in calls:
            need.setdefault((c["meta"]["div"], c["meta"]["season"]), []).append(c)
        out = {}
        for (div, season), cs in need.items():
            url = FD_RESULTS.format(season=season, div=div)
            f = lab.fetch(url, ttl=1800.0, validate=_validate_fd)
            res = {(m["date"].isoformat(), m["home"], m["away"]): m for m in parse_fd(f.text(), season) if m["ftr"]}
            for c in cs:
                m = c["meta"]
                row = res.get((m["date"], m["home"], m["away"]))
                evidence = [{"url": url, "sha256": f.sha256}]
                if row is None:
                    out[c["id"]] = (None, evidence)
                    continue
                late = {}                   # only the lines this call named when it was sealed, and only if the file has them for this match
                for name in c.get("late_baselines") or []:
                    odds = (row.get("close") or {}).get(name)
                    if odds is not None:
                        late[name] = {"p": devig3(odds)[0], "odds": list(odds)}
                out[c["id"]] = (1.0 if row["ftr"] == "H" else 0.0, evidence, late or None)
        return out


def season_code_for(d: dt.date) -> str:
    return season_list(d, 0)[0]


register(SportsAdapter())

# ======================================================================================
# domain: markets  (FRED public CSV; Stooq is behind a browser check, so it is not used)
# ======================================================================================

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}&cosd={since}"
MARKET_HORIZONS = (1, 2, 3, 4)


def _validate_fred(data: bytes):
    if not data[:40].decode("utf-8", "replace").lower().startswith(("observation_date", "date")):
        raise ValueError("not a FRED CSV")


def parse_fred(text: str):
    ords, vals = [], []
    for r in list(csv.reader(io.StringIO(text)))[1:]:
        if len(r) < 2 or r[1].strip() in ("", "."):
            continue
        try:
            d, x = dt.date.fromisoformat(r[0].strip()), float(r[1])
        except ValueError:
            continue
        if x > 0:
            ords.append(d.toordinal())
            vals.append(x)
    return np.array(ords, dtype=np.int64), np.array(vals, float)


def mkt_features(c):
    n = len(c)
    lc = np.log(c)
    F = np.full((n, 5), np.nan)
    r1 = np.full(n, np.nan)
    r1[1:] = lc[1:] - lc[:-1]
    F[:, 0] = r1
    for j, k in ((1, 5), (2, 20), (3, 60)):
        F[k:, j] = lc[k:] - lc[:-k]
    if n > 21:
        F[20:, 4] = np.lib.stride_tricks.sliding_window_view(r1[1:], 20).std(axis=1)
    return F


def mkt_labels(c, h):
    n = len(c)
    y = np.full(n, np.nan)
    if n > h:
        y[: n - h] = (c[h:] > c[: n - h]).astype(float)
    return y


def next_session(now: dt.datetime) -> dt.date:
    """First weekday whose US market open (13:30 UTC, the earlier of winter/summer) is still ahead."""
    d = now.date()
    while True:
        if d.weekday() < 5 and dt.datetime.combine(d, dt.time(13, 30), UTC) > now:
            return d
        d += dt.timedelta(days=1)


class MarketsAdapter(Adapter):
    name = "markets"
    title = "Markets: will the index close higher? (pattern research, not advice)"
    sources = [
        {"name": "FRED public CSV (Federal Reserve Bank of St. Louis)", "url": "https://fred.stlouisfed.org/",
         "note": "free, no key; index levels belong to their owners and are used here for research"},
        {"name": "Stooq daily CSV (the suggested source)", "url": "https://stooq.com/q/d/l/?s=spy.us&i=d",
         "note": "NOT USED: it answers a plain download with a JavaScript browser check, which Prophet does not try to pass"},
    ]
    default_config = DEFAULT_CONFIG["markets"]

    def fetch_history(self, lab, fresh=False):
        cfg = self.config(lab)
        ttl = 1800.0 if fresh else None
        hist = SimpleNamespace(series={}, inputs=[], notes=[])
        for s in cfg["series"]:
            url = FRED_CSV.format(id=s["id"], since=cfg["since"])
            f = lab.fetch(url, ttl=ttl, validate=_validate_fred)
            ords, close = parse_fred(f.text("utf-8"))
            if len(ords) < 800:
                raise FetchError(f"{s['id']}: only {len(ords)} usable closes")
            inp = {"url": url, "sha256": f.sha256}
            hist.series[s["id"]] = SimpleNamespace(id=s["id"], name=s["name"], ords=ords, close=close, inputs=[inp])
            hist.inputs.append(inp)
        return hist

    def backtest(self, lab):
        hist = self.fetch_history(lab)
        rows = []
        START, REFIT = 750, 63
        for sid, s in hist.series.items():
            c, F = s.close, mkt_features(s.close)
            n = len(c)
            for h in MARKET_HORIZONS:
                y = mkt_labels(c, h)
                known = np.nan_to_num(y)
                cum = np.cumsum(known)
                model, last_fit = None, -10 ** 9
                for t in range(START, n - h):
                    if t - last_fit >= REFIT:
                        idx = np.arange(60, t - h + 1)
                        ok = ~np.isnan(F[idx]).any(axis=1) & ~np.isnan(y[idx])
                        model = Logit(l2=5.0).fit(F[idx][ok], y[idx][ok])
                        last_fit = t
                    cnt = t - h + 1                      # labels known at t: indices 0 .. t-h
                    base = (cum[t - h] + 1.0) / (cnt + 2.0)
                    rows.append({"target": f"up_h{h}", "kind": "binary", "key": f"{sid}:{dt.date.fromordinal(int(s.ords[t])).isoformat()}:h{h}",
                                 "as_of": dt.date.fromordinal(int(s.ords[t])).isoformat(), "pred": float(model.prob(F[t])[0]),
                                 "bases": {"up_rate": float(base)}, "primary": "up_rate", "outcome": float(y[t])})
        notes = [f"Series: {', '.join(s.name for s in hist.series.values())}, FRED daily closes (about ten years). Replay starts after three years of history.",
                 "Question: is the close h trading days after day t higher than the close on day t? up_h1 is the next-day call; up_h2 to up_h4 cover the days a source lags behind.",
                 "Pattern: ridge logistic on 1, 5, 20 and 60 day returns and 20-day volatility, refitted every 63 trading days on labels already known. Baseline: always 'up' at the historical up-rate.",
                 "Pattern research only. Nothing here is advice to buy, sell or hold anything."] + hist.notes
        return {"rows": rows, "inputs": hist.inputs, "notes": notes}

    def live_subjects(self, hist, now):
        T = next_session(now)
        subs = []
        for sid, s in hist.series.items():
            R = len(s.close) - 1
            ref_date = dt.date.fromordinal(int(s.ords[R]))
            h = int(np.busday_count(np.datetime64(ref_date.isoformat()), np.datetime64(T.isoformat())))
            if h < 1 or h > max(MARKET_HORIZONS):
                hist.notes.append(f"{sid}: latest close is {ref_date}, {h} sessions before {T}; skipped (too stale)")
                continue
            subs.append(make_subject(
                f"markets:{sid.lower()}:up:{T.isoformat()}", f"up_h{h}", "binary",
                f"Will the {s.name} close on {T} (or the first close after it) be above its {ref_date} close of {s.close[R]:,.2f}? Pattern research only, not advice.",
                f"{(T + dt.timedelta(days=1)).isoformat()}T00:00:00Z",
                {"series": sid, "ref_date": ref_date.isoformat(), "ref_close": float(s.close[R]), "target_date": T.isoformat(), "h": h},
                _s=s, _R=R, _h=h, as_of=iso(now), _inputs=s.inputs))
        return subs

    def features(self, hist, subject, as_of):
        s, R, h = subject["_s"], subject["_R"], subject["_h"]
        F, y = mkt_features(s.close), mkt_labels(s.close, h)
        idx = np.arange(60, R - h + 1)
        ok = ~np.isnan(F[idx]).any(axis=1) & ~np.isnan(y[idx])
        if ok.sum() < 300 or np.isnan(F[R]).any():
            return None
        return {"F": F, "y": y, "idx": idx[ok], "x": F[R], "n_train": int(ok.sum())}

    def predict(self, hist, subject, as_of, feats):
        s, h = subject["_s"], subject["_h"]
        model = Logit(l2=5.0).fit(feats["F"][feats["idx"]], feats["y"][feats["idx"]])
        ref = dt.date.fromordinal(int(s.ords[subject["_R"]])).isoformat()
        return {"value": float(model.prob(feats["x"])[0]),
                "method": f"ridge logistic on 1/5/20/60-day returns and 20-day volatility, trained on {feats['n_train']} {s.id} closes up to {ref}, horizon {h} trading day(s)"}

    def baseline(self, hist, subject, as_of, feats):
        y = feats["y"][feats["idx"]]
        return {"primary": "up_rate", "values": {"up_rate": float((y.sum() + 1.0) / (len(y) + 2.0))}}

    def resolve(self, hist, subject):
        m = subject["meta"]
        s = hist.series.get(m["series"])
        if s is None:
            return None
        j = int(np.searchsorted(s.ords, dt.date.fromisoformat(m["target_date"]).toordinal()))
        return None if j >= len(s.ords) else (1.0 if s.close[j] > m["ref_close"] else 0.0)

    def evidence(self, hist, subject):
        s = hist.series.get(subject["meta"].get("series"))
        return list(s.inputs) if s else []


register(MarketsAdapter())


# ======================================================================================
# building calls, backtests, scoring
# ======================================================================================


def build_call(lab: Lab, adapter: Adapter, subject: dict, fc: dict, bl: dict, inputs: list, now: dt.datetime) -> dict:
    kind = subject["kind"]
    rnd = 4 if kind == "binary" else 2
    value = float(fc["value"])
    if not math.isfinite(value) or (kind == "binary" and not 0.0 <= value <= 1.0):
        raise LabError(f"{subject['key']}: prediction {value} is not valid for a {kind} call")
    prim = bl["primary"]
    values = {k: round(float(v), rnd) for k, v in bl["values"].items()}
    rec = {
        "id": subject["key"], "domain": adapter.name, "target": subject["target"], "kind": kind,
        "question": subject["question"], "made_at": iso(now), "resolves_at": subject["resolves_at"],
        "prediction": round(value, rnd), "baseline_prediction": values[prim], "baselines": values,
        "primary_baseline": prim, "method": fc["method"], "origin": "adapter", "meta": subject["meta"],
        "inputs": [{"url": i["url"], "sha256": i["sha256"]} for i in inputs], "code_sha256": code_sha(),
    }
    if adapter.plugin_file:
        rec["plugin_sha256"] = sha256_bytes(Path(adapter.plugin_file).read_bytes())
    late = list(getattr(adapter, "late_baselines", ()) or ())
    if late:                        # yardsticks that only exist after the event (a closing price): named now, so none can be picked once the result is known
        rec["late_baselines"] = late
    return rec


def write_rows_csv(path: Path, rows: list) -> str:
    names = []
    for r in rows:
        for k in r["bases"]:
            if k not in names:
                names.append(k)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["target", "key", "as_of", "outcome", "pred"] + ["base_" + n for n in names])
    for r in rows:
        w.writerow([r["target"], r["key"], r["as_of"], f"{r['outcome']:g}", f"{r['pred']:.6f}"]
                   + [("" if r["bases"].get(n) is None else f"{r['bases'][n]:.6f}") for n in names])
    text = buf.getvalue()
    atomic_write(path, text)
    return sha256_text(text)


def _example(item, kind):
    r = item["row"]
    return {"key": r["key"], "as_of": r["as_of"], "called": round(r["pred"], 4), "baseline": round(r["bases"][item["primary"]], 4),
            "outcome": r["outcome"], "gap": round(item["bases"][item["primary"]] - item["score"], 4)}


def run_backtest(lab: Lab, adapter: Adapter) -> dict:
    res = adapter.backtest(lab)
    rows = res["rows"]
    if not rows:
        raise LabError(f"{adapter.name}: the backtest produced no scored calls")
    grouped = {}
    for r in rows:
        sc = score_pair(r["kind"], r["pred"], r["outcome"])
        bs = {k: score_pair(r["kind"], v, r["outcome"]) for k, v in r["bases"].items()}
        grouped.setdefault(r["target"], []).append({"score": sc, "bases": bs, "primary": r["primary"], "row": r})
    targets = {}
    for tg, items in grouped.items():
        agg = aggregate(items)
        ranked = sorted(items, key=lambda it: it["bases"][it["primary"]] - it["score"])
        agg["kind"] = items[0]["row"]["kind"]
        agg["first"] = min(it["row"]["as_of"] for it in items)
        agg["last"] = max(it["row"]["as_of"] for it in items)
        agg["misses_worst"] = [_example(it, agg["kind"]) for it in ranked[:3]]
        agg["wins_best"] = [_example(it, agg["kind"]) for it in ranked[::-1][:3]]
        targets[tg] = agg
    csv_rel = f"backtest/{adapter.name}.rows.csv"
    sha = write_rows_csv(lab.root / csv_rel, rows)
    summary = {
        "schema": "prophet-backtest/1", "domain": adapter.name, "status": "ok", "run_at": iso(utcnow()),
        "code_sha256": code_sha(), "mode": "backtest (walk-forward; each call used only data before its date)",
        "inputs": res["inputs"], "targets": targets, "notes": res.get("notes", []),
        "rows_file": csv_rel, "rows": len(rows), "rows_sha256": sha,
    }
    if adapter.plugin_file:
        summary["plugin_sha256"] = sha256_bytes(Path(adapter.plugin_file).read_bytes())
    atomic_write(lab.backtest_dir / f"{adapter.name}.json", json.dumps(summary, indent=1, default=float) + "\n")
    return summary


def target_label(adapter, domain: str, target: str, kind: str) -> str:
    labels = getattr(adapter, "target_labels", None) or {}
    if target in labels:
        return labels[target]
    fixed = {("weather", "tmax"): "next-day high temperature (MAE, deg F)",
             ("weather", "rain"): "next-day rain yes/no (Brier)",
             ("sports", "home_win"): "football home win yes/no (Brier)"}
    if (domain, target) in fixed:
        return fixed[(domain, target)]
    m = re.fullmatch(r"up_h(\d+)", target)
    if domain == "markets" and m:
        h = int(m.group(1))
        return "next-close up yes/no (Brier)" if h == 1 else f"close {h} sessions out up yes/no (Brier)"
    if target == "manual":
        return f"hand-made calls ({SCORE_NAME.get(kind, kind)})"
    return f"{target} ({SCORE_NAME.get(kind, kind)})"


def print_agg(label: str, agg: dict):
    parts = [f"n={agg['n']}", f"Prophet {num(agg['score'])}"]
    for k, v in agg["baselines"].items():
        parts.append(f"{k} {num(v['score'])} (skill {num(v['skill'], 3, True)})")
    print(f"  {label}: " + "  ".join(parts) + f"  -> {agg['verdict']}")


# ---- live state: sealed calls joined with their scores ----------------------------------


def valid_scores(v: Verification) -> dict:
    """{call id: score row} for every score row that passed verification."""
    bad_lines = {line for line, _, _ in v.score_problems}
    scored = {}
    for row in v.scores:
        if row.get("_bad") or row["_line"] in bad_lines or not row.get("id"):
            continue
        scored[row["id"]] = row
    return scored


def live_state(lab: Lab, v: Verification | None = None) -> dict:
    v = v or verify_lab(lab)
    calls = v.log.ok_calls
    scored = valid_scores(v)
    now = utcnow()
    groups = {}
    for c in calls:
        g = groups.setdefault((c["domain"], c["target"]), {"kind": c["kind"], "sealed": 0, "items": [], "waiting": 0, "overdue": 0,
                                                              "next_due": None})
        g["sealed"] += 1
        row = scored.get(c["id"])
        if row is None:
            g["waiting"] += 1
            due = parse_iso(c["resolves_at"])
            if due <= now:
                g["overdue"] += 1
            elif g["next_due"] is None or due < g["next_due"]:
                g["next_due"] = due
            continue
        bases = dict(row.get("baseline_scores") or {})
        prim = row.get("primary_baseline") or c.get("primary_baseline") or "baseline"       # a closing line added at grading becomes the primary
        if prim not in bases:
            bases[prim] = row["baseline_score"]
        late_p = ((row.get("late_baselines") or {}).get(prim) or {}).get("p")
        g["items"].append({"score": row["score"], "bases": bases, "primary": prim,
                           "row": {"key": c["id"], "as_of": c["made_at"][:10], "pred": c["prediction"],
                                   "bases": {prim: c["baseline_prediction"] if late_p is None else late_p}, "outcome": row["outcome"]}})
    for g in groups.values():
        g["agg"] = aggregate(g["items"])
        ranked = sorted(g["items"], key=lambda it: it["bases"][it["primary"]] - it["score"])
        g["misses_worst"] = [_example(it, g["kind"]) for it in ranked[:3] if it["bases"][it["primary"]] - it["score"] < 0]
        g["wins_best"] = [_example(it, g["kind"]) for it in ranked[::-1][:3] if it["bases"][it["primary"]] - it["score"] > 0]
    return {"groups": groups, "verification": v, "calls": calls}


# ======================================================================================
# commands
# ======================================================================================


def pick_domains(arg: str, need_predict: bool = False) -> list:
    if arg in ("all", "*"):
        names = [n for n, a in REGISTRY.items() if a.status == "ready" and (a.can_predict or not need_predict)]
        return names
    if arg not in REGISTRY:
        raise LabError(f"unknown domain '{arg}'. Domains: {', '.join(sorted(REGISTRY))}. Use `call` for a hand-made call on any other question.")
    return [arg]


def cmd_backtest(lab, args):
    names = pick_domains(args.domain)
    failed = 0
    for name in names:
        ad = REGISTRY[name]
        if ad.status != "ready":
            print(f"[{name}] {ad.status.replace('_', ' ')}: {ad.status_note}")
            failed += 1                 # named on purpose and refused: not a success (`all` never lists a domain that is not ready)
            continue
        print(f"[{name}] backtest (walk-forward, data before each date only)")
        started = utcnow()
        try:
            summary = run_backtest(lab, ad)
        except (FetchError, LabError) as e:
            failed += 1
            print(f"[{name}] FAILED: {e}")
            atomic_write(lab.backtest_dir / f"{name}.error.json", json.dumps({"domain": name, "run_at": iso(started), "error": str(e)}, indent=1) + "\n")
            continue
        for tg, agg in summary["targets"].items():
            print_agg(target_label(ad, name, tg, agg["kind"]), agg)
        print(f"  {summary['rows']} scored calls from {min(a['first'] for a in summary['targets'].values())} to {max(a['last'] for a in summary['targets'].values())}; rows in {summary['rows_file']} (sha256 {summary['rows_sha256'][:12]})")
    return 1 if failed else 0


def cmd_predict(lab, args):
    names = pick_domains(args.domain, need_predict=True)
    if args.domain not in ("all", "*") and REGISTRY[args.domain].status != "ready":
        ad = REGISTRY[args.domain]
        print(f"[{args.domain}] {ad.status.replace('_', ' ')}: {ad.status_note}")
        print(f"[{args.domain}] refused: nothing was sealed.")
        return 1
    if args.domain not in ("all", "*") and not REGISTRY[args.domain].can_predict:
        print(f"[{args.domain}] backtest only: it has no live calls.")
        return 0
    now = utcnow()
    problems = 0
    sealed_total = 0
    save_snapshot(lab, SELF)
    for name in names:
        ad = REGISTRY[name]
        if ad.status != "ready":
            print(f"[{name}] {ad.status.replace('_', ' ')}: {ad.status_note}")
            continue
        try:
            hist = ad.fetch_history(lab)
            subjects = ad.live_subjects(hist, now)
        except (FetchError, LabError) as e:
            problems += 1
            print(f"[{name}] could not run: {e}")
            continue
        if ad.plugin_file:
            save_snapshot(lab, Path(ad.plugin_file))
        have = set(read_calls(lab.calls_path).by_id())
        sealed = skipped = 0
        for s in subjects:
            if s["key"] in have:
                skipped += 1
                continue
            as_of = s.get("as_of") or iso(now)
            feats = ad.features(hist, s, as_of)
            if feats is None:
                print(f"  skip {s['key']}: not enough data")
                continue
            fc = ad.predict(hist, s, as_of, feats)
            if fc is None:
                continue
            bl = ad.baseline(hist, s, as_of, feats)
            rec = append_call(lab, build_call(lab, ad, s, fc, bl, s.get("_inputs") or getattr(hist, "inputs", []), now))
            have.add(rec["id"])
            sealed += 1
            later = f"  also graded against {', '.join(rec['late_baselines'])} once the result is in" if rec.get("late_baselines") else ""
            print(f"  sealed {rec['id']}  call {rec['prediction']}  baseline {rec['baseline_prediction']} ({rec['primary_baseline']})  resolves {rec['resolves_at']}  seal {rec['seal'][:12]}{later}")
        for n in getattr(hist, "notes", []):
            if "skipped" in n or "filled" in n:
                print(f"  note: {n}")
        print(f"[{name}] sealed {sealed}, already sealed {skipped}")
        sealed_total += sealed
    print(f"Sealed {sealed_total} new calls into {lab.calls_path.name}.")
    return 1 if problems else 0


def cmd_score(lab, args):
    v = verify_lab(lab)
    if not v.ok:
        print_verification(v, per_call_limit=0)
    scored_ids = {r["id"] for r in v.scores if r.get("id") and not r.get("_bad")}
    now = utcnow()
    open_calls = [c for c in v.log.ok_calls if c["id"] not in scored_ids]
    due = [c for c in open_calls if parse_iso(c["resolves_at"]) <= now]
    print(f"{len(open_calls)} sealed calls have no score yet; {len(due)} are due.")
    by_domain = {}
    manual = []
    for c in due:
        ad = REGISTRY.get(c["domain"])
        if ad is None or (c.get("origin") == "manual" and not getattr(ad, "grades_manual_calls", False)):
            manual.append(c)
        else:
            by_domain.setdefault(c["domain"], []).append(c)
    scored = unresolved = problems = 0
    no_line = {}                        # late yardstick name -> scored calls whose data had no such line for that event
    for dom, cs in by_domain.items():
        ad = REGISTRY[dom]
        try:
            results = ad.resolve_calls(lab, cs)
        except (FetchError, LabError) as e:
            problems += 1
            print(f"[{dom}] could not read outcomes: {e}")
            continue
        for c in cs:
            got = results.get(c["id"], (None, []))
            outcome, evidence = got[0], got[1]
            late = got[2] if len(got) > 2 else None       # yardsticks that exist only after the event (a closing line), see make_score_row
            if outcome is None:
                unresolved += 1
                print(f"  waiting {c['id']}: outcome not in the data yet")
                continue
            row = make_score_row(c, float(outcome), now, "auto", evidence=evidence, late=late)
            append_score(lab, row)
            scored += 1
            extra = "".join(f"; {k} {v['p']:.4f} scores {row['baseline_scores'][k]:.4f}" for k, v in (row.get("late_baselines") or {}).items())
            print(f"  scored {c['id']}: called {c['prediction']}, baseline {c['baseline_prediction']}, outcome {outcome:g}; score {row['score']:.4f} vs baseline {row['baseline_score']:.4f}{extra}")
            for name in c.get("late_baselines") or []:
                if name not in (row.get("late_baselines") or {}):
                    no_line[name] = no_line.get(name, 0) + 1
    for name, n in no_line.items():
        print(f"{n} scored call(s) named the yardstick {name}, but their data has none for that event, so they were graded against the sealed baseline only.")
    if manual:
        print(f"{len(manual)} hand-made calls are past their date. Score each with: resolve <id> --outcome yes|no|<number>")
        for c in manual[:20]:
            print(f"  {c['id']}")
    print(f"Scored {scored} now; {unresolved} still waiting for data.")
    return 1 if (problems or not v.ok) else 0


def cmd_verify(lab, args):
    v = verify_lab(lab)
    print_verification(v)
    return 0 if v.ok else 1


def parse_outcome(kind: str, text: str) -> float:
    t = text.strip().lower()
    if kind == "binary":
        if t in ("yes", "y", "true", "1", "1.0"):
            return 1.0
        if t in ("no", "n", "false", "0", "0.0"):
            return 0.0
        raise LabError("this is a yes/no call: use --outcome yes or --outcome no")
    try:
        x = float(t)
    except ValueError:
        raise LabError("this is a number call: use --outcome <number>")
    if not math.isfinite(x):
        raise LabError("the outcome must be a finite number")
    return x


def parse_resolves(text: str, today: dt.date) -> dt.date:
    """The day a call resolves: YYYY-MM-DD, or the word today or tomorrow (a UTC date, like every date the lab keeps)."""
    word = str(text).strip().lower()
    if word == "today":
        return today
    if word == "tomorrow":
        return today + dt.timedelta(days=1)
    try:
        return dt.date.fromisoformat(word)
    except ValueError:
        raise LabError("--resolves must be a date like 2026-12-31, or the word today or tomorrow (dates are UTC)")


def cmd_call(lab, args):
    if (args.prob is None) == (args.value is None):
        raise LabError("give exactly one of --prob (yes/no call) or --value (number call)")
    now = utcnow()
    resolves = parse_resolves(args.resolves, now.date())
    if resolves < now.date():
        raise LabError("--resolves is in the past: a call must be sealed before its outcome")
    kind = "binary" if args.prob is not None else "number"
    pred = args.prob if kind == "binary" else args.value
    if kind == "binary":
        if not 0.0 <= pred <= 1.0:
            raise LabError("--prob must be between 0 and 1")
        base = 0.5 if args.baseline is None else args.baseline
        if not 0.0 <= base <= 1.0:
            raise LabError("--baseline must be between 0 and 1 for a yes/no call")
        base_name = "coin flip" if args.baseline is None else "stated baseline"
    else:
        if args.baseline is None:
            raise LabError("a number call needs --baseline (what a plain guess would say)")
        base, base_name = args.baseline, "stated baseline"
    domain = slug(args.domain, 32)
    rnd = 4 if kind == "binary" else 6
    rec = {
        "id": f"{domain}:{slug(args.target or 'manual', 24)}:{resolves.isoformat()}:{sha256_text(args.question.strip())[:8]}",
        "domain": domain, "target": args.target or "manual", "kind": kind, "question": args.question.strip(),
        "made_at": iso(now), "resolves_at": day_end_iso(resolves), "prediction": round(float(pred), rnd),
        "baseline_prediction": round(float(base), rnd), "baselines": {base_name: round(float(base), rnd)},
        "primary_baseline": base_name, "method": args.method, "origin": "manual", "meta": {},
        "inputs": [{"url": u, "sha256": None} for u in (args.source or [])], "code_sha256": code_sha(),
    }
    save_snapshot(lab, SELF)
    rec = append_call(lab, rec)
    print(f"Sealed {rec['id']}")
    print(f"  question: {rec['question']}")
    print(f"  call {rec['prediction']}  baseline {rec['baseline_prediction']} ({base_name})  resolves {rec['resolves_at']}  seal {rec['seal']}")
    return 0


def cmd_resolve(lab, args):
    v = verify_lab(lab)
    entry = v.log.by_id().get(args.id)
    if entry is None:
        raise LabError(f"no sealed call with id {args.id}")
    if not entry["ok"]:
        raise LabError(f"call {args.id} is TAMPERED ({'; '.join(entry['problems'])}); it cannot be scored")
    rec = entry["rec"]
    if rec.get("origin") != "manual":
        raise LabError("this call is scored from its data source by `score`, not by hand")
    if any(r.get("id") == args.id for r in v.scores if not r.get("_bad")):
        raise LabError("this call already has a score; a score is never rewritten")
    outcome = parse_outcome(rec["kind"], args.outcome)
    now = utcnow()
    if now < parse_iso(rec["made_at"]):
        raise LabError("the clock is earlier than the call's made_at; check the date on this computer")
    row = make_score_row(rec, outcome, now, "manual", evidence=args.evidence, note=args.note or "")
    append_score(lab, row)
    verdict = "better than" if row["score"] < row["baseline_score"] - BASELINE_EPS else ("worse than" if row["score"] > row["baseline_score"] + BASELINE_EPS else "level with")
    early = " (resolved before its date)" if now < parse_iso(rec["resolves_at"]) else ""
    print(f"Scored {rec['id']}{early}: outcome {outcome:g}; score {row['score']:.4f} vs baseline {row['baseline_score']:.4f}; the call was {verdict} the baseline")
    return 0


def cmd_list(lab, args):
    v = verify_lab(lab)
    scored = {r["id"]: r for r in v.scores if r.get("id") and not r.get("_bad")}
    shown = 0
    for e in v.log.entries:
        rec = e["rec"] or {}
        if not e["ok"]:
            print(f"TAMPERED  {rec.get('id', '(line ' + str(e['line']) + ')')}: {'; '.join(e['problems'])}")
            continue
        row = scored.get(rec["id"])
        if args.open and row or args.scored and not row:
            continue
        status = f"scored {row['score']:.4f} vs {row['baseline_score']:.4f} (outcome {row['outcome']:g})" if row else f"open, resolves {rec['resolves_at']}"
        print(f"{rec['id']}  call {rec['prediction']}  baseline {rec['baseline_prediction']}  {status}")
        shown += 1
    print(f"{shown} shown, {len(v.log.entries)} sealed in total.")
    return 0 if v.ok else 1


def cmd_domains(lab, args):
    for name in sorted(REGISTRY):
        a = REGISTRY[name]
        extra = " (drop-in file)" if a.plugin_file else ""
        print(f"{name:12} {a.status.replace('_', ' '):18} {'live+backtest' if a.can_predict else 'backtest only':14} {a.title}{extra}")
        if a.status_note:
            print(f"{'':12} {a.status_note}")
        for s in a.sources:
            print(f"{'':12} source: {s['name']} - {s['url']} ({s['note']})")
    for name, err in PLUGIN_ERRORS.items():
        print(f"{name}: drop-in file failed to load: {err}")
    print("Any other domain name works with `call` (hand-made calls, scored with `resolve`).")
    return 0


# ======================================================================================
# anchors: prove WHEN the record existed (OpenTimestamps)
# ======================================================================================
# A seal shows that nothing was edited since a call was made. It cannot show WHEN the call was made: the date is this computer's
# clock. `anchor` sends the chain head (the seal of the newest call: a 32-byte SHA-256 that fingerprints the whole list) to two
# public OpenTimestamps calendar servers. Each answers with a receipt that is "pending". Within hours the calendars commit it to
# Bitcoin, and the official `ots` client can then upgrade the receipt into a proof that anyone can check. Only the 32 bytes leave
# this computer: no question, number, file name or call. The calendar servers do see this computer's address and the time, as any
# web server does.

OTS_CALENDARS = ("https://a.pool.opentimestamps.org/digest", "https://b.pool.opentimestamps.org/digest")
ANCHOR_UA = "ProphetLab/1.0 anchor"
OTS_TIMEOUT = 30.0
OTS_RETRY_WAIT = 2.0                 # seconds before the one retry (a network failure or an HTTP 5xx); tests make it short
OTS_MAX_ANSWER = 20_000              # a calendar's answer is a few hundred bytes; anything bigger is not a receipt
OTS_MAGIC = b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"      # the first bytes of every .ots file
OTS_OPS = {0x02: "sha1", 0x03: "ripemd160", 0x08: "sha256", 0x67: "keccak256", 0xF2: "reverse", 0xF3: "hexlify"}     # no argument
OTS_ARG_OPS = {0xF0: "append", 0xF1: "prepend"}                                                                      # one argument
OTS_PENDING = bytes.fromhex("83dfe30d2ef90c8e")          # attestation: "this calendar will commit the hash to Bitcoin"
OTS_BITCOIN = bytes.fromhex("0588960d73d71901")          # attestation: "this hash is in a Bitcoin block" (what `ots upgrade` adds)


class _Bytes:
    """A cursor over bytes. Reading past the end raises ValueError, so a cut-off answer can never parse."""

    def __init__(self, data):
        self.data, self.pos = bytes(data), 0

    def take(self, n):
        if n < 0 or self.pos + n > len(self.data):
            raise ValueError("the data ends too early")
        chunk = self.data[self.pos:self.pos + n]
        self.pos += n
        return chunk

    def byte(self):
        return self.take(1)[0]

    def varuint(self):
        value = shift = 0
        while True:
            b = self.byte()
            value |= (b & 0x7F) << shift
            if not b & 0x80:
                return value
            shift += 7
            if shift > 56:
                raise ValueError("a number is too long")

    def varbytes(self, limit, minimum=0):
        n = self.varuint()
        if n > limit or n < minimum:
            raise ValueError("a length is out of range")
        return self.take(n)

    def at_end(self):
        return self.pos == len(self.data)


def _varuint_bytes(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | 0x80 if n else b)
        if not n:
            return bytes(out)


def ots_read_node(r: _Bytes, depth: int = 0) -> dict:
    """One node of an OpenTimestamps tree: {"atts": [(kind, payload)], "ops": [(op tag, argument, child node)]}. The layout is the
    one the official client reads: items joined by a 0xff marker, the last item without one."""
    if depth > 256:
        raise ValueError("the receipt nests too deeply")
    node = {"atts": [], "ops": []}
    while True:
        tag = r.byte()
        more = tag == 0xFF
        if more:
            tag = r.byte()
        if tag == 0x00:
            node["atts"].append((r.take(8), r.varbytes(8192)))
        elif tag in OTS_ARG_OPS:
            argument = r.varbytes(4096, minimum=1)
            node["ops"].append((tag, argument, ots_read_node(r, depth + 1)))
        elif tag in OTS_OPS:
            node["ops"].append((tag, b"", ots_read_node(r, depth + 1)))
        else:
            raise ValueError("unknown operation 0x%02x" % tag)
        if not more:
            return node


def ots_write_node(node: dict) -> bytes:
    items = [("att", a) for a in sorted(node["atts"])] + [("op", o) for o in sorted(node["ops"], key=lambda o: (o[0], o[1]))]
    if not items:
        raise ValueError("an empty node cannot be written")
    out = bytearray()
    for i, (kind, item) in enumerate(items):
        if i < len(items) - 1:
            out.append(0xFF)
        if kind == "att":
            out += b"\x00" + item[0] + _varuint_bytes(len(item[1])) + item[1]
        else:
            tag, argument, child = item
            out.append(tag)
            if tag in OTS_ARG_OPS:
                out += _varuint_bytes(len(argument)) + argument
            out += ots_write_node(child)
    return bytes(out)


def ots_merge(a: dict, b: dict) -> dict:
    """Two receipts for the same hash, as one tree (what the official client does when it asks several calendars)."""
    ops = {}
    for tag, argument, child in a["ops"] + b["ops"]:
        ops[(tag, argument)] = ots_merge(ops[(tag, argument)], child) if (tag, argument) in ops else child
    return {"atts": sorted(set(a["atts"]) | set(b["atts"])), "ops": [(t, ar, c) for (t, ar), c in sorted(ops.items())]}


def ots_attestations(node: dict):
    for att in node["atts"]:
        yield att
    for _tag, _argument, child in node["ops"]:
        yield from ots_attestations(child)


def ots_parse(answer: bytes) -> dict:
    """A calendar's answer: one whole OpenTimestamps tree for the submitted hash, holding at least one attestation. Anything else
    (an HTML page from a proxy, a cut-off answer, trailing bytes) raises ValueError."""
    r = _Bytes(answer)
    node = ots_read_node(r)
    if not r.at_end():
        raise ValueError("extra bytes after the receipt")
    if not list(ots_attestations(node)):
        raise ValueError("the receipt holds no attestation")
    return node


def ots_pending_uris(node: dict) -> list:
    """The calendar addresses that promised to commit the hash to Bitcoin."""
    found = []
    for kind, payload in ots_attestations(node):
        if kind == OTS_PENDING:
            try:
                uri = _Bytes(payload).varbytes(1000).decode("ascii")
            except (ValueError, UnicodeDecodeError):
                continue
            if uri not in found:
                found.append(uri)
    return found


def ots_bitcoin_blocks(node: dict) -> list:
    """The Bitcoin block heights a receipt claims (present only after `ots upgrade`)."""
    heights = []
    for kind, payload in ots_attestations(node):
        if kind == OTS_BITCOIN:
            try:
                heights.append(_Bytes(payload).varuint())
            except ValueError:
                continue
    return sorted(set(heights))


def ots_file(digest: bytes, node: dict) -> bytes:
    """A detached timestamp file (.ots) for a SHA-256 digest, in the format of the official `ots` client."""
    return OTS_MAGIC + _varuint_bytes(1) + bytes([0x08]) + digest + ots_write_node(node)


def ots_read_file(data: bytes):
    """(digest, tree) of a .ots file that stamps a SHA-256 digest; ValueError for anything else."""
    r = _Bytes(data)
    if r.take(len(OTS_MAGIC)) != OTS_MAGIC:
        raise ValueError("not an OpenTimestamps file")
    if r.varuint() != 1:
        raise ValueError("an OpenTimestamps file of another major version")
    if r.byte() != 0x08:
        raise ValueError("the file stamps a digest that is not SHA-256")
    digest = r.take(32)
    node = ots_read_node(r)
    if not r.at_end():
        raise ValueError("extra bytes after the receipt")
    return digest, node


def ots_submit(url: str, digest: bytes, timeout: float = OTS_TIMEOUT):
    """POST the raw digest to one calendar. Returns (HTTP status, body); an error status comes back as a value, not an exception.
    A network failure raises FetchError. A redirect is never followed: the hash goes to the address that was checked, and nowhere else."""
    request = urllib.request.Request(url, data=digest, method="POST", headers={
        "User-Agent": ANCHOR_UA, "Accept": "application/vnd.opentimestamps.v1", "Content-Type": "application/octet-stream"})
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            return response.status, response.read(OTS_MAX_ANSWER + 1)
    except urllib.error.HTTPError as error:
        try:
            body = error.read(300)
        except Exception:
            body = b""
        finally:
            error.close()
        return error.code, body
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, http.client.HTTPException) as error:
        raise FetchError(f"could not reach {url}: {getattr(error, 'reason', None) or error}")


def ots_ask(url: str, digest: bytes) -> dict:
    """One calendar's answer, with at most one retry (after a network failure, an HTTP 429 or an HTTP 5xx). Any other status is final."""
    answer = {"status": None, "body": b"", "error": "not asked"}
    for attempt in range(2):
        try:
            status, body = ots_submit(url, digest, OTS_TIMEOUT)
            answer = {"status": status, "body": body, "error": None}
        except FetchError as e:
            answer = {"status": None, "body": b"", "error": str(e)}
        if answer["status"] is not None and answer["status"] < 500 and answer["status"] != 429:
            break
        if attempt == 0:
            time.sleep(OTS_RETRY_WAIT)
    return answer


def ots_judge(answer: dict):
    """(accepted, why not, tree) for one calendar's answer. Only an HTTP 200 whose body is a real receipt counts."""
    if answer["error"]:
        return False, answer["error"], None
    if answer["status"] != 200:
        return False, f"HTTP {answer['status']}", None
    body = answer["body"]
    if not body or len(body) > OTS_MAX_ANSWER:
        return False, "HTTP 200 but the answer is empty or too large to be a receipt", None
    try:
        return True, "", ots_parse(body)
    except ValueError as e:
        return False, f"HTTP 200 but the body is not an OpenTimestamps receipt ({e})", None


def anchor_log_path(lab: Lab) -> Path:
    return lab.root / "anchors.jsonl"


def read_anchor_log(lab: Lab) -> list:
    rows = []
    path = anchor_log_path(lab)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def free_name(path: Path) -> Path:
    """path, or path with -2, -3 ... before the extension when it exists: a receipt that was upgraded later is never overwritten."""
    if not path.exists():
        return path
    n = 2
    while True:
        other = path.with_name(f"{path.stem}-{n}{path.suffix}")
        if not other.exists():
            return other
        n += 1


def anchor_state(lab: Lab, v: Verification) -> dict:
    """What the record says about anchors, from anchors.jsonl, the receipt file and the chain as it stands now.
    state: never | current (the newest call is covered) | behind (newer calls exist) | foreign (the anchored head is not in this chain).
    receipt: pending | bitcoin (the file claims a Bitcoin block; the `ots` client must check it) | missing | unreadable | mismatch."""
    rows = [r for r in read_anchor_log(lab) if r.get("ok") and isinstance(r.get("head"), str)]
    if not rows:
        return {"state": "never"}
    last = rows[-1]
    seals = [e["rec"]["seal"] for e in v.log.entries if e["rec"] and e["rec"].get("seal")]
    if last["head"] not in seals:
        state, newer = "foreign", None
    else:
        newer = len(seals) - 1 - seals.index(last["head"])
        state = "current" if newer == 0 else "behind"
    receipt, blocks, uris = "missing", [], []
    path = lab.root / str(last.get("file") or "")
    if last.get("file") and path.is_file():
        try:
            digest, node = ots_read_file(path.read_bytes())
            blocks, uris = ots_bitcoin_blocks(node), ots_pending_uris(node)
            receipt = "mismatch" if digest.hex() != last["head"] else ("bitcoin" if blocks else "pending")
        except (ValueError, OSError):
            receipt = "unreadable"
    ok = [c for c in last.get("calendars", []) if c.get("ok")]
    return {"state": state, "date": last.get("date"), "head": last["head"], "file": last.get("file"), "newer": newer,
            "calendars_ok": len(ok), "calendars_total": len(last.get("calendars", [])), "receipt": receipt,
            "blocks": blocks, "calendar_uris": uris, "anchored_calls": last.get("calls"), "runs": len(rows)}


def anchor_sentence(a: dict) -> str:
    if a["state"] == "never":
        return ("Not anchored yet. A seal shows that nothing was edited, but not when a call was made. "
                "Run `prophet lab anchor` to get a dated receipt from OpenTimestamps.")
    base = f"Last anchored on {a['date']} ({a['calendars_ok']} of {a['calendars_total']} calendars accepted the hash)."
    if a["state"] == "foreign":
        return (base + " WARNING: that chain head is not in this record. The record was changed after it was anchored, "
                       "or this is a different record.")
    where = ("The newest call is covered." if a["state"] == "current"
             else f"{a['newer']} newer call{'s' if a['newer'] != 1 else ''} {'are' if a['newer'] != 1 else 'is'} not anchored yet.")
    if a["receipt"] == "bitcoin":
        blocks = ", ".join(str(b) for b in a["blocks"])
        tail = f" The receipt file now claims Bitcoin block {blocks}. This page does not check that: use the `ots` client."
    elif a["receipt"] == "pending":
        tail = " Status: pending. The calendars hold the hash; Bitcoin has not been checked. Nothing is confirmed yet."
    else:
        tail = f" WARNING: the receipt file is {a['receipt']}."
    return f"{base} {where}{tail}"


def cmd_anchor(lab, args):
    v = verify_lab(lab)
    if not v.log.entries:
        raise LabError("nothing is sealed yet, so there is no chain head to anchor. Seal a call first (`call` or `predict`).")
    if not v.ok:
        print_verification(v, per_call_limit=0)
        raise LabError("the record fails verification, so it is not anchored: a date on a broken record proves nothing. "
                       "Find the cause first (`verify`).")
    head = v.log.head
    digest = bytes.fromhex(head)
    if len(digest) != 32:
        raise LabError("the chain head is not a 32-byte SHA-256 hash")
    for url in OTS_CALENDARS:
        if net.api_entry(urllib.parse.urlsplit(url)) is None:
            raise LabError(f"{url} is not on prophet_net.API_ALLOWLIST, so nothing is sent")
    day = utcnow().strftime("%Y-%m-%d")
    rel = f"anchors/{day}_{head[:12]}.ots-pending"
    target = free_name(lab.root / rel)
    print(f"Prophet anchor{' (dry run)' if args.dry_run else ''}")
    print(f"Chain head (seal of the newest call): {head}")
    print(f"Seal check: ALL SEALS OK ({len(v.log.entries)} sealed call{'' if len(v.log.entries) == 1 else 's'})")
    if args.dry_run:
        print(f"Would send {len(digest)} bytes, the chain head as raw bytes, with POST to:")
        for url in OTS_CALENDARS:
            print(f"  {url}")
        print(f"Headers: User-Agent: {ANCHOR_UA}; Accept: application/vnd.opentimestamps.v1; Content-Type: application/octet-stream")
        print("Only that hash leaves this computer: no question, no number, no file. The servers see this computer's address and the time.")
        print(f"Would save the receipt as {target.relative_to(lab.root).as_posix()} and add one line to anchors.jsonl.")
        print("Dry run: nothing was sent, and no receipt or anchors.jsonl line was written.")
        return 0
    earlier = [r for r in read_anchor_log(lab) if r.get("ok") and r.get("head") == head]
    if earlier and not args.again:
        print(f"This chain head was already anchored on {earlier[-1].get('date')} ({earlier[-1].get('file')}). Nothing is sent.")
        print("Seal a new call to get a new chain head, or use --again to send this one a second time.")
        return 0
    results = []
    for url in OTS_CALENDARS:
        answer = ots_ask(url, digest)
        ok, why, node = ots_judge(answer)
        body = answer["body"] if ok else b""
        results.append({"url": url, "ok": ok, "status": answer["status"], "bytes": len(body) if ok else len(answer["body"]),
                        "sha256": sha256_bytes(body) if ok else None, "error": None if ok else why,
                        "pending": ots_pending_uris(node) if ok else [], "_node": node})
        print(f"  {url}  " + (f"HTTP {answer['status']}  {len(body)} bytes  sha256 {sha256_bytes(body)[:16]}  pending at {', '.join(results[-1]['pending']) or '(not named)'}"
                              if ok else f"FAILED: {why}"))
    good = [r for r in results if r["ok"]]
    if not good:
        raise LabError("no calendar accepted the hash, so nothing was saved. " + "; ".join(f"{r['url']}: {r['error']}" for r in results))
    tree = good[0]["_node"]
    for r in good[1:]:
        tree = ots_merge(tree, r["_node"])
    receipt = ots_file(digest, tree)
    atomic_write_bytes(target, receipt)
    entry = {"schema": "prophet-anchor/1", "date": day, "anchored_at": iso(utcnow()), "head": head, "calls": len(v.log.entries),
             "file": target.relative_to(lab.root).as_posix(), "file_bytes": len(receipt), "file_sha256": sha256_bytes(receipt),
             "calendars": [{k: r[k] for k in ("url", "ok", "status", "bytes", "sha256", "error", "pending")} for r in results],
             "ok": True, "state": "pending"}
    append_line(anchor_log_path(lab), entry)
    for r in results:
        if not r["ok"]:
            print(f"WARNING: {r['url']} did not accept the hash ({r['error']}). The receipt below comes from the other calendar.")
    print(f"Saved {target.relative_to(lab.root).as_posix()} ({len(receipt)} bytes) and one line in anchors.jsonl.")
    print("Status: PENDING. The calendar servers hold the hash and will commit it to Bitcoin, usually within hours. Nothing is confirmed yet.")
    print("Later: copy the receipt to a name ending in .ots, run `ots upgrade` on it (pip install opentimestamps-client), then `ots info`.")
    return 0


# ======================================================================================
# the record: RECORD.md (ask format) and record.html
# ======================================================================================

PLUGIN_EXAMPLE = '''import prophet_lab as lab
class Adapter(lab.Adapter):
    name = "rivers"; title = "River level tomorrow"
    sources = [{"name": "agency CSV", "url": "https://example.org/levels.csv", "note": "free, no key"}]
    def fetch_history(self, lab_, fresh=False): ...   # lab_.fetch(url) gives bytes + sha256; return an object with .inputs
    def features(self, hist, subject, as_of): ...     # use only data before as_of
    def predict(self, hist, subject, as_of, feats): return {"value": 3.2, "method": "how this number was made"}
    def baseline(self, hist, subject, as_of, feats): return {"primary": "persistence", "values": {"persistence": 3.0}}
    def resolve(self, hist, subject): ...             # outcome (1 or 0, or a number) or None while unknown
    def backtest_subjects(self, hist): ...            # lab.make_subject(...) per past case, oldest first, each with "as_of"
    def live_subjects(self, hist, now): ...           # lab.make_subject(...) per case that has not happened yet
ADAPTER = Adapter()'''

TEXT_HOW = ("How it works. Prophet learns patterns from past records: weather history, football results, index closes, and any domain added to it. "
            "For each question it writes a pattern call, a probability for a yes/no question or a number, next to a plain baseline such as 'same as today' or 'the usual rate'. "
            "The call is sealed before the outcome exists: its fields go into calls.jsonl with a sha256 seal over the canonical JSON and a link to the previous seal. "
            "Later the outcome is read from the data source and the call is scored. Yes/no calls use the Brier score and number calls use the absolute error, so lower is better. "
            "Skill is the baseline score minus the Prophet score: positive means better than the baseline.")
TEXT_EVIDENCE = (f"Evidence rule. A domain is shown as 'beats baseline' only when it has at least {MIN_CALLS} scored calls, is ahead of every baseline listed for it, and the 95% range of the gap stays above zero. "
                 "With fewer scored calls nothing is claimed and the scores are printed anyway. Backtests (walk-forward replays where each call uses only data from before its date) and live sealed calls are kept in separate tables and are never mixed. "
                 "A backtest is a rehearsal; only live sealed calls are a track record.")
TEXT_SEAL = ("What a seal proves. Any edit to a sealed call, a removed or reordered call, or a score that does not match its outcome is reported as TAMPERED by `verify`, and every command checks the seals when it reads the log. "
             "A seal alone does not prove when the call was made (the clock is this computer's) and it cannot stop someone who rewrites the whole file and every hash. "
             "`anchor` closes that gap: it sends the chain head (a 32-byte hash, nothing else) to two public OpenTimestamps servers and keeps their receipt under anchors/. "
             "The receipt is pending at first. After it is upgraded to a Bitcoin proof with the official `ots` client, anyone can check that the record existed by that date. Until then nothing is confirmed. "
             "The exact code that made a call is kept in code_snapshots/ under the code_sha256 stored in the call.")
TEXT_PLUGIN = ("How to add a domain (plugin interface). A domain is one small adapter. Save a file named prophet_lab_<name>.py next to prophet_lab.py and the next run picks it up; no other file changes. "
               "The file starts with `import prophet_lab as lab`, defines `class Adapter(lab.Adapter)` with name, title, sources and status, and ends with `ADAPTER = Adapter()`. "
               "The adapter supplies five calls and two work lists. fetch_history(lab) downloads through lab.fetch (polite User-Agent; the site's robots.txt or the official-API allowlist is checked first; cached under raw/, URL, time and sha256 recorded) and returns the parsed history with an inputs list. "
               "features(hist, subject, as_of) builds features from data strictly before as_of. predict(hist, subject, as_of, feats) returns {'value': probability or number, 'method': text}. "
               "baseline(hist, subject, as_of, feats) returns {'primary': name, 'values': {name: value}}. resolve(hist, subject) returns the outcome (1 or 0 for yes/no, otherwise the number) or None while it is unknown. "
               "backtest_subjects(hist) lists past cases oldest first and live_subjects(hist, now) lists what to call right now; lab.make_subject builds each case. "
               "Free public data only, no keys, no scraping of sites that forbid it. If no free source exists, set status = 'needs_data_source' and the record shows that instead of a score. Heavy adapters may override backtest(lab) and resolve_calls(lab, calls).")
TEXT_MANUAL = ("Any question without a data feed. `prophet lab call --domain <any name> --question \"...\" --resolves YYYY-MM-DD --prob 0.62 [--baseline 0.5]` seals a hand-made or agent-made call (use --value and --baseline for a number; --resolves also takes today or tomorrow). "
               "When the answer is known, `prophet lab resolve <id> --outcome yes|no|<number>` scores it. Hand-made calls appear in the live table under their own domain with the target 'manual'.")
TEXT_COMMANDS = ("Commands, all run as `prophet lab <command>` from any folder: backtest <domain|all>, predict <domain|all>, score, call, resolve <id> --outcome, verify, record, domains, list, anchor [--dry-run].")


def describe_agg(agg: dict) -> str:
    bits = []
    for name, b in agg["baselines"].items():
        bits.append(f"{name} {num(b['score'])} (skill {num(b['skill'], 3, True)}, 95% range {num(b['lo'], 3, True)} to {num(b['hi'], 3, True)}, wins {b['wins']}, misses {b['misses']})")
    return f"n={agg['n']}, Prophet {num(agg['score'])}; " + "; ".join(bits) + f". Verdict: {agg['verdict']}."


def table_rows(label: str, agg: dict, lead: list | None = None) -> list:
    lead = lead or []
    if not agg["baselines"]:
        return [[label] + lead + [str(agg["n"]), "", "", "", "", "", "", "", agg["verdict"]]]
    rows = []
    for i, (name, b) in enumerate(agg["baselines"].items()):
        first = i == 0
        rows.append([label if first else ""] + (lead if first else [""] * len(lead)) +
                    [str(agg["n"]) if first else "", num(agg["score"]) if first else "", name, num(b["score"]),
                     num(b["skill"], 3, True), f"{num(b['lo'], 3, True)} to {num(b['hi'], 3, True)}", str(b["wins"]), str(b["misses"]),
                     agg["verdict"] if first else ""])
    return rows


def examples_text(prefix: str, items: list, primary: str) -> str:
    if not items:
        return ""
    bits = [f"{e['key']} ({e['as_of']}) called {e['called']}, {primary} {e['baseline']}, outcome {e['outcome']:g}" for e in items]
    return f"{prefix}: " + "; ".join(bits) + "."


def ordered_domains(live: dict) -> list:
    names = list(REGISTRY)
    for (d, _t) in live["groups"]:
        if d not in names:
            names.append(d)
    return names


def read_summary(lab: Lab, name: str):
    summary = error = None
    p, e = lab.backtest_dir / f"{name}.json", lab.backtest_dir / f"{name}.error.json"
    if p.exists():
        summary = json.loads(p.read_text(encoding="utf-8"))
    if e.exists():
        error = json.loads(e.read_text(encoding="utf-8"))
    return summary, error


def build_blocks(lab: Lab, live: dict, now: dt.datetime) -> list:
    v = live["verification"]
    names = ordered_domains(live)
    B = [("h1", "Prophet Pattern Lab record"),
         ("p", f"Prophet Pattern Lab record, generated {iso(now)} by prophet_lab.py (code sha256 {code_sha()}). It is rewritten by `record` from the sealed files, not from memory. "
               "Prophet makes pattern calls and keeps a track record. It places no bets or trades and gives no betting, trading or investment advice."),
         ("p", TEXT_HOW), ("p", TEXT_EVIDENCE), ("p", TEXT_SEAL), ("h2", "How to add a domain"), ("p", TEXT_PLUGIN), ("pre", PLUGIN_EXAMPLE),
         ("p", TEXT_MANUAL), ("p", TEXT_COMMANDS)]
    # ---- status of every domain
    status_bits, bt_rows, live_rows = [], [], []
    bt_head = ["Domain: target", "n", "Prophet", "Baseline", "Baseline score", "Skill", "95% range", "Wins", "Misses", "Verdict"]
    live_head = ["Domain: target", "Sealed", "Scored", "Prophet", "Baseline", "Baseline score", "Skill", "95% range", "Wins", "Misses", "Verdict"]
    sections = []
    for name in names:
        ad = REGISTRY.get(name)
        summary, error = read_summary(lab, name)
        status = (ad.status.replace("_", " ") if ad else "hand-made calls only")
        if name in PLUGIN_ERRORS:
            status = "drop-in file failed to load"
        status_bits.append(f"{name} {status}" + (" (backtest only)" if ad and not ad.can_predict else ""))
        sec = [("h2", ad.title if ad else f"{name}: hand-made calls")]
        if ad:
            sec.append(("p", f"{name} status: {status}." + (f" {ad.status_note}" if ad.status_note else "")))
        else:
            sec.append(("p", f"{name} status: hand-made calls only. These were sealed with `call` and are scored with `resolve`; there is no data feed behind them."))
        if summary:
            for tg, agg in summary["targets"].items():
                label = target_label(ad, name, tg, agg["kind"])
                bt_rows.extend(table_rows(f"{name}: {label}", agg))
                sec.append(("p", f"{name} backtest, {label}, {agg['first']} to {agg['last']}: " + describe_agg(agg)))
                for txt in (examples_text("Biggest misses against the baseline", agg.get("misses_worst", []), agg["primary"]),
                            examples_text("Biggest wins against the baseline", agg.get("wins_best", []), agg["primary"])):
                    if txt:
                        sec.append(("p", f"{name} backtest {tg}. {txt}"))
            for n in summary.get("notes", []):
                sec.append(("p", f"{name} method note: {n}"))
            hosts = sorted({urllib.parse.urlsplit(i["url"]).netloc or "local files" for i in summary["inputs"]})
            sec.append(("p", f"{name} backtest receipts: {summary['rows']} scored calls in {summary['rows_file']} (sha256 {summary['rows_sha256']}), run {summary['run_at']} with code sha256 {summary['code_sha256'][:16]}; "
                             f"{len(summary['inputs'])} input files with hashes recorded in backtest/{name}.json, from {', '.join(hosts)}."))
        elif error:
            sec.append(("p", f"{name} backtest FAILED at {error['run_at']}: {error['error']}"))
        elif ad and ad.status == "ready":
            sec.append(("p", f"{name} backtest: not run yet. Run `backtest {name}`."))
        if summary and error and error["run_at"] > summary["run_at"]:
            sec.append(("p", f"{name} latest backtest attempt FAILED at {error['run_at']}: {error['error']} (the numbers above are from the earlier run)."))
        # ---- live
        groups = sorted(((t, g) for (d, t), g in live["groups"].items() if d == name), key=lambda x: x[0])
        if not groups:
            if ad and ad.can_predict and ad.status == "ready":
                sec.append(("p", f"{name} live calls: none sealed yet. Run `predict {name}`."))
        for tg, g in groups:
            label = target_label(ad, name, tg, g["kind"])
            lead = [str(g["sealed"])]
            live_rows.extend(table_rows(f"{name}: {label}", g["agg"], lead))
            when = f" Next outcome expected {iso(g['next_due'])}." if g["next_due"] else ""
            txt = f"{name} live, {label}: {g['sealed']} sealed, {g['agg']['n']} scored, {g['waiting']} waiting for an outcome ({g['overdue']} past their date).{when} "
            txt += describe_agg(g["agg"]) if g["agg"]["n"] else "No outcome is known yet, so no live skill is claimed."
            sec.append(("p", txt))
            for t2 in (examples_text("Live misses against the baseline", g["misses_worst"], g["agg"]["primary"] or "baseline"),
                       examples_text("Live wins against the baseline", g["wins_best"], g["agg"]["primary"] or "baseline")):
                if t2:
                    sec.append(("p", f"{name} live {tg}. {t2}"))
        if ad and ad.sources:
            sec.append(("p", f"{name} data sources: " + "; ".join(f"{s['name']} ({s['url']}; {s['note']})" for s in ad.sources) + "."))
        sections.append(sec)
    n_sealed, n_scored = len(live["calls"]), sum(g["agg"]["n"] for g in live["groups"].values())
    B.append(("h2", "At a glance"))
    B.append(("p", "Domains: " + "; ".join(status_bits) + f". Live sealed calls: {n_sealed}. Live calls scored: {n_scored}. "
                   + ("No live call has been scored yet, so there is no live track record to show; the backtests below are rehearsals." if n_scored == 0
                      else f"Live skill is claimed only for domains with at least {MIN_CALLS} scored calls (see the verdict column).")))
    B.append(("h2", "Backtests (rehearsal: walk-forward replay of history)"))
    B.append(("table", bt_head, bt_rows or [["no backtest has been run yet"] + [""] * 9]))
    B.append(("h2", "Live sealed calls (the track record)"))
    B.append(("table", live_head, live_rows or [["no live call has been scored or sealed yet"] + [""] * 10]))
    for sec in sections:
        B.extend(sec)
    for name, err in PLUGIN_ERRORS.items():
        B.append(("p", f"Drop-in file {name}.py failed to load: {err}"))
    B.append(("h2", "Integrity and files"))
    bad = len(v.log.bad) + len(v.score_problems)
    B.append(("p", f"Integrity check at generation time: {len(v.log.entries)} sealed calls, {len(v.log.entries) - len(v.log.bad)} OK, {len(v.log.bad)} TAMPERED; {len(v.scores)} score rows, {len(v.score_problems)} problems. "
                   + ("Result: ALL SEALS OK." if bad == 0 else "Result: TAMPERED. Do not trust this record until the cause is understood.")
                   + f" Chain head (seal of the newest call): {v.log.head or '(empty log)'}."))
    B.append(("p", "Anchor: " + anchor_sentence(anchor_state(lab, v))))
    B.append(("p", "Files in the pattern-lab folder: calls.jsonl (sealed calls, append only), scores.jsonl (scores, append only), backtest/ (summary json and every scored row as csv), "
                   "raw/ (every download with its URL, fetch time and sha256 in raw/manifest.jsonl), code_snapshots/ (the exact code behind each seal), config.json (places, leagues and series, editable), "
                   "anchors/ and anchors.jsonl (OpenTimestamps receipts for chain heads, written by `anchor`)."))
    return B


def _md_cell(s) -> str:
    return str(s).replace("|", "/").replace("\n", " ")


def render_md(lab: Lab, blocks: list) -> str:
    url = (lab.root / "RECORD.md").as_posix()
    head = f'---\ntitle: "Prophet Pattern Lab record"\nchannel: "lab"\nurl: {json.dumps(url)}\n---\n'
    parts = []
    for b in blocks:
        if b[0] == "h1":
            parts.append("# " + b[1])
        elif b[0] == "h2":
            parts.append("## " + b[1])
        elif b[0] == "p":
            parts.append(b[1])
        elif b[0] == "pre":
            parts.append("```\n" + b[1] + "\n```")
        elif b[0] == "table":
            lines = ["| " + " | ".join(_md_cell(h) for h in b[1]) + " |", "|" + "|".join(" --- " for _ in b[1]) + "|"]
            lines += ["| " + " | ".join(_md_cell(c) for c in r) + " |" for r in b[2]]
            parts.append("\n".join(lines))
    return head + "\n\n".join(parts) + "\n"


# ---- record.html: the page people look at ---------------------------------------------------------------------------------
# Built from the sealed files only (calls.jsonl, scores.jsonl, the practice summaries in backtest/, anchors.jsonl). Nothing on it is
# typed by hand. One file: no scripts, no web fonts, nothing loaded from the internet.

DOMAIN_NAMES = {"weather": "Weather", "sports": "Football", "markets": "Stock markets", "geopolitics": "World events",
                "business": "Business", "racing": "Horse racing", "production": "Production"}
PLAIN_BASELINE = {"persistence": "same as today", "climatology": "the usual for that date", "league_home_rate": "the league's usual home-win rate",
                  "up_rate": "the usual up rate", "weekday_rate": "that weekday's usual rate",
                  "coin flip": "a coin flip", "stated baseline": "the baseline you stated",
                  "pinnacle_close": "Pinnacle's closing odds, margin taken out"}
MARKET_LINES = frozenset(CLOSE_LINES)        # baselines that are a market's own price, not a lazy guess: the page does not call them one
PLAIN_TARGET = {("weather", "tmax"): "Tomorrow's high temperature", ("weather", "rain"): "Will it rain tomorrow?",
                ("sports", "home_win"): "Will the home team win?", ("geopolitics", "up_h2"): "Will an event type's share rise?"}
TARGET_UNIT = {("weather", "tmax"): "degrees F"}
VERDICT_WORDS = {"beats": ("Ahead of the lazy guess", "good"), "ahead_unclear": ("Ahead, but not a clear edge", "mid"),
                 "mixed": ("Mixed: ahead of one lazy guess, not the other", "mid"), "behind": ("Not ahead of the lazy guess", "bad"),
                 "too_few": ("Too early to say", "none"), "none": ("Nothing graded yet", "none")}
GLOSSARY = [
    ("Call", "A prediction written down before the answer is known: a chance for a yes or no question (like 70%), or a number."),
    ("Seal", "A fingerprint (SHA-256) of the call, chained to the seal before it. Any later edit changes the fingerprint, and "
             "`prophet lab verify` reports TAMPERED."),
    ("Chain head", "The seal of the newest call. It fingerprints the whole list, so one line stands for the whole record."),
    ("Anchor", "A dated receipt for the chain head from OpenTimestamps. A seal shows that nothing was edited. Only an anchor shows when."),
    ("Lazy guess", "What a plain rule says with no thinking: 'same as today', 'the usual rate for this date' or a coin flip. "
                   "Prophet is graded next to it. Also called the baseline. It is a simple reference point, not a strong test: "
                   "being ahead of it is a starting point, not proof of skill."),
    ("Closing line", "The odds a bookmaker offers just before a match starts, with the bookmaker's margin taken out so the chances add up to 100%. "
                     "It is the market's final answer, so it is a hard one to be ahead of. Football calls are also graded against Pinnacle's closing line "
                     "when the data has it. The call itself never uses odds."),
    ("Brier score", "For a yes or no call: how far the chance was from what happened (1 if it happened, 0 if not), squared. "
                    "0 is perfect. Always saying 50% scores 0.25. Lower is better."),
    ("Average miss", "For a number call, such as tomorrow's high: how far off the call was, in the number's own unit (here degrees F), "
                     "averaged over many calls. Lower is better."),
    ("Win and miss", "A graded call is a win when Prophet's score beat the lazy guess on that call, and a miss when it did not."),
    ("Practice run", "A replay of history, one day at a time, using only what was known before each day. A rehearsal, not a track record."),
    ("Live", f"Calls sealed before their outcome. Only these count as the track record. A domain gets a verdict only after {MIN_CALLS} "
             "graded calls, with a clear gap over every lazy guess."),
]

PAGE_CSS = """:root{color-scheme:dark light;--bg:#16162f;--surface:#0b1633;--text:#fdf5ef;--muted:#8d9096;--accent:#deb077;--accent-ink:#deb077;
--glow:#fedb83;--line:rgba(222,176,119,.24);--soft:rgba(253,245,239,.12);--win-bg:#deb077;--win-fg:#16162f;--lazy:#8d9096;--bad:#fdf5ef}
@media (prefers-color-scheme:light){:root{--bg:#fdf5ef;--surface:#f4ece7;--text:#16162f;--muted:#62646a;--accent:#ba872a;--accent-ink:#7a5818;
--glow:#916921;--line:rgba(22,22,47,.18);--soft:rgba(22,22,47,.1);--win-bg:#deb077;--win-fg:#16162f;--lazy:#62646a;--bad:#16162f}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--text);font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
.wrap{max-width:1120px;margin:0 auto;padding:0 16px}
a{color:var(--accent-ink)}
h1,h2,h3,h4,p,dl,dd,ul{margin:0}
h1,h2,h3,h4{font-weight:600;line-height:1.25}
small{color:var(--muted);font-weight:400}
.top{border-bottom:1px dashed var(--line)}
.top .wrap{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:8px 20px;min-height:56px;padding-top:8px;padding-bottom:8px}
.mark{font-size:.85rem;font-weight:600;letter-spacing:.32em;text-transform:uppercase}
.top nav{display:flex;flex-wrap:wrap;gap:6px 18px}
.top nav a{color:var(--muted);font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;text-decoration:none}
.top nav a:hover{color:var(--text)}
.hero{padding:40px 0 8px}
.eyebrow{color:var(--accent-ink);font-size:.72rem;letter-spacing:.22em;text-transform:uppercase;margin-bottom:10px}
h1{font-size:clamp(2rem,7vw,3.1rem);letter-spacing:.01em}
.lede{color:var(--muted);max-width:64ch;margin-top:12px}
.alert{border:2px solid var(--text);border-radius:12px;padding:14px 16px;margin:20px 0 0;font-weight:600}
.tiles{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:24px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:16px}
.tile b{display:block;font-size:clamp(1.9rem,7vw,2.8rem);line-height:1.1;font-variant-numeric:tabular-nums}
.tile span{display:block;margin-top:2px;font-size:.78rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.tile small{display:block;margin-top:6px;font-size:.82rem}
.proof{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:12px;margin-top:12px}
.box{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:16px}
.box h2{font-size:.72rem;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);margin-bottom:8px}
.chip{display:inline-block;border:1px solid var(--accent);border-radius:999px;padding:2px 12px;font-size:.82rem;font-weight:600;color:var(--accent-ink)}
.chip.bad{border-color:var(--text);color:var(--text);background:transparent;text-decoration:underline}
code,.mono{font:.82rem/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;overflow-wrap:anywhere;word-break:break-all}
.box p+p{margin-top:8px}
section.block{padding:40px 0 0}
section.block>h2{font-size:.78rem;letter-spacing:.22em;text-transform:uppercase;color:var(--muted);margin-bottom:6px}
section.block>h2::after{content:"";display:block;width:40px;height:2px;margin-top:10px;background:var(--accent)}
.intro{color:var(--muted);max-width:70ch;margin:12px 0 18px}
.boards{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,340px),1fr));gap:14px;align-items:start}
.board{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:18px}
.board>h3{font-size:1.15rem;margin-bottom:12px}
.panel+.panel{margin-top:16px;padding-top:14px;border-top:1px dashed var(--line)}
.panel>h4{font-size:.72rem;letter-spacing:.18em;text-transform:uppercase;color:var(--accent-ink);margin-bottom:8px}
.panel>h4 small{letter-spacing:0;text-transform:none;font-size:.8rem;margin-left:6px}
.metric+.metric{margin-top:14px}
.label{font-weight:600}
.label small{display:block;font-size:.8rem}
.bars{margin:8px 0 6px}
.bar{display:grid;grid-template-columns:minmax(0,9.5rem) minmax(0,1fr) auto;align-items:center;gap:4px 10px;margin-top:6px;font-size:.88rem}
.bar .who small{display:block;font-size:.74rem;line-height:1.25}
.bar .track{height:10px;border-radius:5px;background:var(--soft);overflow:hidden}
.bar .track i{display:block;height:100%;border-radius:5px}
.bar.me .track i{background:var(--accent)}
.bar.lazy .track i{background:repeating-linear-gradient(135deg,var(--lazy) 0 5px,transparent 5px 8px);border:1px solid var(--lazy)}
.bar b{font-variant-numeric:tabular-nums}
.meta{color:var(--muted);font-size:.82rem}
.verdict{display:inline-block;margin-top:6px;border-radius:999px;padding:2px 12px;font-size:.82rem;font-weight:600;border:1px solid var(--soft)}
.verdict.good{background:var(--win-bg);color:var(--win-fg);border-color:var(--win-bg)}
.verdict.mid{border:1px solid var(--accent);color:var(--accent-ink)}
.verdict.bad{border:1px solid var(--text);color:var(--text)}
.verdict.none{border:1px dashed var(--muted);color:var(--muted)}
.practice .verdict.good{background:transparent;color:var(--accent-ink);border:1px solid var(--accent)}
details{margin-top:8px}
summary{cursor:pointer;color:var(--accent-ink);font-size:.85rem}
.pairs{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,240px),1fr));gap:12px;margin-top:8px}
.pairs h5{margin:0 0 6px;font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}
.pairs ul{padding:0;list-style:none}
.pairs li{border-left:2px solid var(--accent);padding:2px 0 2px 10px;margin-top:8px;font-size:.88rem}
.pairs .misses li{border-left:2px dashed var(--muted)}
.pairs li small{display:block}
.sbs{display:grid;gap:14px}
.sbs .box h3{font-size:1rem;margin-bottom:4px}
.calls{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,300px),1fr));gap:14px}
.call{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:16px;display:flex;flex-direction:column;gap:10px}
.call.win{border-left:4px solid var(--accent)}
.call.miss{border-left:4px solid var(--muted)}
.call header{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:6px}
.dom{font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}
.pill{display:inline-block;border-radius:999px;padding:2px 12px;font-size:.78rem;font-weight:600;white-space:nowrap}
.pill.waiting{border:1px dashed var(--muted);color:var(--muted)}
.pill.win{background:var(--win-bg);color:var(--win-fg);border:1px solid var(--win-bg)}
.pill.miss{border:1px solid var(--text);color:var(--text)}
.pill.tie{border:1px solid var(--muted);color:var(--muted)}
.q{font-size:.98rem;font-weight:500;line-height:1.4}
.nums{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.nums div{background:var(--bg);border:1px solid var(--soft);border-radius:10px;padding:8px 10px}
.nums small{display:block;font-size:.72rem;letter-spacing:.12em;text-transform:uppercase}
.nums b{display:block;font-size:1.35rem;line-height:1.2;font-variant-numeric:tabular-nums}
.nums em{display:block;font-style:normal;font-size:.74rem;color:var(--muted);line-height:1.25}
.res{font-size:.88rem}
.foot{display:flex;flex-wrap:wrap;justify-content:space-between;gap:4px 12px;margin-top:auto;color:var(--muted);font-size:.8rem}
.gloss{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:12px 28px}
.gloss dt{font-weight:600}
.gloss dd{color:var(--muted);margin-bottom:4px}
.gloss code{font-size:.8rem}
footer{margin-top:48px;border-top:1px dashed var(--line);padding:24px 0 48px;color:var(--muted);font-size:.85rem}
footer p+p{margin-top:6px}
@media (max-width:520px){.tiles{gap:8px}.tile{padding:12px}.bar{grid-template-columns:minmax(0,1fr) auto}.bar .track{grid-column:1/-1;order:3}.nums b{font-size:1.2rem}}
@media (prefers-reduced-motion:no-preference){.call,.board{transition:border-color .2s}.call:hover,.board:hover{border-color:var(--accent)}}
"""


def plain_baseline(name) -> str:
    return PLAIN_BASELINE.get(name, str(name).replace("_", " "))


def domain_name(name: str) -> str:
    if name in DOMAIN_NAMES:
        return DOMAIN_NAMES[name]
    ad = REGISTRY.get(name)
    return ad.title if ad and ad.title else name.replace("_", " ").capitalize() + " (hand-made calls)"


def plain_target(ad, domain: str, target: str, kind: str) -> str:
    if (domain, target) in PLAIN_TARGET:
        return PLAIN_TARGET[(domain, target)]
    m = re.fullmatch(r"up_h(\d+)", target)
    if domain == "markets" and m:
        h = int(m.group(1))
        return "Will the index close up tomorrow?" if h == 1 else f"Will the index close up {h} sessions out?"
    if target == "manual":
        return "Hand-made calls"
    return re.sub(r"\s*\((?:Brier|MAE)[^)]*\)\s*$", "", target_label(ad, domain, target, kind))


def score_words(kind: str, domain: str, target: str) -> str:
    if kind == "binary":
        return "Brier score, lower is better"
    unit = TARGET_UNIT.get((domain, target))
    return "average miss" + (f", {unit}" if unit else "") + ", lower is better"


def fmt_score(kind: str, x: float) -> str:
    return f"{x:.3f}" if kind == "binary" else f"{x:.2f}"


def fmt_pair(kind: str, a: float, b: float, domain: str, target: str):
    """Prophet's call and the lazy guess as text: whole percents (one decimal when the whole percents would look equal), or numbers."""
    if kind == "binary":
        if round(a * 100) == round(b * 100):
            return f"{a * 100:.1f}%", f"{b * 100:.1f}%"
        return f"{a * 100:.0f}%", f"{b * 100:.0f}%"
    unit = TARGET_UNIT.get((domain, target))
    tail = " " + unit.replace("degrees F", "&deg;F") if unit else ""
    return f"{a:.1f}{tail}", f"{b:.1f}{tail}"


def record_cards(v: Verification, now: dt.datetime) -> list:
    """One dict per sealed call that passed verification, joined with its score row (if any)."""
    scored = valid_scores(v)
    cards = []
    for c in v.log.ok_calls:
        row = scored.get(c["id"])
        card = {"id": c["id"], "domain": c["domain"], "target": c["target"], "kind": c["kind"], "question": c["question"],
                "resolves_at": c["resolves_at"], "seal": c["seal"], "pred": c["prediction"], "base": c["baseline_prediction"],
                "base_name": c.get("primary_baseline") or "baseline", "status": "waiting", "due": parse_iso(c["resolves_at"]) <= now}
        if row is not None:
            gap = row["baseline_score"] - row["score"]
            try:
                early = parse_iso(row["scored_at"]) < parse_iso(c["resolves_at"])
            except (KeyError, ValueError):
                early = False
            card.update(status="win" if gap > BASELINE_EPS else ("miss" if gap < -BASELINE_EPS else "tie"), gap=gap,
                        outcome=row["outcome"], score=row["score"], base_score=row["baseline_score"], scored_at=row.get("scored_at", ""), early=early)
        cards.append(card)
    return cards


def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _rich(text) -> str:
    """Escaped text in which `backticked` words are shown as code."""
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", _e(text))


def _bars(kind: str, prophet: float, lazies: list) -> str:
    """Prophet's score and each lazy guess's score as bars on one scale (the longest bar is the biggest miss). Lower is better."""
    top = max([prophet] + [s for _n, s in lazies]) or 1.0

    def bar(who, note, score, cls):
        width = max(2.0, 100.0 * score / top)
        return (f'<div class="bar {cls}"><span class="who">{_e(who)}' + (f"<small>{_e(note)}</small>" if note else "") +
                f'</span><span class="track"><i style="width:{width:.1f}%"></i></span><b>{_e(fmt_score(kind, score))}</b></div>')
    return ('<div class="bars">' + bar("Prophet", "", prophet, "me")
            + "".join(bar("The market" if n in MARKET_LINES else "Lazy guess", plain_baseline(n), s, "lazy") for n, s in lazies) + "</div>")


def _verdict(code: str, agg: dict | None = None) -> str:
    words, tone = VERDICT_WORDS.get(code, VERDICT_WORDS["none"])
    if code == "mixed" and agg and MARKET_LINES & set(agg["baselines"]):       # one of the two reference points is a market price, not a lazy guess
        words = "Mixed: ahead of one reference point, not the other"
    return f'<p class="verdict {tone}">{_e(words)}</p>'


def unit_html(domain: str, target: str) -> str:
    unit = TARGET_UNIT.get((domain, target))
    return " &deg;F" if unit == "degrees F" else (" " + _e(unit) if unit else "")


def _examples(items: list, kind: str, domain: str, target: str, questions: dict | None = None, ref: str = "lazy guess") -> str:
    """Biggest misses next to biggest wins. items come from _example(): key, as_of, called, baseline, outcome, gap.
    A live call is shown by its question; a practice case, which has none, by its key and day. ref names what Prophet is set beside."""
    def one(e):
        text = (questions or {}).get(e["key"]) or f'{e["key"]} ({e["as_of"]})'
        if len(text) > 150:
            text = text[:147] + "..."
        mine, lazy = fmt_pair(kind, e["called"], e["baseline"], domain, target)
        result = (("It happened." if e["outcome"] >= 0.5 else "It did not happen.") if kind == "binary"
                  else f'Result {e["outcome"]:g}{unit_html(domain, target)}.')
        return f"<li>{_e(text)}<small>Prophet {mine}, {ref} {lazy}. {result}</small></li>"
    return "".join(one(e) for e in items) or "<li><small>none</small></li>"


def _practice_panel(ad, name: str, summary, error) -> str:
    if summary is None:
        if error:
            return '<section class="panel practice"><h4>Practice <small>history replay</small></h4><p class="meta">The last practice run failed, so there is nothing to show.</p></section>'
        return ""
    out = ['<section class="panel practice"><h4>Practice <small>a rehearsal on past data, not a track record</small></h4>']
    for tg, agg in summary["targets"].items():
        kind = agg["kind"]
        lazies = [(n, b["score"]) for n, b in list(agg["baselines"].items())[:3]]
        ref = "the market" if agg.get("primary") in MARKET_LINES else "lazy guess"
        out.append('<div class="metric">'
                   f'<p class="label">{_e(plain_target(ad, name, tg, kind))}<small>{_e(score_words(kind, name, tg))}</small></p>'
                   + _bars(kind, agg["score"], lazies) + _verdict(agg["verdict_code"], agg) +
                   f'<p class="meta">{agg["n"]:,} practice calls, {_e(str(agg["first"])[:10])} to {_e(str(agg["last"])[:10])}</p>')
        if agg.get("misses_worst") or agg.get("wins_best"):
            out.append('<details><summary>Biggest misses and wins in the practice run</summary><div class="pairs">'
                       f'<div class="misses"><h5>Biggest misses</h5><ul>{_examples(agg.get("misses_worst", []), kind, name, tg, ref=ref)}</ul></div>'
                       f'<div><h5>Biggest wins</h5><ul>{_examples(agg.get("wins_best", []), kind, name, tg, ref=ref)}</ul></div></div></details>')
        out.append("</div>")
    return "".join(out) + "</section>"


def _live_panel(ad, name: str, groups: list) -> str:
    head = '<section class="panel"><h4>Live <small>sealed before the outcome: the track record</small></h4>'
    if not groups:
        if ad is None or ad.status != "ready":                  # not running: the scoreboard says so in one line instead of showing an empty card
            return ""
        if not ad.can_predict:
            return head + '<p class="meta">This kind of question is practice only. It makes no live calls.</p></section>'
        return head + f'<p class="meta">No live call yet. Make some with <code>prophet lab predict {_e(name)}</code>.</p></section>'
    out = [head]
    for tg, g in groups:
        kind, agg = g["kind"], g["agg"]
        when = f" Next result due {_e(iso(g['next_due'])[:10])}." if g["next_due"] else ""
        out.append('<div class="metric">'
                   f'<p class="label">{_e(plain_target(ad, name, tg, kind))}<small>{_e(score_words(kind, name, tg))}</small></p>')
        if agg["n"]:
            out.append(_bars(kind, agg["score"], [(n, b["score"]) for n, b in list(agg["baselines"].items())[:3]]))
        out.append(_verdict(agg["verdict_code"], agg) +
                   f'<p class="meta">{g["sealed"]} sealed, {agg["n"]} graded, {g["waiting"]} waiting' +
                   (f" ({g['overdue']} past their date)" if g["overdue"] else "") + f".{when}</p></div>")
    return "".join(out) + "</section>"


def _call_card(c: dict) -> str:
    kind = c["kind"]
    mine, lazy = fmt_pair(kind, c["pred"], c["base"], c["domain"], c["target"])
    status = c["status"]
    if status == "waiting":
        pill, word = "waiting", ("Past its date, no result yet" if c["due"] else "Waiting")
    else:
        pill, word = status, {"win": "Graded: win", "miss": "Graded: miss", "tie": "Graded: tie"}[status]
    out = [f'<article class="call {status}"><header><span class="dom">{_e(domain_name(c["domain"]))}</span>'
           f'<span class="pill {pill}">{_e(word)}</span></header>',
           f'<h3 class="q">{_e(c["question"])}</h3>',
           f'<div class="nums"><div><small>Prophet</small><b>{mine}</b></div>'
           f'<div><small>Lazy guess</small><b>{lazy}</b><em>{_e(plain_baseline(c["base_name"]))}</em></div></div>']
    if status != "waiting":
        early = " (graded before its date)" if c["early"] else ""
        if kind == "binary":
            happened = "It happened." if c["outcome"] >= 0.5 else "It did not happen."
            out.append(f'<p class="res">{happened} Brier score {_e(fmt_score(kind, c["score"]))} against '
                       f'{_e(fmt_score(kind, c["base_score"]))} for the lazy guess{early}.</p>')
        else:
            out.append(f'<p class="res">Result: {c["outcome"]:g}{unit_html(c["domain"], c["target"])}. Prophet missed by {c["score"]:.2f}; '
                       f'the lazy guess missed by {c["base_score"]:.2f}{early}.</p>')
    out.append(f'<p class="foot"><span>Resolves {_e(c["resolves_at"][:10])}</span><span title="{_e(c["seal"])}">Seal {_e(c["seal"][:12])}</span></p></article>')
    return "".join(out)


def render_page(lab: Lab, v: Verification, live: dict, cards: list, now: dt.datetime) -> str:
    """The record page. Every number comes from the sealed files; nothing here is typed by hand."""
    anchor = anchor_state(lab, v)
    n_sealed = len(cards)
    n_graded = sum(1 for c in cards if c["status"] != "waiting")
    n_wait = n_sealed - n_graded
    n_due = sum(1 for c in cards if c["status"] == "waiting" and c["due"])
    questions = {c["id"]: c["question"] for c in cards}
    head = v.log.head
    seals_ok = v.ok
    out = ["<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width,initial-scale=1">', '<meta name="color-scheme" content="dark light">',
           "<title>Prophet: the record</title>",
           '<meta name="description" content="Predictions written down before they happen, locked with a seal, and graded against the obvious guess.">',
           f"<style>{PAGE_CSS}</style></head><body>",
           '<header class="top"><div class="wrap"><span class="mark">Prophet</span><nav aria-label="Sections">'
           '<a href="#scoreboard">Scoreboard</a><a href="#sidebyside">Wins and misses</a><a href="#calls">Calls</a><a href="#words">Words</a></nav></div></header>',
           '<main class="wrap"><section class="hero"><p class="eyebrow">A receipt book for predictions</p><h1>The record</h1>'
           "<p class=\"lede\">Each call is written down before it happens, locked with a seal, then graded against the lazy guess. "
           "This page is built from the sealed files. Nobody edits it by hand.</p>"]
    if not seals_ok:
        out.append('<p class="alert" role="alert">TAMPERED. A seal or a score does not match. Do not trust this record until the cause is found '
                   "(run <code>prophet lab verify</code>).</p>")
    out.append(f'<div class="tiles"><div class="tile"><b>{n_sealed}</b><span>Sealed</span></div>'
               f'<div class="tile"><b>{n_graded}</b><span>Graded</span></div>'
               f'<div class="tile"><b>{n_wait}</b><span>Waiting</span>' +
               (f"<small>{n_due} past their date</small>" if n_due else "") + "</div></div>")
    if n_sealed == 0:
        out.append('<p class="intro">Nothing is sealed yet. Make the first call: <code>prophet lab call --domain demo --question "Will my die roll '
                   'show a 5 or a 6?" --resolves today --prob 0.33</code></p>')
    out.append('<div class="proof"><div class="box"><h2>Seals</h2>'
               f'<p><span class="chip{"" if seals_ok else " bad"}">{"All seals OK" if seals_ok else "TAMPERED"}</span></p>'
               f'<p class="meta">{len(v.log.entries)} sealed calls checked, {len(v.log.bad)} tampered. {len(v.scores)} score rows, {len(v.score_problems)} problems.</p>'
               f'<p class="meta">Chain head, the seal of the newest call:</p><p class="mono">{_e(head or "(nothing sealed yet)")}</p></div>'
               f'<div class="box"><h2>Anchor</h2><p>{_rich(anchor_sentence(anchor))}</p></div></div></section>')
    # ---- scoreboard: one card per domain, practice and live apart
    boards, quiet = [], []
    for name in ordered_domains(live):
        ad = REGISTRY.get(name)
        summary, error = read_summary(lab, name)
        groups = sorted(((t, g) for (d, t), g in live["groups"].items() if d == name), key=lambda x: x[0])
        body = _live_panel(ad, name, groups) + _practice_panel(ad, name, summary, error)         # the track record first, the rehearsal second
        if body:
            boards.append(f'<article class="board"><h3>{_e(domain_name(name))}</h3>{body}</article>')
        elif ad is not None:
            quiet.append(f"{domain_name(name)} ({ad.status.replace('_', ' ')})")
    out.append('<section class="block" id="scoreboard"><h2>Scoreboard</h2>'
               '<p class="intro">Prophet next to the lazy guess, one card per kind of question. Live calls and practice runs are kept apart on purpose: '
               f"only live calls are a track record, and nothing is claimed before {MIN_CALLS} graded calls. "
               "A lazy guess is a simple reference point, not a strong test.</p>"
               '<div class="boards">' + ("".join(boards) or '<p class="meta">No practice run and no call yet.</p>') + "</div>")
    if quiet:
        out.append('<p class="meta" style="margin-top:12px">Not running: ' + _e(", ".join(quiet)) + '. <code>prophet lab domains</code> says why.</p>')
    out.append("</section>")
    # ---- wins and misses side by side (live calls only)
    out.append('<section class="block" id="sidebyside"><h2>Wins and misses, side by side</h2>')
    sbs = []
    for (name, tg), g in sorted(live["groups"].items()):
        graded = [c for c in cards if c["domain"] == name and c["target"] == tg and c["status"] != "waiting"]
        if not graded:
            continue
        ad = REGISTRY.get(name)
        ranked = sorted(graded, key=lambda c: c["gap"])
        miss = [{"key": c["id"], "as_of": c["resolves_at"][:10], "called": c["pred"], "baseline": c["base"], "outcome": c["outcome"]} for c in ranked if c["gap"] < 0][:3]
        win = [{"key": c["id"], "as_of": c["resolves_at"][:10], "called": c["pred"], "baseline": c["base"], "outcome": c["outcome"]} for c in ranked[::-1] if c["gap"] > 0][:3]
        sbs.append(f'<div class="box"><h3>{_e(domain_name(name))}: {_e(plain_target(ad, name, tg, g["kind"]))}</h3>'
                   f'<div class="pairs"><div class="misses"><h5>Biggest misses</h5><ul>{_examples(miss, g["kind"], name, tg, questions)}</ul></div>'
                   f'<div><h5>Biggest wins</h5><ul>{_examples(win, g["kind"], name, tg, questions)}</ul></div></div></div>')
    if sbs:
        out.append('<div class="sbs">' + "".join(sbs) + "</div>")
    else:
        out.append('<p class="intro">No live call has been graded yet, so there are no wins or misses to show. '
                   "When results arrive, the biggest misses will sit here next to the biggest wins. Practice-run examples are inside each scoreboard card.</p>")
    out.append("</section>")
    # ---- the calls
    graded_cards = sorted((c for c in cards if c["status"] != "waiting"), key=lambda c: c.get("scored_at", ""), reverse=True)
    waiting_cards = sorted((c for c in cards if c["status"] == "waiting"), key=lambda c: c["resolves_at"])
    limit = 150
    out.append('<section class="block" id="calls"><h2>Calls</h2>')
    for title, group, empty in (("Graded", graded_cards, "No call has been graded yet."), ("Waiting for the result", waiting_cards, "No call is waiting.")):
        out.append(f'<h3 style="margin:18px 0 10px;font-size:1rem">{title} <small>({len(group)})</small></h3>')
        if group:
            out.append('<div class="calls">' + "".join(_call_card(c) for c in group[:limit]) + "</div>")
            if len(group) > limit:
                out.append(f'<p class="meta">Showing {limit} of {len(group)}. The whole list is calls.jsonl.</p>')
        else:
            out.append(f'<p class="meta">{empty}</p>')
    out.append("</section>")
    # ---- words and footer
    out.append('<section class="block" id="words"><h2>Words used here</h2><dl class="gloss">' +
               "".join(f"<div><dt>{_e(t)}</dt><dd>{_rich(d)}</dd></div>" for t, d in GLOSSARY) + "</dl></section></main>")
    out.append(f'<footer><div class="wrap"><p>Generated {_e(iso(now))} by prophet_lab.py from calls.jsonl, scores.jsonl, backtest/ and anchors.jsonl. '
               f"Code sha256 {_e(code_sha()[:16])}.</p>"
               "<p>Pattern research only. Prophet places no bets or trades and gives no betting, trading or investment advice.</p></div></footer>"
               "</body></html>")
    return "\n".join(out) + "\n"


def cmd_record(lab, args):
    now = utcnow()
    v = verify_lab(lab)
    live = live_state(lab, v)
    blocks = build_blocks(lab, live, now)
    md, page = render_md(lab, blocks), render_page(lab, v, live, record_cards(v, now), now)
    atomic_write(lab.root / "RECORD.md", md)
    atomic_write(lab.root / "record.html", page)
    print(f"Wrote {lab.root / 'RECORD.md'} ({len(md)} bytes) and {lab.root / 'record.html'} ({len(page)} bytes)")
    print(f"Integrity: {len(v.log.entries)} sealed calls, {len(v.log.bad)} TAMPERED; " + ("ALL SEALS OK" if v.ok else "TAMPERED, see `verify`"))
    return 0 if v.ok else 1

# ======================================================================================
# command line
# ======================================================================================


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--lab-dir", default=argparse.SUPPRESS, help="data folder (default: the pattern-lab notebook folder)")
    common.add_argument("--calls", default=argparse.SUPPRESS, help="use this calls.jsonl instead of the lab's own (for checks on a copy)")
    common.add_argument("--scores", default=argparse.SUPPRESS, help="use this scores.jsonl instead of the lab's own")
    common.add_argument("--refresh", action="store_true", default=argparse.SUPPRESS, help="download again even if a recent copy is cached (a GDELT day that is already counted is never downloaded again)")
    common.add_argument("--days", type=int, default=argparse.SUPPRESS, help="geopolitics: days of GDELT daily event files to use, ending yesterday (default 120, at least 40)")
    common.add_argument("--gdelt-doc-api", action="store_true", default=argparse.SUPPRESS, help="geopolitics: use the old GDELT DOC API (it often answers HTTP 429) instead of the daily event files")
    common.add_argument("--debug", action="store_true", default=argparse.SUPPRESS, help="show full error traces")
    p = argparse.ArgumentParser(prog="prophet lab", parents=[common], formatter_class=argparse.RawDescriptionHelpFormatter,
                                description="Prophet Pattern Lab: make the call, lock it in, get graded. Calls are sealed before the outcome and graded against "
                                            "reality and the lazy guess (a plain baseline).\n"
                                            "Never touches the network: call, resolve, list, verify, record, domains. Downloads public data: backtest, predict, score.\n"
                                            "Sends one 32-byte hash to two OpenTimestamps servers: anchor (anchor --dry-run sends nothing).\n"
                                            "Pattern research only. No bets, no trades, no advice. Free public data, no keys.",
                                epilog='examples:\n  prophet lab call --domain demo --question "Will my die roll show a 5 or a 6?" --resolves today --prob 0.33\n'
                                       '  prophet lab call --domain politics --question "Will X happen by 2026-12-31?" --resolves 2026-12-31 --prob 0.62\n'
                                       "  prophet lab resolve <id> --outcome yes\n  prophet lab verify\n  prophet lab record\n  prophet lab anchor --dry-run\n"
                                       "  prophet lab backtest all\n  prophet lab predict weather\n  prophet lab score")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="command")

    def add(name, helptext, func):
        sp = sub.add_parser(name, parents=[common], help=helptext, description=helptext)
        sp.set_defaults(func=func)
        return sp
    s = add("backtest", "practice run: replay history walk-forward and score it (a rehearsal, not a track record). Downloads public data", cmd_backtest)
    s.add_argument("domain", help="a domain name, or all")
    s = add("predict", "seal live pattern calls for things that have not happened yet. Downloads public data", cmd_predict)
    s.add_argument("domain", help="a domain name, or all")
    add("score", "score sealed calls whose outcome is now in the data. Downloads public data", cmd_score)
    s = add("call", "seal a hand-made or agent-made call on any question (never touches the network)", cmd_call)
    s.add_argument("--domain", required=True, help="any name, for example politics or business")
    s.add_argument("--question", required=True)
    s.add_argument("--resolves", required=True, help="date the answer will be known, YYYY-MM-DD, or the word today or tomorrow (UTC dates)")
    s.add_argument("--prob", type=float, help="probability of yes, 0 to 1")
    s.add_argument("--value", type=float, help="a number, instead of --prob")
    s.add_argument("--baseline", type=float, help="what a plain guess says (default 0.5 for yes/no; required for numbers)")
    s.add_argument("--method", default="manual call", help="how the call was made, for example 'agent: reasoning from X'")
    s.add_argument("--source", action="append", help="a URL the call relied on (recorded, not downloaded); repeatable")
    s.add_argument("--target", help="a label inside the domain (default manual)")
    s = add("resolve", "score a hand-made call once the answer is known (never touches the network)", cmd_resolve)
    s.add_argument("id")
    s.add_argument("--outcome", required=True, help="yes, no, or a number")
    s.add_argument("--evidence", help="URL or note showing the outcome")
    s.add_argument("--note")
    add("verify", "check every seal and the hash chain; broken ones are reported as TAMPERED", cmd_verify)
    add("record", "write RECORD.md and record.html (the record page), from the sealed files only", cmd_record)
    s = add("anchor", "send the chain head (a 32-byte hash, nothing else) to two OpenTimestamps servers, so the date can be proved later", cmd_anchor)
    s.add_argument("--dry-run", action="store_true", help="show what would be sent and where; send nothing and write nothing")
    s.add_argument("--again", action="store_true", help="send a chain head that was already anchored a second time")
    add("domains", "list domains, their status and free data sources", cmd_domains)
    s = add("list", "list sealed calls", cmd_list)
    s.add_argument("--open", action="store_true", help="only calls without a score")
    s.add_argument("--scored", action="store_true", help="only scored calls")
    return p


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    load_plugins()
    args = build_parser().parse_args(argv)
    try:
        lab = Lab(root=getattr(args, "lab_dir", None), calls=getattr(args, "calls", None), scores=getattr(args, "scores", None),
                  ttl_hours=0.0 if getattr(args, "refresh", False) else 6.0,
                  opts={k: getattr(args, k) for k in ("days", "gdelt_doc_api") if hasattr(args, k)})
        return int(args.func(lab, args) or 0)
    except (LabError, FetchError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
        return 130
    except Exception as e:
        if getattr(args, "debug", False):
            raise
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
