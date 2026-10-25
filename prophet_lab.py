#!/usr/bin/env python3
"""Prophet Pattern Lab.

Learns patterns from past records, seals each prediction (a "pattern call") before the
outcome is known, then scores it against what really happened and against a plain
baseline. It publishes a track record, wins and misses included.

  prophet lab backtest <domain|all>   replay history walk-forward (only data before each date)
  prophet lab predict  <domain|all>   seal live pattern calls for things that have not happened yet
  prophet lab score                   score sealed calls whose outcome is now known
  prophet lab call ...                seal a call on any question by hand (any domain)
  prophet lab resolve <id> --outcome  score a hand-made call
  prophet lab verify                  check every seal; a broken seal is reported as TAMPERED
  prophet lab record                  write RECORD.md and record.html
  prophet lab domains                 what each domain is, and whether it has a free data source

Pattern research only. No bets, no trades, no investment advice. Free public data, no keys.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
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
    def __init__(self, root=None, calls=None, scores=None, ttl_hours: float = 6.0):
        self.root = Path(root or os.environ.get("PROPHET_LAB_DIR") or DEFAULT_LAB)
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
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
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
            except (KeyError, TypeError, ValueError):
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


def make_score_row(call: dict, outcome: float, scored_at: dt.datetime, how: str, evidence=None, note: str = "") -> dict:
    kind = call["kind"]
    sc = score_pair(kind, call["prediction"], outcome)
    bs = {k: score_pair(kind, v, outcome) for k, v in (call.get("baselines") or {}).items()}
    row = {
        "id": call["id"], "seal": call["seal"], "outcome": outcome, "score": round(sc, 9),
        "baseline_score": round(score_pair(kind, call["baseline_prediction"], outcome), 9),
        "baseline_scores": {k: round(v, 9) for k, v in bs.items()},
        "scored_at": iso(scored_at), "how": how,
    }
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
        """Outcomes for sealed calls: {id: (outcome or None, evidence inputs)}."""
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
        out.append({"div": (r.get("Div") or "").strip(), "date": d, "time": (r.get("Time") or "").strip(),
                    "home": home, "away": away, "ftr": ftr if ftr in ("H", "D", "A") else "",
                    "hg": hg, "ag": ag, "season": season})
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
                "note": "free historical results; credit football-data.co.uk. Prophet reads results only, never the bookmaker odds columns"}]
    default_config = DEFAULT_CONFIG["sports"]

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
        rows, per_league = [], []
        for div, ms in hist.matches.items():
            _, res = run_league(ms, True)
            for m, p, base in res:
                rows.append({"target": "home_win", "kind": "binary", "key": f"{div}:{m['date'].isoformat()}:{m['home']}-{m['away']}",
                             "as_of": m["date"].isoformat(), "pred": p, "bases": {"league_home_rate": base},
                             "primary": "league_home_rate", "outcome": 1.0 if m["ftr"] == "H" else 0.0})
            per_league.append(f"{div} {len(res)}")
        notes = [f"Leagues and calls: {', '.join(per_league)}. Seasons {hist.seasons[0]} to {hist.seasons[-1]}; the first season of each league only warms the ratings up.",
                 "Pattern: Elo rating gap (K=20, goal-difference weighting, ratings pulled 25% toward average each new season) plus points in the last five matches, turned into a home-win probability by a ridge logistic refitted every 100 matches. Draws count as not a home win.",
                 "Baseline: the league's home-win rate so far."] + hist.notes
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
            res = {(m["date"].isoformat(), m["home"], m["away"]): m["ftr"] for m in parse_fd(f.text(), season) if m["ftr"]}
            for c in cs:
                m = c["meta"]
                r = res.get((m["date"], m["home"], m["away"]))
                out[c["id"]] = (None if not r else (1.0 if r == "H" else 0.0), [{"url": url, "sha256": f.sha256}])
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


def live_state(lab: Lab, v: Verification | None = None) -> dict:
    v = v or verify_lab(lab)
    calls = v.log.ok_calls
    bad_lines = {line for line, _, _ in v.score_problems}
    scored = {}
    for row in v.scores:
        if row.get("_bad") or row["_line"] in bad_lines or not row.get("id"):
            continue
        scored[row["id"]] = row
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
        prim = c.get("primary_baseline") or "baseline"
        if prim not in bases:
            bases[prim] = row["baseline_score"]
        g["items"].append({"score": row["score"], "bases": bases, "primary": prim,
                           "row": {"key": c["id"], "as_of": c["made_at"][:10], "pred": c["prediction"],
                                   "bases": {prim: c["baseline_prediction"]}, "outcome": row["outcome"]}})
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
            print(f"  sealed {rec['id']}  call {rec['prediction']}  baseline {rec['baseline_prediction']} ({rec['primary_baseline']})  resolves {rec['resolves_at']}  seal {rec['seal'][:12]}")
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
        if c.get("origin") == "manual" or c["domain"] not in REGISTRY:
            manual.append(c)
        else:
            by_domain.setdefault(c["domain"], []).append(c)
    scored = unresolved = problems = 0
    for dom, cs in by_domain.items():
        ad = REGISTRY[dom]
        try:
            results = ad.resolve_calls(lab, cs)
        except (FetchError, LabError) as e:
            problems += 1
            print(f"[{dom}] could not read outcomes: {e}")
            continue
        for c in cs:
            outcome, evidence = results.get(c["id"], (None, []))
            if outcome is None:
                unresolved += 1
                print(f"  waiting {c['id']}: outcome not in the data yet")
                continue
            row = make_score_row(c, float(outcome), now, "auto", evidence=evidence)
            append_score(lab, row)
            scored += 1
            print(f"  scored {c['id']}: called {c['prediction']}, baseline {c['baseline_prediction']}, outcome {outcome:g}; score {row['score']:.4f} vs baseline {row['baseline_score']:.4f}")
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


def cmd_call(lab, args):
    if (args.prob is None) == (args.value is None):
        raise LabError("give exactly one of --prob (yes/no call) or --value (number call)")
    try:
        resolves = dt.date.fromisoformat(args.resolves)
    except ValueError:
        raise LabError("--resolves must be a date like 2026-12-31")
    now = utcnow()
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
             "A seal does not prove when the call was made (the clock is this computer's) and it cannot stop someone who rewrites the whole file and every hash; the chain head printed below is the line to copy somewhere outside this machine if that matters. "
             "The exact code that made a call is kept in code_snapshots/ under the code_sha256 stored in the call.")
TEXT_PLUGIN = ("How to add a domain (plugin interface). A domain is one small adapter. Save a file named prophet_lab_<name>.py next to prophet_lab.py and the next run picks it up; no other file changes. "
               "The file starts with `import prophet_lab as lab`, defines `class Adapter(lab.Adapter)` with name, title, sources and status, and ends with `ADAPTER = Adapter()`. "
               "The adapter supplies five calls and two work lists. fetch_history(lab) downloads through lab.fetch (polite User-Agent; the site's robots.txt or the official-API allowlist is checked first; cached under raw/, URL, time and sha256 recorded) and returns the parsed history with an inputs list. "
               "features(hist, subject, as_of) builds features from data strictly before as_of. predict(hist, subject, as_of, feats) returns {'value': probability or number, 'method': text}. "
               "baseline(hist, subject, as_of, feats) returns {'primary': name, 'values': {name: value}}. resolve(hist, subject) returns the outcome (1 or 0 for yes/no, otherwise the number) or None while it is unknown. "
               "backtest_subjects(hist) lists past cases oldest first and live_subjects(hist, now) lists what to call right now; lab.make_subject builds each case. "
               "Free public data only, no keys, no scraping of sites that forbid it. If no free source exists, set status = 'needs_data_source' and the record shows that instead of a score. Heavy adapters may override backtest(lab) and resolve_calls(lab, calls).")
TEXT_MANUAL = ("Any question without a data feed. `prophet lab call --domain <any name> --question \"...\" --resolves YYYY-MM-DD --prob 0.62 [--baseline 0.5]` seals a hand-made or agent-made call (use --value and --baseline for a number). "
               "When the answer is known, `prophet lab resolve <id> --outcome yes|no|<number>` scores it. Hand-made calls appear in the live table under their own domain with the target 'manual'.")
TEXT_COMMANDS = ("Commands, all run as `prophet lab <command>` from any folder: backtest <domain|all>, predict <domain|all>, score, call, resolve <id> --outcome, verify, record, domains, list.")


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
    B.append(("p", "Files in the pattern-lab folder: calls.jsonl (sealed calls, append only), scores.jsonl (scores, append only), backtest/ (summary json and every scored row as csv), "
                   "raw/ (every download with its URL, fetch time and sha256 in raw/manifest.jsonl), code_snapshots/ (the exact code behind each seal), config.json (places, leagues and series, editable)."))
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


HTML_CSS = """:root{color-scheme:light dark;--bg:#fafaf7;--fg:#1d1d1b;--mute:#5d5d57;--line:#d8d8d0;--head:#ecece4}
@media (prefers-color-scheme:dark){:root{--bg:#161715;--fg:#e8e8e2;--mute:#a4a49c;--line:#3a3b37;--head:#232420}}
body{background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,Segoe UI,sans-serif;margin:0;padding:24px 16px}
main{max-width:1100px;margin:0 auto}h1{font-size:1.7rem;margin:.2em 0}h2{font-size:1.2rem;margin:1.6em 0 .4em}
p{margin:.6em 0;max-width:80ch}.wrap{overflow-x:auto}table{border-collapse:collapse;font-size:.88rem;min-width:100%}
th,td{border:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}th{background:var(--head)}
pre{background:var(--head);padding:12px;overflow-x:auto;font-size:.85rem;border:1px solid var(--line)}"""


def render_html(blocks: list, now: dt.datetime) -> str:
    out = ["<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width,initial-scale=1">', "<title>Prophet Pattern Lab</title>",
           f"<style>{HTML_CSS}</style></head><body><main>"]
    for b in blocks:
        if b[0] == "h1":
            out.append(f"<h1>{html.escape('Prophet Pattern Lab')}</h1>")
        elif b[0] == "h2":
            out.append(f"<h2>{html.escape(b[1])}</h2>")
        elif b[0] == "p":
            out.append(f"<p>{html.escape(b[1])}</p>")
        elif b[0] == "pre":
            out.append(f"<pre>{html.escape(b[1])}</pre>")
        elif b[0] == "table":
            t = ['<div class="wrap"><table><thead><tr>' + "".join(f"<th>{html.escape(str(h))}</th>" for h in b[1]) + "</tr></thead><tbody>"]
            t += ["<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in b[2]]
            t.append("</tbody></table></div>")
            out.append("".join(t))
    out.append(f"<p>Generated {html.escape(iso(now))}. Pattern research only: no bets, no trades, no advice.</p></main></body></html>")
    return "\n".join(out) + "\n"


def cmd_record(lab, args):
    now = utcnow()
    v = verify_lab(lab)
    live = live_state(lab, v)
    blocks = build_blocks(lab, live, now)
    md, page = render_md(lab, blocks), render_html(blocks, now)
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
    common.add_argument("--refresh", action="store_true", default=argparse.SUPPRESS, help="download again even if a recent copy is cached")
    common.add_argument("--debug", action="store_true", default=argparse.SUPPRESS, help="show full error traces")
    p = argparse.ArgumentParser(prog="prophet lab", parents=[common], formatter_class=argparse.RawDescriptionHelpFormatter,
                                description="Prophet Pattern Lab: pattern calls sealed before the outcome, scored against reality and a plain baseline.\n"
                                            "Pattern research only. No bets, no trades, no advice. Free public data, no keys.",
                                epilog='examples:\n  prophet lab backtest all\n  prophet lab predict weather\n  prophet lab score\n'
                                       '  prophet lab call --domain politics --question "Will X happen by 2026-12-31?" --resolves 2026-12-31 --prob 0.62\n'
                                       "  prophet lab resolve <id> --outcome yes\n  prophet lab verify\n  prophet lab record")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="command")

    def add(name, helptext, func):
        sp = sub.add_parser(name, parents=[common], help=helptext, description=helptext)
        sp.set_defaults(func=func)
        return sp
    s = add("backtest", "replay history walk-forward and score it (rehearsal, not a track record)", cmd_backtest)
    s.add_argument("domain", help="a domain name, or all")
    s = add("predict", "seal live pattern calls for things that have not happened yet", cmd_predict)
    s.add_argument("domain", help="a domain name, or all")
    add("score", "score sealed calls whose outcome is now in the data", cmd_score)
    s = add("call", "seal a hand-made or agent-made call on any question", cmd_call)
    s.add_argument("--domain", required=True, help="any name, for example politics or business")
    s.add_argument("--question", required=True)
    s.add_argument("--resolves", required=True, help="date the answer will be known, YYYY-MM-DD")
    s.add_argument("--prob", type=float, help="probability of yes, 0 to 1")
    s.add_argument("--value", type=float, help="a number, instead of --prob")
    s.add_argument("--baseline", type=float, help="what a plain guess says (default 0.5 for yes/no; required for numbers)")
    s.add_argument("--method", default="manual call", help="how the call was made, for example 'agent: reasoning from X'")
    s.add_argument("--source", action="append", help="a URL the call relied on (recorded, not downloaded); repeatable")
    s.add_argument("--target", help="a label inside the domain (default manual)")
    s = add("resolve", "score a hand-made call once the answer is known", cmd_resolve)
    s.add_argument("id")
    s.add_argument("--outcome", required=True, help="yes, no, or a number")
    s.add_argument("--evidence", help="URL or note showing the outcome")
    s.add_argument("--note")
    add("verify", "check every seal and the hash chain; broken ones are reported as TAMPERED", cmd_verify)
    add("record", "write RECORD.md and record.html", cmd_record)
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
                  ttl_hours=0.0 if getattr(args, "refresh", False) else 6.0)
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
