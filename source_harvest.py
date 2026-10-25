"""Prophet source harvester: pulls the approved, verified, keyless sources of the Prophet source registry into local
searchable notebooks that `ask` reads (ask -c sources-<area> <question>). Local files only: no keys, no accounts, no
paid calls, no GPU, no model.

  python -B source_harvest.py --area all                    everything approved (and the optional local God's Eye View clone, if gods_eye_view points at one)
  python -B source_harvest.py --area film_craft --limit 10  one area, at most 10 items per source
  python -B source_harvest.py --source gh_comfyui --dry-run fetch and parse, write nothing
  python -B source_harvest.py --list                        what is approved, what is skipped (fetches nothing)

Rules it keeps
  * Stops if sources.json no longer matches the sha256 in the approval record, or if that record does not say "status":
    "approved" (a pending confirmation, another status or no status at all makes it refuse to run). Only verified=true sources are touched.
  * YouTube sources are never fetched (this tool does not harvest YouTube). SEC sources are marked disabled in the registry:
    SEC wants a dedicated contact e-mail in the User-Agent and none is configured. A row marked "disabled": true is kept in
    the registry but never fetched.
  * Polite: User-Agent "Prophet-source-harvest/1.0 (local research)", no e-mail, robots.txt obeyed (wildcards, Allow,
    Crawl-delay; a robots.txt that answers 429, a 5xx or a refusal, or redirects without end, blocks the site, and only a plain
    404 or 410 means no restrictions), at least 3.2 s between requests to one site (more for arXiv and GDELT), 20 s timeouts, size caps,
    at most 50 newest items per source per run. Several sites run in parallel; one site is never hit in parallel.
  * The one exception to robots.txt is prophet_net.API_ALLOWLIST (project decision), shared with the Pattern Lab: a
    short, explicit list of official, documented API endpoints (arXiv API, Open-Meteo, api.weather.gov and similar). Only a URL
    that matches an entry exactly (https, host, documented path prefix, required query) skips the robots.txt check; no other
    page of those sites is ever fetched, and every other URL is still checked. The same User-Agent, gap, timeout and caps
    apply. There is no switch that skips robots.txt for another site.
  * GitHub: github.com robots.txt blocks its Atom feeds, so releases, tags and commits sources (kind github_releases) are read
    through the official REST API instead: api.github.com, no token, Accept: application/vnd.github+json, the same User-Agent,
    the same 3.2 s gap. GitHub allows 60 such requests an hour per address. The harvester reads X-RateLimit-Remaining; when it
    reaches 0 it stops calling GitHub, records "rate limited, resume later" (with the reset time) for the sources not reached,
    and carries on with other sites. Run it again after the reset. robots.txt is still checked on every host, api.github.com included.
  * Podcasts: episode metadata, show notes and audio links only. No audio download, no Whisper ("audio transcript deferred").
  * Urantia Book area (faith_ub): links and metadata only, never text.
  * Every item carries its licence text. Items whose redistribution text does not clearly allow it are flagged private: true.
  * A ledger (sources/harvest-ledger.json) records the last run, items and errors per source; re-runs add only new items.
  * The God's Eye View repo is read, never written. Secrets files (.env), hidden tool folders, node_modules and build
    output are not read.
"""
import argparse
import collections
import concurrent.futures
import csv
import gzip
import hashlib
import html
import http.client
import io
import json
import os
import re
import socket
import ssl
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
import zlib
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from prophet_net import (API_ALLOWLIST, Robots, RobotsUnreachable, RobotsUnreadable, api_entry, fetch_robots,  # noqa: F401
                         robots_from_answer)           # the shared robots.txt rule: see prophet_net.py

import prophet_config as pcfg

HOME = Path(__file__).resolve().parent
KNOWLEDGE = pcfg.NOTEBOOKS
REGISTRY = HOME / "sources" / "sources.json"      # the list shipped with the code
APPROVAL = KNOWLEDGE / "sources" / "APPROVAL.json"   # written by approve_sources.py
LEGACY_APPROVALS = set()      # every approval record must say "status": "approved"
GEV_ROOT = pcfg.GODS_EYE_VIEW
GEV_ID = "gods-eye-view"
UA = "Prophet-source-harvest/1.0 (local research)"
ROBOTS_TOKEN = "prophet-source-harvest"
MAX_ITEMS = 50
TIMEOUT = 20
DEFAULT_GAP = 3.2
SITE_GAP = {"arxiv.org": 4.0, "gdeltproject.org": 6.0, "loc.gov": 4.0}
MAX_TEXT = 4000
GITHUB_API = "https://api.github.com"
GITHUB_ACCEPT = "application/vnd.github+json"
GITHUB_FEED = re.compile(r"^/([^/]+)/([^/]+)/(?:(releases|tags)|commits/(.+))\.atom$")     # path of a github.com Atom feed
PARA_WIDTH = 1200
METADATA_ONLY_AREAS = {"faith_ub"}
LEDGER_SCHEMA = "prophet-source-harvest-ledger/1"
TWO_LEVEL = {"co.uk", "org.uk", "com.au", "co.jp", "com.br", "co.nz", "co.za"}

# The robots.txt rule lives in prophet_net.py, shared with the Pattern Lab: API_ALLOWLIST (official API endpoints that are read
# WITHOUT a robots.txt check; project decision), api_entry() and the robots.txt parser. Nothing else skips robots.txt.
COLUMN_HINTS = {
    "csv_retrosheet_gamelogs": ["date", "game_number", "weekday", "visitor", "visitor_league", "visitor_game_no", "home",
                                "home_league", "home_game_no", "visitor_score", "home_score", "outs", "day_night", "completion"],
    "api_coinbase_candles": ["time", "low", "high", "open", "close", "volume"],
    "csv_gdelt_lastupdate": ["size_bytes", "md5", "url"],
    "csv_meteostat_bulk": ["date", "tavg", "tmin", "tmax", "prcp", "snow", "wdir", "wspd", "wpgt", "pres", "tsun"],
}
csv.field_size_limit(10 ** 7)


def say(*parts):
    print(*parts, flush=True)


# ----------------------------------------------------------------------------------------------- small helpers
def utcnow():
    return datetime.now(timezone.utc)


def stamp(dt=None):
    return (dt or utcnow()).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha8(text):
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:8]


def sha16(text):
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:16]


def slug(text, limit=60):
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")[:limit].strip("-") or "item"


def short(value, n=160):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        value = round(value, 6)
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "..."


def write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(tmp, path)


def site_key(host):
    parts = host.lower().split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in TWO_LEVEL:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def redistributable(text):
    """Registry redistribution text -> True only when it clearly allows redistribution (with or without attribution)."""
    t = (text or "").lower()
    if any(bad in t for bad in ("do not redistribute", "non-commercial", "rights vary", "check item", "own copyright")):
        return False
    return any(good in t for good in ("may redistribute", "public domain", "cc0", "cc by", "odc-by", "free reuse",
                                      "open (", "open with", "open license", "free with"))


SECRET_PATTERNS = [
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"(?<![A-Za-z0-9])xox[abprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{24,}"),
    re.compile(r"(?i)\b((?:api[_-]?key|secret|token|passwd|password|client[_-]?secret)[\"']?\s*[:=]\s*[\"']?)([A-Za-z0-9_\-./+=]{20,})"),
]


def redact(text):
    count = 0
    for pattern in SECRET_PATTERNS:
        def swap(match, pattern=pattern):
            nonlocal count
            count += 1
            return match.group(1) + "[REDACTED]" if pattern.groups == 2 else "[REDACTED]"
        text = pattern.sub(swap, text)
    return text, count


# ----------------------------------------------------------------------------------------------- dates
def parse_when(value, slash="mdy", strict=False):
    """Datetime (UTC) from RFC 822, ISO 8601, YYYYMMDD, YYYYMM, YYYY-MM, epoch seconds/ms and common text forms, else None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = int(value) if float(value).is_integer() else float(value)
        text = str(value)
    else:
        text = str(value).strip()
    if not text:
        return None
    try:
        if re.fullmatch(r"\d{13}", text):
            return datetime.fromtimestamp(int(text) / 1000.0, timezone.utc)
        if re.fullmatch(r"\d{9,10}(\.\d+)?", text):
            seconds = float(text)
            return datetime.fromtimestamp(seconds, timezone.utc) if 9e8 < seconds < 4.2e9 else None
        if re.fullmatch(r"\d{8}", text):
            return datetime.strptime(text, "%Y%m%d").replace(tzinfo=timezone.utc)
        if re.fullmatch(r"\d{6}", text):
            return datetime.strptime(text, "%Y%m").replace(tzinfo=timezone.utc)
        if re.fullmatch(r"\d{4}", text):
            return datetime(int(text), 1, 1, tzinfo=timezone.utc) if 1800 <= int(text) <= 2100 else None
        if re.fullmatch(r"[A-Za-z]{3}\s+\d{4}", text):
            return datetime.strptime(text, "%b %Y").replace(tzinfo=timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None
    slash_formats = ("%m/%d/%Y", "%d/%m/%Y", "%m/%d/%y", "%d/%m/%y") if slash == "mdy" else ("%d/%m/%Y", "%m/%d/%Y", "%d/%m/%y", "%m/%d/%y")
    if re.match(r"^\d{1,2}/\d{1,2}/\d{2,4}", text):
        first = text.split()[0]
        order = ("%m/%d", "%d/%m") if slash == "mdy" else ("%d/%m", "%m/%d")
        year = "%Y" if len(first.split("/")[-1]) == 4 else "%y"
        for fmt in ([order[0] + "/" + year] if strict else [o + "/" + year for o in order]):
            try:
                return datetime.strptime(first, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(text)
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError, IndexError):
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%Y-%m", "%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y") + slash_formats:
        try:
            return datetime.strptime(text[:32], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def iso_day(dt):
    return dt.strftime("%Y-%m-%d") if dt else ""


def compact_day(dt):
    return dt.strftime("%Y%m%d") if dt else ""


# ----------------------------------------------------------------------------------------------- text conversion
class _Text(HTMLParser):
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "blockquote",
             "pre", "table", "ul", "ol", "hr", "dd", "dt", "figure", "figcaption", "details", "summary"}
    BREAK = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "blockquote", "pre", "table", "ul", "ol", "hr"}
    SKIP = {"script", "style", "svg", "noscript", "head", "template"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n\n" if tag in self.BREAK else "\n")
            if tag == "li":
                self.out.append("- ")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BREAK or tag == "tr":
            self.out.append("\n\n" if tag in self.BREAK else "\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def tidy(text):
    text = text.replace("\xa0", " ").replace("\u200b", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    out, blank = [], False
    for line in lines:
        if not line:
            blank = bool(out)
            continue
        if blank:
            out.append("")
            blank = False
        out.append(line)
    return "\n".join(out).strip()


def html_to_text(value):
    if not value:
        return ""
    if "<" not in value and "&" not in value:
        return tidy(value)
    parser = _Text()
    try:
        parser.feed(value)
        parser.close()
        return tidy("".join(parser.out))
    except Exception:
        return tidy(html.unescape(re.sub(r"<[^>]+>", " ", value)))


def clip(text, limit=MAX_TEXT):
    if len(text) <= limit:
        return text
    cut = text[:limit]
    edge = max(cut.rfind("\n\n"), cut.rfind(". "))
    if edge > limit * 0.6:
        cut = cut[: edge + 1]
    return cut.rstrip() + " [...]"


def paragraphs(text, width=PARA_WIDTH):
    """Blank-line paragraphs of at most about `width` characters, so a search hit points at a small passage."""
    out = []
    for para in re.split(r"\n\s*\n", (text or "").strip()):
        para = para.strip()
        while len(para) > width * 1.4:
            cut = max(para.rfind(". ", 0, width), para.rfind("\n", 0, width), para.rfind(" ", 0, width))
            if cut < width * 0.4:
                cut = width
            out.append(para[: cut + 1].strip())
            para = para[cut + 1:].strip()
        if para:
            out.append(para)
    return "\n\n".join(out)


def fm_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = json.dumps(re.sub(r"\s+", " ", str(value)).strip(), ensure_ascii=False)
    for bad in ("\u2028", "\u2029", "\x85", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e"):
        text = text.replace(bad, "\\u%04x" % ord(bad))
    return text


def front_matter(pairs):
    return "---\n" + "".join("%s: %s\n" % (k, fm_value(v)) for k, v in pairs if v not in (None, "")) + "---\n"


# ----------------------------------------------------------------------------------------------- HTTP, politely
class FetchError(Exception):
    def __init__(self, message, transient=False, kind="http", status=None):
        super().__init__(message)
        self.transient = transient
        self.kind = kind
        self.status = status


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


OPENER = urllib.request.build_opener(NoRedirect())
_fallback = {"opener": None, "tried": False}


def fallback_opener():
    """Same strict TLS verification, but against the certifi or Git-for-Windows CA list, for when the Windows store lacks a root yet."""
    if not _fallback["tried"]:
        _fallback["tried"] = True
        candidates = []
        try:
            import certifi
            candidates.append(certifi.where())
        except Exception:
            pass
        program_files = os.environ.get("ProgramFiles")
        if program_files:
            candidates += [os.path.join(program_files, "Git", "mingw64", "etc", "ssl", "certs", "ca-bundle.crt")]
        for path in candidates:
            if os.path.exists(path):
                ctx = ssl.create_default_context(cafile=path)
                _fallback["opener"] = urllib.request.build_opener(NoRedirect(), urllib.request.HTTPSHandler(context=ctx))
                break
    return _fallback["opener"]


def classify(exc, url):
    host = urllib.parse.urlsplit(url).hostname or url
    if isinstance(exc, FetchError):
        return exc
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return FetchError("timeout after %d s waiting for %s" % (TIMEOUT, host), transient=True, kind="timeout")
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, ssl.SSLCertVerificationError):
            return FetchError("TLS certificate error for %s: %s" % (host, reason), kind="tls")
        if isinstance(reason, (socket.timeout, TimeoutError)):
            return FetchError("timeout after %d s waiting for %s" % (TIMEOUT, host), transient=True, kind="timeout")
        if isinstance(reason, socket.gaierror):
            return FetchError("DNS lookup failed for %s: %s" % (host, reason), transient=True, kind="dns")
        return FetchError("connection failed for %s: %s" % (host, reason), transient=True, kind="connection")
    if isinstance(exc, ssl.SSLCertVerificationError):
        return FetchError("TLS certificate error for %s: %s" % (host, exc), kind="tls")
    if isinstance(exc, (ssl.SSLError, ConnectionError, http.client.HTTPException, OSError)):
        return FetchError("%s while reading %s: %s" % (type(exc).__name__, host, exc), transient=True, kind="connection")
    return FetchError("%s: %s" % (type(exc).__name__, exc), kind="other")


class RateGate:
    """Request budget of an API that reports X-RateLimit-* headers (GitHub: 60 requests an hour per address without a token).
    When the budget reaches 0 no further request is made until the reset time: callers get a FetchError of kind "ratelimit"
    ("rate limited, resume later") instead of a request, and the run carries on with other sites."""

    def __init__(self, name="GitHub API"):
        self.name, self.mu = name, threading.Lock()
        self.limit = self.remaining = self.reset = None
        self.limited = False

    @staticmethod
    def _number(headers, field):
        try:
            return int(headers.get(field))
        except (TypeError, ValueError):
            return None

    def message(self):
        when = stamp(datetime.fromtimestamp(self.reset, timezone.utc)) if self.reset else "an unknown time"
        return "rate limited, resume later: %s allows %s requests an hour without a token; the limit resets at %s" % (self.name, self.limit or 60, when)

    def check(self):
        """Before each request: raises while the budget is spent and the reset time has not passed."""
        with self.mu:
            if self.limited and self.reset and time.time() >= self.reset:
                self.limited, self.remaining = False, None              # a new hour has started
            if self.limited:
                raise FetchError(self.message(), kind="ratelimit")

    def note(self, status, headers, body=b""):
        """After each response (any status): remember the budget. Raises when the server itself refused for rate limiting."""
        limit, remaining, reset = (self._number(headers, "X-RateLimit-" + field) for field in ("Limit", "Remaining", "Reset"))
        retry = self._number(headers, "Retry-After")
        refused = status in (403, 429) and (remaining == 0 or retry is not None or b"rate limit" in (body or b"").lower())
        with self.mu:
            if limit is not None:
                self.limit = limit
            if remaining is not None:
                self.remaining = remaining
            if reset:
                self.reset = reset
            elif refused and retry is not None:
                self.reset = int(time.time()) + retry
            if remaining == 0 or refused:
                self.limited = True
            text = self.message() if refused else None
        if text:
            raise FetchError(text, kind="ratelimit", status=status)


class Response:
    def __init__(self, url, status, headers, body, truncated, seconds):
        self.url, self.status, self.headers, self.body, self.truncated, self.seconds = url, status, headers, body, truncated, seconds

    def text(self):
        charset = None
        match = re.search(r"charset=([\w-]+)", self.headers.get("Content-Type", "") or "", re.I)
        if match:
            charset = match.group(1)
        for enc in ([charset] if charset else []) + ["utf-8-sig", "cp1252"]:
            try:
                return self.body.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue
        return self.body.decode("utf-8", "replace")


class Http:
    def __init__(self, timeout=TIMEOUT, gap=DEFAULT_GAP):
        self.timeout, self.gap = timeout, gap
        self.mu = threading.Lock()
        self.site_locks, self.last, self.robots = {}, {}, {}
        self.api_hits = {}             # host -> requests sent under API_ALLOWLIST this run (robots.txt not consulted for them)
        self.github = RateGate()       # request budget of api.github.com, shared by every GitHub source of the run
        self.requests = 0

    def _lock(self, site):
        with self.mu:
            return self.site_locks.setdefault(site, threading.Lock())

    def _pace(self, site, crawl_delay):
        gap = max(SITE_GAP.get(site, self.gap), min(crawl_delay or 0, 30.0))
        wait = self.last.get(site, 0) + gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)

    def _request(self, url, accept, max_bytes):
        headers = {"User-Agent": UA, "Accept": accept, "Accept-Encoding": "gzip", "Accept-Language": "en"}
        started = time.monotonic()
        with self.mu:
            self.requests += 1
        try:
            try:
                resp = OPENER.open(urllib.request.Request(url, headers=headers), timeout=self.timeout)
            except urllib.error.URLError as exc:
                alt = fallback_opener() if isinstance(exc.reason, ssl.SSLCertVerificationError) else None
                if alt is None:
                    raise
                resp = alt.open(urllib.request.Request(url, headers=headers), timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read(4096)
            except Exception:
                body = b""
            code, hdrs = exc.code, exc.headers
            exc.close()
            return code, hdrs, body, False, time.monotonic() - started
        except Exception as exc:
            raise classify(exc, url)
        try:
            encoding = (resp.headers.get("Content-Encoding") or "").lower()
            inflater = zlib.decompressobj(16 + zlib.MAX_WBITS) if "gzip" in encoding else None
            raw, truncated = bytearray(), False
            while True:
                if time.monotonic() - started > 75:
                    raise FetchError("download from %s exceeded 75 s" % urllib.parse.urlsplit(url).hostname, transient=True, kind="timeout")
                chunk = resp.read(65536)
                if not chunk:
                    break
                raw += inflater.decompress(chunk) if inflater else chunk
                if len(raw) > max_bytes:
                    raw, truncated = raw[:max_bytes], True
                    break
            if inflater and not truncated:
                raw += inflater.flush()
            return resp.status, resp.headers, bytes(raw), truncated, time.monotonic() - started
        except Exception as exc:
            raise classify(exc, url)
        finally:
            resp.close()

    def _count_request(self):
        with self.mu:
            self.requests += 1

    def _robots_open(self, request, timeout):
        """How robots.txt is opened: like every other request here (a redirect is never followed by itself; CA-bundle fallback)."""
        try:
            return OPENER.open(request, timeout=timeout)
        except urllib.error.URLError as exc:
            alt = fallback_opener() if isinstance(exc.reason, ssl.SSLCertVerificationError) else None
            if alt is None:
                raise
            return alt.open(request, timeout=timeout)

    def _robots_for(self, origin, site):
        """The robots.txt rules of one site, read once per run through prophet_net (one home for what an answer means).
        A robots.txt that cannot be read, answers 429 or a 5xx, refuses us or redirects without end blocks the site (FetchError);
        only a plain 404 or 410 means there are no restrictions."""
        cached = self.robots.get(origin)
        if cached is not None:
            return cached
        self._pace(site, None)
        try:
            status, text = fetch_robots(origin, UA, timeout=self.timeout, opener=self._robots_open, on_request=self._count_request)
        except RobotsUnreachable as exc:
            raise FetchError("robots.txt fetch failed at %s (%s)" % (origin, exc), transient=True, kind="robots")
        finally:
            self.last[site] = time.monotonic()
        try:
            self.robots[origin] = robots_from_answer(status, text, ROBOTS_TOKEN)
        except RobotsUnreadable as exc:
            raise FetchError("robots.txt at %s cannot be used: %s" % (origin, exc), transient=exc.transient, kind="robots", status=status)
        return self.robots[origin]

    def get(self, url, accept="*/*", max_bytes=8_000_000, gate=None):
        for _ in range(6):
            parts = urllib.parse.urlsplit(url)
            if parts.scheme not in ("http", "https") or not parts.hostname:
                raise FetchError("unsupported URL: " + url, kind="url")
            origin, site = "%s://%s" % (parts.scheme, parts.netloc), site_key(parts.hostname)
            with self._lock(site):
                if gate is not None:
                    gate.check()                       # budget spent: no robots.txt fetch and no request
                if api_entry(parts) is not None:       # official API endpoint on the allowlist (project decision): robots.txt not consulted
                    with self.mu:
                        self.api_hits[parts.hostname.lower()] = self.api_hits.get(parts.hostname.lower(), 0) + 1
                    crawl_delay = None
                else:
                    robots = self._robots_for(origin, site)
                    allowed, rule = robots.rule_for(parts)
                    if not allowed:                    # no override: only prophet_net.API_ALLOWLIST skips this check
                        raise FetchError("robots.txt disallows %s for %s (rule %s)" % (url, ROBOTS_TOKEN, rule), kind="robots")
                    crawl_delay = robots.delay
                self._pace(site, crawl_delay)
                try:
                    status, headers, body, truncated, seconds = self._request(url, accept, max_bytes)
                finally:
                    self.last[site] = time.monotonic()
                if gate is not None:
                    gate.note(status, headers, body)   # keeps the budget; raises when the server refused for rate limiting
            if status in (301, 302, 303, 307, 308) and headers.get("Location"):
                url = urllib.parse.urljoin(url, headers["Location"])
                continue
            if 200 <= status < 300:
                return Response(url, status, headers, body, truncated, seconds)
            reason = http.client.responses.get(status, "")
            transient = status in (408, 425, 429) or 500 <= status <= 599
            raise FetchError("HTTP %d %s for %s" % (status, reason, url), transient=transient, kind="http", status=status)
        raise FetchError("too many redirects from " + url, kind="redirect")


# ----------------------------------------------------------------------------------------------- feeds
def lname(tag):
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def kids(elem, name):
    return [c for c in elem if lname(c.tag) == name]


def kid(elem, name):
    found = kids(elem, name)
    return found[0] if found else None


def etext(elem):
    return "".join(elem.itertext()).strip() if elem is not None else ""


def parse_xml(body):
    text = body.decode("utf-8", "replace") if isinstance(body, bytes) else body
    text = re.sub(r"^\ufeff?\s*<\?xml[^>]*\?>", "", text, count=1)
    for attempt in range(3):
        try:
            return ET.fromstring(text)
        except ET.ParseError as exc:
            if attempt == 0:
                text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
                text = re.sub(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)", "&amp;", text)
            elif attempt == 1:
                start = min([i for i in (text.find("<rss"), text.find("<feed"), text.find("<rdf:RDF")) if i >= 0] or [0])
                text = text[start:]
            else:
                raise FetchError("XML parse error: %s" % exc, kind="parse")


def clock_text(value):
    value = (value or "").strip()
    if re.fullmatch(r"\d+", value):
        seconds = int(value)
        return "%d:%02d:%02d" % (seconds // 3600, seconds % 3600 // 60, seconds % 60) if seconds >= 3600 else "%d:%02d" % (seconds // 60, seconds % 60)
    return value


def feed_entries(root):
    """Normalised items from RSS 2.0, RSS 1.0 (RDF) and Atom (GitHub, arXiv)."""
    tag = lname(root.tag).lower()
    if tag == "rss":
        channel = kid(root, "channel")
        nodes = kids(channel, "item") if channel is not None else []
    elif tag == "rdf":
        nodes = kids(root, "item")
    elif tag == "feed":
        nodes = kids(root, "entry")
    else:
        raise FetchError("not a feed: root element <%s>" % lname(root.tag), kind="parse")
    entries = []
    for node in nodes:
        title = re.sub(r"\s+", " ", html_to_text(etext(kid(node, "title")))).strip()
        links = [(l.get("rel") or "alternate", l.get("href") or "", l.get("title") or "", l.get("type") or "") for l in kids(node, "link") if l.get("href")]
        url = next((h for rel, h, _, t in links if rel == "alternate" and "html" in (t or "html")), "") or next((h for rel, h, _, _ in links if rel == "alternate"), "")
        if not url and kid(node, "link") is not None and kid(node, "link").text:
            url = kid(node, "link").text.strip()
        guid = etext(kid(node, "guid")) or etext(kid(node, "id"))
        if not url and guid.startswith("http"):
            url = guid
        when_text = next((etext(kid(node, n)) for n in ("pubDate", "published", "updated", "date", "created") if kid(node, n) is not None and etext(kid(node, n))), "")
        body = ""
        for name in ("encoded", "content", "description", "summary", "subtitle"):
            node_text = etext(kid(node, name)) if kid(node, name) is not None else ""
            if node_text and len(html_to_text(node_text)) > len(body):
                body = html_to_text(node_text)
        authors = [etext(kid(a, "name")) or etext(a) for a in kids(node, "author") if (etext(kid(a, "name")) or etext(a))]
        authors += [etext(c) for c in kids(node, "creator") if etext(c)]
        authors = [a for a in (re.sub(r"\s*[\(<]?\S+@\S+\.\S+[\)>]?\s*", " ", a).strip() for a in authors) if a]
        cats = [(c.get("term") or etext(c)).strip() for c in kids(node, "category") if (c.get("term") or etext(c))]
        enclosure = None
        enc = kid(node, "enclosure")
        if enc is not None and enc.get("url"):
            enclosure = {"url": enc.get("url"), "type": enc.get("type") or "", "length": enc.get("length") or ""}
        else:
            for rel, href, _, mime in links:
                if rel == "enclosure":
                    enclosure = {"url": href, "type": mime, "length": ""}
        entries.append({"guid": guid, "title": title, "url": url, "when": when_text, "text": body, "authors": authors[:6],
                        "cats": cats[:8], "enclosure": enclosure, "links": links,
                        "duration": clock_text(etext(kid(node, "duration"))), "episode": etext(kid(node, "episode")),
                        "season": etext(kid(node, "season"))})
    return entries


def newest_first(entries, key):
    stamped = [(key(e), i, e) for i, e in enumerate(entries)]
    if sum(1 for s, _, _ in stamped if s) >= max(1, len(stamped) * 0.6):
        stamped.sort(key=lambda t: (t[0] is None, -(t[0].timestamp()) if t[0] else 0, t[1]))
    return [e for _, _, e in stamped]


def feed_items(src, root, limit):
    entries = feed_entries(root)
    for e in entries:
        e["dt"] = parse_when(e["when"])
    entries = newest_first(entries, lambda e: e["dt"])[:limit]
    meta_only = src["area"] in METADATA_ONLY_AREAS
    podcast = src["id"].startswith("pod_")
    out = []
    for e in entries:
        enclosure = e["enclosure"] if e["enclosure"] and re.match(r"(audio|video)/", e["enclosure"]["type"] or "audio/") else None
        extras = []
        if e["authors"]:
            extras.append("Authors: " + ", ".join(e["authors"]))
        if e["cats"]:
            extras.append("Categories: " + ", ".join(e["cats"]))
        if e["duration"]:
            extras.append("Duration: " + e["duration"])
        if e["season"] or e["episode"]:
            extras.append("Season %s Episode %s" % (e["season"] or "-", e["episode"] or "-"))
        if src["kind"] == "arxiv":
            pdf = next((h for rel, h, t, _ in e["links"] if t == "pdf"), "")
            if pdf:
                extras.append("PDF link (not downloaded): " + pdf)
        item_extra = {}
        label = "feed item"
        if podcast or (enclosure and enclosure["type"].startswith("audio")):
            label = "podcast episode (show notes and links only)"
            if enclosure:
                item_extra.update(audio_url=enclosure["url"], audio_type=enclosure["type"], audio_transcript="deferred")
                extras.append("Audio (not downloaded): " + enclosure["url"] + "; audio transcript deferred")
        if src["kind"] == "github_releases":
            label = "release or commit notes"
        if src["kind"] == "arxiv":
            label = "paper abstract (metadata)"
        text = clip(e["text"]) if not meta_only else ""
        if meta_only:
            label = label.split(" (")[0] + " (metadata and links only; Urantia Book area, no text stored)"
            extras = [x for x in extras if not x.startswith("Categories")]
        key = e["guid"] or e["url"] or (e["title"] + "|" + e["when"])
        if src["kind"] == "arxiv":
            key = re.sub(r"v\d+$", "", key)
        title = e["title"] or short(e["text"], 90) or "untitled"
        out.append({"key": key, "title": title, "url": e["url"] or src["homepage"], "dt": e["dt"], "text": text,
                    "tail": extras, "label": label, "front": item_extra})
    return uniquify_urls(out)


def uniquify_urls(items):
    """ask groups hits by URL, so items that share one link (a show's home page) each get their own fragment."""
    counts = collections.Counter(i["url"] for i in items)
    for i in items:
        if counts[i["url"]] > 1:
            i["url"] = i["url"].split("#")[0] + "#" + sha8(i["key"])
    return items


def fetch_feed(http, src, limit):
    resp = http.get(src["url"], accept="application/atom+xml, application/rss+xml, application/rdf+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.1")
    root = parse_xml(resp.body)
    items = feed_items(src, root, limit)
    return {"items": items, "status": resp.status, "bytes": len(resp.body), "note": "%d entries" % len(items)}


# ----------------------------------------------------------------------------------------------- GitHub (official REST API)
def github_route(feed_url, limit):
    """(kind, API url, owner, repo, branch) for a github.com Atom feed address (releases.atom, tags.atom, commits/<branch>.atom), else None."""
    parts = urllib.parse.urlsplit(feed_url)
    if parts.scheme != "https" or (parts.hostname or "").lower() not in ("github.com", "www.github.com"):
        return None
    match = GITHUB_FEED.match(urllib.parse.unquote(parts.path))
    if not match:
        return None
    owner, repo, listing, branch = match.groups()
    base = "%s/repos/%s/%s/" % (GITHUB_API, urllib.parse.quote(owner, safe=""), urllib.parse.quote(repo, safe=""))
    if branch:
        return "commits", base + "commits?sha=%s&per_page=%d" % (urllib.parse.quote(branch, safe=""), limit), owner, repo, branch
    return listing, base + "%s?per_page=%d" % (listing, 50 if listing == "tags" else limit), owner, repo, None


def markdown_text(value):
    """Release and commit notes are Markdown: keep the words, drop images, HTML comments, layout tags and e-mail addresses."""
    text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"</?(?:img|br|hr|p|div|span|details|summary|picture|source|center|table|thead|tbody|tr|td|th|a)\b[^>]*>", " ", text, flags=re.I)
    text = re.sub(r"\s*<[^<>\s]+@[^<>\s]+>", "", text)                       # "Co-authored-by: Name <mail>" keeps the name only
    return tidy(text)


def release_item(rel, owner, repo):
    tag = (rel.get("tag_name") or "").strip()
    title = re.sub(r"\s+", " ", rel.get("name") or "").strip() or tag or "release"
    login = (rel.get("author") or {}).get("login") or ""
    assets = [a["name"] for a in rel.get("assets") or [] if isinstance(a, dict) and a.get("name")]
    extras = ["Repository: %s/%s" % (owner, repo)]
    if tag:
        extras.append("Tag: " + tag + (" (pre-release)" if rel.get("prerelease") else ""))
    if login and not login.endswith("[bot]"):
        extras.append("Authors: " + login)
    if assets:
        extras.append("Release files (not downloaded): " + ", ".join(assets[:12]) + (", ... (%d files)" % len(assets) if len(assets) > 12 else ""))
    return {"key": "release:" + str(rel.get("id") or rel.get("html_url") or tag), "title": title,
            "url": rel.get("html_url") or "https://github.com/%s/%s/releases/tag/%s" % (owner, repo, urllib.parse.quote(tag)),
            "dt": parse_when(rel.get("published_at") or rel.get("created_at")), "text": clip(markdown_text(rel.get("body"))) or title,
            "tail": extras, "label": "release or commit notes", "front": {}}


def commit_item(rec, owner, repo, branch):
    commit = rec.get("commit") or {}
    sha = rec.get("sha") or ""
    message = (commit.get("message") or "").strip()
    who = commit.get("author") or {}
    login = (rec.get("author") or {}).get("login") or ""
    name = re.sub(r"\s*[\(<]?\S+@\S+\.\S+[\)>]?\s*", " ", who.get("name") or "").strip() or login
    extras = ["Repository: %s/%s" % (owner, repo), "Branch: " + branch, "Commit: " + sha]
    if name:
        extras.append("Authors: " + name + (" (%s)" % login if login and login != name else ""))
    title = short(message.split("\n", 1)[0], 200) or sha[:7] or "commit"
    return {"key": "commit:" + sha, "title": title,
            "url": rec.get("html_url") or "https://github.com/%s/%s/commit/%s" % (owner, repo, sha),
            "dt": parse_when((commit.get("committer") or {}).get("date") or who.get("date")), "text": clip(markdown_text(message)) or title,
            "tail": extras, "label": "release or commit notes", "front": {}}


def tag_item(rec, owner, repo):
    name = rec.get("name") or ""
    sha = (rec.get("commit") or {}).get("sha") or ""
    text = "Tag %s of %s/%s points at commit %s. The GitHub tags list carries no dates or notes." % (name, owner, repo, sha[:12] or "unknown")
    return {"key": "tag:" + name, "title": name or "tag", "url": "https://github.com/%s/%s/releases/tag/%s" % (owner, repo, urllib.parse.quote(name)),
            "dt": None, "text": text, "tail": ["Repository: %s/%s" % (owner, repo), "Commit: " + sha] if sha else ["Repository: %s/%s" % (owner, repo)],
            "label": "release or commit notes", "front": {}}


def version_key(name):
    """Tags carry no dates: order by the numbers in the name, so n9.1 outranks n9.0.2 and v0.6.1 sorts last."""
    return [int(n) for n in re.findall(r"\d+", name)[:6]], name


def fetch_github(http, src, limit):
    """Releases, tags and commits of a github.com repo through api.github.com (keyless), falling back to the feed for any other address."""
    route = github_route(src["url"], limit)
    if route is None:
        return fetch_feed(http, src, limit)
    kind, url, owner, repo, branch = route
    resp = http.get(url, accept=GITHUB_ACCEPT, gate=http.github)
    if resp.truncated:
        raise FetchError("response larger than the 8 MB cap, not parsed: " + url, kind="parse")
    try:
        data = json.loads(resp.text())
    except ValueError as exc:
        raise FetchError("JSON parse error: %s" % exc, kind="parse")
    if not isinstance(data, list):
        raise FetchError("unexpected answer from the GitHub API (expected a list): %s" % short(data, 120), kind="parse")
    rows = [r for r in data if isinstance(r, dict)]
    if kind == "tags":
        rows = sorted(rows, key=lambda r: version_key(r.get("name") or ""), reverse=True)
    rows = rows[:limit]
    if kind == "releases":
        items = [release_item(r, owner, repo) for r in rows]
    elif kind == "commits":
        items = [commit_item(r, owner, repo, branch) for r in rows]
    else:
        items = [tag_item(r, owner, repo) for r in rows]
    left = RateGate._number(resp.headers, "X-RateLimit-Remaining")                 # as reported with this very answer
    return {"items": uniquify_urls(items), "status": resp.status, "bytes": len(resp.body),
            "note": "%d %s via GitHub API%s" % (len(items), kind, "" if left is None else ", %d of %s requests left this hour" % (left, http.github.limit or 60))}


# ----------------------------------------------------------------------------------------------- JSON and tables
TITLE_KEYS = ("title", "headline", "name", "question", "event_title", "label", "displayName", "full_name", "fullName", "official_name", "raceName")
SUMMARY_KEYS = ("summary", "abstract", "description", "body", "text", "excerpt", "snippet", "content", "subtitle", "extract", "rules", "details")
URL_KEYS = ("url", "link", "html_url", "web_url", "permalink", "canonical_url", "href", "uri", "id")
DATE_KEYS = ("published", "publishedAt", "published_at", "publication_date", "date", "datetime", "timestamp", "time", "seendate", "created",
             "created_at", "createdAt", "createdTime", "created_time", "updated", "updated_at", "updatedAt", "modified", "gameDate", "effective",
             "sent", "report_date_as_yyyy_mm_dd", "record_date", "date_start", "start", "startTime", "start_date", "open_time", "close_time",
             "closeTime", "end_date")
ID_KEYS = ("id", "identifier", "guid", "uuid", "slug", "ticker", "event_ticker", "gamePk", "document_number", "key", "conditionId")
URL_BUILDERS = {
    "api_hf_daily_papers": lambda r: "https://huggingface.co/papers/" + str((r.get("paper") or {}).get("id", "")),
    "api_internet_archive": lambda r: "https://archive.org/details/" + str(r.get("identifier", "")),
}


NOISE_KEY = re.compile(r"(^|\.)(_id|avatar\w*|thumbnail\w*|mediaurls?|hidden|ispro|ishf\w*|usertype)(\[\d+\])?(\.|$)|_id$", re.I)
HEX_ID = re.compile(r"[0-9a-f]{20,}", re.I)


def pick(rec, keys):
    """First non-empty scalar among `keys` (case-insensitive), looking at the record and its properties/attributes."""
    for scope in (rec, rec.get("properties") if isinstance(rec.get("properties"), dict) else None,
                  rec.get("attributes") if isinstance(rec.get("attributes"), dict) else None):
        if not scope:
            continue
        lowered = {str(k).lower(): v for k, v in scope.items()}
        for key in keys:
            value = lowered.get(key.lower())
            if value not in (None, "") and not isinstance(value, (dict, list)):
                return key, value
    return None


def leafs(obj, path="", depth=0, out=None, limit=80, prefer=None, maxdepth=3):
    if out is None:
        out = []
    if len(out) >= limit:
        return out
    if isinstance(obj, dict):
        items = list(obj.items())
        nested = sum(1 for _, v in items if isinstance(v, (dict, list)))
        if len(items) > 12 and nested > 0.8 * len(items):          # a big keyed collection: name it, show two members
            out.append((path or "keys", "%d keys: %s ..." % (len(items), ", ".join(str(k) for k, _ in items[:10]))))
            picked = [it for it in items if str(it[0]) in (prefer or ())][:2]
            items = picked + [it for it in items if it not in picked][: 2 - len(picked)]
        elif len(items) > 14 and nested == 0:                     # a long run of scalars (years, dates): show both ends
            items = items[:5] + [("...", "(%d more)" % (len(items) - 10))] + items[-5:]
        for k, v in items:
            p = "%s.%s" % (path, k) if path else str(k)
            if isinstance(v, dict):
                if depth < maxdepth:
                    leafs(v, p, depth + 1, out, limit, prefer, maxdepth)
                else:
                    out.append((p, "{...}"))
            elif isinstance(v, list):
                if v and all(not isinstance(x, (dict, list)) for x in v):
                    out.append((p, "[" + ", ".join(short(x, 40) for x in v[:8]) + (", ... (%d values)" % len(v) if len(v) > 8 else "") + "]"))
                elif v and depth < maxdepth - 1:
                    out.append((p, "%d entries" % len(v)))
                    leafs(v[0], p + "[0]", depth + 1, out, limit, prefer, maxdepth)
            elif v not in (None, ""):
                out.append((p, short(v)))
            if len(out) >= limit:
                break
    elif isinstance(obj, list):
        out.append((path or "list", "%d entries" % len(obj)))
    else:
        out.append((path or "value", short(obj)))
    return out


def find_records(data):
    """Largest list of dict records anywhere in the JSON (depth <= 4); GeoJSON features are unwrapped."""
    best = []

    def walk(node, path, depth):
        if depth > 4:
            return
        if isinstance(node, list):
            dicts = [x for x in node if isinstance(x, dict)]
            if dicts and len(dicts) >= len(node) * 0.8 and any(len(d) >= 2 for d in dicts):
                best.append((len(dicts), -depth, path, dicts))
            for i, x in enumerate(node[:3]):
                if isinstance(x, (dict, list)):
                    walk(x, "%s[%d]" % (path, i), depth + 1)
        elif isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, (dict, list)):
                    walk(v, "%s.%s" % (path, k) if path else str(k), depth + 1)

    walk(data, "", 0)
    if not best:
        return [], ""
    best.sort(key=lambda t: (t[0], t[1]), reverse=True)
    _, _, path, records = best[0]
    flat = []
    for rec in records:
        if isinstance(rec.get("properties"), dict) and rec.get("type") == "Feature":
            merged = dict(rec["properties"])
            if rec.get("id") is not None:
                merged.setdefault("id", rec["id"])
                if isinstance(rec["id"], str) and rec["id"].startswith("http"):
                    merged.setdefault("url", rec["id"])
            geometry = rec.get("geometry")
            if isinstance(geometry, dict) and geometry.get("coordinates") is not None:
                merged["coordinates"] = geometry["coordinates"]
            flat.append(merged)
        else:
            flat.append(rec)
    return flat, path or "(top level)"


def rec_date(rec):
    found = pick(rec, DATE_KEYS)
    return parse_when(found[1]) if found else None


def documentary(records):
    """Records that read like documents: a real title plus a summary, a link or a date (not code lists or structure metadata)."""
    sample = records[:10]
    good = 0
    for rec in sample:
        found = pick(rec, TITLE_KEYS)
        if found and isinstance(found[1], str) and len(found[1]) >= 8 and (pick(rec, SUMMARY_KEYS) or pick(rec, URL_KEYS[:-1]) or pick(rec, DATE_KEYS)):
            good += 1
    return good >= max(2, int(0.7 * len(sample) + 0.5))


def record_items(src, records, limit, api_url):
    ordered = newest_first(records, rec_date)[:limit]
    builder = URL_BUILDERS.get(src["id"])
    skip_keys = {x.lower() for x in TITLE_KEYS + SUMMARY_KEYS}
    date_names = {x.lower() for x in DATE_KEYS}
    out = []
    for n, rec in enumerate(ordered, 1):
        flat = leafs(rec, limit=200)
        found_title = pick(rec, TITLE_KEYS)
        title = str(found_title[1]) if found_title else (next((v for _, v in flat if v and not v.startswith("[")), "") or "record %d" % n)
        url = ""
        if builder:
            try:
                url = builder(rec)
            except Exception:
                url = ""
        if not url:
            for key in URL_KEYS:
                found = pick(rec, (key,))
                if found and isinstance(found[1], str) and found[1].startswith("http"):
                    url = found[1]
                    break
        rid = None
        for key in ID_KEYS:
            found = pick(rec, (key,))
            if found:
                rid = str(found[1])
                break
        if rid is None:
            rid = next((v for k, v in flat if k.endswith(".id") or k.endswith("identifier")), None)
        key = rid or url or (title + "|" + str(rec_date(rec)))
        summary_field = pick(rec, SUMMARY_KEYS)
        summary = html_to_text(str(summary_field[1])) if summary_field else ""
        lines = ["%s: %s" % (k, date_cell(v) if k.split(".")[-1].lower() in date_names else v) for k, v in flat
                 if k.split(".")[-1].lower() not in skip_keys and not NOISE_KEY.search(k) and not HEX_ID.fullmatch(v)][:18]
        text = "\n\n".join(p for p in (clip(summary, 2500), "\n".join(lines)) if p)
        out.append({"key": src["id"] + ":" + key, "title": short(title, 200), "url": url or (src["homepage"] + "#" + sha8(key)),
                    "dt": rec_date(rec), "text": text, "tail": ["Record from API: " + api_url], "label": "open data record", "front": {}})
    return uniquify_urls(out)


# ----------------------------------------------------------------------------------------------- tables (CSV and friends)
def unpack_text(data, url):
    """Text of a CSV/TXT payload that may be a zip or gzip file."""
    note = ""
    if data[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = [i for i in zf.infolist() if not i.is_dir()]
            if not names:
                raise FetchError("zip file is empty: " + url, kind="parse")
            names.sort(key=lambda i: (not i.filename.lower().endswith((".csv", ".txt")), -i.file_size))
            chosen = names[0]
            if chosen.file_size > 40_000_000:
                raise FetchError("zip member %s is larger than 40 MB" % chosen.filename, kind="parse")
            data = zf.read(chosen)
            note = "unzipped member " + chosen.filename
    elif data[:2] == b"\x1f\x8b":
        data = gzip.GzipFile(fileobj=io.BytesIO(data)).read(40_000_000)
        note = "gunzipped"
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc), note
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace"), note


def cell_text(value, n=60):
    value = str(value)
    return short(value, 220) if value.startswith("http") else short(value, n)


def date_cell(value):
    """Epoch-like date cells also show the calendar day."""
    value = str(value).strip()
    if re.fullmatch(r"\d{9,13}", value):
        dt = parse_when(value)
        if dt:
            return "%s (%s)" % (iso_day(dt), value)
    return short(value, 40)


def numeric(cell):
    return bool(re.fullmatch(r"[-+]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?%?", cell.strip())) if cell.strip() else False


def split_row(line, delim):
    if delim == "ws":
        return re.split(r"\s+", line.strip())
    return next(csv.reader([line], delimiter=delim))


def read_table(text, hint_columns=None):
    """(headers, rows, preamble) for CSV, pipe, tab, semicolon or whitespace tables with comment and footer lines."""
    preamble, blocks, current, comments = [], [], [], []
    for line in text.replace("\r", "").split("\n"):
        stripped = line.strip()
        if not stripped:
            if current:
                blocks.append(current)
                current = []
            continue
        if stripped[0] in "#%":
            comments.append(stripped.lstrip("#% ").strip())
            continue
        current.append(line)
    if current:
        blocks.append(current)
    if not blocks:
        raise FetchError("no data rows found", kind="parse")
    block = max(blocks, key=len)
    sample = block[:30]
    best = (0, None, 0)
    for delim in (",", "\t", "|", ";", "ws"):
        try:
            counts = [len(split_row(line, delim)) for line in sample]
        except csv.Error:
            continue
        mode = max(set(counts), key=counts.count)
        score = counts.count(mode) if mode >= 2 else 0
        if score > best[0] or (score == best[0] and delim == "," and best[1] is not None and mode > best[2]):
            best = (score, delim, mode)
    _, delim, width = best
    if not delim:
        return ["text"], [[line] for line in block], comments[:5]
    rows, odd = [], []
    for line in block:
        try:
            cells = split_row(line, delim)
        except csv.Error:
            odd.append(line)
            continue
        (rows if len(cells) == width else odd).append(cells if len(cells) == width else line)
    preamble = [o.strip() for o in odd[:4]] + comments[:4]
    headers = None
    if hint_columns:
        headers = (hint_columns + ["c%d" % (i + 1) for i in range(len(hint_columns), width)])[:width]
    elif rows:
        first = rows[0]
        numeric_share = sum(1 for c in first if numeric(c)) / max(1, len(first))
        if numeric_share < 0.25 and len(set(first)) == len(first):
            headers, rows = [c.strip() or "c%d" % (i + 1) for i, c in enumerate(first)], rows[1:]
        elif comments:
            guess = [c.strip() for c in split_row(comments[-1], delim)] if comments[-1].count(",") or delim == "ws" else []
            if len(guess) == width and sum(1 for c in guess if numeric(c)) == 0:
                headers = guess
    if headers is None:
        headers = ["c%d" % (i + 1) for i in range(width)]
    rows = [r for r in rows if r[0].strip() != headers[0]]
    return headers, rows, preamble


def date_column(headers, rows):
    """(index, parser) of the column that holds dates, else (None, None)."""
    sample = rows[:80]
    if len(sample) < 3:
        return None, None
    best = None
    for ci in range(min(len(headers), 24)):
        values = [r[ci].strip() for r in sample if ci < len(r) and r[ci].strip()]
        if len(values) < len(sample) * 0.8:
            continue
        name = headers[ci].lower()
        named = bool(re.search(r"date|time|day|period|year|yr|month|obs|seas", name))
        for slash in ("mdy", "dmy"):
            parsed = [parse_when(v, slash, True) for v in values]
            good = [v for v, p in zip(values, parsed) if p is not None]
            if len(values) - len(good) > max(2, 0.1 * len(values)):         # a units row or a footer may be unparseable, little else
                continue
            pure = [v for v in good if re.fullmatch(r"\d+", v)]
            if pure and len(pure) == len(good):
                if all(len(v) == 4 for v in good) and not named:
                    continue
                if all(len(v) in (9, 10) for v in good):
                    ordered = [int(v) for v in good]
                    if ordered != sorted(ordered) and ordered != sorted(ordered, reverse=True):
                        continue
                elif not named and not all(len(v) in (6, 8) for v in good):
                    continue
            if len(set(good)) < 3:
                continue
            score = (2 if named else 1, -ci)
            if best is None or score > best[0]:
                best = (score, ci, slash)
            break
    if best is None:
        return None, None
    _, ci, slash = best
    return ci, (lambda v, s=slash: parse_when(v, s, True))


def table_text(src, headers, rows, preamble, limit, truncated):
    ci, parse = date_column(headers, rows)
    shown_cols = min(len(headers), 14)
    lines = []
    if ci is not None:
        dated = [(parse(r[ci]) if ci < len(r) else None, r) for r in rows]
        dated = [(d, r) for d, r in dated if d]                 # rows without a date (units row, footer) are not "newest"
        stamps = [d for d, _ in dated]
        if not stamps:
            ci = None
    if ci is not None:
        first_dt, last_dt = dated[0][0], dated[-1][0]
        ascending = bool(first_dt and last_dt and first_dt <= last_dt)
        picked = [r for _, r in (dated[-limit:][::-1] if ascending else dated[:limit])]
        lines.append("Date column: %s. Range %s to %s (%d dated rows). Newest %d rows follow, newest first." % (
            headers[ci], iso_day(min(stamps)), iso_day(max(stamps)), len(stamps), len(picked)))
        newest_dt = max(stamps)
    else:
        picked = rows[:limit]
        newest_dt = None
        lines.append("No date column found. First %d rows follow." % len(picked))
    summary = ["Columns (%d): %s" % (len(headers), ", ".join(headers[:30]) + (" ..." if len(headers) > 30 else "")),
               "Rows parsed: %d%s." % (len(rows), " (download was cut at the size cap, so this is the start of the file only)" if truncated else "")]
    if preamble:
        summary.append("File notes: " + short(" / ".join(preamble), 400))
    body = []
    for n, row in enumerate(picked):
        cells = ["%s=%s" % (headers[i], date_cell(row[i]) if i == ci else cell_text(row[i])) for i in range(shown_cols) if i < len(row) and row[i].strip() != ""]
        body.append(" | ".join(cells))
        if (n + 1) % 8 == 0:
            body.append("")
    if len(headers) > shown_cols:
        body.append("(+%d more columns not shown)" % (len(headers) - shown_cols))
    return ["\n".join(summary), lines[0], "\n".join(body).strip()], newest_dt


def snapshot_item(src, url, resp, paragraphs_list, newest_dt, shape):
    today = utcnow()
    head = ("%s. %s" % (src["name"], src["notes"])).strip()
    fetched = "Fetched %s: %s (HTTP %d, %d bytes, %s)." % (iso_day(today), url, resp.status, len(resp.body), resp.headers.get("Content-Type", "unknown type"))
    newest = ["Newest data date in this sample: %s." % iso_day(newest_dt)] if newest_dt else []
    text = "\n\n".join([head, fetched, shape] + newest + [p for p in paragraphs_list if p] +
                       ["This is a sample of the verified query for this source. Change the query parameters or the file to get other data."])
    return {"key": "%s:snapshot:%s" % (src["id"], iso_day(today)), "title": "data snapshot %s" % iso_day(today),
            "url": src["homepage"] + ("&" if "#" in src["homepage"] else "#") + "snapshot-" + compact_day(today),
            "dt": today, "text": text, "tail": ["Fetched URL: " + url], "label": "open data snapshot", "front": {"fetched_at": stamp(today)}}


URL_TWEAKS = {"api_cftc_cot": "&$order=report_date_as_yyyy_mm_dd%20DESC"}      # newest rows first (same endpoint, standard SoQL sort)


def bump_limits(url, limit):
    host = urllib.parse.urlsplit(url).hostname or ""
    names = ["limit", "per_page", "maxrecords", "rows", r"\$limit", r"page\[size\]"] + (["c"] if host.endswith("loc.gov") else [])
    rx = re.compile(r"(?<=[?&])(%s)=(\d+)" % "|".join(names))
    return rx.sub(lambda m: "%s=%d" % (m.group(1), limit) if int(m.group(2)) < limit else m.group(0), url)


def xml_outline(root, limit=60):
    lines = []

    def walk(node, path, depth):
        if len(lines) >= limit or depth > 6:
            return
        name = lname(node.tag)
        children = list(node)
        text = (node.text or "").strip()
        if not children and (text or node.attrib):
            attrs = " ".join("%s=%s" % (lname(k), short(v, 40)) for k, v in list(node.attrib.items())[:4])
            lines.append("%s/%s: %s %s" % (path, name, short(text, 120), attrs))
        for child in children[:6]:
            walk(child, path + "/" + name, depth + 1)

    walk(root, "", 0)
    return lines


def fetch_api(http, src, limit):
    url = bump_limits(src["url"], limit)
    if src["id"] in URL_TWEAKS and "$order" not in url:
        url += URL_TWEAKS[src["id"]]
    resp = http.get(url, accept="application/json, application/geo+json;q=0.9, text/plain;q=0.5, application/xml;q=0.4, */*;q=0.1")
    if resp.truncated:
        raise FetchError("response larger than the 8 MB cap, not parsed: " + url, kind="parse")
    text = resp.text().lstrip("\ufeff \t\r\n")
    if text[:1] in "{[":
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise FetchError("JSON parse error: %s" % exc, kind="parse")
        records, where = find_records(data)
        if records and documentary(records):
            items = record_items(src, records, limit, url)
            return {"items": items, "status": resp.status, "bytes": len(resp.body), "note": "%d records" % len(items)}
        shape = ("JSON %s." % ("list of %d" % len(data) if isinstance(data, list) else "object with keys " + ", ".join(list(data)[:20])))
        parts, newest = [], None
        prefer = {t for t in re.split(r"[/?&=,:#]", url) if t}
        if records:
            ordered = newest_first(records, rec_date)
            newest = max((d for d in (rec_date(r) for r in records) if d), default=None)
            shape += " Largest record list: %d records at %s; newest %d shown." % (len(records), where, min(limit, len(ordered)))
            rows = ordered[:limit]
            block = []
            for n, rec in enumerate(rows):
                block.append(" | ".join("%s=%s" % (k, v) for k, v in leafs(rec, limit=10)))
                if (n + 1) % 6 == 0:
                    block.append("")
            parts.append("\n".join(k + ": " + v for k, v in leafs(data, limit=30, prefer=prefer, maxdepth=6)))   # the document itself first (values, units, dates)
            parts.append("\n".join(block).strip())
        else:
            arrays = isinstance(data, list) and data and isinstance(data[0], list)
            if arrays:
                rows = [[cell_text(c) for c in r] for r in data if isinstance(r, list)]
                hint = COLUMN_HINTS.get(src["id"])
                headers = hint or (rows[0] if rows and all(not numeric(c) for c in rows[0]) else ["c%d" % (i + 1) for i in range(len(rows[0]))])
                body_rows = rows if hint or headers is not rows[0] else rows[1:]
                ci, parse = date_column(headers, body_rows)
                if ci is not None:
                    dated = [(parse(r[ci]), r) for r in body_rows if ci < len(r)]
                    dated = [(d, r) for d, r in dated if d]
                    ascending = bool(dated and dated[0][0] <= dated[-1][0])
                    chosen = dated[-limit:][::-1] if ascending else dated[:limit]
                    newest = max((d for d, _ in dated if d), default=None)
                    body_rows = [r for _, r in chosen]
                else:
                    body_rows = body_rows[:limit]
                shape += " Table of %d rows; columns: %s." % (len(rows), ", ".join(headers[:20]))
                if ci is not None:
                    shape += " Date column %s; the newest %d rows follow, newest first." % (headers[ci], len(body_rows))
                block = []
                for n, r in enumerate(body_rows):
                    block.append(" | ".join("%s=%s" % (headers[i], date_cell(r[i]) if i == ci else r[i]) for i in range(min(len(headers), len(r)))))
                    if (n + 1) % 8 == 0:
                        block.append("")
                parts.append("\n".join(block).strip())
            else:
                parts.append("\n".join("%s: %s" % (k, v) for k, v in leafs(data, limit=90, prefer=prefer, maxdepth=5)))
        return {"items": [snapshot_item(src, url, resp, [clip("\n\n".join(parts), 6000)], newest, shape)], "status": resp.status,
                "bytes": len(resp.body), "note": "snapshot"}
    if text[:1] == "<":
        root = parse_xml(resp.body)
        if lname(root.tag).lower() in ("rss", "feed", "rdf"):
            return {"items": feed_items(src, root, limit), "status": resp.status, "bytes": len(resp.body), "note": "feed"}
        outline = xml_outline(root)
        shape = "XML document, root element <%s>." % lname(root.tag)
        return {"items": [snapshot_item(src, url, resp, ["\n".join(outline)], None, shape)], "status": resp.status,
                "bytes": len(resp.body), "note": "xml snapshot"}
    raise FetchError("expected JSON or XML but the response starts with: %r" % text[:60], kind="parse")


def fetch_csv(http, src, limit):
    url = src["url"]
    resp = http.get(url, accept="text/csv, text/plain, application/zip, application/gzip, application/octet-stream;q=0.5, */*;q=0.1", max_bytes=12_000_000)
    packed = resp.body[:4] in (b"PK\x03\x04",) or resp.body[:2] == b"\x1f\x8b"
    if resp.truncated and packed:
        raise FetchError("compressed file larger than the 12 MB cap, not read: " + url, kind="parse")
    text, note = unpack_text(resp.body, url)
    if text.lstrip()[:1] == "<":
        raise FetchError("expected a data table but received markup: %r" % text.lstrip()[:60], kind="parse")
    try:
        headers, rows, preamble = read_table(text, COLUMN_HINTS.get(src["id"]))
        parts, newest = table_text(src, headers, rows, preamble, limit, resp.truncated)
        shape = "Table (%s%s)." % (resp.headers.get("Content-Type", "text"), "; " + note if note else "")
    except FetchError:
        raise
    except Exception as exc:                                   # odd layout: keep a raw head and tail instead of failing
        lines = text.splitlines()
        parts = [clip("\n".join(lines[:30]), 3000), "...", clip("\n".join(lines[-15:]), 1500)]
        newest = None
        shape = "Text file (%d lines); table layout not recognised (%s), raw head and tail shown." % (len(lines), type(exc).__name__)
    return {"items": [snapshot_item(src, url, resp, parts, newest, shape)], "status": resp.status, "bytes": len(resp.body), "note": "table snapshot"}


HANDLERS = {"rss": fetch_feed, "github_releases": fetch_github, "arxiv": fetch_feed, "api": fetch_api, "csv": fetch_csv}


# ----------------------------------------------------------------------------------------------- notebook output
def notebook_for(area):
    return "sources-" + area


def render(src, notebook, item, harvested):
    private = not redistributable(src["redistribution"])
    when = item.get("dt")
    body, hits = redact(paragraphs(item["text"]))
    tail = [t for t in item.get("tail", []) if t]
    notes = re.sub(r"\s*Feed URL from the Apple Podcasts directory \([^)]*\)\.?", "", src.get("notes", "")).strip()
    foot = ["Source: %s (%s) | Area: %s | Published: %s | Link: %s" % (src["name"], src["id"], src["area"], iso_day(when) or "unknown", item["url"])]
    if notes:
        foot.append("About the source: " + notes)
    foot.append("Licence: %s | Redistribution: %s%s" % (src["license_or_terms"], src["redistribution"], " | PRIVATE: do not redistribute" if private else ""))
    last = "\n".join(tail + foot)
    paras = body.split("\n\n") if body else []
    if paras and len(paras[-1]) + len(last) < 1800:          # one search passage for short items, so a hit is not split from its source line
        paras[-1] = paras[-1] + "\n" + last
    else:
        paras.append(last)
    text = "\n\n".join(paras) + "\n"
    pairs = [("channel", notebook), ("notebook", notebook), ("source_id", src["id"]), ("source", src["name"]), ("area", src["area"]),
             ("source_kind", src["kind"]), ("kind", item.get("label", "item")), ("title", "%s / %s" % (src["name"], item["title"])),
             ("url", item["url"]), ("upload_date", compact_day(when)), ("published", iso_day(when)),
             ("licence", src["license_or_terms"]), ("redistribution", src["redistribution"]), ("private", private)]
    pairs += list(item.get("front", {}).items())
    pairs += [("harvested_at", harvested), ("sha256", hashlib.sha256(text.encode("utf-8")).hexdigest())]
    if hits:
        pairs.append(("redacted_secrets", hits))
    return front_matter(pairs) + text


def item_file(root, notebook, source_id, item):
    day = compact_day(item.get("dt")) or "nodate"
    folder = root / notebook if source_id == notebook else root / notebook / source_id
    return folder / ("%s-%s-%s.md" % (day, slug(item["title"], 50), sha8(item["key"])))


class Ledger:
    def __init__(self, path, registry_sha, dry):
        self.path, self.dry = path, dry
        self.data = {"schema": LEDGER_SCHEMA, "registry_sha256": registry_sha, "approval": APPROVAL.name, "run": {}, "sources": {}}
        if path.exists():
            try:
                old = json.loads(path.read_text(encoding="utf-8"))
            except ValueError as exc:
                raise SystemExit("STOP: ledger %s is not valid JSON (%s). Not overwriting it." % (path, exc))
            self.data.update({k: v for k, v in old.items() if k in ("run", "sources")})
        self.data["registry_sha256"] = registry_sha

    def entry(self, sid):
        return self.data["sources"].setdefault(sid, {"seen": {}})

    def save(self):
        if not self.dry:
            self.data["updated"] = stamp()
            write_atomic(self.path, json.dumps(self.data, indent=1, ensure_ascii=False))

    def begin(self, force):
        run = self.data.get("run") or {}
        if run.get("state") == "running" and not force:
            beat = parse_when(run.get("heartbeat"))
            if beat and (utcnow() - beat).total_seconds() < 1200:
                raise SystemExit("STOP: another harvest started %s is still running (ledger heartbeat %s). Wait for it, or use --force." % (run.get("started"), run.get("heartbeat")))
        if not self.dry:
            self.data["run"] = {"state": "running", "pid": os.getpid(), "started": stamp(), "heartbeat": stamp()}
            self.save()

    def beat(self):
        if not self.dry and self.data.get("run"):
            self.data["run"]["heartbeat"] = stamp()

    def finish(self, state, summary=None):
        if not self.dry:
            self.data["run"] = dict(self.data.get("run") or {}, state=state, finished=stamp(), heartbeat=stamp())
            if summary:
                self.data["run"]["summary"] = summary
            self.save()


def store_items(out_root, src, items, led, harvested, dry, refresh=False):
    """Write items that are not in the ledger yet (or whose file is gone). Returns (new, skipped)."""
    notebook = notebook_for(src["area"]) if src["area"] != GEV_ID else GEV_ID
    entry = led.entry(src["id"])
    seen = entry.setdefault("seen", {})
    new = 0
    for item in items:
        keyhash = sha16(item["key"])
        path = item_file(out_root, notebook, src["id"], item)
        rel = path.relative_to(out_root).as_posix()
        text = render(src, notebook, item, harvested)
        digest = sha8(text.split("\n---\n", 1)[-1])
        known = seen.get(keyhash)
        if known and (out_root / known.split("|")[0]).exists() and not (refresh and known.split("|")[1] != digest):
            continue
        if not dry:
            write_atomic(path, text)
        seen[keyhash] = "%s|%s" % (rel, digest)
        new += 1
    return new


def write_index(out_root, notebook, led, sources_in_nb, title_note):
    lines = ["# " + notebook, "", "Generated by Prophet source_harvest.py on %s. %s" % (stamp(), title_note),
             "Notes for private research. Items marked private: true must not be redistributed.", "",
             "| source | kind | status | items | last run | note |", "|---|---|---|---|---|---|"]
    for src in sources_in_nb:
        e = led.data["sources"].get(src["id"], {})
        lines.append("| %s (%s) | %s | %s | %d | %s | %s |" % (src["name"], src["id"], src["kind"], e.get("status", "-"),
                                                          len(e.get("seen", {})), e.get("last_run", "-"), (e.get("error") or e.get("note") or "").replace("|", "/")[:120]))
    write_atomic(out_root / notebook / "INDEX.md", "\n".join(lines) + "\n")


# ----------------------------------------------------------------------------------------------- God's Eye View (read only)
GEV_SKIP_DIRS = {"node_modules", ".git", "dist", "output", "build_cache", "__pycache__", "coverage"}
GEV_CODE_EXT = {".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".py", ".css", ".html", ".glsl", ".wgsl", ".vert", ".frag", ".sh", ".ps1", ".bat", ".cmd", ".yml", ".yaml"}
GEV_PRIVATE_FILES = set()      # third-party files inside the clone that must not be redistributed (none by default)


def gev_walk(root):
    """(relative path, absolute path) for repo files; hidden folders, node_modules, build output and .env* are never entered."""
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in GEV_SKIP_DIRS and not d.startswith("."))
        for name in sorted(files):
            if name.startswith(".env") and name != ".env.example":
                continue
            if name == ".env.example":
                continue
            full = Path(current) / name
            yield full.relative_to(root).as_posix(), full


def gev_read(path, cap=400_000):
    with open(path, "rb") as handle:
        return handle.read(cap).decode("utf-8", "replace")


def header_comment(path):
    try:
        head = gev_read(path, 3000)
    except OSError:
        return ""
    head = re.sub(r"^#!.*\n", "", head.lstrip("\ufeff"))
    match = re.match(r"\s*/\*+(.*?)\*/", head, re.S) or re.match(r"\s*((?://[^\n]*\n\s*){1,6})", head) or re.match(r"\s*(?:\"\"\"|''')(.*?)(?:\"\"\"|''')", head, re.S) or re.match(r"\s*((?:#[^\n]*\n\s*){1,6})", head)
    if not match:
        return ""
    text = re.sub(r"(?m)^\s*(?:/\*+|\*+/?|//+|#+)\s?", "", match.group(1))
    text = re.sub(r"(?i)(copyright|license|licensed|spdx|eslint|@ts-|prettier).*", "", text)
    return short(text, 140)


def md_sections(text, rel):
    """Split a markdown file into one part per level-2 heading so search hits are small."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"<(?!/?(?:code|pre)\b)[^>]+>", "", text)
    parts, current, heading = [], [], rel
    first_title = next((m.group(1).strip() for m in re.finditer(r"(?m)^#\s+(.+)$", text)), rel)
    for line in text.split("\n"):
        if line.startswith("## "):
            if "".join(current).strip():
                parts.append((heading, "\n".join(current)))
            heading, current = line[3:].strip(), []
        else:
            current.append(line)
    if "".join(current).strip():
        parts.append((heading, "\n".join(current)))
    merged = []
    for heading, body in parts:
        if merged and len(body.strip()) < 300:
            merged[-1] = (merged[-1][0], merged[-1][1] + "\n\n## " + heading + "\n" + body)
        else:
            merged.append((heading, body))
    out = []
    for heading, body in merged:
        while len(body) > 9000:
            cut = body.rfind("\n\n", 0, 7000)
            cut = cut if cut > 2000 else 7000
            out.append((heading, body[:cut]))
            body, heading = body[cut:], heading + " (cont.)"
        out.append((heading, body))
    return first_title, out


def harvest_gev(out_root, led, harvested, dry, log):
    """Index the God's Eye View repo into notebooks/gods-eye-view/. Read-only on the repo."""
    root = GEV_ROOT
    if out_root.resolve() == root.resolve() or root.resolve() in out_root.resolve().parents:
        raise SystemExit("STOP: output folder is inside the God's Eye View repo; refusing to write there.")
    if not root.is_dir():
        raise FetchError("God's Eye View repo not found at %s" % root, kind="local")
    license_line = ""
    try:
        lic = gev_read(root / "LICENSE", 600).splitlines()
        license_line = " ".join(l.strip() for l in lic[:3] if l.strip())
    except OSError:
        pass
    remote = ""
    try:
        cfg = gev_read(root / ".git" / "config", 20000)
        m = re.search(r'\[remote "origin"\][^\[]*?url\s*=\s*(\S+)', cfg)
        remote = re.sub(r"://[^/@\s]+@", "://", m.group(1)) if m else ""
    except OSError:
        pass
    head = ""
    try:
        ref = gev_read(root / ".git" / "HEAD", 200).strip()
        if ref.startswith("ref:"):
            refpath = root / ".git" / ref.split(":", 1)[1].strip()
            head = gev_read(refpath, 100).strip() if refpath.exists() else ""
        else:
            head = ref
    except OSError:
        pass
    src = {"id": GEV_ID, "name": "God's Eye View repo", "area": GEV_ID, "kind": "local", "url": root.as_posix(),
           "homepage": remote or root.as_posix(), "license_or_terms": (re.sub(r"\s+Copyright", ", Copyright", license_line or "MIT License") + ". Local read-only clone; repo files keep their own licence."),
           "redistribution": "MIT licence: may redistribute with the copyright notice", "notes": "A local clone of the open-source God's Eye View project."}
    private_src = dict(src, redistribution="private third-party transcript: do not redistribute")
    items_public, items_private = [], []
    redactions = 0
    docs = code_files = 0
    groups = {}
    for rel, full in gev_walk(root):
        ext = os.path.splitext(rel)[1].lower()
        if ext in (".md", ".markdown"):
            try:
                raw = gev_read(full)
            except OSError:
                continue
            text, hits = redact(raw)
            redactions += hits
            title, sections = md_sections(text, rel)
            target = items_private if os.path.basename(rel) in GEV_PRIVATE_FILES else items_public
            when = datetime.fromtimestamp(full.stat().st_mtime, timezone.utc)
            for n, (heading, body) in enumerate(sections):
                body = tidy(body)
                if len(body) < 40:
                    continue
                target.append({"key": "%s#%d" % (rel, n), "title": "%s: %s" % (rel, heading) if heading != rel else rel,
                               "url": "file:///" + (root / rel).as_posix() + ("#" + slug(heading, 30) + "-" + sha8("%s#%d" % (rel, n)) if n else ""), "dt": when,
                               "text": body, "tail": [], "label": "repo document", "front": {"repo_path": rel}})
            docs += 1
        elif ext in GEV_CODE_EXT or (ext == ".json" and rel.count("/") <= 1 and not rel.startswith(("src/", "public/", "dist/"))):
            parts = rel.split("/")
            group = "/".join(parts[:2]) if len(parts) > 2 else (parts[0] if len(parts) == 2 else "(root)")
            groups.setdefault(group, []).append((rel, full))
            code_files += 1
    when_now = utcnow()
    for group, files in sorted(groups.items()):
        lines = []
        for rel, full in files:
            name = os.path.basename(rel)
            stem = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", os.path.splitext(name)[0])
            words = re.sub(r"[-_.]+", " ", stem).lower()
            purpose = header_comment(full) if os.path.splitext(name)[1].lower() in GEV_CODE_EXT else ""
            try:
                size = full.stat().st_size
            except OSError:
                size = 0
            lines.append("%s (%s, %d bytes)%s%s" % (rel, words, size, ": " if purpose else "", purpose))
        chunk = 12
        body = "\n\n".join("\n".join(lines[i:i + chunk]) for i in range(0, len(lines), chunk))
        body, hits = redact(body)
        redactions += hits
        items_public.append({"key": "codefiles:" + group, "title": "code files in %s (%d files)" % (group, len(files)),
                             "url": "file:///" + (root / group).as_posix() + "/" if group != "(root)" else "file:///" + root.as_posix() + "/#code-files-root",
                             "dt": when_now, "text": body, "tail": [], "label": "code file list (names, sizes, first comment)", "front": {"repo_path": group}})
    pkg = root / "package.json"
    if pkg.exists():
        try:
            data = json.loads(gev_read(pkg))
            scripts = "\n".join("%s: %s" % (k, short(v, 120)) for k, v in (data.get("scripts") or {}).items())
            deps = ", ".join("%s %s" % (k, v) for k, v in sorted((data.get("dependencies") or {}).items()))
            dev = ", ".join("%s %s" % (k, v) for k, v in sorted((data.get("devDependencies") or {}).items()))
            text = "package.json: %s %s. %s\n\nScripts:\n%s\n\nDependencies: %s\n\nDev dependencies: %s" % (data.get("name", ""), data.get("version", ""), data.get("description", ""), scripts, deps, dev)
            text, hits = redact(text)
            redactions += hits
            items_public.append({"key": "package.json", "title": "package.json: scripts and dependencies", "url": "file:///" + (root / "package.json").as_posix(),
                                 "dt": datetime.fromtimestamp(pkg.stat().st_mtime, timezone.utc), "text": text, "tail": [], "label": "package manifest", "front": {"repo_path": "package.json"}})
        except ValueError:
            pass
    items_public.append({"key": "repo-facts", "title": "repo facts", "url": "file:///" + root.as_posix() + "/#repo-facts", "dt": when_now,
                         "text": "God's Eye View is a browser spy-satellite style 3D globe of public data (aircraft, ships, satellites, quakes, cameras) with voice control. "
                                 "Local clone at %s of %s at commit %s. Licence: %s. Indexed read-only: markdown docs, a list of code files with first comments, and package.json. "
                                 "Not indexed: .env files, hidden tool folders, node_modules, dist, output, logs, caches, images, PDFs." % (root.as_posix(), remote or "unknown remote", head or "unknown", license_line or "see LICENSE"),
                         "tail": [], "label": "repo facts", "front": {"repo_path": ""}})
    new = store_items(out_root, src, items_public, led, harvested, dry, refresh=True)
    new += store_items(out_root, dict(private_src, id=GEV_ID), items_private, led, harvested, dry, refresh=True) if items_private else 0
    entry = led.entry(GEV_ID)
    entry.update({"area": GEV_ID, "kind": "local", "name": src["name"], "url": root.as_posix(), "status": "ok", "error": None,
                  "last_run": harvested, "last_ok": harvested, "new_last_run": new,
                  "note": "read-only: %d markdown files, %d code/config files listed, %d secret-like strings redacted, head %s" % (docs, code_files, redactions, head[:10])})
    log("[ok  ] %-34s local  %-9s new=%-4d total=%-4d %s" % (GEV_ID, "gods-eye", new, len(entry["seen"]), entry["note"]))
    return new


# ----------------------------------------------------------------------------------------------- run
def skip_reason(src):
    if src["kind"] == "youtube":
        return "youtube: not harvested by this tool"
    if src.get("disabled"):                                   # the row stays in the registry; it is only never fetched
        return "disabled: " + (src.get("disabled_reason") or "no reason recorded")
    terms = (src.get("license_or_terms", "") + " " + src.get("notes", "")).lower()
    if "contact email" in terms:                              # safety net: the SEC rows are also marked disabled in the registry
        return "skipped: SEC fair access wants a contact e-mail in the User-Agent; none is configured"
    return None


def load_registry():
    raw = REGISTRY.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if not APPROVAL.exists():
        raise SystemExit("STOP: the source list has not been approved yet. Read sources/sources.json, then run: python approve_sources.py")
    approval = json.loads(APPROVAL.read_text(encoding="utf-8"))
    if digest != approval.get("sourcesSha256"):
        raise SystemExit("STOP: sources.json sha256 %s does not equal the approval's %s. The list changed since you approved it. Nothing harvested." % (digest, approval.get("sourcesSha256")))
    status = approval.get("status", "approved" if APPROVAL.name in LEGACY_APPROVALS else None)     # a record without a status is not an approval
    if status != "approved":
        raise SystemExit("STOP: the approval record %s %s, not 'approved': this exact list is not approved yet. Nothing harvested." % (
            APPROVAL.name, "has no status" if status is None else "has status %r" % status))
    registry = json.loads(raw.decode("utf-8"))
    approved = [s for s in registry["sources"] if s.get("verified") is True]
    return approved, digest, approval


def interleave(sources):
    """Order sources so neighbours are on different sites (parallel workers rarely wait on one site)."""
    groups = {}
    for s in sources:
        groups.setdefault(site_key(urllib.parse.urlsplit(s["url"]).hostname or s["url"]), []).append(s)
    out = []
    while any(groups.values()):
        for key in list(groups):
            if groups[key]:
                out.append(groups[key].pop(0))
    return out


def work(http, src, limit):
    try:
        return {"ok": True, **HANDLERS[src["kind"]](http, src, limit)}
    except FetchError as exc:
        return {"ok": False, "error": str(exc), "transient": exc.transient, "error_kind": exc.kind, "http_status": exc.status}
    except Exception as exc:                                    # a bug or odd data in one source must not stop the run
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc), "transient": False, "error_kind": "bug", "http_status": None}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="source_harvest", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", help="area id, comma list, 'all', or gods-eye-view")
    ap.add_argument("--source", action="append", default=[], help="source id (repeat or comma list)")
    ap.add_argument("--limit", type=int, default=MAX_ITEMS, help="max items per source per run (hard cap 50)")
    ap.add_argument("--dry-run", action="store_true", help="fetch and parse, write nothing (no notebook files, no ledger)")
    ap.add_argument("--list", action="store_true", help="show approved and skipped sources, fetch nothing")
    ap.add_argument("--workers", type=int, default=4, help="parallel sites (default 4); one site is never hit in parallel")
    ap.add_argument("--no-retry", action="store_true", help="do not re-run transient failures once at the end")
    ap.add_argument("--strict", action="store_true", help="exit 1 when any source failed")
    ap.add_argument("--force", action="store_true", help="ignore a recent 'running' mark in the ledger")
    ap.add_argument("--out", default=str(KNOWLEDGE), help="notebooks folder to write (testing only)")
    args = ap.parse_args(argv)
    if not args.area and not args.source:
        ap.print_help()
        return 2
    limit = max(1, min(args.limit, MAX_ITEMS))
    out_root = Path(args.out)

    approved, reg_sha, approval = load_registry()
    by_id = {s["id"]: s for s in approved}
    areas = sorted({s["area"] for s in approved})
    wanted_ids = {x.strip() for chunk in args.source for x in chunk.split(",") if x.strip()}
    want_gev = False
    if args.area and args.area.lower() == "all":
        chosen = list(approved)
        want_gev = GEV_ROOT.is_dir()
    else:
        chosen = []
        for a in [x.strip() for x in (args.area or "").split(",") if x.strip()]:
            if a == GEV_ID:
                want_gev = True
            elif a not in areas:
                raise SystemExit("Unknown area %r. Areas: %s, %s, all" % (a, ", ".join(areas), GEV_ID))
            else:
                chosen += [s for s in approved if s["area"] == a]
    for sid in wanted_ids:
        if sid == GEV_ID:
            want_gev = True
        elif sid not in by_id:
            raise SystemExit("Source %r is not in the approved set (verified=true in the approved sources.json)." % sid)
        elif by_id[sid] not in chosen:
            chosen.append(by_id[sid])
    say("Prophet source harvest | registry sha256 %s... matches approval %s (%s)" % (reg_sha[:12], approval["at"], approval.get("confirmedBy") or approval.get("by", "unknown")))

    skipped = {s["id"]: skip_reason(s) for s in chosen if skip_reason(s)}
    todo = [s for s in chosen if s["id"] not in skipped]
    if args.list:
        for s in chosen:
            how = "will harvest" + (" (official API on the robots allowlist)" if api_entry(urllib.parse.urlsplit(s["url"])) else "")
            say("%-34s %-26s %-16s %s" % (s["id"], s["area"], s["kind"], skipped.get(s["id"], how)))
        say("approved selected=%d, skipped=%d, to fetch=%d%s" % (len(chosen), len(skipped), len(todo), ", plus gods-eye-view" if want_gev else ""))
        return 0

    led = Ledger(out_root / "sources" / "harvest-ledger.json", reg_sha, args.dry_run)
    led.begin(args.force)
    harvested = stamp()
    http = Http()
    results, started = {}, time.monotonic()
    state = "finished"
    try:
        for s in chosen:
            if s["id"] in skipped:
                e = led.entry(s["id"])
                e.update({"area": s["area"], "kind": s["kind"], "name": s["name"], "url": s["url"], "status": "skipped", "error": None,
                          "note": skipped[s["id"]], "last_run": harvested})
        yt = [s for s in chosen if s["kind"] == "youtube"]
        if yt:
            say("youtube: not harvested by this tool (%d sources)" % len(yt))
        for s in chosen:
            if s["id"] in skipped and s["kind"] != "youtube":
                say("[skip] %-34s %s" % (s["id"], skipped[s["id"]]))
        if want_gev:
            try:
                harvest_gev(out_root, led, harvested, args.dry_run, say)
            except FetchError as exc:
                e = led.entry(GEV_ID)
                e.update({"area": GEV_ID, "kind": "local", "status": "failed", "error": str(exc), "last_run": harvested})
                say("[FAIL] %-34s %s" % (GEV_ID, exc))
            led.save()

        def run_pass(batch, label):
            if not batch:
                return
            say("-- %s: %d sources, %d workers" % (label, len(batch), args.workers))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                futures = {pool.submit(work, http, s, limit): s for s in interleave(batch)}
                for fut in concurrent.futures.as_completed(futures):
                    s = futures[fut]
                    res = fut.result()
                    e = led.entry(s["id"])
                    e.update({"area": s["area"], "kind": s["kind"], "name": s["name"], "url": s["url"], "last_run": harvested,
                              "attempts": e.get("attempts", 0) + 1})
                    if res["ok"]:
                        try:
                            new = store_items(out_root, s, res["items"], led, harvested, args.dry_run)
                        except OSError as exc:
                            res = {"ok": False, "error": "could not write notebook files: %s" % exc, "transient": False, "error_kind": "write", "http_status": None}
                    if res["ok"]:
                        e.update({"status": "ok", "error": None, "error_kind": None, "last_ok": harvested, "new_last_run": new,
                                  "http_status": res["status"], "note": res["note"], "transient": False})
                        results[s["id"]] = ("ok", new)
                        say("[ok  ] %-34s %-15s %-24s new=%-3d total=%-4d %s" % (s["id"], s["kind"], s["area"], new, len(e["seen"]), res["note"]))
                    elif res["error_kind"] == "ratelimit":          # the site's own request budget is spent: not a failure of the source
                        e.update({"status": "deferred", "error": res["error"], "error_kind": "ratelimit", "http_status": res["http_status"],
                                  "new_last_run": 0, "transient": False})
                        results[s["id"]] = ("deferred", 0)
                        say("[WAIT] %-34s %-15s %-24s %s" % (s["id"], s["kind"], s["area"], res["error"]))
                    else:
                        e.update({"status": "failed", "error": res["error"], "error_kind": res["error_kind"], "http_status": res["http_status"],
                                  "new_last_run": 0, "transient": res["transient"]})
                        results[s["id"]] = ("failed", 0)
                        say("[FAIL] %-34s %-15s %-24s %s%s" % (s["id"], s["kind"], s["area"], res["error"], "  (transient)" if res["transient"] else ""))
                    led.beat()
                    led.save()

        run_pass(todo, "pass 1")
        if not args.no_retry:
            again = [s for s in todo if results.get(s["id"], ("", 0))[0] == "failed" and led.entry(s["id"]).get("transient")]
            if again:
                say("-- retrying %d transient failures once after 25 s: %s" % (len(again), ", ".join(s["id"] for s in again)))
                time.sleep(25)
                run_pass(again, "retry")
    except KeyboardInterrupt:
        state = "interrupted"
        say("Interrupted: ledger saved with what finished.")
    except BaseException:
        state = "crashed"
        led.finish(state)
        raise

    # per-notebook index pages and the summary table
    if not args.dry_run:
        for area in sorted({s["area"] for s in chosen}):
            write_index(out_root, notebook_for(area), led, [s for s in approved if s["area"] == area], "Harvested from the approved Prophet source registry.")
        if want_gev:
            write_index(out_root, GEV_ID, led, [{"id": GEV_ID, "name": "God's Eye View repo", "kind": "local"}], "Read-only index of the local God's Eye View repo.")
    rows = []
    for area in sorted({s["area"] for s in chosen}):
        sel = [s for s in chosen if s["area"] == area]
        att = [s for s in sel if s["id"] in results and results[s["id"]][0] != "deferred"]       # deferred sources made no request
        ok = sum(1 for s in att if results[s["id"]][0] == "ok")
        new = sum(results[s["id"]][1] for s in att)
        total = sum(len(led.data["sources"].get(s["id"], {}).get("seen", {})) for s in approved if s["area"] == area)
        rows.append((area, len(sel), sum(1 for s in sel if s["id"] in skipped), len(att), ok, len(att) - ok, new, total))
    if want_gev:
        e = led.data["sources"].get(GEV_ID, {})
        rows.append((GEV_ID + " (local)", 1, 0, 1, 1 if e.get("status") == "ok" else 0, 0 if e.get("status") == "ok" else 1, e.get("new_last_run", 0), len(e.get("seen", {}))))
    say("")
    say("%-28s %8s %8s %9s %4s %7s %10s %12s" % ("area", "approved", "skipped", "attempted", "ok", "failed", "new items", "items total"))
    for r in rows:
        say("%-28s %8d %8d %9d %4d %7d %10d %12d" % r)
    say("%-28s %8d %8d %9d %4d %7d %10d %12d" % (("TOTAL",) + tuple(sum(r[i] for r in rows) for i in range(1, 8))))
    waiting = [s for s in todo if results.get(s["id"], ("", 0))[0] == "deferred"]
    if waiting:
        say("")
        say("RATE LIMITED, resume later (%d sources, not fetched): %s" % (len(waiting), ", ".join(s["id"] for s in waiting)))
        say("  " + str(led.entry(waiting[0]["id"]).get("error")))
        say("  Run the same command again after that time; sources already stored add only new items.")
    failed = [s for s in todo if results.get(s["id"], ("", 0))[0] == "failed"]
    if failed:
        say("")
        say("FAILED sources (%d):" % len(failed))
        for s in failed:
            e = led.entry(s["id"])
            say("  %s [%s, %s, %s, site %s]: %s" % (s["id"], s["area"], s["kind"], e.get("error_kind"),
                                                    site_key(urllib.parse.urlsplit(s["url"]).hostname or ""), e.get("error")))
        kinds = {}
        for s in failed:
            kinds[led.entry(s["id"]).get("error_kind")] = kinds.get(led.entry(s["id"]).get("error_kind"), 0) + 1
        say("  by cause: " + ", ".join("%s=%d" % kv for kv in sorted(kinds.items(), key=lambda kv: -kv[1])))
    gone = [s for s in todo if s["id"] not in results]
    if gone:
        say("NOT RUN (interrupted): " + ", ".join(s["id"] for s in gone))
    if skipped:
        say("")
        say("SKIPPED sources (%d): %s" % (len(skipped), "; ".join("%s (%s)" % (k, v) for k, v in skipped.items() if not v.startswith("youtube"))
                                          + ("; youtube: not harvested by this tool x%d" % sum(1 for v in skipped.values() if v.startswith("youtube")))))
    say("")
    say("%s in %.0f s, %d HTTP requests%s." % (state, time.monotonic() - started, http.requests, " (dry run: nothing written)" if args.dry_run else ""))
    if http.api_hits:
        say("API allowlist (project decision): robots.txt not consulted for %d request(s) to official API endpoints: %s" % (
            sum(http.api_hits.values()), ", ".join("%s x%d" % kv for kv in sorted(http.api_hits.items()))))
    summary = {"attempted": sum(1 for v in results.values() if v[0] != "deferred"), "ok": sum(1 for v in results.values() if v[0] == "ok"), "failed": len(failed)}
    if http.api_hits:
        summary["api_allowlist"] = dict(sorted(http.api_hits.items()))
    if waiting:
        summary["deferred"] = len(waiting)
    if http.github.limit is not None:
        summary["github_api"] = {"limit": http.github.limit, "remaining": http.github.remaining,
                                 "resets": stamp(datetime.fromtimestamp(http.github.reset, timezone.utc)) if http.github.reset else None}
    led.finish(state, summary)
    return 1 if (args.strict and (failed or waiting)) else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
