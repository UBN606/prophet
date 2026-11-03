"""Prophet domain: geopolitics (GDELT, free). A drop-in file for prophet_lab.py.

What it calls: whether one kind of event will make up a higher share of the world events GDELT codes on a coming day than it did on
the last day with data. The kinds are protests (CAMEO root code 14), violence (18 assault, 19 fight, 20 mass violence) and reduced
relations such as sanctions and boycotts (16). That is a count of reported events by type, not a forecast of the events themselves.
For questions about the events, use hand-made calls:
  prophet_lab.py call --domain geopolitics --question "..." --resolves YYYY-MM-DD --prob 0.35

Data, the default: the GDELT 1.0 daily event export, one zip per day at https://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip
(about 7 MB; history back to 2013-04-01; free, no key; credit the GDELT Project). That host is not on the official-API allowlist, so the
lab reads its robots.txt first (a plain 404 means no restrictions) and waits at least 2 seconds between requests. Each file is
unzipped as a stream and its events are counted by EventRootCode (column 28 of the 58 tab-separated columns). One row per day, with
the zip's URL and sha256, goes to gdelt_daily_counts.csv in the lab folder, and the zip is deleted. A day that has a row is never
downloaded again. The newest day or two are published the morning after and may answer 404: that is reported, not an error. The
window is the last 120 days ending yesterday; `--days N` widens or narrows it.

Fallback, off unless `--gdelt-doc-api` is given: GDELT's DOC 2.0 API (news coverage volume of a theme). It answers HTTP 429 to this
lab and keeps only about 90 days of daily history; it allows one request per 5 seconds, so the lab waits at least 6.5 seconds
between requests to it and, when it answers 429, waits 10 seconds, asks once more and then stops with a clear error. Every download is
cached, and older cached copies are merged in, so its history grows as the lab runs. A call remembers which source made it
(meta.source: "gdelt_export_v1" or "gdelt_doc_api") and `score` reads its outcome from that same source. Without the flag the DOC API
is never asked, scoring included: a call the DOC API made stays unscored until a run is given `--gdelt-doc-api`.

Sources tried: Metaculus's public API answers HTTP 403 ("only available to authenticated users"), so it is not used.
"""
import csv
import datetime as dt
import io
import json
import sys
import urllib.parse
import zipfile
import zlib
from types import SimpleNamespace

import numpy as np

import prophet_lab as lab

# ---- the old route: GDELT DOC API, off by default ---------------------------------------
GDELT = "https://api.gdeltproject.org/api/v2/doc/doc?query={q}&mode=timelinevol&format=json&timespan=3m"
GDELT_MIN_GAP = 6.5      # seconds between two requests to GDELT, robots.txt reads included (it asks for one every 5 seconds)
GDELT_RETRY_429 = 1      # on HTTP 429 wait at least 10 s and ask once more; if it still says 429 the domain stops cleanly
SOURCE_DOC = "gdelt_doc_api"

# ---- the default route: GDELT 1.0 daily event export ------------------------------------
SOURCE_EXPORT = "gdelt_export_v1"
EXPORT_URL = "https://data.gdeltproject.org/events/{day}.export.CSV.zip"      # day as YYYYMMDD
EXPORT_MIN_GAP = 2.0         # seconds between two requests to the site, the robots.txt read included
EXPORT_RETRY_429 = 1         # on HTTP 429 wait at least 10 s and ask once more, then stop cleanly
EXPORT_MAX_ZIP = 60_000_000  # a day is about 7 MB zipped; anything over this is not downloaded
EXPORT_TIMEOUT = 120.0
KEEP_SECONDS = 30 * 86400    # a zip left behind by a run that stopped before counting it is reused for this long
DEFAULT_DAYS = 120           # backtest window: this many days ending yesterday (`--days N` changes it)
MIN_DAYS = 40                # fewer counted days than this is not a history
FIRST_DAY = dt.date(2013, 4, 1)      # the first day of the GDELT 1.0 daily export
PENDING_DAYS = 2             # the newest files appear the morning after: a 404 on one of the last two days is "not published yet"
MIN_EVENTS = 5000            # a day with fewer coded events than this (a broken or partial file) is counted but left out of the series
ROOT_COLUMN = 28             # 0-based column of EventRootCode in the 58 tab-separated columns of a GDELT 1.0 event row ("01" to "20")
CHUNK = 1 << 20              # the unzipped member is read this many bytes at a time, never whole
MEMBER_LIMIT = 1_000_000_000     # refuse a zip that unpacks to more than this (a real day is about 50 MB)
PROGRESS_EVERY = 10          # days between progress lines on stderr
COUNTS_FILE = "gdelt_daily_counts.csv"
COUNT_COLUMNS = (["day", "events", "lines", "bad_lines", "zip_bytes", "member_bytes", "sha256", "url", "fetched_at"]
                 + ["root%02d" % n for n in range(1, 21)])
EXPORT_TOPICS = [
    {"id": "protest_events", "name": "protests", "roots": [14]},
    {"id": "violence_events", "name": "violence (assault, fighting, mass violence)", "roots": [18, 19, 20]},
    {"id": "sanctions_events", "name": "reduced relations (sanctions, boycotts, expulsions)", "roots": [16]},
]
DEFAULT = {"topics": [
    {"id": "protest", "name": "protests", "query": "theme:PROTEST"},
    {"id": "armed_conflict", "name": "armed conflict", "query": "theme:ARMEDCONFLICT"},
    {"id": "terror", "name": "terrorism", "query": "theme:TERROR"},
    {"id": "election", "name": "elections", "query": "theme:ELECTION"},
    {"id": "sanctions", "name": "sanctions", "query": "sanctions"},
    {"id": "ceasefire", "name": "ceasefire", "query": "ceasefire"},
    {"id": "coup", "name": "coups", "query": "coup"},
    {"id": "tariffs", "name": "tariffs", "query": "tariffs"},
], "event_topics": EXPORT_TOPICS}
H = 2                 # calls look two days past the last complete day: day t against day t+2
WARMUP = 21           # days of history before the first scored call of a theme
MIN_TRAIN = 60


def _say(message):
    """Progress of the long downloads goes to stderr, so the command's own output stays clean."""
    print(message, file=sys.stderr, flush=True)


def doc_api_on(lab_):
    """True only for a run given --gdelt-doc-api (the flag reaches the lab as lab_.opts["gdelt_doc_api"]). The DOC API is never asked otherwise."""
    return bool((getattr(lab_, "opts", None) or {}).get("gdelt_doc_api"))


# ---- DOC API route: parsing and cached copies --------------------------------------------
def _validate(data):
    j = json.loads(data.decode("utf-8"))
    if "timeline" not in j:
        raise ValueError("not a GDELT timeline")


def _parse(data):
    j = json.loads(data.decode("utf-8"))
    out = {}
    for pt in (j["timeline"][0]["data"] if j.get("timeline") else []):
        out[dt.datetime.strptime(pt["date"][:8], "%Y%m%d").date()] = float(pt["value"])
    return out


def _snapshots(lab_, url):
    """Every distinct cached copy of this URL, oldest first."""
    p = lab_.raw_dir / "manifest.jsonl"
    seen, out = set(), []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("url") == url and e["sha256"] not in seen and (lab_.root / e["file"]).exists():
                seen.add(e["sha256"])
                out.append(e)
    return out


# ---- the model's inputs, the same for both routes ------------------------------------------
def topic_features(v, t):
    """Features at day index t, from values up to and including t."""
    if t < 8 or v[t] <= 0 or v[t - 1] <= 0 or np.isnan(v[t - 7:t + 1]).any():
        return None
    base = v[t - 7:t].mean()
    return [float(np.log(v[t] / v[t - 1])), float(np.log(v[t] / base))] if base > 0 else None


def weekday_onehot(date):
    """Weekday of the target day, Monday to Saturday (Sunday is the reference)."""
    return [1.0 if date.weekday() == i else 0.0 for i in range(6)]


# ---- export route: count one day's events while unzipping ----------------------------------
def _tally(pieces, roots):
    """Add the EventRootCode of each line in `pieces` (lines as bytes) to `roots`. Returns (lines seen, lines that are not an event row)."""
    seen = bad = 0
    for raw in pieces:
        if not raw or raw == b"\r":
            continue
        seen += 1
        fields = raw.split(b"\t", ROOT_COLUMN + 2)
        code = fields[ROOT_COLUMN] if len(fields) > ROOT_COLUMN + 1 else b""
        if code.isdigit() and len(code) <= 2 and 1 <= int(code) <= 20:
            roots[int(code) - 1] += 1
        else:
            bad += 1
    return seen, bad


def count_export_zip(data):
    """Count the events of one GDELT 1.0 daily export (the zip's bytes) by CAMEO root code 01 to 20.

    The one member is read CHUNK bytes at a time and counted line by line, so a day of about 50 MB of text never sits in memory.
    Returns {"roots": [n01, ..., n20], "events", "lines", "bad", "member_bytes"}; a line whose root code is missing or outside
    01 to 20 is counted in "bad". Raises ValueError for bytes that are not a zip with exactly one readable member, or for a
    member that unpacks to more than MEMBER_LIMIT bytes."""
    roots = [0] * 20
    lines = bad = size = 0
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            members = [i for i in zf.infolist() if not i.is_dir()]
            if len(members) != 1:
                raise ValueError("expected one file inside the zip, found %d" % len(members))
            tail = b""
            with zf.open(members[0]) as member:
                while True:
                    chunk = member.read(CHUNK)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MEMBER_LIMIT:
                        raise ValueError("the unzipped file is larger than %d bytes; refused" % MEMBER_LIMIT)
                    pieces = (tail + chunk).split(b"\n")
                    tail = pieces.pop()
                    seen, off = _tally(pieces, roots)
                    lines += seen
                    bad += off
            seen, off = _tally([tail], roots)
            lines += seen
            bad += off
    except (zipfile.BadZipFile, zlib.error, EOFError, NotImplementedError, RuntimeError) as e:
        raise ValueError("not a readable zip file: %s" % e)
    return {"roots": roots, "events": sum(roots), "lines": lines, "bad": bad, "member_bytes": size}


def _validate_zip(data):
    if data[:4] != b"PK\x03\x04":
        raise ValueError("not a zip file (it starts with %r)" % data[:16])


# ---- export route: the counts file, one row per counted day ----------------------------------
def _csv_line(values):
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow(values)
    return buf.getvalue()


def _ends_with_newline(path):
    with open(path, "rb") as f:
        f.seek(-1, 2)
        return f.read(1) == b"\n"


def read_counts(path):
    """{day: row} from the counts file; a row that does not parse is ignored, and when a day is there twice the first row wins."""
    rows = {}
    if not path.exists():
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            try:
                row = SimpleNamespace(day=dt.date.fromisoformat(rec["day"]), events=int(rec["events"]), lines=int(rec["lines"]),
                                      bad=int(rec["bad_lines"]), zip_bytes=int(rec["zip_bytes"]), member_bytes=int(rec["member_bytes"]),
                                      sha256=rec["sha256"], url=rec["url"], fetched_at=rec["fetched_at"],
                                      roots=[int(rec["root%02d" % n]) for n in range(1, 21)])
            except (KeyError, TypeError, ValueError):
                continue
            rows.setdefault(row.day, row)
    return rows


def append_count(path, row):
    """Add one counted day to the counts file (append only). A file whose last row was cut short gets a line break first."""
    values = [row.day.isoformat(), row.events, row.lines, row.bad, row.zip_bytes, row.member_bytes, row.sha256, row.url,
              row.fetched_at] + list(row.roots)
    with lab.file_lock(path):
        if not path.exists() or path.stat().st_size == 0:
            lab.append_line(path, _csv_line(COUNT_COLUMNS))
        elif not _ends_with_newline(path):
            lab.append_line(path, "")
        lab.append_line(path, _csv_line(values))


class _BadDay(Exception):
    """A day's file arrived but is not a readable GDELT export."""


def count_day(lab_, day):
    """Download one day's export through the lab (robots.txt, polite gap, manifest), count it, record the row, delete the zip."""
    url = EXPORT_URL.format(day=day.strftime("%Y%m%d"))
    got = lab_.fetch(url, ttl=KEEP_SECONDS, validate=_validate_zip, min_gap=EXPORT_MIN_GAP, timeout=EXPORT_TIMEOUT,
                     max_bytes=EXPORT_MAX_ZIP, suffix=".zip", retry_429=EXPORT_RETRY_429)
    try:
        counted = count_export_zip(got.data)
    except ValueError as e:
        raise _BadDay("%s: %s" % (url, e))
    finally:
        try:
            got.path.unlink()           # the zip is not kept: its counts, URL and sha256 are
        except OSError:
            pass
    row = SimpleNamespace(day=day, events=counted["events"], lines=counted["lines"], bad=counted["bad"], zip_bytes=len(got.data),
                          member_bytes=counted["member_bytes"], sha256=got.sha256, url=url, fetched_at=got.fetched_at,
                          roots=counted["roots"])
    append_count(lab_.root / COUNTS_FILE, row)
    return row


def load_days(lab_, days, today, notes):
    """The count rows of `days` as ({day: row}, error). A day that already has a row in the counts file is read from there and never
    downloaded again; the others are downloaded newest first, one at a time.

    A 404 means the file does not exist: for the newest PENDING_DAYS days it is "not published yet", for an older day it is a gap; both
    go into `notes` and the day is left out. A file that is not a readable export is left out too. Any other failure (robots.txt, HTTP
    429 or 5xx, a network error, a page that is not a zip) stops the downloading, since asking again at once would not help: the days
    already counted stay counted, `error` is the FetchError that stopped it, and `notes` says so."""
    path = lab_.root / COUNTS_FILE
    have = read_counts(path)
    todo = sorted((d for d in days if d not in have), reverse=True)
    pending, gaps, unreadable, error, new, size = [], [], [], None, 0, 0
    if todo:
        _say("gdelt export: %d of %d days are not counted yet; fetching them newest first, at least %.0f s apart (about %.0f MB)"
             % (len(todo), len(days), EXPORT_MIN_GAP, 7.5 * len(todo)))
    for day in todo:
        try:
            row = count_day(lab_, day)
        except _BadDay as e:
            unreadable.append((day, str(e)))
            continue
        except lab.FetchError as e:
            if e.code == 404:
                (pending if day >= today - dt.timedelta(days=PENDING_DAYS) else gaps).append(day)
                continue
            error = e
            notes.append(f"history is incomplete: downloading stopped at {day} ({e}); the rest was skipped, run the command again to continue")
            break
        have[day] = row
        new += 1
        size += row.zip_bytes
        if new % PROGRESS_EVERY == 0:
            _say("gdelt export: %d of %d days counted (%s), %.0f MB downloaded and deleted so far" % (new, len(todo), day, size / 1e6))
    if todo:
        _say("gdelt export: counted %d new day(s), %.1f MB downloaded and deleted; per-day counts in %s" % (new, size / 1e6, COUNTS_FILE))
    if pending:
        notes.append("not published yet (HTTP 404), left out: " + _days_text(pending))
    if gaps:
        notes.append("no file for " + _days_text(gaps) + " (HTTP 404): a gap in the series, skipped")
    for day, why in unreadable:
        notes.append(f"{day.isoformat()}: the file is not a readable GDELT export ({why}); skipped")
    return {d: have[d] for d in days if d in have}, error


def _days_text(days):
    """Dates for a note: all of them when there are a few, else the first three and the last two with the count."""
    days = [d.isoformat() for d in sorted(days)]
    return ", ".join(days) if len(days) <= 6 else ", ".join(days[:3]) + ", ..., " + ", ".join(days[-2:]) + f" ({len(days)} days)"


def share(row, roots):
    """Percent of the day's coded events that carry one of these root codes."""
    return 100.0 * sum(row.roots[r - 1] for r in roots) / row.events


def event_topics(cfg):
    """The themes of the export route: config.json "event_topics" (id, name, roots) or the built-in three."""
    out, seen = [], set()
    for tp in cfg.get("event_topics") or EXPORT_TOPICS:
        try:
            tid, roots = str(tp["id"]), sorted({int(r) for r in tp["roots"]})
            name = str(tp.get("name") or tid)
        except (KeyError, TypeError, ValueError, AttributeError):
            raise lab.LabError("geopolitics event_topics in config.json: each theme needs an id and a list of root codes")
        if not roots or roots[0] < 1 or roots[-1] > 20 or tid in seen:
            raise lab.LabError(f"geopolitics event_topics in config.json: theme {tid!r} needs unique id and root codes between 1 and 20")
        seen.add(tid)
        out.append({"id": tid, "name": name, "roots": roots})
    return out


class Adapter(lab.Adapter):
    name = "geopolitics"
    title = "Geopolitics: will the daily share of a world-affairs event type rise?"
    sources = [{"name": "GDELT 1.0 daily event export (events counted by CAMEO root code)", "url": "https://data.gdeltproject.org/events/",
                "note": "free, no key; one zip of about 7 MB per day since 2013, counted then deleted; robots.txt read first; "
                        "at least 2 s between requests; --days N sets the window (default 120); credit the GDELT Project"},
               {"name": "GDELT DOC 2.0 API (timeline of coverage volume), fallback", "url": "https://api.gdeltproject.org/api/v2/doc/doc",
                "note": "OFF unless --gdelt-doc-api: it answers HTTP 429; about 90 days of daily history; one request per 5 seconds"},
               {"name": "Metaculus public API (the suggested community-forecast source)", "url": "https://www.metaculus.com/api/posts/",
                "note": "NOT USED: it answers HTTP 403 'The API is only available to authenticated users'"}]
    status_note = ("Calls are about the daily share of world events GDELT codes as protests, violence or reduced relations (sanctions), "
                   "counted from its daily event files, not forecasts of the events themselves. --days N sets the history window; "
                   "--gdelt-doc-api switches to the old DOC API. Metaculus needs an account token, so its community forecasts are not used. "
                   "For event questions use hand-made calls: call --domain geopolitics ...")
    default_config = DEFAULT
    target_labels = {"up_h2": "share two days past the last full day is higher, yes/no (Brier)"}

    # ---- history ---------------------------------------------------------------------
    def fetch_history(self, lab_, fresh=False):
        if doc_api_on(lab_):
            return self._fetch_doc(lab_, fresh)
        return self._fetch_export(lab_)

    def _fetch_export(self, lab_):
        cfg = self.config(lab_)
        n = int((getattr(lab_, "opts", None) or {}).get("days") or DEFAULT_DAYS)
        if n < MIN_DAYS:
            raise lab.LabError(f"--days {n} is too few: the model needs at least {MIN_DAYS} days of GDELT history")
        topics = event_topics(cfg)
        today = lab.utcnow().date()
        end = today - dt.timedelta(days=1)
        start = max(end - dt.timedelta(days=n - 1), FIRST_DAY)
        window = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
        notes = []
        rows, error = load_days(lab_, window, today, notes)
        usable = {d: r for d, r in rows.items() if r.events >= MIN_EVENTS}
        thin = sorted(set(rows) - set(usable))
        if thin:
            notes.append(f"fewer than {MIN_EVENTS} coded events (a broken or partial file), left out of the series: " + _days_text(thin))
        if len(usable) < MIN_DAYS:
            if error is not None:
                raise error
            raise lab.LabError(f"only {len(usable)} days of GDELT events could be counted; at least {MIN_DAYS} are needed. " + " | ".join(notes))
        days = sorted(usable)
        inputs = [{"url": usable[d].url, "sha256": usable[d].sha256} for d in days]
        hist = SimpleNamespace(mode="export", topics={}, inputs=inputs, notes=notes)
        span = [days[0] + dt.timedelta(days=i) for i in range((days[-1] - days[0]).days + 1)]
        for tp in topics:
            vals = np.array([share(usable[d], tp["roots"]) if d in usable else np.nan for d in span])
            hist.topics[tp["id"]] = SimpleNamespace(tp=tp, start=days[0], v=vals, inputs=inputs)
        return hist

    def _fetch_doc(self, lab_, fresh=False):
        if not doc_api_on(lab_):      # the only door to the DOC API, and it stays shut unless the run was given --gdelt-doc-api
            raise lab.LabError("the GDELT DOC API is off unless the command is given --gdelt-doc-api")
        cfg = self.config(lab_)
        ttl = 1800.0 if fresh else 20 * 3600.0
        today = lab.utcnow().date()
        hist = SimpleNamespace(mode="doc", topics={}, inputs=[], notes=[])
        for tp in cfg["topics"]:
            url = GDELT.format(q=urllib.parse.quote(tp["query"], safe=":"))
            lab_.fetch(url, ttl=ttl, validate=_validate, min_gap=GDELT_MIN_GAP, retry_429=GDELT_RETRY_429)
            merged, inputs = {}, []
            for e in _snapshots(lab_, url):             # oldest first; newer copies replace older values
                for d, x in _parse((lab_.root / e["file"]).read_bytes()).items():
                    merged[d] = x
                inputs.append({"url": url, "sha256": e["sha256"]})
            days = sorted(d for d in merged if d < today)    # today's bucket is still filling
            if len(days) < 40:
                hist.notes.append(f"{tp['id']}: only {len(days)} complete days; skipped")
                continue
            lo = days[0]
            vals = np.array([merged.get(lo + dt.timedelta(days=i), np.nan) for i in range((days[-1] - lo).days + 1)])
            hist.topics[tp["id"]] = SimpleNamespace(tp=tp, start=lo, v=vals, inputs=inputs)
            hist.inputs.extend(inputs)
        if not hist.topics:
            raise lab.LabError("no GDELT topic could be read")
        return hist

    def _rows(self, hist):
        """Every labelled pair, as (as-of day, topic id, features, label, target day), oldest first."""
        rows = []
        for tid, tp in hist.topics.items():
            for t in range(0, len(tp.v) - H):
                f = topic_features(tp.v, t)
                if f is None or np.isnan(tp.v[t + H]):
                    continue
                d_target = tp.start + dt.timedelta(days=t + H)
                rows.append((tp.start + dt.timedelta(days=t), tid, f + weekday_onehot(d_target), 1.0 if tp.v[t + H] > tp.v[t] else 0.0, d_target))
        rows.sort(key=lambda r: (r[0], r[1]))
        return rows

    @staticmethod
    def _fit(train):
        X = np.array([r[2] for r in train])
        Y = np.array([r[3] for r in train])
        return lab.Logit(l2=10.0).fit(X, Y), Y

    @staticmethod
    def _bases(train, d_target):
        Y = [r[3] for r in train]
        wk = [r[3] for r in train if r[4].weekday() == d_target.weekday()]
        return {"up_rate": (sum(Y) + 1.0) / (len(Y) + 2.0), "weekday_rate": (sum(wk) + 1.0) / (len(wk) + 2.0)}

    def backtest(self, lab_):
        hist = self.fetch_history(lab_)
        rows = self._rows(hist)
        out = []
        for d, tid, x, y, d_target in rows:
            if (d - hist.topics[tid].start).days < WARMUP:
                continue
            train = [r for r in rows if r[4] <= d]           # labels known on day d: target day already complete
            if len(train) < MIN_TRAIN:
                continue
            model, Y = self._fit(train)
            out.append({"target": "up_h2", "kind": "binary", "key": f"{tid}:{d.isoformat()}", "as_of": d.isoformat(),
                        "pred": float(model.prob(x)[0]), "bases": self._bases(train, d_target), "primary": "up_rate", "outcome": y})
        longest = max(len(t.v) for t in hist.topics.values())
        if hist.mode == "export":
            intro = (f"{len(hist.topics)} event themes ({', '.join(hist.topics)}), each the daily share of the events GDELT coded that day "
                     f"(GDELT 1.0 daily event export, counted by CAMEO root code; per-day counts and file hashes in {COUNTS_FILE}), "
                     f"up to {longest} complete days each, {min(t.start for t in hist.topics.values())} to "
                     f"{max(t.start + dt.timedelta(days=len(t.v) - 1) for t in hist.topics.values())}; the first {WARMUP} days of each theme only train.")
            tail = "Baselines: up_rate (primary), the share of earlier pairs that rose; weekday_rate, the same for that weekday only. Small sample (120 days by default): a rehearsal, not evidence of skill."
        else:
            intro = f"{len(hist.topics)} themes ({', '.join(hist.topics)}), daily GDELT coverage share, up to {longest} complete days each; the first {WARMUP} days of each theme only train."
            tail = "Baselines: up_rate (primary), the share of earlier pairs that rose; weekday_rate, the same for that weekday only. Small sample from about 90 days of history: a rehearsal, not evidence of skill."
        notes = [intro,
                 "Question: is the share two days after the as-of day higher than on the as-of day? Pattern: ridge logistic on the last one-day change, the gap to the 7-day average and the weekday of the target day, pooled over themes and fitted only on pairs whose target day was already complete.",
                 tail] + hist.notes
        return {"rows": out, "inputs": hist.inputs, "notes": notes}

    # ---- live -----------------------------------------------------------------------
    def live_subjects(self, hist, now):
        return self._live_export(hist, now) if hist.mode == "export" else self._live_doc(hist, now)

    def _live_export(self, hist, now):
        """One call per theme: the share two days after the newest counted day against that day. The newest file is
        published the morning after its day, so the newest counted day is yesterday or the day before and the target is today or tomorrow."""
        L = now.date()
        subs = []
        for tid, tp in hist.topics.items():
            R = len(tp.v) - 1
            last = tp.start + dt.timedelta(days=R)
            T = last + dt.timedelta(days=H)
            if T < L:
                hist.notes.append(f"{tid}: the newest counted day is {last}, so its call would be for {T}, which is already over; skipped (newer files are not published or could not be read; run again later)")
                continue
            f = topic_features(tp.v, R)
            if f is None:
                continue
            subs.append(lab.make_subject(
                f"geopolitics:{tid}:up:{T.isoformat()}", "up_h2", "binary",
                f"Will {tp.tp['name']} make up a higher share of the world events GDELT codes on {T} than on {last} ({tp.v[R]:.3f}%)? A count of reported events by type, not a forecast of the events.",
                f"{(T + dt.timedelta(days=1)).isoformat()}T12:00:00Z",
                {"topic": tid, "roots": tp.tp["roots"], "ref_date": last.isoformat(), "ref_value": float(tp.v[R]), "target_date": T.isoformat(),
                 "source": SOURCE_EXPORT},
                _x=f + weekday_onehot(T), _T=T, as_of=lab.iso(now), _inputs=tp.inputs))
        return subs

    def _live_doc(self, hist, now):
        L = now.date()
        T = L + dt.timedelta(days=1)
        subs = []
        for tid, tp in hist.topics.items():
            R = len(tp.v) - 1
            last = tp.start + dt.timedelta(days=R)
            if last != L - dt.timedelta(days=1):
                hist.notes.append(f"{tid}: the latest complete day is {last}, not {L - dt.timedelta(days=1)}; skipped (run with --refresh)")
                continue
            f = topic_features(tp.v, R)
            if f is None:
                continue
            subs.append(lab.make_subject(
                f"geopolitics:{tid}:up:{T.isoformat()}", "up_h2", "binary",
                f"Will GDELT's share of world news coverage matching '{tp.tp['query']}' on {T} be higher than on {last} ({tp.v[R]:.4f}%)? Attention to world affairs, not a forecast of the events.",
                f"{(T + dt.timedelta(days=1)).isoformat()}T06:00:00Z",
                {"topic": tid, "query": tp.tp["query"], "ref_date": last.isoformat(), "ref_value": float(tp.v[R]), "target_date": T.isoformat(),
                 "source": SOURCE_DOC},
                _x=f + weekday_onehot(T), _T=T, as_of=lab.iso(now), _inputs=tp.inputs))
        return subs

    def features(self, hist, subject, as_of):
        train = self._rows(hist)
        return {"train": train} if len(train) >= MIN_TRAIN else None

    def predict(self, hist, subject, as_of, feats):
        model, Y = self._fit(feats["train"])
        p = float(model.prob(subject["_x"])[0])
        kind = "GDELT event-share themes" if hist.mode == "export" else "GDELT themes"
        return {"value": p, "method": f"ridge logistic on one-day change, gap to the 7-day average and target weekday, fitted on {len(Y)} complete-day pairs of {len(hist.topics)} {kind} up to {subject['meta']['ref_date']}"}

    def baseline(self, hist, subject, as_of, feats):
        return {"primary": "up_rate", "values": self._bases(feats["train"], subject["_T"])}

    # ---- outcomes -------------------------------------------------------------------
    def resolve(self, hist, subject):
        m = subject["meta"]
        tp = hist.topics.get(m["topic"])
        if tp is None:
            return None
        i = (dt.date.fromisoformat(m["target_date"]) - tp.start).days
        if i < 0 or i >= len(tp.v) or np.isnan(tp.v[i]):
            return None
        return 1.0 if tp.v[i] > m["ref_value"] else 0.0

    def evidence(self, hist, subject):
        tp = hist.topics.get(subject["meta"].get("topic"))
        return list(tp.inputs) if tp else []

    def resolve_calls(self, lab_, calls):
        """Outcomes {id: (outcome or None, evidence)}. A call is read from the source that made it (meta.source): an export call needs only
        its target day's file, however old the call is; a call without a source was made from the DOC API, and it is read from there only
        when the run was given --gdelt-doc-api: otherwise it comes back as (None, []) and no request is made for it."""
        out = {}
        export = [c for c in calls if (c.get("meta") or {}).get("source") == SOURCE_EXPORT]
        taken = {c["id"] for c in export}
        if export:
            targets = sorted({dt.date.fromisoformat(c["meta"]["target_date"]) for c in export})
            rows, error = load_days(lab_, targets, lab.utcnow().date(), [])
            if error is not None and any(t not in rows for t in targets):
                raise error
            for c in export:
                m = c["meta"]
                row = rows.get(dt.date.fromisoformat(m["target_date"]))
                if row is None or row.events < MIN_EVENTS:
                    out[c["id"]] = (None, [])
                else:
                    out[c["id"]] = (1.0 if share(row, m["roots"]) > m["ref_value"] else 0.0, [{"url": row.url, "sha256": row.sha256}])
        old = [c for c in calls if c["id"] not in taken]
        if old and not doc_api_on(lab_):
            _say(f"geopolitics: {len(old)} call(s) were made from the GDELT DOC API and stay unscored unless the command is given --gdelt-doc-api; no request was made for them")
            out.update({c["id"]: (None, []) for c in old})
        elif old:
            hist = self._fetch_doc(lab_, fresh=True)
            for c in old:
                out[c["id"]] = (self.resolve(hist, lab.call_subject(c)), self.evidence(hist, lab.call_subject(c)))
        return out


ADAPTER = Adapter()
