"""Prophet domain: geopolitics (GDELT news-attention themes, free). A drop-in file for prophet_lab.py.

What it calls: whether the share of world news coverage that matches a theme (protests, armed conflict, sanctions,
ceasefire, elections, coups...) will be higher on a coming day than on the last complete day. That is attention to
world affairs, not a forecast of the events themselves. For questions about the events, use hand-made calls:
  prophet_lab.py call --domain geopolitics --question "..." --resolves YYYY-MM-DD --prob 0.35

Sources tried: Metaculus's public API answers HTTP 403 ("only available to authenticated users"), so it is not used.
GDELT's DOC API is free and keyless but keeps about 90 days of daily history and allows one request per 5 seconds: the lab
waits at least 6.5 seconds between requests to it and, when it answers HTTP 429, waits 10 seconds, asks once more and then
stops with a clear error instead of hammering it.
Every download is cached, and older cached copies are merged in, so the history grows as the lab runs.
"""
import datetime as dt
import json
import urllib.parse
from types import SimpleNamespace

import numpy as np

import prophet_lab as lab

GDELT = "https://api.gdeltproject.org/api/v2/doc/doc?query={q}&mode=timelinevol&format=json&timespan=3m"
GDELT_MIN_GAP = 6.5      # seconds between two requests to GDELT, robots.txt reads included (it asks for one every 5 seconds)
GDELT_RETRY_429 = 1      # on HTTP 429 wait at least 10 s and ask once more; if it still says 429 the domain stops cleanly
DEFAULT = {"topics": [
    {"id": "protest", "name": "protests", "query": "theme:PROTEST"},
    {"id": "armed_conflict", "name": "armed conflict", "query": "theme:ARMEDCONFLICT"},
    {"id": "terror", "name": "terrorism", "query": "theme:TERROR"},
    {"id": "election", "name": "elections", "query": "theme:ELECTION"},
    {"id": "sanctions", "name": "sanctions", "query": "sanctions"},
    {"id": "ceasefire", "name": "ceasefire", "query": "ceasefire"},
    {"id": "coup", "name": "coups", "query": "coup"},
    {"id": "tariffs", "name": "tariffs", "query": "tariffs"},
]}
H = 2                 # calls look two days past the last complete day: day t against day t+2
WARMUP = 21           # days of history before the first scored call of a theme
MIN_TRAIN = 60


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


def topic_features(v, t):
    """Features at day index t, from values up to and including t."""
    if t < 8 or v[t] <= 0 or v[t - 1] <= 0 or np.isnan(v[t - 7:t + 1]).any():
        return None
    base = v[t - 7:t].mean()
    return [float(np.log(v[t] / v[t - 1])), float(np.log(v[t] / base))] if base > 0 else None


def weekday_onehot(date):
    """Weekday of the target day, Monday to Saturday (Sunday is the reference)."""
    return [1.0 if date.weekday() == i else 0.0 for i in range(6)]


class Adapter(lab.Adapter):
    name = "geopolitics"
    title = "Geopolitics: will news attention to a world-affairs theme rise?"
    sources = [{"name": "GDELT DOC 2.0 API (timeline of coverage volume)", "url": "https://api.gdeltproject.org/api/v2/doc/doc",
                "note": "free, no key; about 90 days of daily history; one request per 5 seconds; credit the GDELT Project"},
               {"name": "Metaculus public API (the suggested community-forecast source)", "url": "https://www.metaculus.com/api/posts/",
                "note": "NOT USED: it answers HTTP 403 'The API is only available to authenticated users'"}]
    status_note = ("Calls are about news attention to world-affairs themes (GDELT), not forecasts of the events themselves. "
                   "Metaculus needs an account token, so its community forecasts are not used. For event questions use hand-made calls: call --domain geopolitics ...")
    default_config = DEFAULT
    target_labels = {"up_h2": "theme coverage two days past the last full day is higher, yes/no (Brier)"}

    def fetch_history(self, lab_, fresh=False):
        cfg = self.config(lab_)
        ttl = 1800.0 if fresh else 20 * 3600.0
        today = lab.utcnow().date()
        hist = SimpleNamespace(topics={}, inputs=[], notes=[])
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
        notes = [f"{len(hist.topics)} themes ({', '.join(hist.topics)}), daily GDELT coverage share, up to {max(len(t.v) for t in hist.topics.values())} complete days each; the first {WARMUP} days of each theme only train.",
                 "Question: is the coverage share two days after the as-of day higher than on the as-of day? Pattern: ridge logistic on the last one-day change, the gap to the 7-day average and the weekday of the target day, pooled over themes and fitted only on pairs whose target day was already complete.",
                 "Baselines: up_rate (primary), the share of earlier pairs that rose; weekday_rate, the same for that weekday only. Small sample from about 90 days of history: a rehearsal, not evidence of skill."] + hist.notes
        return {"rows": out, "inputs": hist.inputs, "notes": notes}

    # ---- live -----------------------------------------------------------------------
    def live_subjects(self, hist, now):
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
                {"topic": tid, "query": tp.tp["query"], "ref_date": last.isoformat(), "ref_value": float(tp.v[R]), "target_date": T.isoformat()},
                _x=f + weekday_onehot(T), _T=T, as_of=lab.iso(now), _inputs=tp.inputs))
        return subs

    def features(self, hist, subject, as_of):
        train = self._rows(hist)
        return {"train": train} if len(train) >= MIN_TRAIN else None

    def predict(self, hist, subject, as_of, feats):
        model, Y = self._fit(feats["train"])
        p = float(model.prob(subject["_x"])[0])
        return {"value": p, "method": f"ridge logistic on one-day change, gap to the 7-day average and target weekday, fitted on {len(Y)} complete-day pairs of {len(hist.topics)} GDELT themes up to {subject['meta']['ref_date']}"}

    def baseline(self, hist, subject, as_of, feats):
        return {"primary": "up_rate", "values": self._bases(feats["train"], subject["_T"])}

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


ADAPTER = Adapter()
