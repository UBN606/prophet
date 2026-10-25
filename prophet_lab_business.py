"""Prophet domain: business (SEC EDGAR XBRL, free). A drop-in file for prophet_lab.py.

Question asked of every company: will its revenue for the next quarter be above the same quarter a year earlier?
Pattern: revenue growth momentum (the last two year-over-year growth rates), pooled over all companies.
Baselines: the share of earlier quarters that grew year over year, and "growth continues" (last quarter's sign).
Every quarter is taken as first reported (earliest filing date), so a backtest only uses what was public at the time.
"""
import datetime as dt
import json
from types import SimpleNamespace

import numpy as np

import prophet_lab as lab

EDGAR = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{tag}.json"
DEFAULT = {
    "tags": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "companies": [
        {"ticker": "AAPL", "cik": 320193, "name": "APPLE"}, {"ticker": "MSFT", "cik": 789019, "name": "MICROSOFT"},
        {"ticker": "AMZN", "cik": 1018724, "name": "AMAZON"}, {"ticker": "GOOGL", "cik": 1652044, "name": "ALPHABET"},
        {"ticker": "META", "cik": 1326801, "name": "META PLATFORMS"}, {"ticker": "NVDA", "cik": 1045810, "name": "NVIDIA"},
        {"ticker": "WMT", "cik": 104169, "name": "WALMART"}, {"ticker": "COST", "cik": 909832, "name": "COSTCO"},
        {"ticker": "HD", "cik": 354950, "name": "HOME DEPOT"}, {"ticker": "PEP", "cik": 77476, "name": "PEPSICO"}, {"ticker": "PG", "cik": 80424, "name": "PROCTER"},
        {"ticker": "JNJ", "cik": 200406, "name": "JOHNSON"}, {"ticker": "XOM", "cik": 34088, "name": "EXXON"},
        {"ticker": "CVX", "cik": 93410, "name": "CHEVRON"}, {"ticker": "UNH", "cik": 731766, "name": "UNITEDHEALTH"},
    ],
}
FORMS = ("10-Q", "10-K", "10-Q/A", "10-K/A")
MIN_TRAIN = 80


def _validate(data):
    j = json.loads(data.decode("utf-8"))
    if not isinstance(j, dict) or "units" not in j:
        raise ValueError("not an EDGAR XBRL concept file")


def _day(s):
    return dt.date.fromisoformat(s)


def quarters_from_facts(facts):
    """Discrete quarters as first reported: [(end, value, filed)], oldest first. Q4 = full year minus Q1..Q3."""
    q, fy = {}, {}
    for x in facts:
        if x.get("form") not in FORMS or "start" not in x or "filed" not in x:
            continue
        s, e = _day(x["start"]), _day(x["end"])
        d = (e - s).days
        bucket = q if 80 <= d <= 100 else (fy if 350 <= d <= 380 else None)
        if bucket is None or not x.get("val") or x["val"] <= 0:
            continue
        rec = (float(x["val"]), _day(x["filed"]))
        if (s, e) not in bucket or rec[1] < bucket[(s, e)][1]:
            bucket[(s, e)] = rec
    out = {e: rec for (s, e), rec in q.items()}
    for (s, e), (v, f) in fy.items():
        if e in out:
            continue
        inside = sorted((qe, qv) for (qs, qe), (qv, _qf) in q.items() if qs >= s - dt.timedelta(days=5) and qe <= e - dt.timedelta(days=60))
        if len(inside) != 3:
            continue
        ends = [x[0] for x in inside] + [e]
        if all(75 <= (b - a).days <= 105 for a, b in zip(ends, ends[1:])) and v - sum(x[1] for x in inside) > 0:
            out[e] = (v - sum(x[1] for x in inside), f)
    return sorted((e, v, f) for e, (v, f) in out.items())


def contiguous(series, a, b):
    """True when quarters a..b (inclusive) follow each other, about three months apart."""
    return a >= 0 and all(75 <= (series[i + 1][0] - series[i][0]).days <= 105 for i in range(a, b))


def clip(x, lim=0.6):
    return max(-lim, min(lim, x))


def rows_for(series):
    """One row per quarter k that has a year-ago quarter and two earlier growth rates."""
    out = []
    for k in range(6, len(series)):
        if not contiguous(series, k - 6, k):
            continue
        e, v, f = series[k]
        out.append({"k": k, "end": e, "filed": f, "t": series[k - 1][2],
                    "g1": clip(series[k - 1][1] / series[k - 5][1] - 1.0), "g2": clip(series[k - 2][1] / series[k - 6][1] - 1.0),
                    "y": 1.0 if v > series[k - 4][1] else 0.0, "ya_end": series[k - 4][0], "ya_val": series[k - 4][1]})
    return out


class Adapter(lab.Adapter):
    name = "business"
    title = "Business: will a company's quarterly revenue beat the same quarter last year?"
    sources = [{"name": "SEC EDGAR XBRL API (data.sec.gov, concept files)", "url": "https://data.sec.gov/api/xbrl/",
                "note": "free, no key; U.S. government data. NOT USED for now: EDGAR asks each automated client for a contact e-mail in its User-Agent"}]
    status = "blocked"
    status_note = ("Blocked on purpose: SEC EDGAR asks every automated client to put a contact e-mail in its User-Agent, and Prophet does not send anyone's e-mail "
                   "address without the owner's go. Nothing is downloaded from sec.gov. Hand-made calls still work: call --domain business ... then resolve. "
                   "To switch it on, the owner decides the contact line and sets status = \"ready\" in this file.")
    default_config = DEFAULT
    target_labels = {"rev_up": "quarterly revenue above the same quarter a year earlier, yes/no (Brier)"}

    # ---- history ---------------------------------------------------------------------
    def _company(self, lab_, co, tags, ttl):
        facts, inputs, name = [], [], None
        for tag in tags:
            url = EDGAR.format(cik=co["cik"], tag=tag)
            try:
                f = lab_.fetch(url, ttl=ttl, validate=_validate, min_gap=1.0)
            except lab.FetchError as e:
                if e.code == 404:
                    continue
                raise
            j = json.loads(f.text("utf-8"))
            name = j.get("entityName") or name
            inputs.append({"url": url, "sha256": f.sha256})
            usd = (j.get("units") or {}).get("USD") or []
            facts.append(usd)
        if name is not None and co.get("name") and co["name"].lower() not in name.lower():
            return None, inputs, f"{co['ticker']}: EDGAR says CIK {co['cik']} is '{name}', not '{co['name']}'; skipped"
        # merge tags: prefer the first tag that has a given quarter
        merged = {}
        for usd in facts:
            for e, v, f in quarters_from_facts(usd):
                merged.setdefault(e, (v, f))
        series = sorted((e, v, f) for e, (v, f) in merged.items())
        return series, inputs, None

    def fetch_history(self, lab_, fresh=False):
        cfg = self.config(lab_)
        ttl = 3600.0 if fresh else 86400.0
        hist = SimpleNamespace(companies={}, inputs=[], notes=[])
        for co in cfg["companies"]:
            series, inputs, note = self._company(lab_, co, cfg["tags"], ttl)
            hist.inputs.extend(inputs)
            if note:
                hist.notes.append(note)
                continue
            if len(series) < 8:
                hist.notes.append(f"{co['ticker']}: only {len(series)} quarters found; skipped")
                continue
            hist.companies[co["ticker"]] = SimpleNamespace(co=co, series=series, rows=rows_for(series), inputs=inputs)
        if not hist.companies:
            raise lab.LabError("no company data could be read from EDGAR")
        return hist

    # ---- walk-forward backtest -----------------------------------------------------------
    @staticmethod
    def _fit(rows):
        X = np.array([[r["g1"], r["g2"]] for r in rows])
        y = np.array([r["y"] for r in rows])
        return lab.Logit(l2=3.0).fit(X, y), float((y.sum() + 1.0) / (len(y) + 2.0))

    def backtest(self, lab_):
        hist = self.fetch_history(lab_)
        allrows = sorted(((tk, r) for tk, c in hist.companies.items() for r in c.rows), key=lambda x: (x[1]["t"], x[0]))
        out = []
        for tk, r in allrows:
            train = [x for _, x in allrows if x["filed"] < r["t"]]
            if len(train) < MIN_TRAIN:
                continue
            model, base = self._fit(train)
            out.append({"target": "rev_up", "kind": "binary", "key": f"{tk}:{r['end'].isoformat()}", "as_of": r["t"].isoformat(),
                        "pred": float(model.prob([r["g1"], r["g2"]])[0]),
                        "bases": {"growth_rate": base, "momentum_sign": 1.0 if r["g1"] > 0 else 0.0},
                        "primary": "growth_rate", "outcome": r["y"]})
        names = ", ".join(hist.companies)
        notes = [f"{len(hist.companies)} companies ({names}), revenue by quarter from SEC EDGAR XBRL, each quarter as first reported; {len(allrows)} quarters have a year-ago quarter and two earlier growth rates, and the first {MIN_TRAIN} only train.",
                 "Question: is the quarter's revenue above the same quarter a year earlier? Pattern: ridge logistic on the last two year-over-year growth rates, fitted only on quarters whose results were already filed before the call date.",
                 "Baselines: growth_rate, the share of earlier quarters that grew (primary); momentum_sign, 'last quarter grew, so this one will' stated as certain."] + hist.notes
        return {"rows": out, "inputs": hist.inputs, "notes": notes}

    # ---- live ---------------------------------------------------------------------------
    def live_subjects(self, hist, now):
        allrows = [r for c in hist.companies.values() for r in c.rows]
        subs = []
        for tk, c in hist.companies.items():
            s = c.series
            k = len(s)
            e1 = s[-1][0] + dt.timedelta(days=91)
            if e1 <= now.date():
                hist.notes.append(f"{tk}: the next quarter (ends about {e1}) has already ended; skipped so the call is made before the quarter closes")
                continue
            if not contiguous(s, k - 6, k - 1):
                hist.notes.append(f"{tk}: recent quarters are not contiguous; skipped")
                continue
            ya = s[k - 4]
            feats = {"g1": clip(s[-1][1] / s[-5][1] - 1.0), "g2": clip(s[-2][1] / s[-6][1] - 1.0)}
            meta = {"ticker": tk, "cik": c.co["cik"], "target_end": e1.isoformat(), "ya_end": ya[0].isoformat(), "ya_val": ya[1]}
            subs.append(lab.make_subject(
                f"business:{tk.lower()}:rev_up:{e1.isoformat()}", "rev_up", "binary",
                f"Will {tk}'s revenue for the quarter ending about {e1} be above its revenue for the quarter ended {ya[0]} ({ya[1]:,.0f} USD), as first reported to the SEC?",
                f"{(e1 + dt.timedelta(days=62)).isoformat()}T00:00:00Z", meta, _feats=feats, _rows=allrows, as_of=lab.iso(now), _inputs=c.inputs))
        return subs

    def features(self, hist, subject, as_of):
        rows = [r for r in subject["_rows"] if r["filed"] <= dt.date.fromisoformat(as_of[:10])]
        if len(rows) < MIN_TRAIN:
            return None
        return dict(subject["_feats"], rows=rows)

    def predict(self, hist, subject, as_of, feats):
        model, base = self._fit(feats["rows"])
        p = float(model.prob([feats["g1"], feats["g2"]])[0])
        return {"value": p, "method": f"ridge logistic on the last two year-over-year revenue growth rates ({feats['g1']:+.1%}, {feats['g2']:+.1%}), fitted on {len(feats['rows'])} earlier company-quarters filed before {as_of[:10]}"}

    def baseline(self, hist, subject, as_of, feats):
        _, base = self._fit(feats["rows"])
        return {"primary": "growth_rate", "values": {"growth_rate": base, "momentum_sign": 1.0 if feats["g1"] > 0 else 0.0}}

    def resolve(self, hist, subject):
        m = subject["meta"]
        c = hist.companies.get(m["ticker"])
        if c is None:
            return None
        target = dt.date.fromisoformat(m["target_end"])
        for e, v, f in c.series:
            if abs((e - target).days) <= 12 and e > dt.date.fromisoformat(m["ya_end"]) + dt.timedelta(days=300):
                return 1.0 if v > m["ya_val"] else 0.0
        return None

    def evidence(self, hist, subject):
        c = hist.companies.get(subject["meta"].get("ticker"))
        return list(c.inputs) if c else []

    def resolve_calls(self, lab_, calls):
        """Read only the companies named in the calls."""
        cfg = self.config(lab_)
        wanted = {c["meta"]["ticker"] for c in calls}
        hist = SimpleNamespace(companies={}, inputs=[], notes=[])
        for co in cfg["companies"]:
            if co["ticker"] in wanted:
                series, inputs, note = self._company(lab_, co, cfg["tags"], 3600.0)
                if series:
                    hist.companies[co["ticker"]] = SimpleNamespace(co=co, series=series, rows=[], inputs=inputs)
        return {c["id"]: (self.resolve(hist, lab.call_subject(c)), self.evidence(hist, lab.call_subject(c))) for c in calls}


ADAPTER = Adapter()
