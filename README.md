# Prophet

Prophet finds the exact moment in what you have collected, and calls what comes next with an honest track record.

It is a toolbox of small command-line programs in plain Python. Everything runs on your own computer: it reads the
files you point it at and writes to one data folder. There is no server, no account and no API key.

> **Name note.** Meta also publishes an open-source forecasting library called Prophet (`facebook/prophet`).
> This project is a different thing and shares nothing with it but the word.

## What it does

| Command | What it does |
|---|---|
| `prophet ask` | Search every transcript, note and finding; answers carry a time code or path:line |
| `prophet ub` | Urantia Book passages with exact citations from a local book tool you provide, plus your own notebooks |
| `prophet lab` | Pattern Lab: sealed calls on sports, weather, markets, anything; scored against a baseline |

Run `prophet` with no arguments to print this list with a status for each skill. `building` marks a skill that is newer and
not yet marked proven by its author's own checks.

**Studio-specific skills are not included.** Prophet also has skills that only make sense inside the author's own film studio
setup (they work on its cut documents, renders and private notes). They are left out of this repository on purpose, and the
`prophet` command lists only the skills that are here.

Two scripts sit beside the front door:

| Script | What it does |
|---|---|
| `source_harvest.py` | Collects the public sources in `sources/sources.json` into local, searchable notebooks |
| `approve_sources.py` | Records that you reviewed the source list; the harvester will not run without it |

`prophet_net.py` is a library, not a script: it holds the download rule that the harvester and the Pattern Lab share (see
"The honest-record rules", item 7).

## Install

You need:

* Python 3.11 or newer (`python --version`)
* the Python packages in requirements.txt (numpy): `pip install -r requirements.txt`

Optional, only for the parts that use them:

* `node` and a local copy of the Urantia Book search tools (`ub-search.js`, `ub-verify.js`) for `prophet ub`. The book text is not part of this repo.

```
git clone <this repository>
cd prophet
pip install -r requirements.txt
python prophet.py
```

`prophet.cmd` (Windows) and `prophet` (macOS and Linux; run `chmod +x prophet` once) are thin launchers for
`python -B prophet.py`. Put the folder on your PATH and you can type `prophet` anywhere.

Prophet was built and tested on Windows. The code is portable Python and the shell launcher covers macOS and Linux, but those
two have not been exercised yet.

## Try it first (no network, nothing to set up)

The Pattern Lab can seal and score a call on any question by hand. This takes seconds and touches only your data folder:

```
python prophet.py lab call --domain demo --question "Will it rain in Oslo on 2027-01-01?" --resolves 2027-01-01 --prob 0.7
python prophet.py lab list
python prophet.py lab verify
python prophet.py lab record
```

The call is sealed with a plain baseline beside it. `lab record` writes `RECORD.md` and `record.html` under
`<data folder>/notebooks/pattern-lab/`. When the answer is known, score the call with
`python prophet.py lab resolve <id> --outcome yes` (or `no`); the id is printed by `lab call` and `lab list`.

## Where Prophet keeps things

One data folder holds everything Prophet writes. By default it is `~/prophet-data` (on Windows `%USERPROFILE%\prophet-data`).
Move it with the environment variable `PROPHET_HOME`, or copy `prophet.example.toml` to `prophet.toml` and edit it. The same
file (or `PROPHET_<KEY>` variables) points the optional parts at your own folders. To see what Prophet resolved:

```
python prophet_config.py
```

Nothing Prophet writes ever goes into this repository folder.

## Quickstart

### Search everything Prophet knows: `prophet ask`

```
prophet ask camera moves for game animation
prophet ask -c sources-sports home advantage    # only one channel (part of the name is enough)
prophet ask --json "dead air"                   # machine readable
prophet ask --stats                             # what is searchable right now
```

`ask` searches the Markdown files in your notebooks folder (captions, transcripts, notes, findings). Each answer carries a
time code or a file path; a YouTube answer links to the exact second.

### Pattern Lab: `prophet lab`

```
prophet lab domains                  # what each domain is and where its free data comes from
prophet lab backtest weather         # replay history walk-forward (a rehearsal, not a track record)
prophet lab predict weather          # seal live calls for things that have not happened yet
prophet lab score                    # score sealed calls whose outcome is now in the data
prophet lab call --domain politics --question "Will X happen by 2027-06-30?" --resolves 2027-06-30 --prob 0.62
prophet lab resolve <id> --outcome yes
prophet lab verify                   # check every seal and the hash chain
prophet lab record                   # write RECORD.md and record.html
```

Built-in domains: `weather` (tomorrow's high and rain, Open-Meteo), `sports` (football home win, football-data.co.uk results),
`markets` (index closes, the public FRED CSV), `geopolitics` (news attention to a theme, from GDELT's timeline API), `business`
(SEC EDGAR; off until its owner chooses the contact line EDGAR requires) and `racing` (no free data source was found, so it
only takes hand-made calls). Every download the lab makes passes the same robots.txt rule as the harvester (see item 7 of the
honest-record rules). The `geopolitics` domain keeps GDELT's rate limit: at least 6.5 seconds between requests, and on HTTP 429
one retry after 10 seconds, after which the domain stops with an error and seals nothing.
Any other question can be sealed by hand with `call` and `resolve`. A new domain is one file named
`prophet_lab_<name>.py` next to `prophet_lab.py`; `prophet lab domains` shows how. The lab keeps its files in
`<data folder>/notebooks/pattern-lab/` (override with `--lab-dir` or `PROPHET_LAB_DIR`).

### Book passages: `prophet ub`

```
prophet ub prayer                    # top passages for a word or phrase
prophet ub --ref 180:2.1 180:2.2     # the exact paragraph text for a citation
prophet ub prayer --with-notebooks   # the same words also asked of your notebooks
```

Needs `node` and the Urantia Book search tools in the `ub_tools` folder. All text comes from that local tool, never from
memory and never from this repo.

### Collect public sources

```
python approve_sources.py                                 # read the list, then say yes (writes an approval file)
python source_harvest.py --list                           # what would be fetched, fetches nothing
python source_harvest.py --area weather_climate --limit 5 # one area, a few items per source
python source_harvest.py --source gh_comfyui --dry-run    # fetch and parse one source, write nothing
```

Output lands in `<data folder>/notebooks/sources-<area>/`, and `ask` finds it. `--list` marks the sources that are read
through an official API on the robots allowlist, and the ones that are disabled. Start small with `--limit`; the hard cap is
50 items per source per run.

## The honest-record rules

The Pattern Lab (`prophet lab`) is built so the record cannot flatter itself:

1. **Sealed before the outcome.** Each call goes into an append-only log with a SHA-256 seal chained to the one before it,
   before the answer is known. `prophet lab verify` re-checks every seal and the chain; an edited, removed or reordered
   call, or a score that does not match its outcome, is reported as TAMPERED. A seal does not prove *when* a call was made
   (the clock is your computer's): copy the chain head that `lab verify` prints somewhere outside your machine if that matters.
2. **Scored against a baseline.** Every call carries what a plain rule would have said (climatology, persistence, the base
   rate, a coin flip). Both scores are shown. A domain is marked "beats baseline" only with at least 30 scored calls, ahead of
   every baseline, and a 95% range that stays above zero; below that, scores are printed and no skill is claimed.
3. **Rehearsals are not a record.** `backtest` replays history walk-forward, using only data from before each date, and is
   labelled a rehearsal. Only calls sealed before their outcome count as the track record, in a separate table.
4. **Wins and misses together.** `lab record` writes `RECORD.md` and `record.html` from the sealed files. Nothing is edited
   by hand, and the biggest misses are listed next to the biggest wins.
5. **No bets, no trades.** Prophet places none, connects to no betting or brokerage site, and gives no betting, trading or
   investment advice. It makes pattern calls and scores them.
6. **Free public data only.** No key, no account, no paid call. A source that needs one is not used (SEC EDGAR, for example,
   stays switched off until its owner chooses the contact line EDGAR requires).
7. **Polite to the sites it reads.** Every automatic download follows one rule, kept in `prophet_net.py` and used by both the
   source harvester and the Pattern Lab: the site's `robots.txt` (wildcards, Allow and Crawl-delay) must allow it. A
   `robots.txt` that cannot be read blocks the site (HTTP 429, any 5xx, a refusal such as 401 or 403, redirects that never
   end); only a plain 404 or 410 means there are no restrictions. The one exception is a short, explicit allowlist (`API_ALLOWLIST`) of
   official, documented API endpoints: the arXiv API, Open-Meteo, api.weather.gov, GDELT's DOC API and a few like them. Only an
   exact API URL on that list skips the `robots.txt` check; no other page of those sites is ever fetched, and there is no
   switch that overrides the rule. Both programs identify themselves, wait between requests to one site and cap what they take.
   The Pattern Lab also keeps a local cache of what it downloaded, asks GDELT for no more than one request every 6.5 seconds,
   and when GDELT answers HTTP 429 it waits 10 seconds, asks once more and then stops with a clear error.
8. **A check that could not run is a failure.** Machine flags are pointers to look at, never approvals.

## The source list

`sources/sources.json` lists 204 free public sources:

| Area | Sources | Kinds |
|---|---|---|
| Film craft | 50 | youtube 35, rss 15 |
| AI tools | 33 | github_releases 20, rss 9, youtube 3, api 1 |
| Forecasting methods | 20 | arxiv 9, rss 6, api 4, github_releases 1 |
| Sports | 19 | api 8, csv 7, github_releases 4 |
| Weather and climate | 20 | api 14, csv 5, rss 1 |
| Markets and business | 22 | api 15, csv 5, rss 2 |
| Geopolitics, history and news | 28 | api 16, rss 10, csv 2 |
| Faith and the Urantia Book (links and metadata only) | 12 | rss 9, youtube 3 |

Each entry has a name, URL, area, kind (rss, youtube, github_releases, csv, api, arxiv), cost, the licence or terms that apply,
and whether its endpoint answered as expected on the day it was checked. It is a list only: no harvested content is shipped.
Only entries marked verified and not `disabled` are ever fetched, and only after you run `python approve_sources.py` (an
approval record without `"status": "approved"` does not count).
4 entries are switched off (`disabled`): `api_sec_submissions`, `api_sec_companyconcept`, `rss_sec_8k`, `api_gdelt_doc` stay
in the list with the reason, for the record, and are never fetched. The source harvester reads Sky News World
(`rss_sky_news_world`) instead of the GDELT DOC API, which is disabled. The Pattern Lab's geopolitics domain is separate: it
still uses GDELT's timeline API, under its own rate limit (see the Pattern Lab section).
The Urantia Book area is links and metadata only; the harvester never stores book text.

**YouTube channels are not harvested by the public tool.** The only way to read a channel's captions is a downloader that
cannot honour YouTube's robots.txt, and every download in this repository has to. The 41 YouTube rows of the list
are kept as pointers and marked `disabled`; follow those channels yourself.

Content you collect stays on your machine and stays subject to the terms listed beside each source. Items marked
`private: true` in the harvested notebooks must not be redistributed.

## What is not in this repository

No collected transcripts, notebooks, ledgers, track records or media; no keys, tokens or addresses; no text of the Urantia
Book (`prophet ub` calls a separate local tool that you provide); and none of the Studio-specific skills described above.

## Tests

```
python -B -m unittest discover -s tests -v
```

## Licence

Licence: not chosen yet (owner decides before release).
