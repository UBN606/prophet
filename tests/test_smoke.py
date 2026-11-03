"""Smoke tests: the front door starts, the help pages print, and everything Prophet writes goes to the data folder.

  python -B -m unittest discover -s tests -v
"""
import contextlib
import datetime as dt
import hashlib
import http.server
import importlib
import io
import json
import math
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.parse
import zipfile
from html.parser import HTMLParser
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load(name):
    """Import one of the scripts beside this folder (plain files, not a package). Importing them writes nothing."""
    os.environ.setdefault("no_proxy", "127.0.0.1")        # the robots test talks to a server on this machine; set before the first import
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    return importlib.import_module(name)


def run(data, *args, script="prophet.py", env_extra=None, home=True):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PROPHET_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if home:
        env["PROPHET_HOME"] = str(data)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-B", str(REPO / script), *args], cwd=str(data), env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=120)


def run_code(data, code):
    """Run a line of Python with the repo on the path and the data folder set; used to look inside a script without running it."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PROPHET_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", PROPHET_HOME=str(data), PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, "-B", "-c", code], cwd=str(data), env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=120)


class Smoke(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_card(self):
        r = run(self.data)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("PROPHET", r.stdout)
        self.assertIn("lab", r.stdout)

    def test_ask_help(self):
        r = run(self.data, "ask", "--help")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("usage", r.stdout.lower())

    def test_lab_help(self):
        r = run(self.data, "lab", "--help")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Pattern Lab", r.stdout)

    def test_config_follows_prophet_home(self):
        r = run(self.data, script="prophet_config.py")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(str(self.data), r.stdout)

    def test_toml_file_sets_the_folders_and_the_environment_wins(self):
        elsewhere = self.data / "elsewhere"
        cfg = self.data / "settings.toml"
        cfg.write_text('home = "%s"\n[paths]\nnotebooks = "my-notes"\n' % (self.data / "h").as_posix(), encoding="utf-8")
        r = run(self.data, script="prophet_config.py", home=False, env_extra={"PROPHET_CONFIG": str(cfg)})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(str(self.data / "h" / "my-notes"), r.stdout)
        r = run(self.data, script="prophet_config.py", home=False, env_extra={"PROPHET_CONFIG": str(cfg), "PROPHET_NOTEBOOKS": str(elsewhere)})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(str(elsewhere), r.stdout)

    def test_ask_stats_writes_only_to_the_data_folder(self):
        r = run(self.data, "ask", "--stats")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((self.data / "notebooks").is_dir())

    def test_lab_domains_works_offline(self):
        r = run(self.data, "lab", "domains")
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_front_door_lists_only_skills_that_are_in_this_copy(self):
        prophet = load("prophet")
        for name, (script, _lead, _desc) in prophet.SKILLS.items():
            self.assertTrue((REPO / script).is_file(), "%s is listed but %s is not in this copy" % (name, script))
        card = run(self.data).stdout
        for name in prophet.SKILLS:
            self.assertIn(name, card)
        self.assertEqual(set(prophet.installed()), set(prophet.SKILLS))


class Site:
    """A throwaway web site on this machine, so these tests never touch the internet. Its robots.txt forbids /private. It has an
    open page, a private page, two redirects and a page that always answers 429 (too many requests). Every request is recorded."""

    def __init__(self, robots_status=200, robots_loop=False):
        self.hits, self.stamps, outer = [], [], self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                outer.hits.append(self.path)
                outer.stamps.append(time.monotonic())
                if self.path == "/robots.txt":
                    if robots_loop:                                     # a robots.txt that redirects to itself, for ever
                        return self.reply(302, b"", {"Location": "/robots.txt"})
                    return self.reply(robots_status, b"User-agent: *\nDisallow: /private\n")
                if self.path == "/to-private":
                    return self.reply(302, b"", {"Location": "/private/data.csv"})
                if self.path == "/to-open":
                    return self.reply(302, b"", {"Location": "/open.csv"})
                if self.path == "/busy":
                    return self.reply(429, b"Please limit requests to one every 5 seconds", {"Retry-After": "0"})
                return self.reply(200, b"ok" if self.path.startswith("/open") else b"secret")

            def reply(self, code, body, headers=None):
                self.send_response(code)
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass
        self.server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.base = "http://127.0.0.1:%d" % self.server.server_port
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class Harvester(unittest.TestCase):
    """The source harvester's rules, checked without the network."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_api_allowlist_is_strict(self):
        sh = load("source_harvest")

        def listed(url):
            return sh.api_entry(urllib.parse.urlsplit(url)) is not None
        for url in ("https://export.arxiv.org/api/query?search_query=all:x&max_results=5",
                    "https://api.open-meteo.com/v1/forecast?latitude=1&longitude=2",
                    "https://api.weather.gov/points/40.7,-74.0",
                    "https://www.loc.gov/search/?q=x&fo=json",
                    "https://api.gdeltproject.org/api/v2/doc/doc?query=x&mode=timelinevol",
                    "https://api.openf1.org/v1/sessions?year=2024&session_type=Race",
                    "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=86400",
                    "https://a.pool.opentimestamps.org/digest",                 # the two calendars `lab anchor` posts its hash to
                    "https://b.pool.opentimestamps.org/digest"):
            self.assertTrue(listed(url), url)
        for url in ("https://export.arxiv.org/abs/2101.00001",           # another page of a listed site
                    "http://export.arxiv.org/api/query",                   # not https
                    "https://api.open-meteo.com/",                        # the site, not its API
                    "https://open-meteo.com/en/docs",
                    "https://api.weather.gov/stations",                   # an endpoint nobody listed
                    "https://www.loc.gov/search/?q=x",                    # the web page, not the JSON API
                    "https://export.arxiv.org/api/../abs/1",              # dot segments
                    "https://someone" + chr(64) + "export.arxiv.org/api/query",   # credentials in the URL
                    "https://export.arxiv.org:8443/api/query",            # another port
                    "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500",   # a download link of a website is not an API
                    "https://www.football-data.co.uk/fixtures.csv",
                    "https://api.openf1.org/v1/laps?session_key=1",         # another endpoint on a listed host
                    "https://api.openf1.org/v1/sessions/extra",
                    "https://api.openf1.org/",
                    "https://api.exchange.coinbase.com/products/BTC-USD/book",
                    "https://api.exchange.coinbase.com/products/BTC-USD/candles/extra",
                    "https://api.exchange.coinbase.com/products//candles",   # the * stands for one real segment
                    "https://api.exchange.coinbase.com/orders",
                    "https://www.coinbase.com/products/BTC-USD/candles",     # the website, not the API host
                    "https://a.pool.opentimestamps.org/",                      # a calendar's other addresses are not listed
                    "https://a.pool.opentimestamps.org/timestamp/00ff",
                    "http://a.pool.opentimestamps.org/digest",
                    "https://c.pool.opentimestamps.org/digest",
                    "https://example.com/api/query"):
            self.assertFalse(listed(url), url)

    def test_openf1_and_coinbase_skip_robots_txt_only_for_their_own_endpoint(self):
        sh = load("source_harvest")
        robots_reads, requests = [], []
        original = sh.fetch_robots, sh.Http._request
        sh.fetch_robots = lambda origin, user_agent, **kwargs: (robots_reads.append(origin) or (403, ""))      # these sites refuse robots.txt
        sh.Http._request = lambda self, url, accept, max_bytes: (requests.append(url) or (200, {}, b"[]", False, 0.0))
        try:
            for host, listed, other in (("api.openf1.org", "/v1/sessions?year=2024&session_type=Race", "/v1/laps?session_key=1"),
                                        ("api.exchange.coinbase.com", "/products/BTC-USD/candles?granularity=86400", "/products/BTC-USD/book")):
                with self.subTest(host=host):
                    robots_reads.clear()
                    requests.clear()
                    client = sh.Http(gap=0.0)
                    self.assertEqual(client.get("https://" + host + listed).status, 200)
                    self.assertEqual((robots_reads, requests), ([], ["https://" + host + listed]))       # on the list: robots.txt is not read
                    with self.assertRaises(sh.FetchError) as caught:                                       # not on the list: it is, and it blocks
                        client.get("https://" + host + other)
                    self.assertEqual(caught.exception.kind, "robots")
                    self.assertEqual(robots_reads, ["https://" + host])
                    self.assertEqual(requests, ["https://" + host + listed])                                # the other path was never requested
        finally:
            sh.fetch_robots, sh.Http._request = original

    def test_harvester_and_lab_share_one_rule(self):
        net, sh, pl = load("prophet_net"), load("source_harvest"), load("prophet_lab")
        self.assertIs(sh.API_ALLOWLIST, net.API_ALLOWLIST)
        self.assertIs(pl.net.API_ALLOWLIST, net.API_ALLOWLIST)
        self.assertIs(sh.Robots, net.Robots)
        self.assertIs(pl.net.Robots, net.Robots)

    def test_robots_txt_is_still_obeyed_for_everything_not_listed(self):
        sh = load("source_harvest")
        site = Site()
        try:
            http_client = sh.Http(gap=0.0)
            self.assertEqual(http_client.get(site.base + "/open.csv").body, b"ok")           # robots.txt allows it
            with self.assertRaises(sh.FetchError) as caught:
                http_client.get(site.base + "/private/data.csv")                               # robots.txt forbids it
            self.assertEqual(caught.exception.kind, "robots")
            self.assertNotIn("/private/data.csv", site.hits)
        finally:
            site.close()

    def test_there_is_no_switch_that_skips_robots_txt(self):
        removed = "--robots-" + "exempt"                                # the switch that used to exist; written in two parts on purpose
        r = run(self.data, removed, "open-meteo.com", "--list", script="source_harvest.py")
        self.assertEqual(r.returncode, 2)
        self.assertIn("unrecognized arguments", r.stderr)

    def test_disabled_and_waiting_sources_are_never_fetched(self):
        sh = load("source_harvest")
        rows = json.loads((REPO / "sources" / "sources.json").read_text(encoding="utf-8"))["sources"]
        for row in rows:
            if row.get("disabled"):
                self.assertTrue(sh.skip_reason(row).startswith(("disabled:", "youtube:")), row["id"])
        waiting = [row for row in rows if "contact email" in (row.get("license_or_terms", "") + " " + row.get("notes", "")).lower()]
        self.assertTrue(waiting)
        for row in waiting:                                             # a source that needs a contact email is off until one is set
            self.assertTrue(row.get("disabled"), row["id"])
            self.assertIn("contact email", row["disabled_reason"])
        gdelt = next(row for row in rows if row["id"] == "api_gdelt_doc")
        self.assertTrue(gdelt.get("disabled"))
        self.assertIn("429", gdelt["disabled_reason"])

    def test_a_pending_approval_stops_the_harvester(self):
        sha = hashlib.sha256((REPO / "sources" / "sources.json").read_bytes()).hexdigest()
        record = self.data / "notebooks" / "sources" / "APPROVAL.json"
        record.parent.mkdir(parents=True)
        record.write_text(json.dumps({"status": "pending", "sourcesSha256": sha}), encoding="utf-8")
        r = run(self.data, "--area", "all", "--list", script="source_harvest.py")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not 'approved'", r.stdout + r.stderr)
        record.write_text(json.dumps({"status": "approved", "at": "2026-01-01", "by": "test", "sourcesSha256": sha}), encoding="utf-8")
        r = run(self.data, "--area", "all", "--list", script="source_harvest.py")
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_an_approval_record_without_a_status_does_not_count(self):
        sha = hashlib.sha256((REPO / "sources" / "sources.json").read_bytes()).hexdigest()
        record = self.data / "notebooks" / "sources" / "APPROVAL.json"
        record.parent.mkdir(parents=True)
        record.write_text(json.dumps({"at": "2026-01-01", "by": "test", "sourcesSha256": sha}), encoding="utf-8")      # no "status" key
        r = run(self.data, "--area", "all", "--list", script="source_harvest.py")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("has no status", r.stdout + r.stderr)
        self.assertEqual(run(self.data, "--status", script="approve_sources.py").returncode, 1)       # the approval script agrees

    def test_the_robots_decision_table(self):
        net = load("prophet_net")
        forbidden = urllib.parse.urlsplit("https://example.org/private/x")
        deny = "User-agent: *\nDisallow: /private\n"
        self.assertFalse(net.robots_from_answer(200, deny).allows(forbidden))              # a file that forbids it
        for status in (404, 410):
            self.assertTrue(net.robots_from_answer(status, deny).allows(forbidden), status)      # no file: no restrictions
        for status in (301, 304, 401, 403, 429, 500, 502, 503):                                 # cannot be read: blocked
            with self.assertRaises(net.RobotsUnreadable, msg=status):
                net.robots_from_answer(status, "")

    def test_a_robots_txt_that_cannot_be_read_blocks_the_site(self):
        sh = load("source_harvest")
        for status in (429, 500, 502, 503, 403):
            with self.subTest(status=status):
                site = Site(robots_status=status)
                try:
                    with self.assertRaises(sh.FetchError) as caught:
                        sh.Http(gap=0.0).get(site.base + "/open.csv")
                    self.assertEqual(caught.exception.kind, "robots")
                    self.assertNotIn("/open.csv", site.hits)
                finally:
                    site.close()

    def test_a_robots_txt_that_is_plainly_not_there_means_no_restrictions(self):
        sh = load("source_harvest")
        for status in (404, 410):
            with self.subTest(status=status):
                site = Site(robots_status=status)
                try:
                    self.assertEqual(sh.Http(gap=0.0).get(site.base + "/private/data.csv").body, b"secret")
                finally:
                    site.close()

    def test_robots_txt_redirects_that_never_end_block_the_site(self):
        sh, net = load("source_harvest"), load("prophet_net")
        site = Site(robots_loop=True)
        try:
            with self.assertRaises(sh.FetchError) as caught:
                sh.Http(gap=0.0).get(site.base + "/open.csv")
            self.assertEqual(caught.exception.kind, "robots")                              # fail closed, not "no restrictions"
            self.assertNotIn("/open.csv", site.hits)
            self.assertEqual(site.hits.count("/robots.txt"), net.MAX_ROBOTS_REQUESTS)       # it gave up after a fixed number of tries
        finally:
            site.close()

    def test_youtube_rows_are_listed_but_not_fetchable(self):
        sh = load("source_harvest")
        rows = json.loads((REPO / "sources" / "sources.json").read_text(encoding="utf-8"))["sources"]
        youtube = [row for row in rows if row["kind"] == "youtube"]
        self.assertTrue(youtube)
        for row in youtube:
            self.assertTrue(row.get("disabled"), row["id"])
            self.assertIn("not harvested", row["disabled_reason"])
            self.assertTrue(sh.skip_reason(row).startswith("youtube:"), row["id"])

    def test_there_is_no_youtube_harvester_in_this_copy(self):
        self.assertFalse((REPO / ("channel_" + "harvest.py")).exists())
        self.assertIn("YouTube channels are not harvested", (REPO / "README.md").read_text(encoding="utf-8"))

    def test_a_zip_member_over_the_cap_is_refused_unless_the_source_asks_for_a_head_sample(self):
        sh = load("source_harvest")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("day.export.CSV", "".join("%d\tcol\t%d\n" % (n, n * 2) for n in range(600)))
        data = buf.getvalue()
        self.assertEqual(sh.ZIP_MEMBER_CAP, 40_000_000)                  # the cap itself is unchanged
        self.assertEqual(sh.STREAM_HEAD_SOURCES, {"csv_gdelt_v1_daily_events": 400})
        self.assertEqual(len(sh.COLUMN_HINTS["csv_gdelt_v1_daily_events"]), 58)
        original = sh.ZIP_MEMBER_CAP
        sh.ZIP_MEMBER_CAP = 1000                                         # a small cap stands in for the real one
        try:
            with self.assertRaises(sh.FetchError) as caught:
                sh.unpack_text(data, "https://example.org/day.zip")      # a source that did not ask: refused as before
            self.assertIn("larger than 40 MB", str(caught.exception))
            text, note = sh.unpack_text(data, "https://example.org/day.zip", head_lines=25)
        finally:
            sh.ZIP_MEMBER_CAP = original
        self.assertEqual(len(text.splitlines()), 25)
        self.assertTrue(text.startswith("0\tcol\t0\n"))
        self.assertIn("first 25 of 600 lines", note)


class LabDownloads(unittest.TestCase):
    """Every download of the Pattern Lab passes the same robots.txt rule as the harvester, and GDELT's rate limit is kept."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.pl = load("prophet_lab")
        self.lab = self.pl.Lab(root=self._tmp.name)
        self.lab.wait_429 = 0.01                                        # the real first wait is 10 s; the test does not wait for it

    def tearDown(self):
        self._tmp.cleanup()

    def test_an_allowed_page_is_fetched_and_robots_txt_is_read_once(self):
        site = Site()
        try:
            first = self.lab.fetch(site.base + "/open.csv", min_gap=0.0)
            second = self.lab.fetch(site.base + "/open2.csv", min_gap=0.0)
            self.assertEqual((first.data, second.data), (b"ok", b"ok"))
            self.assertEqual(site.hits.count("/robots.txt"), 1)
            last = json.loads((self.lab.raw_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(last["allowed_by"], "robots.txt allows")
            receipts = (self.lab.raw_dir / "robots.jsonl").read_text(encoding="utf-8").splitlines()       # one receipt per site read
            self.assertEqual(len(receipts), 1)
            self.assertEqual(json.loads(receipts[0])["origin"], site.base)
        finally:
            site.close()

    def test_a_page_robots_txt_forbids_is_never_requested(self):
        site = Site()
        try:
            with self.assertRaises(self.pl.FetchError) as caught:
                self.lab.fetch(site.base + "/private/data.csv", min_gap=0.0)
            self.assertIn("disallows", str(caught.exception))
            self.assertNotIn("/private/data.csv", site.hits)
        finally:
            site.close()

    def test_a_redirect_is_checked_before_it_is_followed(self):
        site = Site()
        try:
            with self.assertRaises(self.pl.FetchError) as caught:
                self.lab.fetch(site.base + "/to-private", min_gap=0.0)
            self.assertIn("disallows", str(caught.exception))
            self.assertIn("/to-private", site.hits)
            self.assertNotIn("/private/data.csv", site.hits)
            self.assertEqual(self.lab.fetch(site.base + "/to-open", min_gap=0.0).data, b"ok")
        finally:
            site.close()

    def test_a_robots_txt_that_cannot_be_read_is_a_failure_not_a_pass(self):
        for status in (429, 500, 502, 503, 403):
            with self.subTest(status=status):
                site = Site(robots_status=status)
                try:
                    with self.assertRaises(self.pl.FetchError) as caught:
                        self.lab.fetch(site.base + "/open.csv", min_gap=0.0)
                    self.assertIn("HTTP %d" % status, str(caught.exception))
                    self.assertNotIn("/open.csv", site.hits)
                finally:
                    site.close()

    def test_a_robots_txt_that_is_plainly_not_there_means_no_restrictions(self):
        for status in (404, 410):
            with self.subTest(status=status):
                site = Site(robots_status=status)
                try:
                    self.assertEqual(self.lab.fetch(site.base + "/private/data.csv", min_gap=0.0).data, b"secret")
                finally:
                    site.close()

    def test_robots_txt_redirects_that_never_end_block_the_download(self):
        net = load("prophet_net")
        site = Site(robots_loop=True)
        try:
            with self.assertRaises(self.pl.FetchError) as caught:
                self.lab.fetch(site.base + "/open.csv", min_gap=0.0)
            self.assertIn("could not be read", str(caught.exception))
            self.assertNotIn("/open.csv", site.hits)
            self.assertEqual(site.hits.count("/robots.txt"), net.MAX_ROBOTS_REQUESTS)
        finally:
            site.close()

    def test_the_gap_between_requests_counts_the_robots_txt_read(self):
        site = Site()
        try:
            self.lab.fetch(site.base + "/open.csv", min_gap=0.4)
            self.assertGreaterEqual(site.stamps[site.hits.index("/open.csv")] - site.stamps[site.hits.index("/robots.txt")], 0.35)
        finally:
            site.close()

    def test_an_official_api_on_the_list_needs_no_robots_txt(self):
        for url in ("https://archive-api.open-meteo.com/v1/archive?latitude=1&longitude=2",
                    "https://api.open-meteo.com/v1/forecast?latitude=1&longitude=2",
                    "https://api.gdeltproject.org/api/v2/doc/doc?query=theme%3AX&mode=timelinevol&format=json&timespan=3m",
                    "https://api.openf1.org/v1/sessions?year=2024&session_type=Race",
                    "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=86400"):
            self.assertEqual(self.lab._robots_basis(url, 0.0), ("official API (allowlist)", 0.0), url)   # decided with no request at all

    def test_openf1_and_coinbase_other_paths_still_go_through_robots_txt(self):
        net = load("prophet_net")
        reads, original = [], net.fetch_robots
        net.fetch_robots = lambda origin, user_agent, **kwargs: (reads.append(origin) or (403, ""))      # these sites refuse robots.txt
        try:
            for host, other in (("api.openf1.org", "/v1/laps?session_key=1"), ("api.exchange.coinbase.com", "/products/BTC-USD/book")):
                with self.subTest(host=host):
                    reads.clear()
                    with self.assertRaises(self.pl.FetchError) as caught:
                        self.lab._robots_basis("https://" + host + other, 0.0)
                    self.assertIn("HTTP 403", str(caught.exception))
                    self.assertEqual(reads, ["https://" + host])
        finally:
            net.fetch_robots = original

    def test_429_is_retried_once_for_gdelt_then_the_download_stops_cleanly(self):
        site = Site()
        try:
            with self.assertRaises(self.pl.FetchError) as caught:
                self.lab.fetch(site.base + "/busy", min_gap=0.0, retry_429=1)
            self.assertEqual(caught.exception.code, 429)
            self.assertIn("giving up cleanly", str(caught.exception))
            self.assertEqual(site.hits.count("/busy"), 2)               # the first request and exactly one retry
        finally:
            site.close()

    def test_429_default_is_two_retries(self):
        site = Site()
        try:
            with self.assertRaises(self.pl.FetchError):
                self.lab.fetch(site.base + "/busy", min_gap=0.0)
            self.assertEqual(site.hits.count("/busy"), 3)
        finally:
            site.close()

    def test_geopolitics_keeps_gdelts_rate_limit(self):
        geo = load("prophet_lab_geopolitics")
        self.assertGreaterEqual(geo.GDELT_MIN_GAP, 6.0)
        seen = []

        def stop_here(url, **kwargs):
            seen.append(kwargs)
            raise self.pl.FetchError("stop here")
        self.lab.fetch = stop_here
        self.lab.opts = {"gdelt_doc_api": True}                         # the DOC API is only the fallback; it is off unless asked for
        with self.assertRaises(self.pl.FetchError):
            geo.ADAPTER.fetch_history(self.lab)
        self.assertEqual(seen[0]["retry_429"], 1)
        self.assertGreaterEqual(seen[0]["min_gap"], 6.0)


def fake_export_zip(day, counts, extra=()):
    """One GDELT 1.0 daily export in miniature: a zip with one tab-separated member and a row of 58 columns per event. Column 28 holds
    the event root code and column 56 the day the event was added. counts is {"14": 3, ...}; extra are raw lines added at the end."""
    lines = []
    for root, n in counts.items():
        for _ in range(n):
            cols = ["x"] * 58
            cols[0] = str(1000 + len(lines))
            cols[26] = cols[27] = root + "0"
            cols[28] = root
            cols[56] = day
            lines.append("\t".join(cols))
    lines.extend(extra)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(day + ".export.CSV", "\n".join(lines) + "\n")
    return buf.getvalue()


class EventSite:
    """A throwaway stand-in for the GDELT daily-export host, on this machine: /events/YYYYMMDD.export.CSV.zip for each day it holds,
    404 for every other day, and a robots.txt (404 by default, like the real host). Every request path is recorded."""

    def __init__(self, days=None, robots_status=404, robots_body=b""):
        self.days, self.hits, outer = dict(days or {}), [], self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                outer.hits.append(self.path)
                if self.path == "/robots.txt":
                    return self.reply(robots_status, robots_body)
                day = self.path.rsplit("/", 1)[-1].split(".", 1)[0]
                if self.path.startswith("/events/") and self.path.endswith(".export.CSV.zip") and day in outer.days:
                    return self.reply(200, outer.days[day], {"Content-Type": "application/zip"})
                return self.reply(404, b"not found")

            def reply(self, code, body, headers=None):
                self.send_response(code)
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass
        self.server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.base = "http://127.0.0.1:%d" % self.server.server_port
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    @property
    def zip_hits(self):
        return [p for p in self.hits if p.endswith(".export.CSV.zip")]

    def url(self):
        return self.base + "/events/{day}.export.CSV.zip"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class GdeltExport(unittest.TestCase):
    """The geopolitics domain reads GDELT's daily event files: each is downloaded through the lab (robots.txt first), counted while it is
    unzipped, recorded as one row of the counts file and deleted. These tests use a throwaway site on this machine, never the internet."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.pl = load("prophet_lab")
        self.geo = load("prophet_lab_geopolitics")
        self.lab = self.pl.Lab(root=self._tmp.name)
        self.original_url = self.geo.EXPORT_URL
        for module, name in ((self.geo, "EXPORT_URL"), (self.geo, "EXPORT_MIN_GAP"), (self.geo, "_say"), (self.geo, "CHUNK"),
                             (self.geo, "MEMBER_LIMIT"), (self.geo, "MIN_EVENTS"), (self.pl, "utcnow")):
            self.addCleanup(setattr, module, name, getattr(module, name))        # every module setting a test changes is put back
        self.geo.EXPORT_MIN_GAP = 0.0
        self.geo._say = lambda message: None
        self.sites = []

    def tearDown(self):
        for site in self.sites:
            site.close()
        self._tmp.cleanup()

    def site(self, days=None, **kwargs):
        site = EventSite(days, **kwargs)
        self.sites.append(site)
        self.geo.EXPORT_URL = site.url()
        return site

    @staticmethod
    def day(n):
        """The n-th day of October 2026 as (date, 'YYYYMMDD')."""
        d = dt.date(2026, 10, n)
        return d, d.strftime("%Y%m%d")

    def test_events_are_counted_by_root_code_while_unzipping(self):
        extra = ["only\tthree\tcolumns", "\t".join(["x"] * 28 + ["99"] + ["y"] * 29)]        # not an event row; a root code outside 01 to 20
        got = self.geo.count_export_zip(fake_export_zip("20261001", {"14": 3, "18": 2, "19": 4, "16": 1, "04": 5, "20": 1}, extra))
        roots = got["roots"]
        self.assertEqual((roots[13], roots[17], roots[18], roots[15], roots[3], roots[19]), (3, 2, 4, 1, 5, 1))
        self.assertEqual((got["events"], got["bad"], got["lines"]), (16, 2, 18))

    def test_lines_that_cross_a_read_boundary_are_counted_once(self):
        data = fake_export_zip("20261001", {"14": 7, "18": 5, "01": 9}, ["a\tb"])
        whole = self.geo.count_export_zip(data)
        for size in (1, 7, 100, 301):
            self.geo.CHUNK = size                                        # the member is read this many bytes at a time
            self.assertEqual(self.geo.count_export_zip(data), whole, size)

    def test_a_file_that_is_not_a_readable_export_is_refused(self):
        with self.assertRaises(ValueError):
            self.geo.count_export_zip(b"<html>busy</html>")
        two = io.BytesIO()
        with zipfile.ZipFile(two, "w") as zf:
            zf.writestr("a.CSV", "x\n")
            zf.writestr("b.CSV", "y\n")
        with self.assertRaises(ValueError):
            self.geo.count_export_zip(two.getvalue())
        self.geo.MEMBER_LIMIT = 500                                      # a zip that unpacks to more than the limit is refused, not read on
        with self.assertRaises(ValueError) as caught:
            self.geo.count_export_zip(fake_export_zip("20261001", {"14": 50}))
        self.assertIn("larger than", str(caught.exception))

    def test_a_counted_day_is_never_downloaded_again_and_its_zip_is_deleted(self):
        (d1, s1), (d2, s2) = self.day(1), self.day(2)
        z1, z2 = fake_export_zip(s1, {"14": 3, "04": 7}), fake_export_zip(s2, {"14": 5, "19": 2})
        site = self.site({s1: z1, s2: z2})
        rows, error = self.geo.load_days(self.lab, [d1, d2], dt.date(2026, 10, 3), [])
        self.assertIsNone(error)
        self.assertEqual(sorted(rows), [d1, d2])
        self.assertEqual(len(site.zip_hits), 2)
        self.assertEqual((rows[d1].events, rows[d1].roots[13], rows[d2].events, rows[d2].roots[18]), (10, 3, 7, 2))
        self.assertEqual(rows[d1].sha256, hashlib.sha256(z1).hexdigest())
        self.assertEqual(rows[d1].url, site.base + "/events/" + s1 + ".export.CSV.zip")
        self.assertEqual(list(self.lab.raw_dir.rglob("*.zip")), [])                    # the zips are gone; their counts, URLs and hashes stay
        counts = (Path(self._tmp.name) / "gdelt_daily_counts.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(counts), 3)                                                # a header and one row per day
        self.assertTrue(counts[0].startswith("day,events,lines,bad_lines,zip_bytes,member_bytes,sha256,url,fetched_at,root01"))
        again, _ = self.geo.load_days(self.pl.Lab(root=self._tmp.name), [d1, d2], dt.date(2026, 10, 3), [])       # a new run, same folder
        self.assertEqual(len(site.zip_hits), 2)                                         # the site was not asked for either day again
        self.assertEqual(site.hits.count("/robots.txt"), 1)
        self.assertEqual(again[d2].roots, rows[d2].roots)

    def test_a_day_without_a_file_is_reported_and_left_out_not_fatal(self):
        today = dt.date(2026, 10, 10)
        site = self.site({self.day(n)[1]: fake_export_zip(self.day(n)[1], {"14": n, "04": 5}) for n in (1, 2, 3, 4, 5, 6, 8)})
        days = [self.day(n)[0] for n in range(1, 10)]                    # the 7th has no file (a gap); the 9th, yesterday, is not published yet
        notes = []
        rows, error = self.geo.load_days(self.lab, days, today, notes)
        self.assertIsNone(error)
        self.assertEqual(sorted(d.day for d in rows), [1, 2, 3, 4, 5, 6, 8])
        self.assertIn("not published yet (HTTP 404), left out: 2026-10-09", notes)
        self.assertIn("no file for 2026-10-07 (HTTP 404): a gap in the series, skipped", notes)
        before = len(site.zip_hits)
        self.geo.load_days(self.lab, days, today, [])
        self.assertEqual(len(site.zip_hits) - before, 2)                 # only the two days with no file are asked for again
        site.days[self.day(9)[1]] = fake_export_zip(self.day(9)[1], {"14": 9, "04": 5})        # the file appears
        rows, _ = self.geo.load_days(self.lab, days, today, [])
        self.assertIn(self.day(9)[0], rows)

    def test_robots_txt_is_read_first_and_obeyed_and_the_host_is_not_on_the_api_list(self):
        net = load("prophet_net")
        self.assertIsNone(net.api_entry(urllib.parse.urlsplit(self.original_url.format(day="20261023"))))     # a data file, not an API: robots.txt applies
        d, s = self.day(1)
        site = self.site({s: fake_export_zip(s, {"14": 3})})
        rows, _ = self.geo.load_days(self.lab, [d], dt.date(2026, 10, 3), [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(site.hits[0], "/robots.txt")                    # read before the first file is asked for
        receipts = (self.lab.raw_dir / "robots.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual((len(receipts), json.loads(receipts[0])["status"]), (1, 404))
        last = json.loads((self.lab.raw_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(last["allowed_by"], "robots.txt allows")
        with tempfile.TemporaryDirectory() as other:                     # a robots.txt that forbids the files: nothing is downloaded
            blocked = self.site({s: fake_export_zip(s, {"14": 3})}, robots_status=200, robots_body=b"User-agent: *\nDisallow: /events/\n")
            notes = []
            rows, error = self.geo.load_days(self.pl.Lab(root=other), [d], dt.date(2026, 10, 3), notes)
            self.assertEqual((rows, blocked.zip_hits), ({}, []))
            self.assertIn("disallows", str(error))
            self.assertIn("downloading stopped", " ".join(notes))

    def test_a_day_with_too_few_events_is_counted_but_left_out_of_the_series(self):
        self.geo.MIN_EVENTS = 50                                         # a real day has over 100,000 events; the test days are tiny
        first, served = dt.date(2026, 8, 16), {}
        for i in range(48):                                              # 16 Aug to 2 Oct; the 3rd, yesterday, is not published yet
            s = (first + dt.timedelta(days=i)).strftime("%Y%m%d")
            served[s] = fake_export_zip(s, {"04": 60, "14": 3 + i % 4, "19": 8, "16": 2})
        served["20260901"] = fake_export_zip("20260901", {"14": 2, "04": 3})        # a broken, partial file
        self.site(served)
        self.pl.utcnow = lambda: dt.datetime(2026, 10, 4, 8, 0, 0, tzinfo=dt.timezone.utc)
        self.lab.opts = {"days": 49}
        hist = self.geo.ADAPTER.fetch_history(self.lab)
        tp = hist.topics["protest_events"]
        self.assertEqual((len(tp.v), len(hist.inputs)), (48, 47))
        self.assertTrue(math.isnan(tp.v[16]))                            # 1 September is a hole in the series, not a tiny number
        self.assertIn("fewer than 50 coded events (a broken or partial file), left out of the series: 2026-09-01", hist.notes)
        counted = (Path(self._tmp.name) / "gdelt_daily_counts.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(counted), 1 + 48)                           # the count of that day is still kept

    def test_history_backtest_live_call_and_score_work_from_the_counts(self):
        self.geo.MIN_EVENTS = 50
        now = dt.datetime(2026, 10, 25, 8, 0, 0, tzinfo=dt.timezone.utc)
        self.pl.utcnow = lambda: now
        first, served, table = dt.date(2026, 8, 16), {}, {}
        for i in range((dt.date(2026, 10, 23) - first).days + 1):       # 69 days, 16 Aug to 23 Oct; the 24th is not published yet
            d = first + dt.timedelta(days=i)
            s = d.strftime("%Y%m%d")
            table[s] = {"04": 60, "01": 20, "14": 3 + (i * 7) % 5 + (4 if d.weekday() >= 5 else 0), "19": 8 + (i * 3) % 4, "16": 2 + i % 3}
            served[s] = fake_export_zip(s, table[s])
        site = self.site(served)
        self.lab.opts = {"days": 70}
        adapter = self.geo.ADAPTER
        hist = adapter.fetch_history(self.lab)
        self.assertEqual(list(hist.topics), ["protest_events", "violence_events", "sanctions_events"])
        tp = hist.topics["protest_events"]
        self.assertEqual((tp.start, len(tp.v), len(hist.inputs)), (first, 69, 69))
        self.assertAlmostEqual(tp.v[0], 100.0 * table["20260816"]["14"] / sum(table["20260816"].values()))      # protests as a percent of the day's events
        self.assertAlmostEqual(hist.topics["violence_events"].v[1], 100.0 * table["20260817"]["19"] / sum(table["20260817"].values()))
        self.assertEqual(hist.inputs[0]["sha256"], hashlib.sha256(served["20260816"]).hexdigest())
        self.assertIn("not published yet (HTTP 404), left out: 2026-10-24", hist.notes)
        result = adapter.backtest(self.lab)
        self.assertGreater(len(result["rows"]), 30)
        self.assertTrue(all(r["target"] == "up_h2" and 0.0 <= r["pred"] <= 1.0 and set(r["bases"]) == {"up_rate", "weekday_rate"} for r in result["rows"]))
        self.assertIn("gdelt_daily_counts.csv", " ".join(result["notes"]))
        subjects = adapter.live_subjects(hist, now)
        self.assertEqual(sorted(s["key"] for s in subjects), ["geopolitics:protest_events:up:2026-10-25", "geopolitics:sanctions_events:up:2026-10-25",
                                                              "geopolitics:violence_events:up:2026-10-25"])
        s = next(x for x in subjects if x["meta"]["topic"] == "protest_events")
        self.assertEqual((s["resolves_at"], s["meta"]["source"], s["meta"]["ref_date"], s["meta"]["target_date"], s["meta"]["roots"]),
                         ("2026-10-26T12:00:00Z", "gdelt_export_v1", "2026-10-23", "2026-10-25", [14]))
        feats = adapter.features(hist, s, s["as_of"])
        rec = self.pl.build_call(self.lab, adapter, s, adapter.predict(hist, s, s["as_of"], feats), adapter.baseline(hist, s, s["as_of"], feats), s["_inputs"], now)
        self.assertEqual((rec["domain"], rec["target"], rec["kind"], len(rec["inputs"])), ("geopolitics", "up_h2", "binary", 69))
        # the next day's file appears, and two days later the call is scored from the source that made it, whatever flag is set now
        late = fake_export_zip("20261025", {"04": 60, "01": 20, "14": 40, "19": 8, "16": 2})
        site.days["20261025"] = late
        self.pl.utcnow = lambda: now + dt.timedelta(days=2)
        missing = dict(rec, id=rec["id"] + "-later", meta=dict(rec["meta"], target_date="2026-10-26"))
        done = adapter.resolve_calls(self.pl.Lab(root=self._tmp.name, opts={"gdelt_doc_api": True}), [rec, missing])
        self.assertEqual(done[rec["id"]][0], 1.0)                        # 40 of 130 events is a bigger share than the 5 to 11 of about 100 on the 23rd
        self.assertEqual(done[rec["id"]][1], [{"url": site.base + "/events/20261025.export.CSV.zip", "sha256": hashlib.sha256(late).hexdigest()}])
        self.assertEqual(done[missing["id"]], (None, []))                # no file for that day yet: waiting, not wrong
        self.assertEqual(site.zip_hits.count("/events/20261025.export.CSV.zip"), 1)      # only the target day was fetched to score the call

    def test_the_doc_api_is_off_unless_asked_for(self):
        seen = []

        def stop_here(url, **kwargs):
            seen.append(url)
            raise self.pl.FetchError("stop here")
        self.lab.fetch = stop_here
        with self.assertRaises(self.pl.FetchError):
            self.geo.ADAPTER.fetch_history(self.lab)
        self.assertEqual(len(seen), 1)                                   # the first download failed, so the others were not tried
        self.assertTrue(seen[0].startswith("https://data.gdeltproject.org/events/"), seen)
        seen.clear()
        self.lab.opts = {"gdelt_doc_api": True}
        with self.assertRaises(self.pl.FetchError):
            self.geo.ADAPTER.fetch_history(self.lab)
        self.assertTrue(seen[0].startswith("https://api.gdeltproject.org/api/v2/doc/doc?"), seen)

    def outside_requests(self):
        """Replace the lab's network function: a request to this machine goes through, a request to any other address is recorded and
        refused. Returns the list of refused addresses, so a test can prove that nothing was asked of the internet."""
        real, outside = self.lab._http_get, []

        def guarded(url, min_gap, timeout, max_bytes, retry_429=2):
            if urllib.parse.urlsplit(url).hostname == "127.0.0.1":
                return real(url, min_gap, timeout, max_bytes, retry_429)
            outside.append(url)
            raise self.pl.FetchError("test: a request to %s was refused" % url)
        self.lab._http_get = guarded
        return outside

    @staticmethod
    def calls_to_score():
        """Three sealed calls that are due: one made from the daily event files, one made from the DOC API, and one old call with no source."""
        export = {"id": "geopolitics:protest_events:up:2026-10-26", "domain": "geopolitics",
                  "meta": {"source": "gdelt_export_v1", "topic": "protest_events", "roots": [14], "ref_value": 5.0, "target_date": "2026-10-26"}}
        doc = {"id": "geopolitics:protest:up:2026-01-12", "domain": "geopolitics",
               "meta": {"source": "gdelt_doc_api", "topic": "protest", "query": "theme:PROTEST", "ref_value": 0.05, "ref_date": "2026-01-10",
                        "target_date": "2026-01-12"}}
        bare = {"id": "geopolitics:terror:up:2026-01-12", "domain": "geopolitics",
                "meta": {"topic": "terror", "query": "theme:TERROR", "ref_value": 0.05, "ref_date": "2026-01-10", "target_date": "2026-01-12"}}
        return export, doc, bare

    def test_scoring_makes_no_doc_api_request_unless_the_flag_is_given(self):
        export, doc, bare = self.calls_to_score()
        self.geo.MIN_EVENTS = 50                                         # a real day has over 100,000 events; the test day is tiny
        site = self.site({"20261026": fake_export_zip("20261026", {"04": 60, "01": 20, "14": 40, "19": 8, "16": 2})})      # 40 of 130 events
        self.pl.utcnow = lambda: dt.datetime(2026, 10, 28, 8, 0, 0, tzinfo=dt.timezone.utc)
        said = []
        self.geo._say = said.append
        outside = self.outside_requests()                                # no flag: the lab was started with no switches
        self.assertEqual(self.lab.opts, {})
        done = self.geo.ADAPTER.resolve_calls(self.lab, [export, doc, bare])
        self.assertEqual(outside, [])                                    # nothing was asked of any address outside this machine, the DOC API included
        self.assertEqual(done[export["id"]][0], 1.0)                     # the call made from the daily files is still scored from its target day's file
        self.assertEqual(site.zip_hits, ["/events/20261026.export.CSV.zip"])
        self.assertEqual(done[doc["id"]], (None, []))                    # a call the DOC API made waits, unscored
        self.assertEqual(done[bare["id"]], (None, []))                   # so does an old call with no source
        self.assertTrue(any("--gdelt-doc-api" in message for message in said), said)       # and the log says why they wait

    def test_scoring_asks_the_doc_api_when_the_flag_is_given(self):
        _export, doc, _bare = self.calls_to_score()
        self.lab.opts = {"gdelt_doc_api": True}
        outside = self.outside_requests()
        with self.assertRaises(self.pl.FetchError):                      # the refusal above stops it after the first request
            self.geo.ADAPTER.resolve_calls(self.lab, [doc])
        self.assertEqual(len(outside), 1)
        self.assertTrue(outside[0].startswith("https://api.gdeltproject.org/api/v2/doc/doc?"), outside)

    def test_the_doc_api_door_stays_shut_for_any_caller_without_the_flag(self):
        outside = self.outside_requests()
        with self.assertRaises(self.pl.LabError) as caught:
            self.geo.ADAPTER._fetch_doc(self.lab)
        self.assertIn("--gdelt-doc-api", str(caught.exception))
        self.assertEqual(outside, [])

    def test_the_gdelt_doc_api_flag_reaches_the_score_command(self):
        data = Path(self._tmp.name) / "cli"
        data.mkdir()
        code = ("import json\nimport prophet_lab as pl\nseen = []\n"
                "pl.cmd_score = lambda lab, args: seen.append(lab.opts) or 0\n"
                "for extra in ([], ['--gdelt-doc-api']):\n"
                "    pl.main(['score', '--lab-dir', %r] + extra)\n"
                "print(json.dumps(seen))\n" % str(data))
        r = run_code(data, code)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout.strip().splitlines()[-1]), [{}, {"gdelt_doc_api": True}])

    def test_lab_domains_shows_the_export_and_the_flags_are_wired(self):
        data = Path(self._tmp.name) / "data"
        data.mkdir()
        r = run(data, "lab", "domains")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("https://data.gdeltproject.org/events/", r.stdout)
        self.assertIn("OFF unless --gdelt-doc-api", r.stdout)
        helped = run(data, "lab", "backtest", "--help")
        self.assertIn("--days", helped.stdout)
        self.assertIn("--gdelt-doc-api", helped.stdout)
        r = run(data, "lab", "backtest", "geopolitics", "--days", "10")  # refused before anything is downloaded
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("at least 40", r.stdout + r.stderr)


def fake_receipt(uri=b"https://alice.btc.calendar.opentimestamps.org", nonce=b"\x11" * 16):
    """What a calendar answers to a submitted hash, in miniature: append a nonce, SHA-256, then a pending attestation."""
    payload = bytes([len(uri)]) + uri                       # an attestation payload is the address, with its length in front
    return bytes([0xF0, len(nonce)]) + nonce + bytes([0x08, 0x00]) + bytes.fromhex("83dfe30d2ef90c8e") + bytes([len(payload)]) + payload


class CalendarSite:
    """A throwaway stand-in for an OpenTimestamps calendar, on this machine. It records every request and answers as told."""

    def __init__(self, status=200, body=None, redirect=None):
        self.requests, outer = [], self
        answer = fake_receipt() if body is None else body

        class Handler(http.server.BaseHTTPRequestHandler):
            def handle_any(self):
                size = int(self.headers.get("Content-Length") or 0)
                outer.requests.append({"method": self.command, "path": self.path, "headers": dict(self.headers.items()),
                                       "body": self.rfile.read(size) if size else b""})
                if redirect:
                    self.send_response(302)
                    self.send_header("Location", redirect)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(status)
                self.send_header("Content-Length", str(len(answer)))
                self.end_headers()
                self.wfile.write(answer)

            do_POST = do_GET = handle_any

            def log_message(self, *args):
                pass
        self.server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.base = "http://127.0.0.1:%d" % self.server.server_port
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class LabHarness(unittest.TestCase):
    """A throwaway lab folder. The clock is fixed and the OpenTimestamps network call is replaced, so nothing here uses the internet."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name) / "lab"
        self.pl = load("prophet_lab")
        self.now = dt.datetime(2026, 10, 25, 10, 0, 0, tzinfo=dt.timezone.utc)
        self.sent = []                                      # (url, digest) for every submission the code tried
        self.answers = {}                                   # url -> (status, body) or an exception; the default is a good receipt
        self.real_submit = self.pl.ots_submit
        for name in ("ots_submit", "OTS_RETRY_WAIT", "OTS_CALENDARS", "utcnow"):
            self.addCleanup(setattr, self.pl, name, getattr(self.pl, name))
        self.pl.OTS_RETRY_WAIT = 0.0
        self.pl.utcnow = lambda: self.now

        def fake(url, digest, timeout=30.0):
            self.sent.append((url, digest))
            answer = self.answers.get(url, (200, fake_receipt()))
            if isinstance(answer, Exception):
                raise answer
            return answer
        self.pl.ots_submit = fake

    def tearDown(self):
        self._tmp.cleanup()

    def lab(self, *args):
        """Run `prophet lab <args>` in this process against the throwaway folder. Returns (exit code, stdout, stderr)."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = self.pl.main([*args, "--lab-dir", str(self.dir)])
        return code, out.getvalue(), err.getvalue()

    def call(self, question="Will it rain in Oslo today?", prob="0.3", resolves="today", domain="demo"):
        code, out, err = self.lab("call", "--domain", domain, "--question", question, "--resolves", resolves, "--prob", prob)
        self.assertEqual(code, 0, err)
        return out.split()[1]                               # the first line is "Sealed <id>"

    def head(self):
        return self.pl.verify_lab(self.pl.Lab(root=self.dir)).log.head

    def page(self):
        self.assertEqual(self.lab("record")[0], 0)
        return (self.dir / "record.html").read_text(encoding="utf-8")


class Anchor(LabHarness):
    """prophet lab anchor: what is sent, what is saved, and what happens when a calendar says no."""

    def receipts(self):
        folder = self.dir / "anchors"
        return sorted(folder.glob("*.ots-pending")) if folder.is_dir() else []

    def log_lines(self):
        path = self.dir / "anchors.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    def test_the_hash_sent_is_the_chain_head_and_only_that(self):
        self.call()
        self.call("Will it snow in Oslo today?")
        head = self.head()
        code, out, err = self.lab("anchor")
        self.assertEqual(code, 0, err)
        self.assertEqual([url for url, _digest in self.sent], list(self.pl.OTS_CALENDARS))
        for _url, digest in self.sent:
            self.assertEqual(digest, bytes.fromhex(head))     # the 32 raw bytes of the chain head, nothing else
            self.assertEqual(len(digest), 32)
        self.assertIn("Chain head (seal of the newest call): " + head, out)
        self.assertIn("PENDING", out)
        self.assertIn("Nothing is confirmed yet", out)

    def test_the_receipt_is_a_real_opentimestamps_file_and_the_log_says_what_happened(self):
        self.call()
        head = self.head()
        alice, bob = b"https://alice.btc.calendar.opentimestamps.org", b"https://bob.btc.calendar.opentimestamps.org"
        first, second = self.pl.OTS_CALENDARS
        self.answers = {first: (200, fake_receipt(alice, b"\x01" * 16)), second: (200, fake_receipt(bob, b"\x02" * 16))}
        self.assertEqual(self.lab("anchor")[0], 0)
        (receipt,) = self.receipts()
        self.assertEqual(receipt.name, "2026-10-25_%s.ots-pending" % head[:12])
        data = receipt.read_bytes()
        self.assertTrue(data.startswith(self.pl.OTS_MAGIC))
        digest, tree = self.pl.ots_read_file(data)
        self.assertEqual(digest.hex(), head)                  # the receipt stamps the chain head
        self.assertEqual(sorted(self.pl.ots_pending_uris(tree)), sorted(u.decode() for u in (alice, bob)))      # both calendars are in it
        (line,) = self.log_lines()
        self.assertEqual((line["head"], line["date"], line["calls"], line["ok"], line["file"]), (head, "2026-10-25", 1, True, "anchors/" + receipt.name))
        self.assertEqual([(c["url"], c["status"], c["ok"]) for c in line["calendars"]], [(u, 200, True) for u in self.pl.OTS_CALENDARS])
        for c, answer in zip(line["calendars"], (fake_receipt(alice, b"\x01" * 16), fake_receipt(bob, b"\x02" * 16))):
            self.assertEqual((c["bytes"], c["sha256"]), (len(answer), hashlib.sha256(answer).hexdigest()))
            self.assertIn(answer, data)                           # each server's answer is in the receipt exactly as it came

    def test_a_dry_run_sends_nothing_and_writes_nothing(self):
        self.call()
        head = self.head()
        code, out, err = self.lab("anchor", "--dry-run")
        self.assertEqual(code, 0, err)
        self.assertEqual(self.sent, [])
        self.assertFalse((self.dir / "anchors").exists())
        self.assertFalse((self.dir / "anchors.jsonl").exists())
        self.assertIn(head, out)
        for url in self.pl.OTS_CALENDARS:
            self.assertIn(url, out)
        self.assertIn("nothing was sent", out)

    def test_a_calendar_that_says_no_is_an_error_and_nothing_is_saved(self):
        self.call()
        for status, tries in ((403, 1), (404, 1), (429, 2), (500, 2), (503, 2)):       # a 429 or a 5xx gets one retry; anything else none
            with self.subTest(status=status):
                self.sent.clear()
                self.answers = {url: (status, b"no") for url in self.pl.OTS_CALENDARS}
                code, out, err = self.lab("anchor")
                self.assertEqual(code, 2)
                self.assertIn("no calendar accepted", err)
                self.assertIn("HTTP %d" % status, err)
                self.assertEqual(len(self.sent), 2 * tries)
                self.assertEqual((self.receipts(), self.log_lines()), ([], []))
        self.sent.clear()
        self.answers = {url: self.pl.FetchError("could not reach " + url) for url in self.pl.OTS_CALENDARS}      # no network at all
        code, out, err = self.lab("anchor")
        self.assertEqual(code, 2)
        self.assertEqual(len(self.sent), 4)                                              # one try and one retry each, not more
        self.assertEqual((self.receipts(), self.log_lines()), ([], []))

    def test_a_200_that_is_not_a_receipt_is_an_error(self):
        self.call()
        good = fake_receipt()
        for body in (b"<html>Sign in to the Wi-Fi</html>", good[:-5], good + b"x", b""):
            with self.subTest(body=body[:12]):
                self.answers = {url: (200, body) for url in self.pl.OTS_CALENDARS}
                code, out, err = self.lab("anchor")
                self.assertEqual(code, 2)
                self.assertIn("no calendar accepted", err)
                self.assertEqual((self.receipts(), self.log_lines()), ([], []))

    def test_one_calendar_down_still_anchors_and_the_failure_is_recorded(self):
        self.call()
        down = self.pl.OTS_CALENDARS[1]
        self.answers = {down: (500, b"oops")}
        code, out, err = self.lab("anchor")
        self.assertEqual(code, 0, err)
        self.assertIn("WARNING", out)
        (line,) = self.log_lines()
        self.assertEqual([(c["ok"], c["status"]) for c in line["calendars"]], [(True, 200), (False, 500)])
        self.assertEqual(len(self.receipts()), 1)

    def test_a_record_that_fails_verification_is_not_anchored(self):
        self.call()
        path = self.dir / "calls.jsonl"
        path.write_text(path.read_text(encoding="utf-8").replace("Oslo", "Rome"), encoding="utf-8")
        code, out, err = self.lab("anchor")
        self.assertEqual(code, 2)
        self.assertIn("fails verification", err)
        self.assertEqual((self.sent, self.receipts()), ([], []))

    def test_an_empty_record_is_not_anchored(self):
        code, out, err = self.lab("anchor")
        self.assertEqual(code, 2)
        self.assertIn("nothing is sealed yet", err)
        self.assertEqual(self.sent, [])

    def test_the_same_chain_head_is_not_sent_twice_and_a_receipt_is_never_overwritten(self):
        self.call()
        self.assertEqual(self.lab("anchor")[0], 0)
        self.sent.clear()
        code, out, err = self.lab("anchor")
        self.assertEqual(code, 0, err)
        self.assertEqual(self.sent, [])
        self.assertIn("already anchored", out)
        self.call("Will it snow in Oslo today?")                 # a new call makes a new chain head, which is sent
        self.assertEqual(self.lab("anchor")[0], 0)
        self.assertEqual((len(self.sent), len(self.receipts())), (2, 2))
        self.sent.clear()
        self.assertEqual(self.lab("anchor", "--again")[0], 0)    # on request the same head is sent again, into a new file
        self.assertEqual((len(self.sent), len(self.receipts()), len(self.log_lines())), (2, 3, 3))

    def test_only_the_two_listed_addresses_can_receive_the_hash(self):
        net = load("prophet_net")
        for url in self.pl.OTS_CALENDARS:
            self.assertIsNotNone(net.api_entry(urllib.parse.urlsplit(url)), url)
        for url in ("https://a.pool.opentimestamps.org/", "https://a.pool.opentimestamps.org/timestamp/00ff",
                    "http://a.pool.opentimestamps.org/digest", "https://c.pool.opentimestamps.org/digest",
                    "https://opentimestamps.org/digest", "https://a.pool.opentimestamps.org:8443/digest",
                    "https://a.pool.opentimestamps.org/digest/../x"):
            self.assertIsNone(net.api_entry(urllib.parse.urlsplit(url)), url)
        self.call()
        self.pl.OTS_CALENDARS = ("https://example.org/digest",)  # not on the list: nothing is sent
        code, out, err = self.lab("anchor")
        self.assertEqual(code, 2)
        self.assertIn("API_ALLOWLIST", err)
        self.assertEqual(self.sent, [])

    def test_the_post_carries_the_raw_hash_and_a_redirect_is_not_followed(self):
        digest = hashlib.sha256(b"a chain head").digest()
        site = CalendarSite()
        try:
            self.assertEqual(self.real_submit(site.base + "/digest", digest, 5), (200, fake_receipt()))
            (request,) = site.requests
            headers = {k.lower(): v for k, v in request["headers"].items()}
            self.assertEqual((request["method"], request["path"], request["body"]), ("POST", "/digest", digest))
            self.assertEqual(headers["user-agent"], self.pl.ANCHOR_UA)
            self.assertEqual((headers["accept"], headers["content-length"]), ("application/vnd.opentimestamps.v1", "32"))
        finally:
            site.close()
        elsewhere = CalendarSite()
        moved = CalendarSite(redirect=elsewhere.base + "/digest")
        try:
            status, _body = self.real_submit(moved.base + "/digest", digest, 5)
            self.assertEqual(status, 302)                         # reported, not followed
            self.assertEqual((len(moved.requests), elsewhere.requests), (1, []))
        finally:
            moved.close()
            elsewhere.close()
        with socket.socket() as probe:                            # a port nothing listens on: a network failure is an exception, not a status
            probe.bind(("127.0.0.1", 0))
            dead = probe.getsockname()[1]
        with self.assertRaises(self.pl.FetchError):
            self.real_submit("http://127.0.0.1:%d/digest" % dead, digest, 2)

    def test_the_record_page_says_pending_until_the_receipt_file_holds_a_bitcoin_block(self):
        self.call()
        self.assertEqual(self.lab("anchor")[0], 0)
        page = self.page()
        self.assertIn("Status: pending", page)
        self.assertIn("Nothing is confirmed yet", page)
        self.assertNotIn("Bitcoin block", page)
        (receipt,) = self.receipts()                              # what `ots upgrade` would do: the file gains a Bitcoin attestation
        digest, tree = self.pl.ots_read_file(receipt.read_bytes())
        upgraded = self.pl.ots_merge(tree, {"atts": [(self.pl.OTS_BITCOIN, self.pl._varuint_bytes(123456))], "ops": []})
        receipt.write_bytes(self.pl.ots_file(digest, upgraded))
        page = self.page()
        self.assertIn("claims Bitcoin block 123456", page)
        self.assertIn("This page does not check that", page)
        self.assertNotIn("Status: pending", page)

    def test_an_anchored_head_that_is_not_in_the_record_is_flagged(self):
        self.call()
        entry = {"ok": True, "head": "00" * 32, "date": "2026-10-24", "calendars": [{"ok": True}], "file": "anchors/none.ots-pending"}
        (self.dir / "anchors.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
        self.assertIn("that chain head is not in this record", self.page())


class RecordPage(LabHarness):
    """record.html: one self-contained page, built from the sealed files only."""

    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.found, self.tags = [], []

        def handle_starttag(self, tag, attrs):
            self.tags.append(tag)
            for name, value in attrs:
                if name in ("src", "href", "srcset", "action", "data", "poster"):
                    self.found.append((tag, name, value))

    def three_calls(self):
        win = self.call("Will the die show a 5 or a 6? (first)", "0.3", resolves="tomorrow")
        miss = self.call("Will the die show a 5 or a 6? (second)", "0.9")
        self.call("Will the die show a 5 or a 6? (third)", "0.5")
        self.assertEqual(self.lab("resolve", win, "--outcome", "no")[0], 0)
        self.assertEqual(self.lab("resolve", miss, "--outcome", "no")[0], 0)

    def test_the_page_is_one_self_contained_file_and_shows_each_state(self):
        self.three_calls()
        page = self.page()
        parser = self.Links()
        parser.feed(page)
        self.assertEqual([item for item in parser.found if not item[2].startswith("#")], [])      # no link, picture or file outside the page
        for tag in ("script", "link", "img", "iframe", "form", "object", "embed"):
            self.assertNotIn(tag, parser.tags)
        for text in ("url(", "@import", "http://", "https://"):
            self.assertNotIn(text, page)
        self.assertIn('name="viewport"', page)
        self.assertIn("prefers-color-scheme:light", page)
        for text in ("Graded: win", "Graded: miss", "Waiting", "graded before its date", "Words used here", "Brier score"):
            self.assertIn(text, page)
        for number, word in ((3, "Sealed"), (2, "Graded"), (1, "Waiting")):
            self.assertIn("<b>%d</b><span>%s</span>" % (number, word), page)
        self.assertIn(self.head(), page)
        self.assertIn("All seals OK", page)
        for tag in ("article", "section", "details", "main", "header", "footer"):
            self.assertEqual(page.count("<%s" % tag), page.count("</%s>" % tag), tag)

    def test_the_page_follows_the_sealed_files_and_says_tampered_when_a_seal_breaks(self):
        self.three_calls()
        self.assertEqual(self.page().count('<article class="call'), 3)
        path = self.dir / "calls.jsonl"
        path.write_text(path.read_text(encoding="utf-8").replace("(second)", "(2nd)"), encoding="utf-8")
        code, out, err = self.lab("record")
        self.assertEqual(code, 1)
        page = (self.dir / "record.html").read_text(encoding="utf-8")
        self.assertIn("TAMPERED", page)
        self.assertIn('role="alert"', page)
        self.assertNotIn("Graded: miss", page)                    # a call that fails its seal is not shown as a result

    def test_text_that_looks_like_markup_is_escaped(self):
        self.call('<script>alert(1)</script> "quoted" & more', "0.4")
        page = self.page()
        self.assertNotIn("<script>alert", page)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", page)

    def test_record_md_is_still_written_for_ask(self):
        self.call()
        self.assertEqual(self.lab("record")[0], 0)
        text = (self.dir / "RECORD.md").read_text(encoding="utf-8")
        self.assertTrue(text.startswith('---\ntitle: "Prophet Pattern Lab record"\nchannel: "lab"\n'))
        self.assertIn("Anchor: Not anchored yet", text)

    def test_an_empty_lab_still_makes_a_page_that_says_how_to_start(self):
        page = self.page()
        self.assertIn("Nothing is sealed yet", page)
        self.assertIn("--resolves today", page)

    def test_a_domain_that_is_not_running_gets_one_line_and_no_empty_card(self):
        self.call()
        page = self.page()
        self.assertIn("Not running:", page)
        self.assertIn("Horse racing (needs data source)", page)
        self.assertNotIn("<h3>Horse racing</h3>", page)
        self.assertIn("<h3>Weather</h3>", page)                  # a domain that can run, with no call yet, says how to start
        self.assertIn("prophet lab predict weather", page)


class Resolves(unittest.TestCase):
    def test_words_and_dates(self):
        pl = load("prophet_lab")
        today = dt.date(2026, 10, 25)
        self.assertEqual(pl.parse_resolves("today", today), today)
        self.assertEqual(pl.parse_resolves(" Tomorrow ", today), dt.date(2026, 10, 26))
        self.assertEqual(pl.parse_resolves("2027-01-02", today), dt.date(2027, 1, 2))
        for junk in ("yesterday", "", "2026-13-40", "soon"):
            with self.assertRaises(pl.LabError):
                pl.parse_resolves(junk, today)


class NoNetwork(unittest.TestCase):
    """The README names the commands that never use the network. Here each of them runs with sockets that log any attempt."""

    HOOK = """import os, socket
def refuse(*args, **kwargs):
    with open(os.environ["NET_ATTEMPTS"], "a") as f:
        f.write(repr(args)[:200] + chr(10))
    raise OSError("the network is blocked in this test")
for name in ("connect", "connect_ex"):
    setattr(socket.socket, name, refuse)
socket.getaddrinfo = refuse
socket.create_connection = refuse
"""

    def test_the_commands_the_readme_calls_offline_make_no_connection(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, hook, log = Path(tmp) / "data", Path(tmp) / "hook", Path(tmp) / "attempts.log"
            data.mkdir()
            hook.mkdir()
            (hook / "sitecustomize.py").write_text(self.HOOK, encoding="utf-8")
            env = {"PYTHONPATH": str(hook), "NET_ATTEMPTS": str(log)}
            control = subprocess.run([sys.executable, "-c", "import urllib.request; urllib.request.urlopen('http://example.org', timeout=2)"],
                                     env=dict(os.environ, **env), capture_output=True, text=True, timeout=60)
            self.assertNotEqual(control.returncode, 0)            # the block works: a plain download fails and is logged
            self.assertTrue(log.exists() and log.read_text(encoding="utf-8").strip())
            log.unlink()
            question = ["--question", "Will it rain?", "--resolves", "today", "--prob", "0.4"]
            for args in (("ask", "--stats"), ("lab", "domains"), ("lab", "call", "--domain", "demo", *question), ("lab", "list"),
                         ("lab", "verify"), ("lab", "record"), ("lab", "anchor", "--dry-run")):
                r = run(data, *args, env_extra=env)
                self.assertEqual(r.returncode, 0, "%s: %s" % (" ".join(args), r.stderr))
            call_id = run(data, "lab", "list", env_extra=env).stdout.split()[0]
            r = run(data, "lab", "resolve", call_id, "--outcome", "no", env_extra=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertFalse(log.exists(), log.read_text(encoding="utf-8") if log.exists() else "")


class Readme(unittest.TestCase):
    """The README has to say what the program does. These tests read it and check it against the program."""

    def setUp(self):
        self.text = (REPO / "README.md").read_text(encoding="utf-8")
        self._tmp = tempfile.TemporaryDirectory()
        self.data = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_it_leads_with_the_idea_and_not_with_practice_results(self):
        top = "\n".join(self.text.splitlines()[:45])
        for line in ("Prophet is a receipt book for predictions.", "Make the call. Lock it in. Get graded.",
                     "Prophet writes your predictions down before they happen, then scores them against the obvious guess.",
                     "Free, on your own computer, no account."):
            self.assertIn(line, top)
        for word in ("football", "weather", "stock", "beats"):
            self.assertNotIn(word, top.lower())                   # practice results are a small note further down, never the headline
        self.assertNotIn("benchmark", self.text.lower())          # a lazy guess is a simple reference point, not a benchmark

    def test_the_command_count_matches_the_front_door(self):
        prophet = load("prophet")
        found = re.search(r"Prophet has (\w+) commands? \(([^)]*)\) and one script", self.text)
        self.assertIsNotNone(found)
        words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
        self.assertEqual(words.get(found.group(1), found.group(1)), len(prophet.SKILLS))
        for name in prophet.SKILLS:
            self.assertIn("`%s`" % name, found.group(2))

    def test_the_practice_table_shows_every_kind_including_the_one_that_did_not_work(self):
        table = [line for line in self.text.splitlines() if line.startswith("| ")]
        for kind in ("Weather:", "Football:", "Stock markets:", "World events:"):
            self.assertTrue(any(line.startswith("| " + kind) for line in table), kind)
        self.assertTrue(any("Not ahead of the lazy guess" in line for line in table))

    def test_the_ask_example_is_what_ask_prints(self):
        sample = re.search(r"Each result is three lines:\n\n```\n(.*?)\n```", self.text, re.S).group(1).splitlines()
        note = re.search(r"between two `---` lines:\n\n```\n(---\n.*?)\n```", self.text, re.S).group(1) + "\n"
        folder = self.data / "notebooks" / "my-notes"
        folder.mkdir(parents=True)
        (folder / "call-with-sam.md").write_text(note, encoding="utf-8")
        r = run(self.data, "ask", "-c", "my-notes", "radar")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.splitlines()[:3], sample)
        shown = json.loads(run(self.data, "ask", "--json", "-c", "my-notes", "radar").stdout)["results"][0]
        for key in ("channel", "title", "date", "kind", "start", "link", "score", "snippet", "file"):    # what --json adds, as the README says
            self.assertIn(key, shown)

    def test_the_demo_is_what_the_commands_print(self):
        def shape(text):
            text = re.sub(r"demo:manual:\d{4}-\d\d-\d\d:[0-9a-f]{8}", "<id>", text)
            text = re.sub(r"\d{4}-\d\d-\d\dT23:59:59Z", "<resolves>", text)
            return re.sub(r"\b[0-9a-f]{64}\b", "<seal>", text)
        lines = self.text.splitlines()
        sealed = [shape(line) for line in lines if line.startswith(("Sealed demo:", "  question: Will my die", "  call 0.33"))]
        scored = [shape(line.strip()) for line in lines if line.strip().startswith("Scored demo:")]
        command = next(line for line in lines if line.startswith("python prophet.py lab call --domain demo"))
        r = run(self.data, *shlex.split(command)[2:])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual([shape(line) for line in r.stdout.splitlines()], sealed)
        call_id = r.stdout.split()[1]
        r = run(self.data, "lab", "resolve", call_id, "--outcome", "no")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual([shape(r.stdout.strip())], scored)


if __name__ == "__main__":
    unittest.main()
