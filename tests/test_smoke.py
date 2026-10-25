"""Smoke tests: the front door starts, the help pages print, and everything Prophet writes goes to the data folder.

  python -B -m unittest discover -s tests -v
"""
import hashlib
import http.server
import importlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.parse
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
                    "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=86400"):
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
        with self.assertRaises(self.pl.FetchError):
            geo.ADAPTER.fetch_history(self.lab)
        self.assertEqual(seen[0]["retry_429"], 1)
        self.assertGreaterEqual(seen[0]["min_gap"], 6.0)


if __name__ == "__main__":
    unittest.main()
