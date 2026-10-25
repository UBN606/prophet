"""Prophet network rules, shared by the source harvester (source_harvest.py) and the Pattern Lab (prophet_lab.py).

One rule for every automatic download of web data (project decision):

  * an official, documented API endpoint listed in API_ALLOWLIST may be fetched without reading the site's robots.txt;
  * every other URL may be fetched only if the site's robots.txt allows it. A robots.txt that cannot be read is a failure,
    not a pass: only a plain "not found" (404 or 410) means there are no restrictions; 429, any 5xx, a refusal or redirects
    that never end all block (robots_from_answer, fetch_robots).

This file is the one home of the list, of the matcher, of the robots.txt parser and of the decision what an answer to
GET /robots.txt means. It downloads nothing except fetch_robots(), which reads one robots.txt.
"""
import http.client
import re
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request
import zlib

# Official API endpoints that are read WITHOUT a robots.txt check.   (project decision)
# Why: these hosts publish a robots.txt written for web crawlers ("Disallow: /" or a rule over the API path), while the
# provider's own documentation tells programs to call exactly this API. We use the documented API endpoint only and never
# crawl another page of the site. A URL skips the check only if ALL of these hold: https on the default port, no credentials
# in the URL, the host spelled exactly as below, a path equal to the entry's path or (when the entry ends in "/") under it, no
# dot segments, and every required query pair present. A path segment written as * in an entry stands for exactly one non-empty
# segment (an id inside one documented endpoint, such as the product in Coinbase's candles) and nothing more. A redirect is a
# new URL and is matched again; one that leaves the
# list is robots-checked like any other. Everything not listed here still obeys robots.txt. Politeness is unchanged: the
# same User-Agent, the same gap between requests to one site, the same timeout and size caps.
# An endpoint belongs here only if it is an official, documented API. A download link on a website (a CSV button, a data
# file) is not one, however convenient: it goes through the robots.txt check like any page.
# Entry: (host, path, {required query: value}, the provider's documentation that makes it an API)
API_ALLOWLIST = (
    ("export.arxiv.org", "/api/", {}, "https://info.arxiv.org/help/api/user-manual.html"),
    ("api.open-meteo.com", "/v1/", {}, "https://open-meteo.com/en/docs"),
    ("air-quality-api.open-meteo.com", "/v1/", {}, "https://open-meteo.com/en/docs/air-quality-api"),
    ("archive-api.open-meteo.com", "/v1/", {}, "https://open-meteo.com/en/docs/historical-weather-api"),
    ("climate-api.open-meteo.com", "/v1/", {}, "https://open-meteo.com/en/docs/climate-api"),
    ("ensemble-api.open-meteo.com", "/v1/", {}, "https://open-meteo.com/en/docs/ensemble-api"),
    ("historical-forecast-api.open-meteo.com", "/v1/", {}, "https://open-meteo.com/en/docs/historical-forecast-api"),
    ("api.weather.gov", "/alerts/", {}, "https://www.weather.gov/documentation/services-web-api"),
    ("api.weather.gov", "/points/", {}, "https://www.weather.gov/documentation/services-web-api"),
    ("power.larc.nasa.gov", "/api/temporal/", {}, "https://power.larc.nasa.gov/docs/services/api/"),
    ("www.loc.gov", "/search/", {"fo": "json"}, "https://www.loc.gov/apis/json-and-yaml/"),
    ("query.wikidata.org", "/sparql", {"format": "json"}, "https://www.mediawiki.org/wiki/Wikidata_Query_Service/User_Manual"),
    ("en.wikipedia.org", "/api/rest_v1/", {}, "https://en.wikipedia.org/api/rest_v1/"),
    ("api.gdeltproject.org", "/api/v2/doc/doc", {}, "https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/"),
    # project decision: the two official, documented data APIs the sources api_openf1 and
    # api_coinbase_candles read. Only the endpoint each source uses; every other path on these hosts goes through robots.txt.
    ("api.openf1.org", "/v1/sessions", {}, "https://openf1.org/docs/"),
    ("api.exchange.coinbase.com", "/products/*/candles", {},
     "https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles"),
)


def api_entry(parts):
    """The API_ALLOWLIST entry that covers this exact URL (a urllib.parse.SplitResult), or None. Strict on purpose: see API_ALLOWLIST."""
    if parts.scheme != "https" or parts.username is not None or parts.password is not None:
        return None
    try:
        port = parts.port
    except ValueError:
        return None
    if port not in (None, 443):
        return None
    host = (parts.hostname or "").lower()
    path = urllib.parse.unquote(parts.path or "/")
    if "\\" in path or any(segment in (".", "..") for segment in path.split("/")):
        return None
    query = dict(urllib.parse.parse_qsl(parts.query, keep_blank_values=True))
    for entry in API_ALLOWLIST:
        entry_host, entry_path, required, _documentation = entry
        if "*" in entry_path.split("/"):                    # one documented endpoint with an id in it: every other segment must match exactly
            shape = "/".join("[^/]+" if segment == "*" else re.escape(segment) for segment in entry_path.split("/"))
            under = re.fullmatch(shape, path) is not None
        else:
            under = path == entry_path or (entry_path.endswith("/") and path.startswith(entry_path))
        if host == entry_host and under and all(query.get(name) == value for name, value in required.items()):
            return entry
    return None


class Robots:
    """robots.txt for one agent token: Allow/Disallow with * and $, longest match wins (Allow on ties), Crawl-delay."""

    def __init__(self, text=None, token="prophet"):
        self.rules, self.delay = [], None
        self.token = token.lower()
        if text:
            self._parse(text)

    def _parse(self, text):
        groups, current, after_agent = [], None, False
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            field, _, value = line.partition(":")
            field, value = field.strip().lower(), value.strip()
            if field == "user-agent":
                if current is None or not after_agent:
                    current = {"agents": [], "rules": [], "delay": None}
                    groups.append(current)
                current["agents"].append(value.lower())
                after_agent = True
                continue
            after_agent = False
            if current is None:
                continue
            if field in ("allow", "disallow"):
                if field == "disallow" and not value:
                    continue
                current["rules"].append((field == "allow", value))
            elif field == "crawl-delay":
                try:
                    current["delay"] = float(value)
                except ValueError:
                    pass
        specific = [g for g in groups if any(a and a != "*" and a in self.token for a in g["agents"])]
        chosen = specific or [g for g in groups if "*" in g["agents"]]
        for group in chosen:
            for allow, pattern in group["rules"]:
                end = pattern.endswith("$")
                body = pattern[:-1] if end else pattern
                rx = "^" + "".join(".*" if c == "*" else re.escape(c) for c in body) + ("$" if end else "")
                self.rules.append((allow, re.compile(rx), len(pattern), ("Allow: " if allow else "Disallow: ") + pattern))
            if group["delay"] is not None:
                self.delay = group["delay"] if self.delay is None else max(self.delay, group["delay"])

    def rule_for(self, parts):
        """(allowed, the robots.txt line that decided it or None)."""
        target = (parts.path or "/") + ("?" + parts.query if parts.query else "")
        best = None
        for allow, rx, length, text in self.rules:
            if rx.match(target) and (best is None or length > best[1] or (length == best[1] and allow)):
                best = (allow, length, text)
        return (True, None) if best is None else (best[0], best[2])

    def allows(self, parts):
        return self.rule_for(parts)[0]


class RobotsUnreachable(Exception):
    """robots.txt could not be read at all (network error, timeout, bad TLS, redirects that never end). Nothing may be fetched."""


class RobotsUnreadable(Exception):
    """robots.txt answered, but not in a way that tells us the rules (429, 5xx, a refusal...). Nothing may be fetched."""

    def __init__(self, status, message, transient):
        super().__init__(message)
        self.status, self.transient = status, transient


# The only answers that mean "this site has no robots.txt". RFC 9309 lets a crawler treat every 4xx like this; we do not: a 403 or a 401
# can be a site refusing us, so only a plain "not found" lifts the restrictions. To follow the RFC to the letter, add 400, 401, 403, 405.
NOT_FOUND = (404, 410)


def robots_from_answer(status, text, token="prophet"):
    """The rules of one site from the answer to GET /robots.txt (the one decision for the harvester and the Pattern Lab).

      2xx        the file: parse it.
      404, 410   the file does not exist (NOT_FOUND): no restrictions (RFC 9309).
      anything else, 429 and every 5xx included (the site is refusing or failing), and also 400/401/403/405 and a redirect that led
                 nowhere: we cannot know the rules, and that is a block, not a pass. Raises RobotsUnreadable; the caller must not fetch."""
    if 200 <= status < 300:
        return Robots(text, token)
    if status in NOT_FOUND:
        return Robots(None, token)
    raise RobotsUnreadable(status, "robots.txt answered HTTP %d, so the rules cannot be known; nothing is fetched" % status,
                           transient=(status == 429 or status >= 500))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect())
MAX_ROBOTS_REQUESTS = 6          # the first request plus five redirects (RFC 9309 asks for at least five); after that it is a failure


def _default_open(request, timeout):
    return _OPENER.open(request, timeout=timeout)


def fetch_robots(origin, user_agent, timeout=20.0, max_bytes=500_000, opener=None, on_request=None):
    """GET <origin>/robots.txt and return (status, text); an error answer comes back as its status with an empty text.

    Sends the headers some CDNs insist on before they answer at all (Accept, Accept-Encoding, Accept-Language) and reads at most
    max_bytes. Redirects are followed by hand, at most MAX_ROBOTS_REQUESTS requests in all; redirects that never end raise
    RobotsUnreachable (fail closed), as does a network error or timeout. opener(request, timeout) is the caller's way of opening a
    request (the harvester's has a CA-bundle fallback); it must not follow redirects itself. on_request() is called before every
    request so that the caller can count them."""
    open_request = opener or _default_open
    url = origin.rstrip("/") + "/robots.txt"
    for _ in range(MAX_ROBOTS_REQUESTS):
        if on_request is not None:
            on_request()
        request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "text/plain, */*;q=0.1",
                                                       "Accept-Encoding": "gzip", "Accept-Language": "en"})
        try:
            with open_request(request, timeout) as response:
                raw = response.read(max_bytes + 1)[:max_bytes]
                packed = "gzip" in (response.headers.get("Content-Encoding") or "").lower()
                status = response.status
        except urllib.error.HTTPError as error:
            code, location = error.code, (error.headers.get("Location") if error.headers else None)
            error.close()
            if code in (301, 302, 303, 307, 308) and location:
                url = urllib.parse.urljoin(url, location)
                continue
            return code, ""
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, ssl.SSLError, http.client.HTTPException, OSError) as error:
            raise RobotsUnreachable("%s: %s" % (type(error).__name__, getattr(error, "reason", None) or error))
        if packed:
            try:
                raw = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw, max_bytes)
            except zlib.error as error:
                raise RobotsUnreachable("robots.txt did not decompress: %s" % error)
        return status, raw.decode("utf-8", "replace")
    raise RobotsUnreachable("robots.txt redirects %d times without ever answering" % (MAX_ROBOTS_REQUESTS - 1))
