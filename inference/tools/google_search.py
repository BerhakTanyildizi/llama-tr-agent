# -*- coding: utf-8 -*-
import html
import re
import urllib.parse
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .. import config
from html.parser import HTMLParser


# Copied verbatim from the training data. It was briefly reworded to nudge the
# model toward num_results=3; that failed on both counts - the model ignores it
# (see MIN_RESULTS below) and the edit made production, training and eval use
# three different schemas.
SCHEMA = {"type": "function", "function": {"name": "google_search", "description": "Search the web for current information about a query. Returns top-ranked result snippets.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query."}, "num_results": {"type": "integer", "description": "Number of results to return."}}, "required": ["query"]}}}


# Empty on purpose: 'query' is COMPOSED by the model, not extracted from the
# user's words, so grounding it in the conversation would reject valid queries.
GROUNDED_PARAMS: frozenset[str] = frozenset()


# --- trimming budget -------------------------------------------------------

# Vulkan prompt processing measured at ~175 tok/s, so every 1000 chars of
# observation costs ~1.4 s of prompt processing on EVERY subsequent turn, not
# just once. The budget is derived from USER WAITING TIME, not from VRAM.
#
# Measured with page_content enabled: a 3-source observation is ~4700 chars
# (~1175 tokens). With the orchestrator's 5734-token context budget that is
# about 4 searches before pruning starts. The richer answers are worth it, but
# these numbers are the reason the ceiling exists.

MIN_RESULTS = 3          # floor, see run(): the model habitually asks for 1
MAX_RESULTS = 5
MAX_SNIPPET_CHARS = 300
MAX_TITLE_CHARS = 120
MAX_URL_CHARS = 300
MAX_PAGE_CHARS = 900     # per source
MAX_TOTAL_CHARS = 6000
TIMEOUT = 12             # search request
PAGE_TIMEOUT = 5         # one page fetch; kept low because N run in parallel
MAX_PAGE_BYTES = 300_000 # never pull an entire PDF/large asset into memory
OVERFETCH = 2            # extra candidates fetched so blocked sites don't waste a slot


class _DDGParser(HTMLParser):
    """HTML Parser for DuckDuckGo Lite search results."""
    def __init__(self):
        super().__init__()
        self.results = []
        self._current = {}
        self._in_title = False
        self._in_snippet = False
        self._title_chunks = []
        self._snippet_chunks = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "a":
            classes = attrs_dict.get("class", "").split()
            if "result-link" in classes:
                self._current = {"url": attrs_dict.get("href", ""), "title": "", "snippet": ""}
                self._in_title = True
                self._title_chunks = []
                
        elif tag == "td":
            classes = attrs_dict.get("class", "").split()
            if "result-snippet" in classes:
                self._in_snippet = True
                self._snippet_chunks = []

    def handle_endtag(self, tag):
        if tag == "a" and self._in_title:
            self._in_title = False
            self._current["title"] = " ".join("".join(self._title_chunks).split())
            
        elif tag == "td" and self._in_snippet:
            self._in_snippet = False
            self._current["snippet"] = " ".join("".join(self._snippet_chunks).split())
            if self._current.get("url"):
                self.results.append(self._current)
                self._current = {}

    def handle_data(self, data):
        if self._in_title:
            self._title_chunks.append(data)
        elif self._in_snippet:
            self._snippet_chunks.append(data)


class _PageExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self._ignore = 0
        self._ignore_tags = {"script", "style", "noscript", "nav", "footer", "header", "svg"}
        
    def handle_starttag(self, tag, attrs):
        if tag in self._ignore_tags:
            self._ignore += 1

    def handle_endtag(self, tag):
        if tag in self._ignore_tags:
            self._ignore = max(0, self._ignore - 1)
        elif tag in {"p", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"}:
            self.text.append("\n")

    def handle_data(self, data):
        if self._ignore == 0:
            t = data.strip()
            if t:
                self.text.append(t + " ")

# Measured: a bare "Mozilla/5.0" is rejected with 403 by several sites that
# accept a full browser header set. It does not defeat Cloudflare-class blocks
# (cyberciti, stackoverflow stay closed) - that is what OVERFETCH is for.
_BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "tr,en-US;q=0.8,en;q=0.6",
}
# Wikimedia asks API clients to identify themselves rather than pose as a browser.
_WIKI_AGENT = "MyAgent/1.0 (local Turkish function-calling agent)"


def _read_url(url: str, max_chars: int) -> str:
    try:
        req = urllib.request.Request(url, headers=_BROWSER_HEADERS)
        with urllib.request.urlopen(req, timeout=PAGE_TIMEOUT) as r:
            charset = r.headers.get_content_charset() or "utf-8"
            # Bounded read: an unbounded r.read() would download a 50 MB PDF in
            # full only to throw all but max_chars of it away.
            html = r.read(MAX_PAGE_BYTES).decode(charset, "replace")
            
        parser = _PageExtractor()
        parser.feed(html)
        text = "".join(parser.text)
        
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(line for line in lines if line)
        
        if len(text) > max_chars:
            return text[:max_chars] + "... [truncated]"
        return text
    except Exception:
        return ""


def _domain(url: str) -> str:
    """Extract a normalized hostname from a URL."""
    try:
        domain = urllib.parse.urlparse(url).netloc.lower()

        if domain.startswith("www."):
            domain = domain[4:]

        return domain
    except Exception:
        return ""



# Defence in depth. Tavily's error bodies do not echo the key today, but those
# bodies are pasted into an observation the model sees and into --verbose logs.
# One upstream change or one proxy in the way would be enough to leak it, so
# anything key-shaped is redacted on the way out.
_SECRET = re.compile(r"\b(tvly|sk|pk|api)[-_][A-Za-z0-9_-]{8,}", re.I)


def _redact(text: str) -> str:
    return _SECRET.sub("[REDACTED]", text)


class SearchBlocked(RuntimeError):
    """Provider served a bot challenge instead of results.

    A separate type on purpose. MEASURED: DuckDuckGo answers HTTP 202 with a
    CAPTCHA page ("Select all squares containing a duck") after ~7 requests on
    /lite and after 2 on /html. The old code read that as a successful response
    with zero hits and told the model "nothing was found" - a silent lie that
    the model then relayed to the user as fact.
    """


def _blocked(status: int, page: str) -> bool:
    low = page.lower()
    return status != 200 or "captcha" in low or "following challenge" in low


def _ddg_lite(query: str) -> list[dict]:
    body = urllib.parse.urlencode({"q": query}).encode()
    req = urllib.request.Request(
        "https://lite.duckduckgo.com/lite/", data=body,
        headers={**_BROWSER_HEADERS, "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        status, page = r.status, r.read().decode("utf-8", "replace")
    if _blocked(status, page):
        raise SearchBlocked(f"duckduckgo returned HTTP {status} with a bot challenge")
    parser = _DDGParser()
    parser.feed(page)
    return [{"title": r["title"], "snippet": r["snippet"], "url": r["url"],
            "domain": _domain(r["url"])} for r in parser.results]


def _wikipedia(query: str) -> list[dict]:
    """Keyless, official, and it returns the article text itself.

    MEASURED 14/15 under a burst where DuckDuckGo was fully blocked, and the
    intro extract arrives clean (3205 chars for "Bergen"), so results from this
    provider need no page fetch at all. It covers encyclopedic questions, not
    news or prices - which is why it is a fallback, not the primary.
    """
    lang = "tr" if any(c in query for c in "çğıöşüÇĞİÖŞÜ") else "en"
    q = urllib.parse.urlencode({
        "action": "query", "format": "json", "prop": "extracts",
        "exintro": 1, "explaintext": 1, "generator": "search",
        "gsrsearch": query, "gsrlimit": MAX_RESULTS})
    req = urllib.request.Request(f"https://{lang}.wikipedia.org/w/api.php?{q}",
                                headers={"User-Agent": _WIKI_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise SearchBlocked("wikipedia rate limit (HTTP 429)") from None
        raise
    out = []
    for page in data.get("query", {}).get("pages", {}).values():
        extract = (page.get("extract") or "").strip()
        if not extract:
            continue
        title = page["title"]
        out.append({
            "title": title,
            "snippet": extract[:MAX_SNIPPET_CHARS],
            "url": f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
            "domain": f"{lang}.wikipedia.org",
            # Already the article body: no _read_url needed for this provider.
            "page_content": extract[:MAX_PAGE_CHARS],
        })
    return out



# Tavily returns Markdown, chrome included: the first ~200 characters of a page
# body were a logo image link and a "Skip to content" breadcrumb. With a 900
# character per-source budget, that is a quarter of the budget spent on
# navigation the model cannot use.
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
# No trailing \b: it would never match an alternative ending in punctuation,
# which is exactly how breadcrumbs look ("Home » Explore Norway » ...").
_CHROME = re.compile(
    r"^(skip to (main )?content|home\s*[»>|]|menu|share this|advertisement|"
    r"subscribe|sign in|log in|cookie|accept all|follow us|related posts?)", re.I)


def _strip_markdown(text: str) -> str:
    text = _MD_IMAGE.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    kept = []
    for line in text.splitlines():
        # Collapse runs of spaces BEFORE the length test. A stripped link row
        # ("Instagram     Facebook-f   Youtube") is mostly padding, and measuring
        # it uncollapsed pushed it past the 60-character threshold that was
        # meant to catch it.
        line = re.sub(r"[ \t]+", " ", line).strip().lstrip("#*>-|").strip()
        if not line or _CHROME.match(line):
            continue
        # Nav rows are wordless: "Instagram Facebook-f Youtube Search" carries no
        # sentence punctuation. Requiring punctuation below ~60 characters drops
        # them while keeping any real sentence, which almost always ends in one.
        if len(line) < 60 and not any(ch in line for ch in ".!?:"):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _tavily(query: str) -> list[dict]:
    """Primary provider when TAVILY_API_KEY is set.

    Chosen after measuring that HTML scraping is unworkable here: DuckDuckGo
    serves a CAPTCHA (HTTP 202) after ~7 requests on /lite and after 2 on
    /html, and Mojeek does the same. An official API has no bot challenge.

    It also returns the page text itself in `raw_content`, so results from
    this provider skip _read_url entirely - which removes the ~25% of page
    fetches that were being refused with HTTP 403.

    search_depth stays "basic": it costs one credit per search against a
    1000-credit monthly tier, and "advanced" costs two for depth the 900-char
    per-source budget would throw away anyway.
    """
    body = json.dumps({
        "query": query,
        "search_depth": "basic",
        "max_results": MAX_RESULTS + OVERFETCH,
        "include_raw_content": True,
        "include_answer": False,
        "include_images": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.tavily.com/search", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {TAVILY_API_KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = _redact(e.read().decode("utf-8", "replace"))[:160]
        if e.code in (401, 403):
            raise SearchBlocked(f"tavily rejected the API key (HTTP {e.code}): {detail}") from None
        if e.code == 429:
            raise SearchBlocked(f"tavily quota or rate limit (HTTP 429): {detail}") from None
        raise
    out = []
    for r in data.get("results", []):
        url = r.get("url", "")
        # raw_content is the page body; content is Tavily's relevant extract.
        # Falling back to content matters: raw_content comes back null for
        # pages Tavily could not fetch either.
        page_text = _strip_markdown(r.get("raw_content") or r.get("content") or "")
        out.append({
            "title": (r.get("title") or "").strip(),
            "snippet": _strip_markdown(r.get("content") or "")[:MAX_SNIPPET_CHARS],
            "url": url,
            "domain": _domain(url),
            "page_content": page_text[:MAX_PAGE_CHARS],
        })
    return out


# Order matters. Tavily first when a key exists: it is the only provider here
# that is not fighting a bot filter. DuckDuckGo second because it covers news
# and prices; Wikipedia last because it is reliable but encyclopedic only.
# Adding another keyed API means writing ONE function and listing it here.
TAVILY_API_KEY = config.get("TAVILY_API_KEY")

PROVIDERS = ((_tavily,) if TAVILY_API_KEY else ()) + (_ddg_lite, _wikipedia)


def _search(query: str) -> tuple[list[dict], str]:
    """Returns (results, provider name). Raises SearchBlocked if all are blocked."""
    blocked = []
    for provider in PROVIDERS:
        try:
            results = provider(query)
        except SearchBlocked as e:
            blocked.append(f"{provider.__name__}: {e}")
            continue
        except Exception as e:
            blocked.append(f"{provider.__name__}: {type(e).__name__}")
            continue
        if results:
            return results, provider.__name__
    if len(blocked) == len(PROVIDERS):
        raise SearchBlocked("; ".join(blocked))
    return [], "none"


def run(query: str, num_results: int = MIN_RESULTS) -> dict:
    try:
        raw, provider = _search(query)
    except SearchBlocked as e:
        # Every failure message states outright what the model must not do.
        # MEASURED: the model claimed "arama sonuçları incelendi" after a failed
        # search, and claimed it had searched when it never called the tool. The
        # no_results message already carried such an instruction and held up, so
        # the same wording is used on every failure path.
        return {"status": "blocked", "error": "search_blocked", "query": query,
                "message": ("Every search provider refused the request with a bot "
                            "challenge, so NO search was performed and there are no "
                            "results. Tell the user plainly that the search could not "
                            "be run and they may retry shortly. Do NOT claim you "
                            "searched, do NOT invent results, and do NOT answer the "
                            "question from prior knowledge as if it came from a search."),
                "detail": _redact(str(e))[:160]}
    except Exception as e:
        return {"status": "error", "error": "search_unavailable", "query": query,
                "message": ("The search tool failed, so NO search was performed. Tell "
                            "the user the search could not be completed. Do NOT claim "
                            "you searched and do NOT invent results."),
                "detail": _redact(f"{type(e).__name__}: {str(e)}")[:140]}

    # num_results is treated as a FLOOR, not a ceiling. In the training data
    # google_search was called with num_results=1 in 34 of 55 cases, so the
    # model asks for a single source out of habit and the answer ends up
    # resting on one page. How many sources are useful is the tool's call, not
    # the model's - the same principle as trimming: control lives in code.
    n = max(MIN_RESULTS, min(int(num_results or MIN_RESULTS), MAX_RESULTS))

    # One result per domain, so three results mean three sites - but only when
    # the provider actually spans several. Wikipedia returns every article under
    # one domain, and de-duplicating by domain there collapsed a good three-hit
    # answer down to a single source.
    single_domain = len({r.get("domain", "") for r in raw}) <= 1
    picked, seen = [], set()
    for r in raw:
        dedup_key = r.get("url", "") if single_domain else r.get("domain", "")
        if dedup_key and dedup_key in seen:
            continue
        picked.append(r)
        if dedup_key:
            seen.add(dedup_key)
        if len(picked) >= n + OVERFETCH:
            break

    # Fetched in parallel: sequentially, worst-case latency was N x PAGE_TIMEOUT.
    # Some providers (Wikipedia) hand back the article body already; fetching
    # their URL again would cost a request and return worse text.
    if picked:
        with ThreadPoolExecutor(max_workers=len(picked)) as pool:
            pages = list(pool.map(
                lambda r: r.get("page_content") or _read_url(r["url"], MAX_PAGE_CHARS),
                picked))
    else:
        pages = []

    # Measured: ~25% of pages answer 403 regardless of headers. Sources whose
    # body arrived are ranked first so a blocked site costs an extra fetch, not
    # a slot - otherwise the answer rests on a 300-char snippet.
    pairs = sorted(zip(picked, pages), key=lambda rp: not rp[1])[:n]

    results, total = [], 0
    for r, page in pairs:
        item = {
            "title": r["title"][:MAX_TITLE_CHARS],
            "snippet": r["snippet"][:MAX_SNIPPET_CHARS],
            "url": r["url"][:MAX_URL_CHARS],
            "domain": r.get("domain", ""),
        }
        if page:
            item["page_content"] = page

        size = sum(len(item.get(k, "")) for k in ("title", "snippet", "url", "page_content"))
        # `continue`, not `break`: a later result may be small enough to fit.
        if total + size > MAX_TOTAL_CHARS:
            continue
        results.append(item)
        total += size

    if not results:
        # I06 lesson: do not leave the emptiness silent, say it.
        return {
            "status": "no_results",
            "query": query,
            "results": [],
            "source_count": 0,
            "message": ("The search ran but returned no results. Tell the user "
                        "nothing was found. Do NOT answer from prior knowledge "
                        "and do NOT invent results."),
        }
    return {
        "status": "ok",
        "query": query,
        "provider": provider,
        "results": results,
        "source_count": len(results),
    }
