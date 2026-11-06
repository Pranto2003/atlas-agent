"""Tools agents can call. Each agent receives only the tools it is allowed to use."""
from __future__ import annotations
import re
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
import httpx
from bs4 import BeautifulSoup
from .config import settings

_client: httpx.Client | None = None


def client() -> httpx.Client:
    global _client
    if _client is None:
        s = settings()
        _client = httpx.Client(timeout=s.http_timeout, follow_redirects=True,
                               headers={"User-Agent": s.user_agent, "Accept-Language": "en"})
    return _client


_cache: dict = {}
_host_lock = {"en.wikipedia.org": threading.Semaphore(2), "export.arxiv.org": threading.Semaphore(1)}
_last = {"export.arxiv.org": 0.0}


def _request(method: str, url: str, ttl: int = 600, **kw) -> httpx.Response:
    """HTTP with TTL cache, per-host concurrency limits and Retry-After handling for 429s."""
    key = (method, url, repr(sorted((kw.get("params") or kw.get("data") or {}).items())))
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    host = urllib.parse.urlparse(url).netloc
    sem = _host_lock.get(host)
    if sem:
        sem.acquire()
    try:
        if host == "export.arxiv.org":  # arXiv asks for ≤1 request / 3 s
            time.sleep(max(0.0, 3.0 - (time.time() - _last[host])))
            _last[host] = time.time()
        for attempt in range(3):
            r = client().request(method, url, **kw)
            if r.status_code == 429 and attempt < 2:
                time.sleep(min(float(r.headers.get("retry-after", 2 * (attempt + 1))), 8))
                continue
            break
        r.raise_for_status()
        _cache[key] = (time.time(), r)
        return r
    finally:
        if sem:
            sem.release()


def arxiv_search(query: str, limit: int = 3) -> list[dict]:
    r = _request("GET", "https://export.arxiv.org/api/query", params={
        "search_query": "all:" + " AND all:".join(query.split()[:6]), "max_results": limit, "sortBy": "relevance"})
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in ET.fromstring(r.text).findall("a:entry", ns):
        title = " ".join((e.findtext("a:title", "", ns)).split())
        link = e.findtext("a:id", "", ns).replace("http://", "https://")
        summary = " ".join((e.findtext("a:summary", "", ns)).split())
        if title and summary:
            out.append({"title": title, "url": link, "snippet": summary, "origin": "arxiv"})
    return out


def wikipedia_search(query: str, limit: int = 4) -> list[dict]:
    r = _request("GET", "https://en.wikipedia.org/w/api.php", params={
        "action": "query", "list": "search", "srsearch": query, "srlimit": limit,
        "format": "json", "utf8": 1})
    r.raise_for_status()
    out = []
    for h in r.json().get("query", {}).get("search", []):
        title = h["title"]
        snippet = BeautifulSoup(h.get("snippet", ""), "html.parser").get_text()
        out.append({"title": title, "snippet": snippet, "origin": "wikipedia",
                    "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))})
    return out


def wikipedia_text(title: str) -> str:
    r = _request("GET", "https://en.wikipedia.org/w/api.php", ttl=3600, params={
        "action": "query", "prop": "extracts", "explaintext": 1, "exsectionformat": "plain",
        "titles": title, "format": "json", "redirects": 1, "utf8": 1})
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    return next(iter(pages.values()), {}).get("extract", "") or ""


def web_search(query: str, limit: int = 4) -> list[dict]:
    r = _request("POST", "https://html.duckduckgo.com/html/", data={"q": query})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    out = []
    for a in soup.select("a.result__a"):
        href = a.get("href", "")
        if "uddg=" in href:
            href = urllib.parse.unquote(href.split("uddg=")[1].split("&")[0])
        if not href.startswith("http") or "duckduckgo.com" in href:
            continue
        sn = a.find_parent(class_="result")
        snippet = sn.select_one(".result__snippet").get_text(" ", strip=True) if sn and sn.select_one(".result__snippet") else ""
        out.append({"title": a.get_text(" ", strip=True), "url": href, "snippet": snippet, "origin": "web"})
        if len(out) >= limit:
            break
    return out


def fetch_page(url: str, max_chars: int = 60_000) -> str:
    r = client().get(url)
    r.raise_for_status()
    if "html" not in r.headers.get("content-type", "html"):
        raise ValueError("unsupported content type")
    soup = BeautifulSoup(r.text[:1_500_000], "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg"]):
        t.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    paras = [p.get_text(" ", strip=True) for p in root.find_all(["p", "li"])]
    return re.sub(r"[ \t]+", " ", "\n".join(p for p in paras if len(p) > 40))[:max_chars]


TOOLS = {"arxiv_search": arxiv_search, "wikipedia_search": wikipedia_search, "wikipedia_text": wikipedia_text,
         "web_search": web_search, "fetch_page": fetch_page}

# Least-privilege tool access per agent.
PERMISSIONS = {
    "planner": set(),
    "searcher": {"wikipedia_search", "web_search", "arxiv_search"},
    "extractor": {"wikipedia_text", "fetch_page"},
    "synthesizer": set(),
    "citer": set(),
}


def call(agent: str, tool: str, *a, **kw):
    if tool not in PERMISSIONS.get(agent, set()):
        raise PermissionError(f"agent '{agent}' may not use tool '{tool}'")
    return TOOLS[tool](*a, **kw)
