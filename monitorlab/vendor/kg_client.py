# kg_client.py
from __future__ import annotations
import asyncio, json, os, time, hashlib
from functools import lru_cache
from typing import Optional, Dict, List, Tuple

try:
    import aiohttp
except Exception:
    aiohttp = None  # optional async

DBPEDIA_SPARQL = "https://dbpedia.org/sparql"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

def _sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

class KGClient:
    """
    Minimal KG client with:
      - DBpedia + Wikidata SPARQL
      - in-memory LRU cache
      - optional on-disk cache (JSON files)
      - async (aiohttp) or sync (requests-fallback) modes
    """
    def __init__(self, cache_dir: Optional[str] = ".kg_cache", timeout: int = 8):
        self.cache_dir = cache_dir
        self.timeout = timeout
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    # ---------- Cache helpers ----------
    def _cache_get(self, key: str) -> Optional[dict]:
        if not self.cache_dir:
            return None
        p = os.path.join(self.cache_dir, f"{_sha(key)}.json")
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _cache_set(self, key: str, value: dict) -> None:
        if not self.cache_dir:
            return
        p = os.path.join(self.cache_dir, f"{_sha(key)}.json")
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(value, f)
        except Exception:
            pass

    # ---------- Public API ----------
    async def a_get_types(self, surface: str) -> List[str]:
        """
        Return ontology URIs (dbo:*) or WD types for an English label.
        We try DBpedia first, then Wikidata.
        """
        types = await self._a_dbpedia_types(surface)
        if types:
            return types
        return await self._a_wikidata_types(surface)

    def _sync_dbpedia_types(self, surface: str) -> List[str]:
        """Synchronous DBpedia lookup used as safe fallback."""
        try:
            import urllib.parse, urllib.request, json as _json
            q = f'''
            SELECT ?type WHERE {{
              ?entity rdfs:label "{surface}"@en ;
                      rdf:type ?type .
              FILTER(STRSTARTS(STR(?type), "http://dbpedia.org/ontology/"))
            }} LIMIT 20
            '''
            url = DBPEDIA_SPARQL + "?query=" + urllib.parse.quote(q) + "&format=json"
            key = "sync:" + url
            c = self._cache_get(key)
            if c is None:
                with urllib.request.urlopen(url, timeout=self.timeout) as r:
                    data = _json.loads(r.read().decode("utf-8"))
                    self._cache_set(key, data)
                    c = data
            return [b["type"]["value"] for b in c.get("results", {}).get("bindings", [])]
        except Exception:
            return []

    def get_types(self, surface: str) -> List[str]:
        """
        Sync wrapper (uses asyncio if aiohttp present; else basic urllib).
        """
        if not aiohttp:
            return self._sync_dbpedia_types(surface)
        try:
            loop = asyncio.get_event_loop()
            # If already inside a running event loop, avoid run_until_complete
            # and use a synchronous HTTP fallback to prevent un-awaited coroutine warnings.
            if loop.is_running():
                return self._sync_dbpedia_types(surface)
            return loop.run_until_complete(self.a_get_types(surface))
        except Exception:
            return self._sync_dbpedia_types(surface)

    # ---------- Internals (async) ----------
    async def _a_fetch_json(self, session, url: str) -> dict:
        key = "async:" + url
        c = self._cache_get(key)
        if c is not None:
            return c
        async with session.get(url, timeout=self.timeout, headers={"User-Agent": "KGClient/1.0"}) as resp:
            resp.raise_for_status()
            data = await resp.json()
            self._cache_set(key, data)
            return data

    async def _a_dbpedia_types(self, surface: str) -> List[str]:
        if not aiohttp:
            return self.get_types(surface)
        import urllib.parse
        q = f'''
        SELECT ?type WHERE {{
          ?entity rdfs:label "{surface}"@en ;
                  rdf:type ?type .
          FILTER(STRSTARTS(STR(?type), "http://dbpedia.org/ontology/"))
        }} LIMIT 20
        '''
        url = DBPEDIA_SPARQL + "?query=" + urllib.parse.quote(q) + "&format=json"
        try:
            async with aiohttp.ClientSession() as s:
                data = await self._a_fetch_json(s, url)
            return [b["type"]["value"] for b in data.get("results", {}).get("bindings", [])]
        except Exception:
            return []

    async def _a_wikidata_types(self, surface: str) -> List[str]:
        if not aiohttp:
            return []
        import urllib.parse
        # Find entity by English label then return subclasses-of (P279*) or instance-of (P31) types
        q = f'''
        SELECT DISTINCT ?type WHERE {{
          ?e rdfs:label "{surface}"@en.
          OPTIONAL {{ ?e wdt:P31 ?type. }}
        }} LIMIT 20
        '''
        url = WIKIDATA_SPARQL + "?query=" + urllib.parse.quote(q) + "&format=json"
        try:
            async with aiohttp.ClientSession() as s:
                data = await self._a_fetch_json(s, url)
            return [b["type"]["value"] for b in data.get("results", {}).get("bindings", [])]
        except Exception:
            return []

