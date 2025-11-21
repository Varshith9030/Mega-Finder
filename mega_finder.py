#!/usr/bin/env python3
"""
mega_finder.py (patched single-file)

- All-in-one recon & exposure scanner (MEGA). Single-file.
- Supports external wordlist (--wordlist path.txt)
- Generates internal permutation-based wordlist (aggressiveness)
- Merges both lists, dedups, and fuzzes targets
- Domain-restricted crawling (HTML + JS)
- GraphQL detection, .env detection, .git checks, backups, swagger, phpinfo, CORS checks, JS secret search, directory listing detection
- Async with aiohttp and throttling
- Saves per-target results in structured folders (summary.json, summary.csv, raw saved files)

Requirements:
    pip install aiohttp beautifulsoup4 tqdm

Usage:
    python3 mega_finder.py -i targets.txt -o results_dir --wordlist mega_wordlist_1m.txt --concurrency 40 --aggressiveness 3 --max-probes 200000

Be conservative when scanning (start with low concurrency and low max-probes).
"""

import argparse
import asyncio
import csv
import json
import os
import random
import re
import string
import time
from pathlib import Path
from typing import List, Set, Tuple, Dict
from urllib.parse import urljoin, urlparse

import aiohttp
from aiohttp import ClientTimeout
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm

# ---------------- Config defaults ----------------
DEFAULT_HEADERS = {"User-Agent": "MegaFinder/1.0 (+https://example.com)"}
DEFAULT_CONCURRENCY = 20
DEFAULT_TIMEOUT = 12
DEFAULT_MAX_PAGES = 200
DEFAULT_MAX_DEPTH = 2
DEFAULT_CONCURRENCY_PER_HOST = 10
DEFAULT_PER_TARGET_DELAY = 0.0
DEFAULT_AGGRESSIVENESS = 3  # 1..5 => how many generated permutations

# ---------------- Static path lists ----------------
COMMON_GIT_PATHS = [
    "/.git", "/.git/HEAD", "/.git/config", "/.git/index", "/.git/packed-refs", "/.git/logs/HEAD", "/.git/logs/"
]
COMMON_ENV_PATHS = [
    "/.env", "/.env.local", "/.env.production", "/.env.dev", "/env", "/environment", "/config/.env", "/.env.bak"
]
COMMON_BACKUPS = [
    "/backup.zip", "/backup.tar.gz", "/backup.sql", "/db.sql", "/dump.sql", "/site-backup.zip", "/site.zip", "/site.bak"
]
COMMON_SWAGGER = [
    "/swagger", "/swagger.json", "/swagger-ui", "/openapi.json", "/openapi.yaml", "/api/docs", "/v2/api-docs"
]
COMMON_PHPINFO = ["/phpinfo.php", "/info.php", "/phpinfo"]
COMMON_ADMIN_PANELS = [
    "/admin", "/admin/login", "/administrator", "/manage", "/dashboard", "/controlpanel", "/wp-admin"
]
COMMON_LOGS = ["/error.log", "/access.log", "/debug.log", "/app.log", "/logs/"]
COMMON_CONFIGS = [
    "/config.json", "/config.yaml", "/appsettings.json", "/application.properties", "/web.config", "/settings.py"
]
COMMON_WORDPRESS_BACKUPS = [
    "/wp-config.php.bak", "/wp-config.php~", "/wp-config.php.save", "/wp-config.php.old"
]
COMMON_DIR_LIST_CHECK = ["/", "/uploads/", "/files/", "/backup/", "/static/", "/assets/"]
GRAPHQL_COMMON = ["/graphql", "/api/graphql", "/v1/graphql", "/graphql.php", "/graphiql", "/playground", "/gql"]
CI_PATHS = ["/jenkins", "/jenkins/login", "/gitlab", "/gitlab/admin", "/nexus", "/sonarqube", "/hudson"]
S3_CANDIDATES_SUFFIXES = ["s3.amazonaws.com", "s3.us-east-1.amazonaws.com", "storage.googleapis.com"]

SECRET_INDICATORS = [
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "DB_PASSWORD", "DB_USER", "DB_USERNAME", "PASSWORD=", "PRIVATE_KEY",
    "JWT_SECRET", "SECRET_KEY", "API_KEY", "ACCESS_TOKEN", "AUTH_TOKEN", "MONGODB_URI", "DATABASE_URL"
]

ENV_LINE_RE = re.compile(r"^[A-Za-z0-9_]+\s*=\s*.+", re.M)
PRIVATE_KEY_PFX = "-----BEGIN PRIVATE KEY-----"
RSA_KEY_PFX = "-----BEGIN RSA PRIVATE KEY-----"
JS_SECRET_KEYWORDS = ["api_key", "apikey", "access_key", "secret", "token", "auth_token", "firebase", "aws"]

# ---------------- Utilities ----------------


def normalize_url(u: str) -> str:
    if not u:
        return u
    u = u.strip()
    if u.startswith("//"):
        return "https:" + u
    p = urlparse(u)
    if not p.scheme:
        return "https://" + u
    return u


def same_domain(a: str, b: str) -> bool:
    try:
        return urlparse(a).netloc == urlparse(b).netloc
    except Exception:
        return False


def sane_filename(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]", "_", s)[:200]


# ---------------- Wordlist helpers ----------------


def load_wordlist(path: str) -> List[str]:
    """
    Load user-supplied external wordlist lines and normalize them to start with '/' where appropriate.
    If a line looks like a full URL (starts with http), keep it as-is.
    """
    if not path:
        return []
    words = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("http://") or s.startswith("https://"):
                    words.append(s)
                else:
                    # ensure leading slash
                    if not s.startswith("/"):
                        s = "/" + s
                    words.append(s)
    except Exception:
        return []
    return words


def generate_internal_wordlist(aggressiveness: int = DEFAULT_AGGRESSIVENESS) -> List[str]:
    """
    Generate a reasonable internal mega wordlist with permutations.
    aggressiveness 1..5 increases the number of random/derived permutations.
    """
    prefixes = ["", ".", "app", "prod", "stage", "dev", "local", "backup", "old", "private", "secret"]
    suffixes = ["", ".bak", ".backup", "~", ".old", ".1", ".2", ".zip", ".tar.gz", ".json", ".sql"]
    base_files = [
        ".env", "env", "environment", "config.json", "config.yaml", "settings.py",
        "error.log", "access.log", "backup.zip", "dump.sql", "swagger.json", "openapi.json",
        "graphql", "api/graphql", "wp-config.php", "composer.json", "package.json", "credentials.json"
    ]
    dirs = ["", "app", "api", "admin", "config", "private", "server", "src", "dev", "test", "stage", "prod", "public", "static"]
    entries = set()

    # combine commons
    for d in dirs:
        for bf in base_files:
            path = f"/{d}/{bf}" if d else f"/{bf}"
            entries.add(re.sub(r"//+", "/", path))

    # add the standard lists
    for p in COMMON_ENV_PATHS + COMMON_GIT_PATHS + COMMON_BACKUPS + COMMON_SWAGGER + COMMON_PHPINFO + COMMON_ADMIN_PANELS + COMMON_LOGS + COMMON_CONFIGS + COMMON_WORDPRESS_BACKUPS + GRAPHQL_COMMON + CI_PATHS + COMMON_DIR_LIST_CHECK:
        entries.add(p if p.startswith("/") else "/" + p)

    # permutations of prefixes/suffixes
    for pre in prefixes:
        for bf in base_files:
            for suf in suffixes:
                name = (pre + bf + suf) if pre else (bf + suf)
                entries.add("/" + name)

    # random machine-generated names scaled by aggressiveness
    random.seed(0)
    add = aggressiveness * 2000  # aggressiveness scales size
    alphabet = string.ascii_lowercase + string.digits
    for _ in range(add):
        segs = random.randint(1, 3)
        name = "_".join("".join(random.choices(alphabet, k=random.randint(3, 10))) for _ in range(segs))
        if random.random() < 0.4:
            name = "." + name
        if random.random() < 0.5:
            name += random.choice([".env", ".bak", ".backup", ".old", ".zip", ".sql"])
        d = random.choice(dirs + [""])
        path = f"/{d}/{name}" if d else f"/{name}"
        entries.add(re.sub(r"//+", "/", path))

    return sorted(entries)


# ---------------- HTTP helper functions ----------------


async def fetch_text(session: aiohttp.ClientSession, url: str, timeout=DEFAULT_TIMEOUT) -> Tuple[str, int, Dict[str, str]]:
    """
    Return (text, status, headers dict lowercase keys)
    """
    try:
        async with session.get(url, timeout=ClientTimeout(total=timeout), allow_redirects=True) as r:
            text = await r.text(errors="ignore")
            headers = {k.lower(): v for k, v in r.headers.items()}
            return text, r.status, headers
    except Exception:
        return "", None, {}


async def post_json(session: aiohttp.ClientSession, url: str, payload: dict, timeout=DEFAULT_TIMEOUT) -> Tuple[int, str, Dict[str, str]]:
    try:
        async with session.post(url, json=payload, timeout=ClientTimeout(total=timeout), allow_redirects=True) as r:
            text = await r.text(errors="ignore")
            headers = {k.lower(): v for k, v in r.headers.items()}
            return r.status, text, headers
    except Exception:
        return None, "", {}


# ---------------- Crawler & JS parsing ----------------


class Crawler:
    def __init__(self, session: aiohttp.ClientSession, root: str, max_pages: int = DEFAULT_MAX_PAGES, max_depth: int = DEFAULT_MAX_DEPTH):
        self.session = session
        self.root = normalize_url(root)
        self.to_visit = asyncio.Queue()
        self.to_visit.put_nowait((self.root, 0))
        self.visited: Set[str] = set()
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.findings: Set[str] = set()

    async def _parse_js(self, js_text: str, base_url: str):
        if not js_text:
            return
        # quoted paths
        for m in re.findall(r'["\'`]\s*(\/[^\s"\'`\\]+)\s*["\'`]', js_text):
            if "env" in m.lower() or "config" in m.lower() or "secret" in m.lower() or "credential" in m.lower():
                cand = urljoin(base_url, m)
                if same_domain(cand, self.root):
                    self.findings.add(cand)
        # keyword hints
        low = js_text.lower()
        for kw in JS_SECRET_KEYWORDS:
            if kw in low:
                self.findings.add(base_url + " #contains:" + kw)
        # explicit .env references
        for m in re.findall(r'\.env[^\s"\'<>)]*', js_text, flags=re.IGNORECASE):
            cand = urljoin(base_url, m)
            if same_domain(cand, self.root):
                self.findings.add(cand)

    async def _crawl_one(self, url: str, depth: int):
        if url in self.visited:
            return
        self.visited.add(url)
        html, status, headers = await fetch_text(self.session, url)
        if not html:
            return
        soup = BeautifulSoup(html, "html.parser")
        # anchors
        for a in soup.find_all("a", href=True):
            href = a.get("href").strip()
            try:
                full = normalize_url(urljoin(url, href))
            except Exception:
                continue
            if same_domain(full, self.root) and depth + 1 <= self.max_depth:
                await self.to_visit.put((full, depth + 1))
        # scripts
        for s in soup.find_all("script", src=True):
            src = s.get("src").strip()
            try:
                js_url = normalize_url(urljoin(url, src))
            except Exception:
                continue
            if same_domain(js_url, self.root):
                self.findings.add(js_url)
                js_text, st, hs = await fetch_text(self.session, js_url)
                await self._parse_js(js_text, js_url)
        # inline scripts
        for s in soup.find_all("script"):
            if s.string:
                await self._parse_js(s.string, url)
        # raw .env patterns in HTML
        for m in re.findall(r'\/[A-Za-z0-9_\-\/\.]*\.env[^\s"\'<>]*', html, flags=re.IGNORECASE):
            cand = normalize_url(urljoin(url, m))
            if same_domain(cand, self.root):
                self.findings.add(cand)

    async def run(self) -> List[str]:
        pages = 0
        while not self.to_visit.empty() and pages < self.max_pages:
            url, depth = await self.to_visit.get()
            try:
                await self._crawl_one(url, depth)
            except Exception:
                pass
            pages += 1
        return list(self.findings)


# ---------------- Detection helpers ----------------


def looks_like_env(text: str) -> bool:
    if not text:
        return False
    if PRIVATE_KEY_PFX in text or RSA_KEY_PFX in text:
        return True
    if ENV_LINE_RE.search(text):
        # require indicators or multiple key lines
        low = text.lower()
        if any(tok.lower() in low for tok in SECRET_INDICATORS):
            return True
        if len(re.findall(ENV_LINE_RE, text)) >= 3:
            return True
    if re.search(r'"(aws|access|secret|password|token|private|key)"\s*:', text, flags=re.IGNORECASE):
        return True
    return False


async def is_graphql_endpoint(session: aiohttp.ClientSession, url: str) -> Tuple[bool, str]:
    status, text, headers = await post_json(session, url, {"query": "{ __typename }"})
    if status in (200, 400) and text:
        l = text.lower()
        if "graphql" in l or "__typename" in l or "errors" in l:
            return True, text
    return False, ""


async def check_cors(session: aiohttp.ClientSession, url: str) -> Tuple[bool, Dict[str, str]]:
    try:
        async with session.options(url, timeout=ClientTimeout(total=6)) as r:
            headers = {k.lower(): v for k, v in r.headers.items()}
            aco = headers.get("access-control-allow-origin", "")
            acc = headers.get("access-control-allow-credentials", "")
            if aco == "*" and acc == "true":
                return True, headers
            if aco and ("http://" in aco or "https://" in aco) and ("localhost" in aco or "127.0.0.1" in aco or "*" in aco):
                return True, headers
            return False, headers
    except Exception:
        return False, {}


# ---------------- Orchestration ----------------


async def probe_url(session: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore, timeout: int = DEFAULT_TIMEOUT) -> Dict:
    async with sem:
        try:
            text, status, headers = await fetch_text(session, url, timeout=timeout)
            return {"url": url, "status": status, "text": text or "", "headers": headers}
        except Exception:
            return {"url": url, "status": None, "text": "", "headers": {}}


async def probe_candidates(session: aiohttp.ClientSession, candidates: List[str], concurrency: int, per_target_delay: float, max_probes: int = 0) -> List[Dict]:
    sem = asyncio.Semaphore(concurrency)
    tasks = []
    if max_probes and len(candidates) > max_probes:
        # sample or slice to not blow up
        candidates = candidates[:max_probes]
    for c in candidates:
        tasks.append(asyncio.create_task(probe_url(session, c, sem)))
    results = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Probing"):
        try:
            res = await f
            results.append(res)
        except Exception:
            pass
        if per_target_delay > 0:
            await asyncio.sleep(per_target_delay * random.random())
    return results


async def scan_target(session: aiohttp.ClientSession, target: str, outdir: Path, concurrency: int, max_pages: int, max_depth: int, per_target_delay: float, max_probes: int, extra_wordlist: List[str], aggressiveness: int):
    target = normalize_url(target)
    parsed = urlparse(target)
    domain = parsed.netloc.replace(":", "_")
    target_dir = outdir / domain
    target_dir.mkdir(parents=True, exist_ok=True)
    summary = {"target": target, "found": []}

    # 1) crawler hints
    crawler = Crawler(session, target, max_pages=max_pages, max_depth=max_depth)
    try:
        hints = await crawler.run()
    except Exception:
        hints = []
    # core candidate set
    candidate_paths: Set[str] = set()

    # include builtins
    for p in COMMON_ENV_PATHS + COMMON_GIT_PATHS + COMMON_BACKUPS + COMMON_SWAGGER + COMMON_PHPINFO + COMMON_ADMIN_PANELS + COMMON_LOGS + COMMON_CONFIGS + COMMON_WORDPRESS_BACKUPS + GRAPHQL_COMMON + CI_PATHS + COMMON_DIR_LIST_CHECK:
        candidate_paths.add(urljoin(target, p))

    # include hints (crawler)
    for h in hints:
        if isinstance(h, str):
            if h.startswith("http://") or h.startswith("https://"):
                candidate_paths.add(h)
            else:
                candidate_paths.add(urljoin(target, h))

    # include external wordlist items (full urls or relative)
    for item in (extra_wordlist or []):
        if item.startswith("http://") or item.startswith("https://"):
            candidate_paths.add(item)
        else:
            candidate_paths.add(urljoin(target, item))

    # include internal generated list scaled by aggressiveness
    internal = generate_internal_wordlist(aggressiveness)
    # sample internal list if we want to limit overall size later; but keep full now
    for p in internal:
        candidate_paths.add(urljoin(target, p))

    # derived dirs from hints (parents)
    for h in hints:
        if isinstance(h, str) and h.startswith("http"):
            p = urlparse(h)
            base = f"{p.scheme}://{p.netloc}"
            path = p.path or "/"
            # progressively add parents
            parts = [seg for seg in path.split("/") if seg]
            accum = "/"
            for i in range(len(parts)):
                accum = "/" + "/".join(parts[: i + 1]) + "/"
                candidate_paths.add(urljoin(base, accum))

    # robots & sitemap
    candidate_paths.add(urljoin(target, "/robots.txt"))
    candidate_paths.add(urljoin(target, "/sitemap.xml"))

    # s3 like attempts
    hostname = parsed.netloc.split(":")[0]
    for suf in S3_CANDIDATES_SUFFIXES:
        candidate_paths.add("https://" + hostname + "." + suf)
        candidate_paths.add("http://" + hostname + "." + suf)

    # normalize and filter domain-only (keep full URLs only and domain restricted)
    final_candidates = []
    for c in candidate_paths:
        try:
            full = normalize_url(c)
        except Exception:
            continue
        # keep only same-domain HTTP/S URLs or explicit external (user provided)
        # If the candidate is host-like (https://host.s3...), keep it
        try:
            parsed_c = urlparse(full)
            if not parsed_c.scheme:
                continue
            # include if same-domain or looks like bucket / explicit url
            if same_domain(full, target) or ("s3." in full or ".storage.googleapis.com" in full):
                final_candidates.append(full)
        except Exception:
            continue

    # deduplicate and shuffle
    final_candidates = sorted(set(final_candidates))
    random.shuffle(final_candidates)

    # limit total probes if requested
    probes_to_run = final_candidates if (max_probes <= 0) else final_candidates[:max_probes]

    # 2) probe candidates
    probe_results = await probe_candidates(session, probes_to_run, concurrency, per_target_delay, max_probes)

    findings = []
    # analyze probe results
    for r in probe_results:
        url = r.get("url")
        status = r.get("status")
        text = (r.get("text") or "")
        headers = r.get("headers") or {}
        try:
            ppath = urlparse(url).path or "/"
        except Exception:
            ppath = "/"

        if status and 200 <= status < 300:
            # .env detection
            if looks_like_env(text):
                fname = target_dir / (sane_filename(ppath.strip("/").replace("/", "_") or "root") + "_env.txt")
                try:
                    with open(fname, "w", errors="ignore") as fh:
                        fh.write(text)
                except Exception:
                    pass
                findings.append({"type": ".env", "url": url, "status": status, "file": str(fname)})

            # git exposures
            if "/.git" in url:
                findings.append({"type": "git", "url": url, "status": status})

            # backups
            if any(url.lower().endswith(ext) for ext in [".zip", ".tar", ".tar.gz", ".bak", ".old", ".sql"]):
                findings.append({"type": "backup", "url": url, "status": status})

            # swagger/openapi
            if "swagger" in url.lower() or "openapi" in url.lower() or "api-docs" in url.lower():
                if "swagger" in text.lower() or "openapi" in text.lower() or '"openapi":' in text.lower():
                    fname = target_dir / (sane_filename(ppath.strip("/") or "openapi") + ".json")
                    try:
                        with open(fname, "w", errors="ignore") as fh:
                            fh.write(text)
                    except Exception:
                        pass
                    findings.append({"type": "openapi", "url": url, "status": status, "file": str(fname)})

            # phpinfo
            if "phpinfo" in url.lower() or "phpinfo" in text.lower() or "php.ini" in text.lower():
                findings.append({"type": "phpinfo", "url": url, "status": status})

            # directory listing
            if "<title>index of" in text.lower() or re.search(r"(?i)parent directory</a>", text):
                findings.append({"type": "directory-listing", "url": url, "status": status})

            # logs/configs
            if any(kw in url.lower() for kw in ["log", "error", "debug", "config", "settings", "credentials"]) or any(kw.lower() in text.lower() for kw in ["password", "secret", "aws", "mongodb", "db_password"]):
                findings.append({"type": "possible-secret-or-config", "url": url, "status": status})

            # GraphQL detection (fast test)
            if any(g in url.lower() for g in ["graphql", "/graphiql", "/playground", "/gql"]):
                ok, raw = await is_graphql_endpoint(session, url)
                if ok:
                    findings.append({"type": "graphql", "url": url, "status": status})

            # CORS check
            cors_vuln, cors_headers = await check_cors(session, url)
            if cors_vuln:
                findings.append({"type": "cors", "url": url, "status": status, "headers": cors_headers})

    # 3) JS deep-scan: fetch discovered JS (from hints) and search for secret keywords and .env references
    js_candidates = [c for c in final_candidates if c.lower().endswith(".js")]
    if js_candidates:
        # smaller concurrency for JS
        js_results = await probe_candidates(session, js_candidates, max(2, concurrency // 2), per_target_delay, max_probes=2000)
        for jr in js_results:
            jtext = jr.get("text") or ""
            jurl = jr.get("url")
            low = jtext.lower()
            for kw in JS_SECRET_KEYWORDS:
                if kw in low:
                    findings.append({"type": "js-secret-hint", "url": jurl, "keyword": kw})
            for m in re.findall(r'\/[A-Za-z0-9_\-\/\.]*\.env[^\s"\'<>]*', jtext, flags=re.IGNORECASE):
                full = normalize_url(urljoin(jurl, m))
                findings.append({"type": ".env-hint-in-js", "url": full})

    # 4) robots & sitemap hints
    try:
        robots_text, st, _ = await fetch_text(session, urljoin(target, "/robots.txt"))
        if robots_text and st and st < 400:
            for line in robots_text.splitlines():
                if line.strip().lower().startswith("disallow") or ".env" in line.lower() or "backup" in line.lower():
                    findings.append({"type": "robots-hint", "url": urljoin(target, "/robots.txt"), "line": line.strip()})
    except Exception:
        pass

    # save per-target summary
    summary_path = target_dir / "summary.json"
    try:
        with open(summary_path, "w", encoding="utf-8") as sf:
            json.dump({"target": target, "found": findings}, sf, indent=2)
    except Exception:
        pass

    csv_path = target_dir / "summary.csv"
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow(["type", "url", "status", "extra"])
            for f in findings:
                extra = {k: v for k, v in f.items() if k not in ("type", "url", "status")}
                writer.writerow([f.get("type"), f.get("url"), f.get("status"), json.dumps(extra)])
    except Exception:
        pass

    return findings


async def run_targets(targets: List[str], outdir: Path, concurrency: int, max_pages: int, max_depth: int, per_target_delay: float, parallel_targets: int, max_probes: int, wordlist_path: str, aggressiveness: int):
    # load external wordlist if provided
    external = load_wordlist(wordlist_path) if wordlist_path else []
    # create aiohttp session per worker
    results = {}
    sem = asyncio.Semaphore(parallel_targets)

    async def worker(t):
        async with sem:
            timeout = ClientTimeout(total=None)
            connector = aiohttp.TCPConnector(limit_per_host=DEFAULT_CONCURRENCY_PER_HOST, ssl=False)
            headers = DEFAULT_HEADERS.copy()
            async with aiohttp.ClientSession(timeout=timeout, connector=connector, headers=headers) as session:
                try:
                    res = await scan_target(session, t, outdir, concurrency, max_pages, max_depth, per_target_delay, max_probes, external, aggressiveness)
                    results[t] = res
                except Exception as e:
                    results[t] = {"error": str(e)}
            await asyncio.sleep(0.05)

    tasks = [asyncio.create_task(worker(t)) for t in targets]
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Targets"):
        try:
            await f
        except Exception:
            pass

    # global summary
    try:
        with open(outdir / "global_summary.json", "w", encoding="utf-8") as gf:
            json.dump(results, gf, indent=2)
    except Exception:
        pass
    return results


# ---------------- CLI & main ----------------


def parse_args():
    p = argparse.ArgumentParser(description="MEGA-Finder (patched single-file) - Use only on authorized targets")
    p.add_argument("-i", "--input", required=True, help="File with target base URLs (one per line)")
    p.add_argument("-o", "--output", default="mega_results", help="Output directory")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Concurrent HTTP probes")
    p.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES, help="Max pages to crawl per target")
    p.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH, help="Crawl depth per target")
    p.add_argument("--per-target-delay", type=float, default=DEFAULT_PER_TARGET_DELAY, help="Jitter delay between probes")
    p.add_argument("--parallel-targets", type=int, default=2, help="How many targets to scan in parallel")
    p.add_argument("--max-probes", type=int, default=0, help="Limit candidate probes per target (0 = unlimited)")
    p.add_argument("--wordlist", help="Optional external wordlist file (one path per line or full URLs)")
    p.add_argument("--aggressiveness", type=int, default=DEFAULT_AGGRESSIVENESS, help="1..5 internal generation aggressiveness")
    return p.parse_args()


def load_targets(path: str) -> List[str]:
    t = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if s:
                t.append(normalize_url(s))
    return t


def main():
    args = parse_args()
    targets = load_targets(args.input)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[+] Targets: {len(targets)} — output: {outdir.resolve()}")
    start = time.time()
    try:
        asyncio.run(run_targets(targets, outdir, args.concurrency, args.max_pages, args.max_depth, args.per_target_delay, args.parallel_targets, args.max_probes, args.wordlist, args.aggressiveness))
    except KeyboardInterrupt:
        print("[!] Interrupted by user")
    elapsed = time.time() - start
    print(f"[+] Completed in {elapsed:.1f}s — results in {outdir.resolve()}")


if __name__ == "__main__":
    main()
