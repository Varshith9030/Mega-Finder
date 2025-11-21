#!/usr/bin/env python3
"""
mega_finder.py (patched single-file) — segmented delivery

INSTRUCTIONS:
- Collect segments 1..10 in order and concatenate into a single file named mega_finder.py
- Or say "assemble" after I finish and I'll write the file for you and provide a download link.

Legal: Use only on targets you are authorized to test.
"""

# ---------------- imports ----------------
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

        # external JS files
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

        # inline JS
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

    # Private keys
    if PRIVATE_KEY_PFX in text or RSA_KEY_PFX in text:
        return True

    # KEY=VALUE patterns
    if ENV_LINE_RE.search(text):
        low = text.lower()
        # Strong indicators
        if any(tok.lower() in low for tok in SECRET_INDICATORS):
            return True
        # If we find multiple environment-pattern lines
        if len(re.findall(ENV_LINE_RE, text)) >= 3:
            return True

    # JSON-like secrets
    if re.search(r'"(aws|access|secret|password|token|private|key)"\s*:', text, flags=re.IGNORECASE):
        return True

    return False


async def is_graphql_endpoint(session: aiohttp.ClientSession, url: str) -> Tuple[bool, str]:
    # Try a minimal introspection signal
    status, text, headers = await post_json(session, url, {"query": "{ __typename }"})
    if status in (200, 400) and text:
        l = text.lower()
        if "graphql" in l or "__typename" in l or "errors" in l:
            return True, text
    return False, ""


async def check_cors(session: aiohttp.ClientSession, url: str) -> Tuple[bool, Dict[str, str]]:
    """
    Detect Access-Control-Allow-Origin: * with credentials=true
    """
    try:
        async with session.options(url, timeout=ClientTimeout(total=6)) as r:
            headers = {k.lower(): v for k, v in r.headers.items()}
            aco = headers.get("access-control-allow-origin", "")
            acc = headers.get("access-control-allow-credentials", "")

            # worst misconfig
            if aco == "*" and acc == "true":
                return True, headers

            # suspicious origins
            if aco and ("http://" in aco or "https://" in aco):
                if "localhost" in aco or "127.0.0.1" in aco or "*" in aco:
                    return True, headers

            return False, headers

    except Exception:
        return False, {}
# ---------------- Orchestration ----------------

async def probe_url(session: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore, timeout: int = DEFAULT_TIMEOUT) -> Dict:
    """
    Perform a GET request on a candidate URL under semaphore control.
    Returns dict: {url, status, text, headers}.
    """
    async with sem:
        try:
            text, status, headers = await fetch_text(session, url, timeout=timeout)
            return {"url": url, "status": status, "text": text or "", "headers": headers}
        except Exception:
            return {"url": url, "status": None, "text": "", "headers": {}}


async def probe_candidates(
    session: aiohttp.ClientSession,
    candidates: List[str],
    concurrency: int,
    per_target_delay: float,
    max_probes: int = 0
) -> List[Dict]:
    """
    Probes all given candidate URLs asynchronously with concurrency limit.
    """
    sem = asyncio.Semaphore(concurrency)
    tasks = []

    if max_probes and len(candidates) > max_probes:
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


async def scan_target(
    session: aiohttp.ClientSession,
    target: str,
    outdir: Path,
    concurrency: int,
    max_pages: int,
    max_depth: int,
    per_target_delay: float,
    max_probes: int,
    extra_wordlist: List[str],
    aggressiveness: int
):
    """
    Full scan on one target domain:
    - crawl HTML & JS
    - gather candidate URLs
    - merge internal + external wordlists
    - async probe all candidates
    - detect secrets, backup files, openapi, graphql, cors, git, logs, config leaks
    - save findings into summary.json / summary.csv
    """

    target = normalize_url(target)
    parsed = urlparse(target)
    domain = parsed.netloc.replace(":", "_")

    target_dir = outdir / domain
    target_dir.mkdir(parents=True, exist_ok=True)
    summary = {"target": target, "found": []}

    # 1. CRAWLER HINTS
    crawler = Crawler(session, target, max_pages=max_pages, max_depth=max_depth)
    try:
        hints = await crawler.run()
    except Exception:
        hints = []

    candidate_paths: Set[str] = set()

    # Built-in common paths
    for p in (
        COMMON_ENV_PATHS + COMMON_GIT_PATHS + COMMON_BACKUPS + COMMON_SWAGGER +
        COMMON_PHPINFO + COMMON_ADMIN_PANELS + COMMON_LOGS + COMMON_CONFIGS +
        COMMON_WORDPRESS_BACKUPS + GRAPHQL_COMMON + CI_PATHS + COMMON_DIR_LIST_CHECK
    ):
        candidate_paths.add(urljoin(target, p))

    # Crawler hints
    for h in hints:
        if isinstance(h, str):
            if h.startswith("http://") or h.startswith("https://"):
                candidate_paths.add(h)
            else:
                candidate_paths.add(urljoin(target, h))

    # External wordlist (provided by user)
    for item in (extra_wordlist or []):
        if item.startswith("http://") or item.startswith("https://"):
            candidate_paths.add(item)
        else:
            candidate_paths.add(urljoin(target, item))

    # Internal auto-generated permutations
    internal = generate_internal_wordlist(aggressiveness)
    for p in internal:
        candidate_paths.add(urljoin(target, p))

    # Derive parent directories from crawler-found URLs
    for h in hints:
        if isinstance(h, str) and h.startswith("http"):
            p = urlparse(h)
            base = f"{p.scheme}://{p.netloc}"
            path = p.path or "/"
            parts = [seg for seg in path.split("/") if seg]
            accum = "/"
            for i in range(len(parts)):
                accum = "/" + "/".join(parts[: i + 1]) + "/"
                candidate_paths.add(urljoin(base, accum))

    # robots & sitemap
    candidate_paths.add(urljoin(target, "/robots.txt"))
    candidate_paths.add(urljoin(target, "/sitemap.xml"))

    # Cloud bucket candidates
    hostname = parsed.netloc.split(":")[0]
    for suf in S3_CANDIDATES_SUFFIXES:
        candidate_paths.add("https://" + hostname + "." + suf)
        candidate_paths.add("http://" + hostname + "." + suf)

    # Normalize & filter by domain or bucket domains
    final_candidates = []
    for c in candidate_paths:
        try:
            full = normalize_url(c)
        except Exception:
            continue

        try:
            parsed_c = urlparse(full)
            if not parsed_c.scheme:
                continue

            if same_domain(full, target) or ("s3." in full or ".storage.googleapis.com" in full):
                final_candidates.append(full)
        except Exception:
            continue

    # Dedup + shuffle
    final_candidates = sorted(set(final_candidates))
    random.shuffle(final_candidates)

    # Cut by max-probes if needed
    probes_to_run = final_candidates if (max_probes <= 0) else final_candidates[:max_probes]

    # 2. PROBE CANDIDATES
    probe_results = await probe_candidates(
        session, probes_to_run, concurrency, per_target_delay, max_probes
    )

    findings = []
    # 3. DETECT LEAKS
    for r in probe_results:
        url = r["url"]
        st = r["status"]
        text = r["text"]
        headers = r["headers"]

        if st in (None, -1):
            continue

        # --- ENV / SECRET LEAKS ---
        if st == 200 and text and looks_like_env(text):
            fn = sane_filename(url) + "_env.txt"
            outfile = target_dir / fn
            try:
                with open(outfile, "w", encoding="utf-8", errors="ignore") as f:
                    f.write(text)
            except Exception:
                pass
            findings.append(
                {"type": "env_leak", "url": url, "saved": str(outfile.name), "status": st}
            )

        # --- GIT REPO EXPOSURE ---
        if st == 200 and ".git" in url and ("HEAD" in url or "config" in url or "logs" in url):
            fn = sane_filename(url) + "_git.txt"
            outfile = target_dir / fn
            try:
                with open(outfile, "w", encoding="utf-8", errors="ignore") as f:
                    f.write(text)
            except Exception:
                pass
            findings.append(
                {"type": "git_exposed", "url": url, "saved": outfile.name, "status": st}
            )

        # --- BACKUP / CONFIG / LOG leaks by filename heuristic ---
        lname = url.lower()
        if st == 200:
            if any(ext in lname for ext in [".zip", ".tar", ".gz", ".sql", ".bak", ".old", ".backup"]):
                fn = sane_filename(url) + "_backup"
                outfile = target_dir / fn
                try:
                    with open(outfile, "wb") as f:
                        f.write(text.encode("utf-8", errors="ignore"))
                except Exception:
                    pass
                findings.append(
                    {"type": "backup_file", "url": url, "saved": outfile.name, "status": st}
                )

        # --- LOG leaks ---
        if st == 200 and ("log" in lname or lname.endswith(".log")):
            if len(text) > 30:
                fn = sane_filename(url) + "_log.txt"
                outfile = target_dir / fn
                try:
                    with open(outfile, "w", encoding="utf-8", errors="ignore") as f:
                        f.write(text)
                except Exception:
                    pass
                findings.append(
                    {"type": "log_leak", "url": url, "saved": outfile.name, "status": st}
                )

        # --- SWAGGER / OPENAPI ---
        if st == 200 and (
            "swagger" in lname or lname.endswith("swagger.json") or lname.endswith("openapi.json") or lname.endswith("openapi.yaml")
        ):
            fn = sane_filename(url) + "_api.json"
            outfile = target_dir / fn
            try:
                with open(outfile, "w", encoding="utf-8", errors="ignore") as f:
                    f.write(text)
            except Exception:
                pass
            findings.append(
                {"type": "openapi", "url": url, "saved": outfile.name, "status": st}
            )

        # --- PHPINFO ---
        if st == 200 and ("phpinfo" in lname or "php info" in text.lower()):
            fn = sane_filename(url) + "_phpinfo.html"
            outfile = target_dir / fn
            try:
                with open(outfile, "w", encoding="utf-8", errors="ignore") as f:
                    f.write(text)
            except Exception:
                pass
            findings.append(
                {"type": "phpinfo", "url": url, "saved": outfile.name, "status": st}
            )

        # --- Directory Listing ---
        if st == 200 and "Index of" in text:
            findings.append({"type": "dir_listing", "url": url, "status": st})

        # --- CORS Misconfig ---
        is_cors, cors_headers = await check_cors(session, url)
        if is_cors:
            findings.append(
                {"type": "cors_misconfig", "url": url, "details": cors_headers}
            )

        # --- GRAPHQL ---
        if st in (200, 400) and "graphql" in lname:
            is_gql, gql_text = await is_graphql_endpoint(session, url)
            if is_gql:
                fn = sane_filename(url) + "_graphql.txt"
                outfile = target_dir / fn
                try:
                    with open(outfile, "w", encoding="utf-8", errors="ignore") as f:
                        f.write(gql_text)
                except Exception:
                    pass

                findings.append(
                    {"type": "graphql", "url": url, "saved": outfile.name, "status": st}
                )

    # Final summary
    summary["found"] = findings

    # Save summary.json
    sumfile = target_dir / "summary.json"
    with open(sumfile, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save summary.csv
    csvfile = target_dir / "summary.csv"
    with open(csvfile, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "url", "saved", "status"])
        for row in findings:
            w.writerow([
                row.get("type"),
                row.get("url"),
                row.get("saved", ""),
                row.get("status", "")
            ])

    return summary
# ---------------- Global multi-target scan ----------------

async def scan_all(
    targets: List[str],
    output_dir: Path,
    concurrency: int,
    per_target_delay: float,
    max_pages: int,
    max_depth: int,
    max_probes: int,
    wordlist: List[str],
    aggressiveness: int
):
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    async with aiohttp.ClientSession(headers=DEFAULT_HEADERS) as session:
        sem = asyncio.Semaphore( min(concurrency, len(targets)) )

        async def _one(tg):
            async with sem:
                try:
                    return await scan_target(
                        session, tg, output_dir, concurrency,
                        max_pages, max_depth,
                        per_target_delay,
                        max_probes,
                        wordlist,
                        aggressiveness
                    )
                except Exception as e:
                    return {"target": tg, "error": str(e), "found": []}

        tasks = [asyncio.create_task(_one(t)) for t in targets]

        results = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Targets"):
            try:
                r = await f
                results.append(r)
            except Exception:
                pass

    # Save global summary
    gsum = output_dir / "global_summary.json"
    with open(gsum, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nCompleted scanning {len(targets)} targets in {elapsed:.1f}s")
    print(f"Global summary saved to: {gsum}")

    return results


# ---------------- Argument parser ----------------

def parse_args():
    p = argparse.ArgumentParser(description="Mega Finder - All-in-one recon scanner")
    p.add_argument("-i", "--input", required=True, help="File containing target URLs/domains")
    p.add_argument("-o", "--output", required=True, help="Output directory")
    p.add_argument("--wordlist", help="Optional extra wordlist file")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Concurrent HTTP probes")
    p.add_argument("--parallel-targets", type=int, default=5, help="Number of targets to scan in parallel")
    p.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES, help="Max crawler pages per target")
    p.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH, help="Crawler depth limit")
    p.add_argument("--per-target-delay", type=float, default=DEFAULT_PER_TARGET_DELAY, help="Random delay between probes per target")
    p.add_argument("--max-probes", type=int, default=0, help="Limit number of URLs probed per target (0 = unlimited)")
    p.add_argument("--aggressiveness", type=int, default=DEFAULT_AGGRESSIVENESS, help="1..5 internal wordlist strength")
    return p


# ---------------- Entry point + MF Banner ----------------

def print_mf_banner():
    print(r"""
 __  __ _____ 
|  \/  |  ___|
| |\/| | |_  
| |  | |  _| 
|_|  |_|_|   

   M F
MEGA FINDER
""")


def main():
    print_mf_banner()

    args = parse_args().parse_args() if hasattr(parse_args, "parse_args") else parse_args().parse_args()
    # This is a workaround for argparse in inline packaging.


    # Load targets
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            targets = [normalize_url(x.strip()) for x in f if x.strip()]
    except Exception:
        print("ERROR: Could not read input file.")
        return

    # Load wordlist
    extra_list = load_wordlist(args.wordlist) if args.wordlist else []

    print(f"[+] Loaded {len(targets)} targets")
    if args.wordlist:
        print(f"[+] Loaded external wordlist: {len(extra_list)} entries")
    print("[+] Starting scan...\n")

    outdir = Path(args.output)

    asyncio.run(
        scan_all(
            targets,
            outdir,
            concurrency=args.parallel_targets,
            per_target_delay=args.per_target_delay,
            max_pages=args.max_pages,
            max_depth=args.max_depth,
            max_probes=args.max_probes,
            wordlist=extra_list,
            aggressiveness=args.aggressiveness
        )
    )


if __name__ == "__main__":
    main()
# ---------------- Additional helpers & safety patches ----------------

def safe_join(base: str, path: str) -> str:
    """
    urljoin wrapper that avoids double slashes and ensures safe normalization.
    """
    try:
        full = urljoin(base, path)
        full = re.sub(r"//+", "/", full.replace(":/", "://"))
        return full
    except Exception:
        return path


def dedupe_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def sanitize_candidates(candidates: List[str]) -> List[str]:
    """
    Remove malformed or duplicate URLs.
    """
    out = []
    for c in candidates:
        try:
            parsed = urlparse(c)
            if not parsed.scheme or not parsed.netloc:
                continue
            out.append(c)
        except Exception:
            continue
    return dedupe_preserve_order(out)


def looks_like_bucket(url: str) -> bool:
    """
    Simple check for S3/GCS bucket candidates.
    """
    u = url.lower()
    return (
        "s3.amazonaws.com" in u
        or ".s3." in u
        or "storage.googleapis.com" in u
        or u.endswith(".s3.amazonaws.com")
    )
# ---------------- Extended S3 / GCS bucket testing ----------------

async def check_bucket_listable(session: aiohttp.ClientSession, url: str) -> Tuple[bool, str]:
    """
    Basic bucket listing indicator:
    - XML listing
    - 'AccessDenied' vs 'ListBucketResult'
    """
    try:
        text, status, headers = await fetch_text(session, url, timeout=8)
        if not text:
            return False, ""

        low = text.lower()

        # AWS XML listing
        if "<listbucketresult" in low:
            return True, "public-listing"

        # Google Storage listing
        if "<listbuckets" in low or "<prefix>" in low:
            return True, "public-listing"

        # Access Denied but bucket exists
        if "accessdenied" in low or "allaccessdisabled" in low:
            return True, "exists-denied"

        # If JSON error with bucket info
        if "bucket" in low and ("notfound" in low or "forbidden" in low):
            return True, "exists"

        return False, ""
    except Exception:
        return False, ""


# ---------------- robots.txt & sitemap.xml extraction ----------------

def extract_from_robots(text: str, root: str) -> List[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("allow:") or line.lower().startswith("disallow:"):
            try:
                part = line.split(":", 1)[1].strip()
                if part.startswith("/"):
                    out.append(urljoin(root, part))
            except Exception:
                continue
    return out


def extract_from_sitemap(text: str) -> List[str]:
    """
    Extract <loc> URLs from sitemap XML.
    """
    urls = re.findall(r"<loc>(.*?)</loc>", text, flags=re.IGNORECASE)
    out = []
    for u in urls:
        u = u.strip()
        if u.startswith("http://") or u.startswith("https://"):
            out.append(u)
    return out
# ---------------- Integrate robots.txt & sitemap.xml into candidate lists ----------------

async def integrate_robot_sitemap(session: aiohttp.ClientSession, base: str) -> List[str]:
    """
    Fetches /robots.txt and /sitemap.xml and extracts URLs.
    This function is optional — some sites block these requests.
    """
    collected = []
    robots_url = urljoin(base, "/robots.txt")
    sitemap_url = urljoin(base, "/sitemap.xml")

    # robots.txt
    try:
        text, st, hdr = await fetch_text(session, robots_url, timeout=6)
        if st == 200 and text:
            extracted = extract_from_robots(text, base)
            collected.extend(extracted)
    except Exception:
        pass

    # sitemap.xml
    try:
        text, st, hdr = await fetch_text(session, sitemap_url, timeout=6)
        if st == 200 and text:
            extracted = extract_from_sitemap(text)
            collected.extend(extracted)
    except Exception:
        pass

    return collected


# ---------------- Deduplicate & sanitize ----------------

def normalize_final_candidates(base: str, candidates: List[str]) -> List[str]:
    """
    Normalize URLs, enforce domain-matching rules, and filter malformed entries.
    """
    out = []
    for c in candidates:
        try:
            full = normalize_url(c)
        except Exception:
            continue

        parsed = urlparse(full)
        if not parsed.scheme or not parsed.netloc:
            continue

        # allow domain or S3/GCS bucket
        if same_domain(base, full) or looks_like_bucket(full):
            out.append(full)

    return dedupe_preserve_order(out)
# ---------------- Final URL utilities & cleanup ----------------

def ensure_slash_prefix(p: str) -> str:
    if not p.startswith("/"):
        return "/" + p
    return p


def unique_paths(paths: List[str]) -> List[str]:
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


