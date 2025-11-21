# MEGA FINDER
All-in-one Automated Recon & Misconfiguration Scanner

MEGA FINDER is a high-performance, async-based recon engine that automatically detects high-impact exposures across web applications, including:

- .env leaks
- GraphQL endpoint discovery
- .git repository exposure
- Backup files (zip, sql, bak, etc.)
- Swagger / OpenAPI files
- Logs & config files
- Directory listing
- JS secret disclosure
- CI/CD exposed panels
- CORS misconfigurations
- Hidden endpoints from JS & crawling
- S3 / GCS bucket discovery
- and more…

WARNING:
Use this tool only on assets you own or are authorized to test. Unauthorized scanning is illegal.

----------------------------------------------------
FEATURES
----------------------------------------------------

High-Impact Vulnerability Discovery
- Detects .env files with credential extraction
- GraphQL endpoint detection
- .git exposure scanning
- Backup detection (zip, sql, tar.gz, bak)
- Swagger / OpenAPI harvesting
- Config & log file leaks
- JS secret extraction (API keys, tokens, Firebase configs, etc.)

Intelligent Recon Engine
- Domain-restricted HTML crawler
- Deep JavaScript analyzer
- Auto-generated wordlist permutations
- Supports external wordlists

Async & High Speed
- Async HTTP (aiohttp)
- Concurrency controls
- Rate limiting & max probe limits

Organized Output
- Per-target folders with summary.json, summary.csv, leaked files

Single-File Tool
- No frameworks
- Pure Python

----------------------------------------------------
INSTALLATION
----------------------------------------------------

Install dependencies:

```bash
pip install aiohttp beautifulsoup4 tqdm
```

Clone repository:

```bash
git clone https://github.com/Varshith9030/Mega-finder.git

cd Mega-finder
```

----------------------------------------------------
USAGE
----------------------------------------------------

Basic scan:

```bash
python3 mega_finder.py -i targets.txt -o results
```

With external wordlist:

```bash
python3 mega_finder.py -i targets.txt -o results --wordlist mega_wordlist_1m.txt
```

Deeper scan:

```bash
python3 mega_finder.py -i targets.txt -o results --concurrency 60 --parallel-targets 4 --max-probes 200000
```

Fast/light scan:

```bash
python3 mega_finder.py -i targets.txt --max-probes 5000 --max-pages 50 --max-depth 1
```

----------------------------------------------------
OUTPUT STRUCTURE
----------------------------------------------------

results/
  target.com/
    summary.json
    summary.csv
    *.env
    *.log
    swagger.json
  global_summary.json

----------------------------------------------------
ARGUMENTS
----------------------------------------------------

-i, --input            File with targets
-o, --output           Output directory
--wordlist             External wordlist file
--concurrency          Concurrent HTTP probes
--parallel-targets     Domains scanned in parallel
--max-pages            Crawler limit
--max-depth            Crawler depth
--max-probes           Limit probes per target
--aggressiveness       Internal permutation aggressiveness

----------------------------------------------------
DETECTION CAPABILITIES
----------------------------------------------------

CRITICAL:
- Exposed .env
- Git repo leaks
- SQL dumps
- Secrets/tokens
- GraphQL introspection
- Cloud bucket exposures

HIGH:
- OpenAPI / Swagger
- CI/CD panels
- Log leaks
- Directory listing
- JS secret leaks

MEDIUM:
- phpinfo
- Debug endpoints
- Version leaks

----------------------------------------------------
TEST ENVIRONMENT
----------------------------------------------------

- Python 3.8+
- Linux / macOS / Windows supported

----------------------------------------------------
CONTRIBUTING
----------------------------------------------------

Pull requests welcome:
- Add detectors
- Improve JS analysis
- Optimize scanner
- Add config support

----------------------------------------------------
LICENSE
----------------------------------------------------

MIT License

----------------------------------------------------
CREDITS
----------------------------------------------------

Made by Bonagiri Varshith
For Red Team recon & pentesting automation.
