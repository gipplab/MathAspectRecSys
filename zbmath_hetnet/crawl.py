"""
Harvest all zbMATH Open documents and/or authors from the zbMATH REST API.

Raw API records are stored unchanged as gzipped JSON lines, one shard per partition query:
    <out>/raw/<endpoint>/plan.json              partition plan (leaf queries + hit counts)
    <out>/raw/<endpoint>/shards/<hash>.jsonl.gz one finished shard per leaf query

The crawl is resumable: re-running skips the planning step (unless --replan) and all finished shards.

Examples:
    python crawl.py all --out-dir output
    python crawl.py documents --out-dir sample --query "py:2020 & cc:*35K"   # small test crawl
"""
import argparse
import gzip
import hashlib
import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from partition import plan_partitions
from zbmath_client import DEFAULT_RESULTS_PER_PAGE, MAX_RESULT_WINDOW, MAX_RESULTS_PER_PAGE, ZbMathClient

log = logging.getLogger("crawl")


def shard_path(shard_dir, query):
    return os.path.join(shard_dir, hashlib.sha1(query.encode("utf8")).hexdigest()[:20] + ".jsonl.gz")


def load_or_plan(client, endpoint, raw_dir, root_query, workers, replan):
    plan_file = os.path.join(raw_dir, "plan.json")
    if os.path.exists(plan_file) and not replan:
        with open(plan_file, encoding="utf8") as f:
            plan = json.load(f)
        if plan["root_query"] == root_query:
            log.info("[%s] using existing plan %s (%d leaves)", endpoint, plan_file, len(plan["leaves"]))
            return plan["leaves"]
        log.info("[%s] root query changed, re-planning", endpoint)
    total = client.count(endpoint, root_query)
    log.info("[%s] %d hits for %r, planning partitions ...", endpoint, total, root_query)
    leaves = plan_partitions(client, endpoint, root_query, workers)
    planned = sum(l["count"] for l in leaves)
    truncated = [l for l in leaves if l["truncated"]]
    log.info("[%s] %d leaf queries covering %d hits (API total %d); %d truncated leaves",
             endpoint, len(leaves), planned, total, len(truncated))
    tmp = plan_file + ".tmp"
    with open(tmp, "w", encoding="utf8") as f:
        json.dump({"root_query": root_query, "api_total": total, "planned_total": planned, "leaves": leaves},
                  f, indent=1, ensure_ascii=False)
    os.replace(tmp, plan_file)
    return leaves


def fetch_leaf(client, endpoint, leaf, path, page_size):
    """Download every page of one leaf query into a shard file. Returns the number of records written."""
    pages = math.ceil(min(leaf["count"], MAX_RESULT_WINDOW) / page_size)
    tmp = path + ".tmp"
    n = 0
    with gzip.open(tmp, "wt", encoding="utf8") as f:
        for page in range(pages):
            results, _ = client.search(endpoint, leaf["query"], page=page, results_per_page=page_size)
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += len(results)
            if len(results) < page_size:
                break
    os.replace(tmp, path)
    return n


def crawl(client, endpoint, out_dir, root_query, workers, replan, max_leaves=None,
          page_size=DEFAULT_RESULTS_PER_PAGE):
    raw_dir = os.path.join(out_dir, "raw", endpoint)
    shard_dir = os.path.join(raw_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    leaves = load_or_plan(client, endpoint, raw_dir, root_query, workers, replan)
    if max_leaves:
        leaves = leaves[:max_leaves]
    todo = [l for l in leaves if not os.path.exists(shard_path(shard_dir, l["query"]))]
    log.info("[%s] %d of %d shards left to download", endpoint, len(todo), len(leaves))

    written = 0
    with ThreadPoolExecutor(workers) as pool, \
            tqdm(total=sum(min(l["count"], MAX_RESULT_WINDOW) for l in todo), unit="rec", desc=endpoint) as bar:
        futures = {pool.submit(fetch_leaf, client, endpoint, l, shard_path(shard_dir, l["query"]), page_size): l
                   for l in todo}
        for fut in as_completed(futures):
            leaf = futures[fut]
            try:
                n = fut.result()
            except Exception:
                log.exception("[%s] failed leaf %r; re-run to retry", endpoint, leaf["query"])
                continue
            written += n
            bar.update(min(leaf["count"], MAX_RESULT_WINDOW))
    missing = sum(not os.path.exists(shard_path(shard_dir, l["query"])) for l in leaves)
    log.info("[%s] wrote %d records this run; %d shards still missing", endpoint, written, missing)
    return missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["documents", "authors", "all"])
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--query", default="*", help="zbMATH search string restricting the documents crawl (default: all)")
    ap.add_argument("--author-query", default="*", help="search string restricting the authors crawl (default: all)")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests")
    ap.add_argument("--min-interval", type=float, default=0.0, help="min seconds between requests (rate limit)")
    ap.add_argument("--page-size", type=int, default=DEFAULT_RESULTS_PER_PAGE,
                    help=f"records per request (max {MAX_RESULTS_PER_PAGE}; must divide {MAX_RESULT_WINDOW})")
    ap.add_argument("--replan", action="store_true", help="recompute the partition plan")
    ap.add_argument("--max-leaves", type=int, help="only download the first N partitions (testing)")
    args = ap.parse_args()
    if not 0 < args.page_size <= MAX_RESULTS_PER_PAGE or MAX_RESULT_WINDOW % args.page_size:
        ap.error(f"--page-size must be in 1..{MAX_RESULTS_PER_PAGE} and divide {MAX_RESULT_WINDOW}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    client = ZbMathClient(min_interval=args.min_interval)
    missing = 0
    if args.what in ("documents", "all"):
        missing += crawl(client, "document", args.out_dir, args.query, args.workers, args.replan, args.max_leaves,
                         args.page_size)
    if args.what in ("authors", "all"):
        missing += crawl(client, "author", args.out_dir, args.author_query, args.workers, args.replan,
                         args.max_leaves, args.page_size)
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
