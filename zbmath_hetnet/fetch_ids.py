"""
Download specific zbMATH documents by ID (Zbl number like 0940.14001, or internal id like 1363213), optionally
following their references for a number of hops. Useful to build a graph around a set of documents (e.g. the
GoldRiM/SilverRiM seeds and recommendations) without crawling all of zbMATH Open.

Records are written in the same format as crawl.py (<out>/raw/document/shards/ids-*.jsonl.gz), so
build_graph.py picks them up together with any crawled shards. Documents already present in a shard are skipped.

Example:
    python fetch_ids.py --ids-file ../achgnn_work/dataset_ids.txt --out-dir output --citation-hops 1
"""
import argparse
import glob
import gzip
import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from zbmath_client import ZbMathClient

log = logging.getLogger("fetch_ids")
IDS_PER_QUERY = 50


def existing_documents(shard_dir):
    """Return ({internal id}, {Zbl id}) of all documents already stored in shards."""
    ids, zbl = set(), set()
    for path in glob.glob(os.path.join(shard_dir, "*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf8") as f:
            for line in f:
                doc = json.loads(line)
                ids.add(str(doc["id"]))
                zbl.add(doc.get("identifier"))
    return ids, zbl


def fetch_batch(client, batch):
    query = "an:(" + " | ".join(batch) + ")"
    results, total = client.search("document", query, page=0, results_per_page=max(len(batch) * 2, 10))
    return results


def fetch(client, ids, shard_dir, workers, hop):
    """Fetch the given ids into one shard; returns the fetched records."""
    batches = [ids[i:i + IDS_PER_QUERY] for i in range(0, len(ids), IDS_PER_QUERY)]
    records = []
    with ThreadPoolExecutor(workers) as pool:
        for res in tqdm(pool.map(lambda b: fetch_batch(client, b), batches), total=len(batches),
                        desc=f"hop {hop}", unit="batch"):
            records.extend(res)
    name = hashlib.sha1("\n".join(sorted(ids)).encode()).hexdigest()[:20]
    path = os.path.join(shard_dir, f"ids-{name}.jsonl.gz")
    with gzip.open(path + ".tmp", "wt", encoding="utf8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)
    log.info("hop %d: requested %d ids, got %d documents -> %s", hop, len(ids), len(records), path)
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids-file", required=True, help="text file, one Zbl number or internal document id per line")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--citation-hops", type=int, default=0,
                    help="also fetch documents referenced by the fetched ones, this many hops deep")
    ap.add_argument("--max-per-hop", type=int, default=0,
                    help="cap the number of cited documents fetched per citation hop (0: no cap)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    shard_dir = os.path.join(args.out_dir, "raw", "document", "shards")
    os.makedirs(shard_dir, exist_ok=True)
    have_ids, have_zbl = existing_documents(shard_dir)
    with open(args.ids_file, encoding="utf8") as f:
        wanted = [l.strip() for l in f if l.strip()]
    todo = sorted({i for i in wanted if i not in have_ids and i not in have_zbl})
    log.info("%d ids requested, %d not yet downloaded", len(set(wanted)), len(todo))

    client = ZbMathClient()
    for hop in range(args.citation_hops + 1):
        if args.max_per_hop and hop > 0:  # the requested ids themselves are always fetched
            todo = todo[:args.max_per_hop]
        if not todo:
            break
        records = fetch(client, todo, shard_dir, args.workers, hop)
        for doc in records:
            have_ids.add(str(doc["id"]))
            have_zbl.add(doc.get("identifier"))
        cited = {str((r.get("zbmath") or {}).get("document_id"))
                 for doc in records for r in doc.get("references") or []
                 if (r.get("zbmath") or {}).get("document_id")}
        todo = sorted(cited - have_ids)


if __name__ == "__main__":
    main()
