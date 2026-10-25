"""
Build the heterogeneous author / document citation network from the raw crawl.

Node types
    document  documents.csv.gz  one row per zbMATH document (crawled, or only known as a cited reference)
    author    authors.csv.gz    one row per zbMATH author code (from the author API, or only seen on a document)
Edge types
    author -[writes]-> document    writes.csv.gz   (author_code, doc_id, author_position)
    document -[cites]-> document   cites.csv.gz    (citing_doc_id, cited_doc_id, reference_position)

Authors are identified by their zbMATH author code (e.g. "serre.jean-pierre"), documents by their
zbMATH document id (e.g. 3639144). Raw records stay available in <out>/raw for any further attributes.

Example:
    python build_graph.py --out-dir output
"""
import argparse
import csv
import glob
import gzip
import json
import logging
import os

from tqdm import tqdm

log = logging.getLogger("build_graph")

DOC_COLUMNS = ["doc_id", "zbl_id", "title", "year", "document_type", "language", "msc", "keywords", "source",
               "doi", "arxiv", "zbmath_url", "n_authors", "n_references", "n_resolved_references", "in_crawl"]
DOC_TEXT_COLUMNS = ["review"]
AUTHOR_COLUMNS = ["author_code", "name", "main_fields", "spellings", "external_ids", "awards", "zbmath_url",
                  "in_author_api"]


# placeholder the API returns instead of fields it may not redistribute (titles, reviews, reference texts, ...)
LICENSE_MASK = "zbMATH Open Web Interface contents unavailable due to conflicting licenses."


def scrub(x):
    """Replace license placeholders by None, recursively."""
    if isinstance(x, dict):
        return {k: scrub(v) for k, v in x.items()}
    if isinstance(x, list):
        return [scrub(v) for v in x]
    return None if x == LICENSE_MASK else x


def iter_records(raw_dir, endpoint):
    files = sorted(glob.glob(os.path.join(raw_dir, endpoint, "shards", "*.jsonl.gz")))
    if not files:
        log.warning("no %s shards found under %s", endpoint, raw_dir)
    for path in tqdm(files, desc=f"reading {endpoint} shards", unit="shard"):
        with gzip.open(path, "rt", encoding="utf8") as f:
            for line in f:
                yield scrub(json.loads(line))


def open_csv(path, columns):
    f = gzip.open(path, "wt", encoding="utf8", newline="")
    w = csv.writer(f)
    w.writerow(columns)
    return f, w


def join(values):
    return ";".join(str(v) for v in values if v not in (None, ""))


def link_id(doc, link_type):
    for link in doc.get("links") or []:
        if link.get("type") == link_type:
            return link.get("identifier")
    return None


def document_row(doc, with_text):
    title = doc.get("title") or {}
    refs = doc.get("references") or []
    authors = (doc.get("contributors") or {}).get("authors") or []
    row = [
        doc["id"], doc.get("identifier"), title.get("title"), doc.get("year"),
        (doc.get("document_type") or {}).get("code"),
        join((doc.get("language") or {}).get("languages") or []),
        join(m.get("code") for m in doc.get("msc") or []),  # first code is the primary MSC
        join(doc.get("keywords") or []),
        (doc.get("source") or {}).get("source"),
        link_id(doc, "doi"), link_id(doc, "arxiv"), doc.get("zbmath_url"),
        len(authors), len(refs), sum(1 for r in refs if (r.get("zbmath") or {}).get("document_id")), True,
    ]
    if with_text:
        reviews = [c.get("text") for c in doc.get("editorial_contributions") or [] if c.get("text")]
        row.append("\n\n".join(reviews))
    return row


def author_row(a, in_api=True):
    return [
        a["code"], a.get("name"),
        join(m.get("field") for m in a.get("main_fields") or []),
        join(s.get("name") for s in a.get("spellings") or []),
        json.dumps({e["type"]: e["external_id"] for e in a.get("external_ids") or [] if e.get("type")},
                   ensure_ascii=False) if a.get("external_ids") else "",
        json.dumps(a.get("awards"), ensure_ascii=False) if a.get("awards") else "",
        a.get("zbmath_url"), in_api,
    ]


def build(raw_dir, graph_dir, with_text=False):
    os.makedirs(graph_dir, exist_ok=True)
    stats = {}

    # ---- authors from the author API
    authors_f, authors_w = open_csv(os.path.join(graph_dir, "authors.csv.gz"), AUTHOR_COLUMNS)
    api_authors = set()
    for a in iter_records(raw_dir, "author"):
        code = a.get("code")
        if not code or code in api_authors:
            continue
        api_authors.add(code)
        authors_w.writerow(author_row(a))
    stats["authors_from_api"] = len(api_authors)

    # ---- documents, writes and cites edges
    docs_f, docs_w = open_csv(os.path.join(graph_dir, "documents.csv.gz"),
                              DOC_COLUMNS + (DOC_TEXT_COLUMNS if with_text else []))
    writes_f, writes_w = open_csv(os.path.join(graph_dir, "writes.csv.gz"), ["author_code", "doc_id", "author_position"])
    cites_f, cites_w = open_csv(os.path.join(graph_dir, "cites.csv.gz"),
                                ["citing_doc_id", "cited_doc_id", "reference_position"])
    crawled, cited = set(), set()
    stub_authors = {}  # author codes seen on documents but missing in the author API -> name
    n_writes = n_cites = n_unresolved = 0
    for doc in iter_records(raw_dir, "document"):
        doc_id = doc.get("id")
        if doc_id is None or doc_id in crawled:
            continue
        crawled.add(doc_id)
        docs_w.writerow(document_row(doc, with_text))

        authors = (doc.get("contributors") or {}).get("authors") or []
        for pos, a in enumerate(authors, 1):
            for code in a.get("codes") or []:
                writes_w.writerow([code, doc_id, pos])
                n_writes += 1
                if code not in api_authors and code not in stub_authors:
                    stub_authors[code] = a.get("name")

        seen_refs = set()
        for ref in doc.get("references") or []:
            cited_id = (ref.get("zbmath") or {}).get("document_id")
            if not cited_id:
                n_unresolved += 1
                continue
            if cited_id in seen_refs:
                continue
            seen_refs.add(cited_id)
            cited.add(cited_id)
            cites_w.writerow([doc_id, cited_id, ref.get("position")])
            n_cites += 1
    stats.update(documents_crawled=len(crawled), writes_edges=n_writes, cites_edges=n_cites,
                 unresolved_references=n_unresolved)

    # ---- cited documents that are not part of the crawl become stub nodes, built from the reference data
    missing = cited - crawled
    stats["documents_cited_only"] = len(missing)
    if missing:
        log.info("%d cited documents not in the crawl, adding them as stub nodes", len(missing))
        for doc in iter_records(raw_dir, "document"):
            for ref in doc.get("references") or []:
                z = ref.get("zbmath") or {}
                cited_id = z.get("document_id")
                if cited_id not in missing:
                    continue
                missing.discard(cited_id)
                row = [cited_id, None, None, z.get("year"), None, None, join(z.get("msc") or []), None, ref.get("text"),
                       ref.get("doi"), None, f"https://zbmath.org/{cited_id}", len(z.get("author_codes") or []),
                       None, None, False]
                docs_w.writerow(row + ([None] if with_text else []))
                for pos, code in enumerate(z.get("author_codes") or [], 1):
                    writes_w.writerow([code, cited_id, pos])
                    stats["writes_edges"] += 1
                    if code not in api_authors and code not in stub_authors:
                        stub_authors[code] = None
            if not missing:
                break

    for code, name in stub_authors.items():
        authors_w.writerow([code, name, "", "", "", "", f"https://zbmath.org/authors/{code}", False])
    stats["authors_not_in_api"] = len(stub_authors)
    stats["author_nodes"] = len(api_authors) + len(stub_authors)
    stats["document_nodes"] = stats["documents_crawled"] + stats["documents_cited_only"]

    for f in (authors_f, docs_f, writes_f, cites_f):
        f.close()
    with open(os.path.join(graph_dir, "graph_stats.json"), "w", encoding="utf8") as f:
        json.dump(stats, f, indent=1)
    log.info("graph written to %s: %s", graph_dir, stats)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="output", help="same --out-dir as used for crawl.py")
    ap.add_argument("--with-text", action="store_true", help="also store review/summary texts of documents")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    build(os.path.join(args.out_dir, "raw"), os.path.join(args.out_dir, "graph"), args.with_text)


if __name__ == "__main__":
    main()
