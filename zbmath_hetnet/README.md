# zbMATH Open author–document heterogeneous network

Harvests **all** documents and authors from the [zbMATH Open API](https://api.zbmath.org/v1) and builds a
heterogeneous graph in which documents are connected through **citations** and through their **authors**.

```
author ──writes──▶ document ──cites──▶ document
```

| Node type  | ID                                             | Source                                                   |
|------------|------------------------------------------------|----------------------------------------------------------|
| `author`   | zbMATH author code, e.g. `serre.jean-pierre`   | `/v1/author/_search` (+ codes seen on documents)         |
| `document` | zbMATH document id, e.g. `3639144`             | `/v1/document/_search` (+ ids seen only in references)   |

| Edge type  | From → to            | Source                                                       |
|------------|----------------------|--------------------------------------------------------------|
| `writes`   | author → document    | `contributors.authors[].codes` of each document              |
| `cites`    | document → document  | `references[].zbmath.document_id` of each document           |

## Usage

```bash
pip install -r requirements.txt
bash run_all.sh output          # = the three steps below
```

1. **Crawl** – `python crawl.py all --out-dir output [--workers 4]`
   Stores the raw API records unchanged (gzipped JSON lines) in `output/raw/{document,author}/shards/`.
2. **Build** – `python build_graph.py --out-dir output [--with-text]`
   Writes node/edge tables to `output/graph/`.
3. **Export** (optional) – `python export_graph.py --out-dir output --format pyg|networkx`
   Saves a PyTorch Geometric `HeteroData` (`hetnet.pyg.pt`) or a `networkx.MultiDiGraph` (`hetnet.nx.pkl`).

Try it on a small slice first:

```bash
python crawl.py documents --out-dir sample --query "py:2020 & cc:*35K"
python crawl.py authors --out-dir sample --author-query "ai:bh*"
python build_graph.py --out-dir sample
python export_graph.py --out-dir sample --format networkx
```

## Output (`output/graph/`)

| File                | Columns |
|---------------------|---------|
| `documents.csv.gz`  | doc_id, zbl_id, title, year, document_type, language, msc (`;`-separated, primary first), keywords, source, doi, arxiv, zbmath_url, n_authors, n_references, n_resolved_references, in_crawl (+ review with `--with-text`) |
| `authors.csv.gz`    | author_code, name, main_fields (2-digit MSC), spellings, external_ids (JSON: orcid, wikidata, mgp, …), awards, zbmath_url, in_author_api |
| `writes.csv.gz`     | author_code, doc_id, author_position |
| `cites.csv.gz`      | citing_doc_id, cited_doc_id, reference_position |
| `graph_stats.json`  | node/edge counts |

`in_crawl = False` marks documents only known from another document's reference list; `in_author_api = False`
marks author codes that appear on documents but were not returned by the author endpoint. Any field not in the
tables is still available in the raw records. Fields the API withholds for licensing reasons
("contents unavailable due to conflicting licenses") are left empty.

## How the crawl covers everything

The API returns at most 10,000 hits per query (`page * results_per_page ≤ 10,000`, ≤ 500 per page) and has no
cursor. `partition.py` therefore splits the search into disjoint sub-queries of ≤ 10,000 hits each:

- documents: decade → year (`py:2020`) → primary MSC (`cc:*35`, `cc:*35K`, `cc:*35K55`) → document type (`dt:j`)
- authors: author-code prefix (`ai:a*`, `ai:ab*`, …)

Every split also adds a residual query `parent & !(child1 | child2 | …)`, so records that match no child (e.g.
documents without MSC code) are not lost. The plan (with hit counts) is saved in `raw/<endpoint>/plan.json`; leaves
that could not be split below 10,000 hits are flagged `"truncated": true` and logged.

## Runtime and size

The API spends roughly 30 ms per returned document, so a full crawl of ~5.2M documents takes on the order of a day
or more, depending on `--workers` and server load; authors (~1.4M) are much faster. Raw document shards take about
2 KB per document gzipped (~10 GB in total). The crawl is resumable: interrupt and re-run the same command, finished
shards are skipped. Please keep `--workers` moderate (and use `--min-interval` if the API asks you to slow down) –
this is a shared public service.
