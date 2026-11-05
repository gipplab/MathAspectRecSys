# zbMATH Open Author–Document Network

Harvests documents and authors from the [zbMATH Open API](https://api.zbmath.org/v1) and builds a heterogeneous network in which documents are connected through citations and authors:

```
author ──writes──▶ document ──cites──▶ document
```

| Node type  | Identifier                                     | Source                                                 |
|------------|------------------------------------------------|--------------------------------------------------------|
| `author`   | zbMATH author code, e.g. `serre.jean-pierre`   | `/v1/author/_search` and author codes on documents     |
| `document` | zbMATH document id, e.g. `3639144`             | `/v1/document/_search` and ids in reference lists      |

| Edge type  | From → to            | Source                                               |
|------------|----------------------|------------------------------------------------------|
| `writes`   | author → document    | `contributors.authors[].codes`                       |
| `cites`    | document → document  | `references[].zbmath.document_id`                    |

## Usage

### Try it on a small slice first:

```bash
python crawl.py documents --out-dir sample --query "py:2020 & cc:*35K"
python crawl.py authors --out-dir sample --author-query "ai:bh*"
python build_graph.py --out-dir sample
python export_graph.py --out-dir sample --format networkx
```

you can use test_graph.py to check for sample node and its connections.

### Full run

```bash
pip install -r requirements.txt
python crawl.py all --out-dir output                      # all documents and authors (resumable)
python build_graph.py --out-dir output [--with-text]      # node and edge tables
python export_graph.py --out-dir output --format pyg      # optional: PyTorch Geometric or networkx object
```

`run_all.sh` runs the three steps in sequence.

Specific documents can be fetched by Zbl number or document id, optionally including the documents they cite:

```bash
python fetch_ids.py --ids-file ids.txt --out-dir output --citation-hops 1
```

A crawl can be restricted with a zbMATH search string, e.g. `--query "py:2020 & cc:*35K"` for documents and `--author-query "ai:bh*"` for authors.

## Output

Raw API records are stored unchanged as gzipped JSON lines in `output/raw/{document,author}/shards/`. `build_graph.py` writes:

| File (`output/graph/`) | Columns |
|------------------------|---------|
| `documents.csv.gz` | doc_id, zbl_id, title, year, document_type, language, msc (`;`-separated, primary first), keywords, source, doi, arxiv, zbmath_url, n_authors, n_references, n_resolved_references, in_crawl (+ review with `--with-text`) |
| `authors.csv.gz` | author_code, name, main_fields, spellings, external_ids (JSON), awards, zbmath_url, in_author_api |
| `writes.csv.gz` | author_code, doc_id, author_position |
| `cites.csv.gz` | citing_doc_id, cited_doc_id, reference_position |
| `graph_stats.json` | node and edge counts |

Documents known only from another document's reference list have `in_crawl = False`; author codes that occur on documents but not in the author endpoint have `in_author_api = False`. Fields withheld by the API for licensing reasons are left empty.

## Harvesting Strategy

The API returns at most 10,000 hits per query (at most 500 per page) and offers no cursor. `partition.py` therefore splits the search space into disjoint sub-queries with at most 10,000 hits each:

- documents: decade → year → primary MSC class (2, 3 and 5 characters) → document type
- authors: author-code prefix (`ai:a*`, `ai:ab*`, …)

Each split adds a residual query `parent & !(child_1 | … | child_n)`, so the sub-queries always cover their parent completely. The plan with its hit counts is stored in `raw/<endpoint>/plan.json`; finished shards are skipped when a crawl is restarted.

## Runtime and Storage

A full crawl of about 5.2 million documents takes on the order of one to two days, depending on `--workers` and server load; the 1.4 million authors take considerably less. Raw document shards require about 2 KB per document (≈ 10 GB in total). Please keep the number of parallel requests moderate and use `--min-interval` when requested by the service.
