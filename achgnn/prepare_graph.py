"""
Step 3: turn the zbMATH network (zbmath_hetnet/build_graph.py --with-text) into the AchGNN graph (paper, Sec. 3.2.1):

    nodes   papers P (text = title + abstract with in-abstract references removed), authors A (name)
    edges   citations E_c (paper - paper, undirected), authorship E_a (author - paper)
    labels  aspect-labelled pairs E_as from GoldRiM/SilverRiM, mapped to paper indices

    python achgnn/prepare_graph.py --graph-dir zbmath_hetnet/output/graph --work work
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import ASPECTS, Work, log, remove_document_references, save_json, setup_logging


def author_display_name(code, name):
    if isinstance(name, str) and name.strip():
        return name.strip()
    # e.g. "tate.john-torrence" -> "Tate, John Torrence"
    last, _, first = str(code).partition(".")
    return f"{last.replace('-', ' ').title()}, {first.replace('-', ' ').title()}".strip(", ")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph-dir", required=True, help="output/graph of zbmath_hetnet (built with --with-text)")
    ap.add_argument("--work", default="work")
    ap.add_argument("--english-only", action="store_true",
                    help="restrict the retrieval corpus to English documents (as in the paper)")
    ap.add_argument("--min-words", type=int, default=0,
                    help="papers whose text has fewer words are kept as graph nodes but are not retrieval candidates")
    args = ap.parse_args()
    setup_logging()
    work = Work(args.work)
    g = lambda name: os.path.join(args.graph_dir, name)

    docs = pd.read_csv(g("documents.csv.gz"), dtype=str, keep_default_na=False)
    if "review" not in docs.columns:
        raise SystemExit("documents.csv.gz has no review column: run build_graph.py with --with-text")
    docs["doc_id"] = docs["doc_id"].astype("int64")
    docs = docs.drop_duplicates("doc_id").reset_index(drop=True)
    docs["text"] = [
        (f"{t}. " if t else "") + remove_document_references(r) for t, r in zip(docs["title"], docs["review"])
    ]
    docs["text"] = docs["text"].str.strip()
    candidate = (docs["in_crawl"] == "True") & (docs["text"].str.split().str.len() >= max(args.min_words, 1))
    if args.english_only:
        candidate &= docs["language"].str.contains("English", na=False)
    docs["candidate"] = candidate
    paper_index = pd.Series(np.arange(len(docs)), index=docs["doc_id"])
    docs[["doc_id", "zbl_id", "title", "year", "msc", "text", "candidate"]].to_csv(work.papers, index=False)
    np.save(work.candidates, np.nonzero(candidate.values)[0].astype(np.int64))

    authors = pd.read_csv(g("authors.csv.gz"), dtype=str, keep_default_na=False).drop_duplicates("author_code")
    authors["name"] = [author_display_name(c, n) for c, n in zip(authors["author_code"], authors["name"])]
    authors = authors[["author_code", "name"]].reset_index(drop=True)
    author_index = pd.Series(np.arange(len(authors)), index=authors["author_code"])
    authors.to_csv(work.authors, index=False)

    cites = pd.read_csv(g("cites.csv.gz"), usecols=["citing_doc_id", "cited_doc_id"], dtype="int64")
    writes = pd.read_csv(g("writes.csv.gz"), usecols=["author_code", "doc_id"], dtype={"author_code": str},
                         keep_default_na=False)
    writes = writes[writes["author_code"].isin(author_index.index) & writes["doc_id"].isin(paper_index.index)]
    cites = cites[cites["citing_doc_id"].isin(paper_index.index) & cites["cited_doc_id"].isin(paper_index.index)]
    c_src, c_dst = paper_index.loc[cites["citing_doc_id"]].values, paper_index.loc[cites["cited_doc_id"]].values
    w_author, w_paper = author_index.loc[writes["author_code"]].values, paper_index.loc[writes["doc_id"]].values
    np.savez(work.graph, cite_src=c_src.astype(np.int64), cite_dst=c_dst.astype(np.int64),
             write_author=w_author.astype(np.int64), write_paper=w_paper.astype(np.int64),
             num_papers=len(docs), num_authors=len(authors))

    # ---- aspect-labelled pairs
    pairs = pd.read_csv(work.pairs_raw, dtype={"seed": str, "rec": str})
    zbl_index = pd.Series(np.arange(len(docs)), index=docs["zbl_id"].where(docs["zbl_id"] != "", None))
    zbl_index = zbl_index[zbl_index.index.notna() & ~zbl_index.index.duplicated()]
    pairs["seed_idx"] = pairs["seed"].map(zbl_index)
    pairs["rec_idx"] = pairs["rec"].map(zbl_index)
    missing = pairs["seed_idx"].isna() | pairs["rec_idx"].isna()
    if missing.any():
        log.warning("%d of %d pairs reference documents that are not in the graph and are dropped "
                    "(fetch them with zbmath_hetnet/fetch_ids.py)", missing.sum(), len(pairs))
    pairs = pairs[~missing].astype({"seed_idx": "int64", "rec_idx": "int64"})
    pairs["aspect_id"] = pairs["aspect"].map({a: i for i, a in enumerate(ASPECTS)})
    pairs.to_csv(work.pairs, index=False)

    stats = {
        "papers": len(docs), "candidate_papers": int(candidate.sum()), "authors": len(authors),
        "citation_edges": int(len(c_src)), "authorship_edges": int(len(w_author)),
        "pairs_gold": int(pairs["gold"].sum()), "pairs_silver": int(pairs["silver"].sum()),
        "pairs_dropped": int(missing.sum()), "aspects": ASPECTS,
    }
    save_json(stats, work.path("graph_stats.json"))
    log.info("AchGNN graph: %s", stats)


if __name__ == "__main__":
    main()
