"""
Load the CSV graph produced by build_graph.py as an in-memory graph object.

    --format networkx  -> <out>/graph/hetnet.nx.pkl   networkx.MultiDiGraph, node attribute `type` in {author, document}
    --format pyg       -> <out>/graph/hetnet.pyg.pt   torch_geometric HeteroData (needs torch + torch_geometric)

Node ids: documents are "d:<doc_id>", authors "a:<author_code>" (networkx). For PyG, node order follows
documents.csv.gz / authors.csv.gz; the id mappings are stored as `data[type].ids`.
A networkx graph of the full zbMATH network needs a lot of RAM (tens of GB); PyG is much more compact.
"""
import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd

log = logging.getLogger("export_graph")


def load_tables(graph_dir):
    read = lambda name, **kw: pd.read_csv(os.path.join(graph_dir, name), **kw)
    docs = read("documents.csv.gz", dtype=str, keep_default_na=False).astype({"doc_id": "int64"})
    authors = read("authors.csv.gz", dtype=str, keep_default_na=False)
    writes = read("writes.csv.gz", dtype={"author_code": str, "doc_id": "int64", "author_position": "Int64"},
                  keep_default_na=False)
    cites = read("cites.csv.gz", dtype={"citing_doc_id": "int64", "cited_doc_id": "int64", "reference_position": "Int64"})
    return docs, authors, writes, cites


def to_networkx(docs, authors, writes, cites):
    import networkx as nx

    g = nx.MultiDiGraph()
    g.add_nodes_from((f"d:{r['doc_id']}", {"type": "document", **r}) for r in docs.to_dict("records"))
    g.add_nodes_from((f"a:{r['author_code']}", {"type": "author", **r}) for r in authors.to_dict("records"))
    g.add_edges_from((f"a:{a}", f"d:{d}", {"type": "writes", "position": p})
                     for a, d, p in writes.itertuples(index=False))
    g.add_edges_from((f"d:{s}", f"d:{t}", {"type": "cites", "position": p})
                     for s, t, p in cites.itertuples(index=False))
    return g


def to_pyg(docs, authors, writes, cites):
    import torch
    from torch_geometric.data import HeteroData

    doc_index = pd.Series(range(len(docs)), index=docs["doc_id"])
    author_index = pd.Series(range(len(authors)), index=authors["author_code"])
    data = HeteroData()
    data["document"].num_nodes = len(docs)
    data["document"].ids = docs["doc_id"].tolist()
    data["document"].year = torch.tensor(pd.to_numeric(docs["year"], errors="coerce").fillna(-1).astype("int64").values)
    data["author"].num_nodes = len(authors)
    data["author"].ids = authors["author_code"].tolist()

    w = torch.from_numpy(np.stack([author_index.loc[writes["author_code"]].values,
                                   doc_index.loc[writes["doc_id"]].values]))
    c = torch.from_numpy(np.stack([doc_index.loc[cites["citing_doc_id"]].values,
                                   doc_index.loc[cites["cited_doc_id"]].values]))
    data["author", "writes", "document"].edge_index = w
    data["document", "written_by", "author"].edge_index = w.flip(0)
    data["document", "cites", "document"].edge_index = c
    data["document", "cited_by", "document"].edge_index = c.flip(0)
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--format", choices=["networkx", "pyg"], default="pyg")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    graph_dir = os.path.join(args.out_dir, "graph")
    tables = load_tables(graph_dir)
    if args.format == "networkx":
        g = to_networkx(*tables)
        path = os.path.join(graph_dir, "hetnet.nx.pkl")
        with open(path, "wb") as f:
            pickle.dump(g, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("networkx graph with %d nodes / %d edges saved to %s", g.number_of_nodes(), g.number_of_edges(), path)
    else:
        import torch

        data = to_pyg(*tables)
        path = os.path.join(graph_dir, "hetnet.pyg.pt")
        torch.save(data, path)
        log.info("PyG HeteroData saved to %s:\n%s", path, data)


if __name__ == "__main__":
    main()
