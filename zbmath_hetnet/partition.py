"""
Split a zbMATH search into disjoint sub-queries that each return <= 10,000 hits.

The zbMATH API only serves the first 10,000 results of any query (page * results_per_page <= 10,000),
so harvesting all ~5M documents / ~1.4M authors requires partitioning the search space:

documents: decade -> year -> primary MSC (2 digits) -> primary MSC (3 chars) -> primary MSC (5 chars)
           -> document type
authors:   author-code prefix, extended one character at a time (a*, ab*, abc*, ...)

Every split also adds a *residual* query `base & !(child_1 | child_2 | ...)`, so the children always
cover the parent completely (e.g. documents without MSC code, or MSC codes like `35-01`).
"""
import datetime
import logging
import string
from concurrent.futures import ThreadPoolExecutor

from zbmath_client import MAX_RESULT_WINDOW

log = logging.getLogger(__name__)

# level names (in order) used when splitting document queries
DOC_LEVELS = ["decade", "year", "msc2", "msc3", "msc5", "doctype"]
AUTHOR_CODE_CHARS = string.ascii_lowercase + string.digits + "-."


def _and(base, term):
    if base in ("*", ""):
        return term
    if "|" in base:  # keep a user query like "py:2020 | py:2021" together; & binds tighter than |
        base = f"({base})"
    return f"{base} & {term}"


def _doc_terms(level, ctx):
    """Return (terms, next_level, ctx_per_term) for splitting a document query at `level`."""
    if level == "decade":
        last = datetime.date.today().year + 1
        decades = list(range(1800, last + 1, 10))
        terms = ["py:0-1799"] + [f"py:{d}-{d + 9}" for d in decades]
        ctxs = [{"years": (0, 1799)}] + [{"years": (d, d + 9)} for d in decades]
        return terms, "year", ctxs
    if level == "year":
        lo, hi = ctx["years"]
        years = range(lo, hi + 1)
        return [f"py:{y}" for y in years], "msc2", [{} for _ in years]
    if level == "msc2":
        codes = [f"{i:02d}" for i in range(100)]
        return [f"cc:*{c}" for c in codes], "msc3", [{"msc": c} for c in codes]
    if level == "msc3":
        codes = [ctx["msc"] + c for c in string.ascii_uppercase]
        return [f"cc:*{c}" for c in codes], "msc5", [{"msc": c} for c in codes]
    if level == "msc5":
        codes = [ctx["msc"] + f"{i:02d}" for i in range(100)]
        return [f"cc:*{c}" for c in codes], "doctype", [{} for _ in codes]
    if level == "doctype":
        types = ["j", "a", "b"]
        return [f"dt:{t}" for t in types], None, [{} for _ in types]
    return [], None, []


def _author_terms(level, ctx):
    prefix = ctx.get("prefix", "")
    if len(prefix) >= 30:
        return [], None, []
    nxt = [prefix + c for c in AUTHOR_CODE_CHARS]
    return [f"ai:{p}*" for p in nxt], "prefix", [{"prefix": p} for p in nxt]


def _children(endpoint, node):
    """Split a node into child nodes (including the residual node). Empty list = cannot split further."""
    level, ctx, base = node["level"], node.get("ctx", {}), node["query"]
    if endpoint == "document":
        terms, next_level, ctxs = _doc_terms(level, ctx)
    else:
        terms, next_level, ctxs = _author_terms(level, ctx)
    if not terms:
        return []
    # author prefixes are nested (ai:ab* lies within ai:a*), so they are combined with the root query only
    term_base = node.get("root", base) if endpoint == "author" else base
    kids = [
        {"query": _and(term_base, t), "level": next_level, "ctx": c, "root": node.get("root", base)}
        for t, c in zip(terms, ctxs)
    ]
    # Residual: whatever the children above do not cover. It has no structure we know of, so it skips the
    # remaining structural levels and is only split by document type (documents) or not at all (authors).
    residual = _and(base, f"!({' | '.join(terms)})") if base not in ("*", "") else f"!({' | '.join(terms)})"
    kids.append({"query": residual, "level": "doctype" if endpoint == "document" and level != "doctype" else None,
                 "ctx": {}, "residual": True})
    return kids


def plan_partitions(client, endpoint, root_query="*", workers=4):
    """
    Return a list of leaf queries [{query, count, truncated}] whose result sets are disjoint and
    together cover `root_query`. Leaves with count > 10,000 that cannot be split further are marked
    truncated (only the first 10,000 hits will be harvested).
    """
    root_level = "decade" if endpoint == "document" else "prefix"
    frontier = [{"query": root_query, "level": root_level, "ctx": {}}]
    leaves = []
    depth = 0
    with ThreadPoolExecutor(workers) as pool:
        while frontier:
            counts = list(pool.map(lambda n: client.count(endpoint, n["query"]), frontier))
            next_frontier = []
            for node, count in zip(frontier, counts):
                if count == 0:
                    continue
                if count <= MAX_RESULT_WINDOW:
                    leaves.append({"query": node["query"], "count": count, "truncated": False})
                    continue
                kids = _children(endpoint, node) if node["level"] else []
                if kids:
                    next_frontier.extend(kids)
                else:
                    log.warning("cannot split %r (%d hits) further; only %d will be harvested",
                                node["query"], count, MAX_RESULT_WINDOW)
                    leaves.append({"query": node["query"], "count": count, "truncated": True})
            depth += 1
            log.info("[%s] partition depth %d: %d leaves so far (%d hits), %d nodes to split next",
                     endpoint, depth, len(leaves), sum(l["count"] for l in leaves), len(next_frontier))
            frontier = next_frontier
    return leaves
