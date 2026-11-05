import pickle
import random

with open("sample/graph/hetnet.nx.pkl", "rb") as f:
    g = pickle.load(f)

print(g)  # e.g. MultiDiGraph with N nodes and M edges

# Pick a sample node: a random document, or use a known id like "d:1234567"
docs = [n for n, t in g.nodes(data="type") if t == "document"]
node = random.choice(docs)
print(node, g.nodes[node])  # node id and its attributes (title, year, ...)

# Outgoing: documents this one cites (for an author: documents they wrote)
out_nbrs = list(g.successors(node))

# Incoming: documents citing this one, plus its authors
in_nbrs = list(g.predecessors(node))

# Split by edge type
cites     = [v for _, v, t in g.out_edges(node, data="type") if t == "cites"]
cited_by  = [u for u, _, t in g.in_edges(node, data="type") if t == "cites"]
authors   = [u for u, _, t in g.in_edges(node, data="type") if t == "writes"]

print("cites:", cites[:10])
print("cited by:", cited_by[:10])
print("authors:", authors)

author = "a:smith.john"  # replace with a real author_code
papers = [v for _, v, t in g.out_edges(author, data="type") if t == "writes"]
coauthors = {u for p in papers for u in g.predecessors(p)
             if g.nodes[u]["type"] == "author"} - {author}