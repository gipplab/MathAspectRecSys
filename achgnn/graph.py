"""
Heterogeneous paper/author graph with fixed-size neighbourhood sampling (GraphSAGE style), in plain numpy/PyTorch
so it runs on any machine with or without a GPU and without torch_geometric.

Relations used for message passing (paper, Sec. 3.2.2, R = {c, a}):
    ("paper",  "cites",     "paper")   citation edges, treated as undirected
    ("author", "writes",    "paper")   authorship, author -> paper
    ("paper",  "writes_by", "author")  authorship, paper -> author (so authors get updated too)
"""
import numpy as np
import torch

from common import Work

RELATIONS = [("paper", "cites", "paper"), ("author", "writes", "paper"), ("paper", "writes_by", "author")]
NODE_TYPES = ["paper", "author"]


class CSR:
    """For every destination node, the list of its source neighbours."""

    def __init__(self, src, dst, num_dst):
        order = np.argsort(dst, kind="stable")
        self.indices = src[order].astype(np.int64)
        self.indptr = np.zeros(num_dst + 1, dtype=np.int64)
        np.cumsum(np.bincount(dst, minlength=num_dst), out=self.indptr[1:])

    def sample(self, nodes, fanout, rng):
        """
        Sample up to `fanout` neighbours of each node (all of them if the degree is smaller, otherwise uniformly
        with replacement). Returns (dst_position, src_node) arrays; dst_position indexes into `nodes`.
        """
        start = self.indptr[nodes]
        deg = self.indptr[nodes + 1] - start
        cnt = np.minimum(deg, fanout) if fanout > 0 else deg
        total = int(cnt.sum())
        if total == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        dst_pos = np.repeat(np.arange(len(nodes), dtype=np.int64), cnt)
        offs = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        big = np.repeat(deg > cnt, cnt)
        if big.any():
            rand = (rng.random(int(big.sum())) * np.repeat(deg, cnt)[big]).astype(np.int64)
            offs[big] = rand
        return dst_pos, self.indices[np.repeat(start, cnt) + offs]


class HeteroGraph:
    def __init__(self, work: Work, drop_authorship=False, random_authorship=False, seed=0):
        g = np.load(work.graph)
        self.num_nodes = {"paper": int(g["num_papers"]), "author": int(g["num_authors"])}
        cs, cd = g["cite_src"], g["cite_dst"]
        keep = cs != cd
        und_src, und_dst = np.concatenate([cs[keep], cd[keep]]), np.concatenate([cd[keep], cs[keep]])
        pairs = np.unique(np.stack([und_src, und_dst], 1), axis=0) if len(und_src) else np.zeros((0, 2), np.int64)
        wa, wp = g["write_author"], g["write_paper"]
        if random_authorship:  # ablation (paper, Sec. 4.4.2): authors assigned to random papers
            wp = np.random.default_rng(seed).integers(0, self.num_nodes["paper"], len(wp))
        if drop_authorship:   # ablation: no authorship edges
            wa, wp = wa[:0], wp[:0]
        self.csr = {
            RELATIONS[0]: CSR(pairs[:, 0], pairs[:, 1], self.num_nodes["paper"]),
            RELATIONS[1]: CSR(wa, wp, self.num_nodes["paper"]),
            RELATIONS[2]: CSR(wp, wa, self.num_nodes["author"]),
        }

    def sample_blocks(self, seeds, fanouts, num_layers, rng):
        """
        Build the computation graph of `num_layers` message-passing layers for the given seed nodes.

        seeds: array of paper ids, or {type: ids}. fanouts: {relation: max sampled neighbours (0 = all)}.
        Returns (input_nodes, blocks) where input_nodes[type] are the global ids whose h^(0) is needed and
        blocks[l] describes layer l: its destination counts and, per relation, (src_local, dst_local) index
        arrays. Destination nodes are always the first entries of the source nodes.
        """
        if not isinstance(seeds, dict):
            seeds = {"paper": seeds}
        dst = {t: np.asarray(seeds.get(t, []), dtype=np.int64) for t in NODE_TYPES}
        blocks = []
        for _ in range(num_layers):
            sampled = {}
            for rel in RELATIONS:
                s_type, _, d_type = rel
                pos, nbr = self.csr[rel].sample(dst[d_type], fanouts[rel], rng)
                sampled[rel] = (pos, nbr)
            src = {}
            for t in NODE_TYPES:
                nbrs = [nbr for (s_type, _, _), (_, nbr) in sampled.items() if s_type == t]
                new = np.setdiff1d(np.concatenate(nbrs) if nbrs else np.empty(0, np.int64), dst[t])
                src[t] = np.concatenate([dst[t], new])
            edges = {}
            for rel, (pos, nbr) in sampled.items():
                s_nodes = src[rel[0]]
                order = np.argsort(s_nodes, kind="stable")
                src_local = order[np.searchsorted(s_nodes[order], nbr)].astype(np.int64)
                edges[rel] = (torch.from_numpy(src_local), torch.from_numpy(pos.astype(np.int64)))
            blocks.append({"num_dst": {t: len(dst[t]) for t in NODE_TYPES}, "edges": edges})
            dst = src
        blocks.reverse()  # blocks[0] consumes the input features
        return dst, blocks


class FeatureStore:
    """
    Float16 node features h^(0). By default rows are read on demand from the memory-mapped .npy files; with
    `resident_device` (e.g. cuda:0) the full matrices are copied to that device once and gathered there, which
    removes disk reads from the training loop when the features fit into GPU memory.
    """

    def __init__(self, work: Work, resident_device=None):
        self.arrays = {"paper": np.load(work.paper_emb, mmap_mode="r"), "author": np.load(work.author_emb, mmap_mode="r")}
        self.dim = self.arrays["paper"].shape[1]
        assert self.arrays["author"].shape[1] == self.dim, "paper and author embeddings must have the same dimension"
        g = np.load(work.graph)
        for t, n in (("paper", int(g["num_papers"])), ("author", int(g["num_authors"]))):
            if self.arrays[t].shape[0] != n:
                raise SystemExit(f"{t} embeddings have {self.arrays[t].shape[0]} rows but the graph has {n} {t}s: "
                                 "re-run embed.py after prepare_graph.py")
        self.resident = None
        if resident_device is not None:
            self.resident = {}
            for t, arr in self.arrays.items():
                buf = torch.empty(arr.shape, dtype=torch.float16, device=resident_device)
                for s in range(0, arr.shape[0], 500_000):
                    buf[s:s + 500_000] = torch.from_numpy(np.ascontiguousarray(arr[s:s + 500_000])).to(resident_device)
                self.resident[t] = buf

    @property
    def nbytes(self):
        return sum(a.shape[0] * a.shape[1] * 2 for a in self.arrays.values())

    @staticmethod
    def fits_on(device, fraction=0.6, nbytes=0):
        """Whether `nbytes` of features fit into `fraction` of the device memory."""
        if device.type != "cuda":
            return False
        return nbytes < fraction * torch.cuda.get_device_properties(device).total_memory

    def gather(self, node_type, ids, device):
        ids = np.asarray(ids, dtype=np.int64)
        if self.resident is not None:
            src = self.resident[node_type]
            return src[torch.from_numpy(ids).to(src.device)].to(device).float()
        if len(ids) == 0:
            return torch.zeros(0, self.dim, device=device)
        order = np.argsort(ids)  # sorted reads are much faster on memory-mapped files
        rows = np.empty((len(ids), self.dim), dtype=np.float32)
        rows[order] = self.arrays[node_type][ids[order]]
        return torch.from_numpy(rows).to(device, non_blocking=True)


def open_features(work: Work, device, mode="auto", log=None):
    """FeatureStore that is GPU-resident when requested (mode "on") or when it fits (mode "auto")."""
    feats = FeatureStore(work)
    resident = mode == "on" or (mode == "auto" and FeatureStore.fits_on(device, nbytes=feats.nbytes))
    if resident:
        if log:
            log.info("loading %.1f GB of node features onto %s", feats.nbytes / 1e9, device)
        feats = FeatureStore(work, resident_device=device)
    return feats
