"""AchGNN: Aspect-conditioned heterogeneous GNN (paper, Sec. 3.2)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph import NODE_TYPES, RELATIONS


def rel_key(rel):
    return "__".join(rel)


class HeteroSAGELayer(nn.Module):
    """
    One layer of Eq. (2):  h_v^(l+1) = sigma( W_0 h_v^(l) + sum_r mean_{u in N_r(v)} W_r h_u^(l) )
    with a self transform per node type and a transform per relation.
    """

    def __init__(self, in_dim, out_dim, activation=True, dropout=0.0):
        super().__init__()
        self.self_lin = nn.ModuleDict({t: nn.Linear(in_dim, out_dim) for t in NODE_TYPES})
        self.rel_lin = nn.ModuleDict({rel_key(r): nn.Linear(in_dim, out_dim, bias=False) for r in RELATIONS})
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_src, block):
        out = {t: self.self_lin[t](h_src[t][: block["num_dst"][t]]) for t in NODE_TYPES}
        for rel, (src_local, dst_local) in block["edges"].items():
            s_type, _, d_type = rel
            n_dst = block["num_dst"][d_type]
            if n_dst == 0 or len(src_local) == 0:
                continue
            src_local, dst_local = src_local.to(h_src[s_type].device), dst_local.to(h_src[s_type].device)
            x = h_src[s_type][src_local]
            agg = torch.zeros(n_dst, x.size(1), device=x.device, dtype=x.dtype).index_add_(0, dst_local, x)
            deg = torch.bincount(dst_local, minlength=n_dst).clamp(min=1).unsqueeze(1).to(x.dtype)
            # mean_u W_r h_u == W_r mean_u h_u (W_r is linear), aggregating first is cheaper
            out[d_type] = out[d_type] + self.rel_lin[rel_key(rel)](agg / deg)
        if self.activation:
            out = {t: F.relu(v) for t, v in out.items()}
        return {t: self.dropout(v) for t, v in out.items()}


class AchGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, num_layers=2, num_aspects=4, aspect_dim=64, dropout=0.1,
                 final_activation=True, normalize=False):
        super().__init__()
        self.config = dict(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_aspects=num_aspects,
                           aspect_dim=aspect_dim, dropout=dropout, final_activation=final_activation,
                           normalize=normalize)
        self.normalize = normalize
        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList([
            HeteroSAGELayer(dims[i], dims[i + 1], activation=(i < num_layers - 1) or final_activation,
                            dropout=dropout if i < num_layers - 1 else 0.0)
            for i in range(num_layers)
        ])
        # e_{as_k}: learnable aspect embeddings (Eq. 3)
        self.aspect_emb = nn.Embedding(num_aspects, aspect_dim)
        # f_score: MLP with 2 hidden layers and scalar output (Eq. 4, Fig. 2)
        self.f_score = nn.Sequential(
            nn.Linear(3 * hidden_dim + aspect_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # f_aspect(z) = W_a sigma(W_z z + b_z) + b_a (Eq. 7)
        self.f_aspect = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, num_aspects))

    # ---- message passing
    def encode(self, h0, blocks):
        """h0: {type: features of the input nodes}; returns the final embeddings of the seed papers."""
        h = h0
        for l, block in enumerate(blocks):
            h = self.layer_forward(l, h, block)
        return h["paper"]

    def layer_forward(self, l, h_src, block):
        out = self.layers[l](h_src, block)
        if self.normalize and l == len(self.layers) - 1:  # unit-length final embeddings
            out = {t: F.normalize(v, dim=-1) for t, v in out.items()}
        return out

    # ---- scoring heads
    @staticmethod
    def interaction(hq, hc):
        return torch.cat([hq, hc, hq * hc], dim=-1)  # [h_q || h_c || h_q ⊙ h_c] (Eq. 5)

    def score(self, hq, hc, aspect):
        """s^(as)_{q,c} = f_score([h_q || h_c || h_q ⊙ h_c || e_as]) (Eqs. 3-4); aspect: LongTensor [n]."""
        z = torch.cat([self.interaction(hq, hc), self.aspect_emb(aspect)], dim=-1)
        return self.f_score(z).squeeze(-1)

    def score_all_aspects(self, hq, hc):
        """Scores for every aspect: [n, num_aspects]."""
        n, a = hq.size(0), self.aspect_emb.num_embeddings
        inter = self.interaction(hq, hc).unsqueeze(1).expand(n, a, -1)
        e = self.aspect_emb.weight.unsqueeze(0).expand(n, a, -1)
        return self.f_score(torch.cat([inter, e], dim=-1)).squeeze(-1)

    def aspect_logits(self, hq, hc):
        """ŷ_{q,c} = f_aspect(z_{q,c}) (Eq. 6)."""
        return self.f_aspect(self.interaction(hq, hc))

    # ---- objective (Eqs. 8-10)
    def loss(self, hq, hpos, hneg, aspect, lam):
        """
        hq, hpos: [m, d] query and positive embeddings; hneg: [r*m, d] r negatives per positive, stacked in r
        blocks of m. Every positive is contrasted with each of its negatives (BPR, Eq. 9).
        """
        reps = hneg.size(0) // hq.size(0)
        s_pos = self.score(hq, hpos, aspect).repeat(reps)
        s_neg = self.score(hq.repeat(reps, 1), hneg, aspect.repeat(reps))
        l_rank = -F.logsigmoid(s_pos - s_neg).mean()
        l_aspect = F.cross_entropy(self.aspect_logits(hq, hpos).float(), aspect)  # positives only
        return l_rank + lam * l_aspect, l_rank.detach(), l_aspect.detach()

    def forward(self, h0, blocks, inv, m, aspect, lam):
        """
        Training step entry point (used through DistributedDataParallel): message passing over the sampled
        blocks, then the loss. inv maps [queries | positives | negatives] to rows of the encoded batch nodes.
        """
        h = self.encode(h0, blocks)
        return self.loss(h[inv[:m]], h[inv[m:2 * m]], h[inv[2 * m:]], aspect, lam)
