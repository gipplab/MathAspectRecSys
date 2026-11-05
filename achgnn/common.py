"""Shared helpers: paths inside the work directory, device selection, ID normalisation, text cleaning."""
import json
import logging
import os
import re

import numpy as np

log = logging.getLogger("achgnn")

ASPECTS = ["Specialization / Restriction", "Generalization / Modification", "Results", "Prove / Cases"]


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class Work:
    """File layout of the work directory shared by all pipeline steps."""

    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def path(self, *parts):
        p = os.path.join(self.root, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    pairs_raw = property(lambda s: s.path("pairs_raw.csv"))        # dataset.py: normalised GoldRiM/SilverRiM pairs
    dataset_ids = property(lambda s: s.path("dataset_ids.txt"))    # dataset.py: Zbl ids to fetch
    papers = property(lambda s: s.path("papers.csv.gz"))           # prepare_graph.py: paper nodes + text
    candidates = property(lambda s: s.path("candidates.npy"))      # prepare_graph.py: retrieval corpus (paper ids)
    authors = property(lambda s: s.path("authors.csv.gz"))         # prepare_graph.py: author nodes + names
    graph = property(lambda s: s.path("graph.npz"))                # prepare_graph.py: edge index arrays
    pairs = property(lambda s: s.path("pairs.csv"))                # prepare_graph.py: pairs mapped to node indices
    paper_emb = property(lambda s: s.path("emb", "paper.npy"))     # embed.py: f_text(paper)
    author_emb = property(lambda s: s.path("emb", "author.npy"))   # embed.py: f_text(author)

    def run(self, name, *parts):
        return self.path("runs", name, *parts)


def get_device(name="auto"):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available; use --device cpu")
    return dev


def normalize_zbl(x):
    """
    Restore a Zbl number that lost leading/trailing zeros (e.g. read as a float): 688.46025 -> 0688.46025,
    1491.6507 -> 1491.65070. Other identifiers (e.g. internal ids) are returned unchanged.
    """
    x = str(x).strip()
    m = re.fullmatch(r"(\d{1,4})\.(\d{1,5})", x)
    if not m:
        return x
    return f"{m.group(1).zfill(4)}.{m.group(2).ljust(5, '0')}"


# Mentions of other zbMATH documents inside reviews, e.g. "[Zbl 0451.13008]", "(see Zbl 1436.13034)", "JFM 12.0345.01"
_ZBL_REF = r"(?:Zbl|JFM|ERAM)\s*\d+[.\d]*[A-Z]?"
_BRACKETED_REF = re.compile(r"[\[(][^\[\]()]*" + _ZBL_REF + r"[^\[\]()]*[\])]")
_BARE_REF = re.compile(_ZBL_REF)
_SPACES = re.compile(r"\s+")


def remove_document_references(text):
    """Remove explicit mentions of other zbMATH documents from an abstract/review (paper, Sec. 4.2)."""
    if not text:
        return ""
    text = _BRACKETED_REF.sub(" ", text)
    text = _BARE_REF.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def save_json(obj, path):
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


def load_json(path):
    with open(path, encoding="utf8") as f:
        return json.load(f)


def seed_split(seeds, fold, test_fraction=0.25):
    """
    Deterministic 75/25 train/test split of SilverRiM seeds; `fold` (0..4) selects one of the five repetitions
    used for cross-validation. Returns the set of test seeds.
    """
    seeds = np.unique(np.asarray(seeds))
    rng = np.random.default_rng(1000 + fold)
    perm = rng.permutation(seeds)
    return set(perm[: int(round(len(perm) * test_fraction))].tolist())
