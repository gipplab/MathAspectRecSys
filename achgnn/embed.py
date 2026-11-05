"""
Initial node features h^(0) = f_text(.) (Eq. 1).

Papers are embedded from "Title: ...\nAbstract: ..." and authors from their names with the same encoder, so both
node types share one dimensionality. Embeddings are L2-normalised and stored as float16 .npy files that later
steps memory-map. Writing happens in chunks, so an interrupted run resumes where it stopped.

Encoders
    --encoder hf     HuggingFace embedding model with last-token pooling (default: Qwen/Qwen3-Embedding-8B).
                     --devices selects one or more GPUs; with several, the rows are split into one contiguous
                     range per GPU, each processed by its own worker process.
    --encoder tfidf  TF-IDF + truncated SVD on the CPU (no model download).

    python achgnn/embed.py --work work --encoder hf --devices all --batch-size 64
    python achgnn/embed.py --work work --encoder tfidf --dim 256
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import Work, get_device, load_json, log, save_json, setup_logging

DEFAULT_HF_MODEL = "Qwen/Qwen3-Embedding-8B"
CHUNK_ROWS = 20_000


def load_texts(work, kind):
    if kind == "paper":
        papers = pd.read_csv(work.papers, dtype=str, keep_default_na=False, usecols=["title", "text"])
        return [f"Title: {t}\nAbstract: {x}" for t, x in zip(papers["title"], papers["text"])]
    return pd.read_csv(work.authors, dtype=str, keep_default_na=False, usecols=["name"])["name"].tolist()


class HFEncoder:
    """Last-token pooled decoder embeddings (Qwen3-Embedding style)."""

    def __init__(self, model_name, device, max_length):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch, self.device, self.max_length = torch, device, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
        kwargs = {"trust_remote_code": True}
        if device.type == "cuda":
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            try:
                import flash_attn  # noqa: F401

                kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                kwargs["attn_implementation"] = "sdpa"
        else:
            kwargs["torch_dtype"] = torch.float32
        self.model = AutoModel.from_pretrained(model_name, **kwargs).to(device).eval()
        self.dim = self.model.config.hidden_size

    def __call__(self, texts):
        torch = self.torch
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length,
                               return_tensors="pt").to(self.device)
        with torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"]
        if bool((mask[:, -1] == 1).all()):  # left padding: the last position is the last real token
            emb = hidden[:, -1]
        else:
            emb = hidden[torch.arange(hidden.size(0), device=hidden.device), mask.sum(1) - 1]
        return torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy()


def encode_range(encoder, texts, out_path, lo, hi, batch_size, progress_path, desc, position=0):
    """Encode rows [lo, hi) into the existing .npy at out_path, resuming from progress_path."""
    out = np.load(out_path, mmap_mode="r+")
    done = int(load_json(progress_path)["rows"]) if os.path.exists(progress_path) else int(lo)
    with tqdm(total=hi - lo, initial=done - lo, desc=desc, unit="doc", position=position) as bar:
        for start in range(done, hi, CHUNK_ROWS):
            end = min(start + CHUNK_ROWS, hi)
            # sort by length inside the chunk so batches contain similarly long texts
            order = start + np.argsort([len(t) for t in texts[start:end]], kind="stable")
            for b in range(0, len(order), batch_size):
                rows = order[b:b + batch_size]
                out[rows] = encoder([texts[i] for i in rows]).astype(np.float16)
                bar.update(len(rows))
            out.flush()
            save_json({"rows": int(end)}, progress_path)
    del out


def _hf_worker(worker, device_name, args, kind, lo, hi, out_path):
    import torch

    setup_logging()
    work = Work(args.work)
    texts = load_texts(work, kind)
    device = torch.device(device_name)
    encoder = HFEncoder(args.model, device, args.max_length if kind == "paper" else 32)
    batch = args.batch_size if kind == "paper" else args.batch_size * 8
    encode_range(encoder, texts, out_path, lo, hi, batch, f"{out_path}.progress.{worker}.json",
                 f"{kind}s [{device_name}]", position=worker)


def resolve_devices(spec):
    import torch

    if spec == "all":
        if not torch.cuda.is_available():
            return ["cpu"]
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if spec == "auto":
        return [str(get_device("auto"))]
    return [d.strip() for d in spec.split(",") if d.strip()]


def prepare_output(path, n, dim):
    """Create (or validate) the output file. Returns True if it is already complete."""
    progress = glob.glob(path + ".progress.*.json")
    if os.path.exists(path):
        shape = np.load(path, mmap_mode="r").shape
        if shape == (n, dim) and not progress:
            log.info("%s already complete", path)
            return True
        if shape != (n, dim):
            log.info("%s has shape %s, expected %s: recomputing", path, shape, (n, dim))
            os.remove(path)
            for p in progress:
                os.remove(p)
    if not os.path.exists(path):
        np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n, dim)).flush()
    return False


def embed_hf(args, work, kind, path, n):
    import torch.multiprocessing as mp
    from transformers import AutoConfig

    dim = AutoConfig.from_pretrained(args.model, trust_remote_code=True).hidden_size
    if prepare_output(path, n, dim):
        return
    devices = resolve_devices(args.devices)
    bounds = np.linspace(0, n, len(devices) + 1).astype(int)
    log.info("embedding %d %ss on %s", n, kind, ", ".join(devices))
    if len(devices) == 1:
        _hf_worker(0, devices[0], args, kind, 0, n, path)
    else:
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=_hf_worker,
                             args=(i, d, args, kind, int(bounds[i]), int(bounds[i + 1]), path))
                 for i, d in enumerate(devices)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        failed = [d for d, p in zip(devices, procs) if p.exitcode != 0]
        if failed:
            raise SystemExit(f"embedding workers failed on {failed}; re-run to resume")
    for p in glob.glob(path + ".progress.*.json"):
        os.remove(p)


class TfidfEncoder:
    """Fit-on-corpus TF-IDF + SVD encoder (papers: word n-grams, authors: character n-grams)."""

    def __init__(self, dim, analyzer, ngram_range):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.dim = dim
        self.vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, sublinear_tf=True,
                                          max_features=200_000, min_df=1)

    def fit(self, texts):
        from sklearn.decomposition import TruncatedSVD

        x = self.vectorizer.fit_transform(texts)
        k = min(self.dim, x.shape[1] - 1, x.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=max(k, 1), random_state=0).fit(x)
        return self

    def __call__(self, texts):
        v = self.svd.transform(self.vectorizer.transform(texts))
        if v.shape[1] < self.dim:  # pad tiny corpora to the requested dimensionality
            v = np.pad(v, ((0, 0), (0, self.dim - v.shape[1])))
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def embed_tfidf(args, work, kind, path, n):
    if prepare_output(path, n, args.dim):
        return
    texts = load_texts(work, kind)
    enc = TfidfEncoder(args.dim, *(("word", (1, 2)) if kind == "paper" else ("char_wb", (2, 4)))).fit(texts)
    progress = path + ".progress.0.json"
    encode_range(enc, texts, path, 0, n, 4096, progress, f"{kind}s")
    if os.path.exists(progress):
        os.remove(progress)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--encoder", choices=["hf", "tfidf"], default="hf")
    ap.add_argument("--model", default=DEFAULT_HF_MODEL, help="HF model name or local path (--encoder hf)")
    ap.add_argument("--devices", default="auto",
                    help="auto | all (every visible GPU) | comma-separated list, e.g. cuda:0,cuda:1 or cpu")
    ap.add_argument("--batch-size", type=int, default=32, help="paper texts per forward pass and device")
    ap.add_argument("--max-length", type=int, default=512, help="max tokens per paper text")
    ap.add_argument("--dim", type=int, default=256, help="embedding size for --encoder tfidf")
    args = ap.parse_args()
    setup_logging()
    work = Work(args.work)

    counts = {"paper": len(pd.read_csv(work.papers, usecols=["doc_id"])),
              "author": len(pd.read_csv(work.authors, usecols=["author_code"]))}
    for kind, path in (("paper", work.paper_emb), ("author", work.author_emb)):
        if args.encoder == "hf":
            embed_hf(args, work, kind, path, counts[kind])
        else:
            embed_tfidf(args, work, kind, path, counts[kind])
    save_json({"encoder": args.encoder, "model": args.model if args.encoder == "hf" else "tfidf",
               "dim": int(np.load(work.paper_emb, mmap_mode="r").shape[1])}, work.path("emb", "info.json"))
    log.info("embeddings written to %s", os.path.dirname(work.paper_emb))


if __name__ == "__main__":
    main()
