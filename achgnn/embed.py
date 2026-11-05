"""
Step 4: initial node features h^(0) = f_text(.) (paper, Eq. 1).

Papers are embedded from "Title: ...\nAbstract: ..." and authors from their names, with the same encoder so both
node types share one dimensionality. Embeddings are L2-normalised and stored as float16 .npy files (memory-mapped
later), written in chunks so an interrupted run resumes where it stopped.

Encoders
    --encoder hf     any HuggingFace embedding model with last-token pooling; default Qwen/Qwen3-Embedding-8B
                     (the Qwen3 embedding model used in the paper). Use a GPU for this.
    --encoder tfidf  TF-IDF + truncated SVD; no model download, CPU only. Meant for quick tests of the pipeline.

    python achgnn/embed.py --work work --encoder hf --device cuda --batch-size 16
    python achgnn/embed.py --work work_sample --encoder tfidf --dim 256
"""
import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import Work, get_device, load_json, log, save_json, setup_logging

DEFAULT_HF_MODEL = "Qwen/Qwen3-Embedding-8B"


class HFEncoder:
    """Last-token pooled decoder embeddings (Qwen3-Embedding style), as in src/create_indexes/index_Qwen3_7B.py."""

    def __init__(self, model_name, device, max_length):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch, self.device, self.max_length = torch, device, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True).to(device).eval()
        self.dim = self.model.config.hidden_size

    def __call__(self, texts):
        torch = self.torch
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length,
                               return_tensors="pt").to(self.device)
        with torch.no_grad():
            hidden = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"]
        if bool((mask[:, -1] == 1).all()):  # left padding: last position is the last real token
            emb = hidden[:, -1]
        else:
            emb = hidden[torch.arange(hidden.size(0), device=hidden.device), mask.sum(1) - 1]
        emb = torch.nn.functional.normalize(emb.float(), dim=-1)
        return emb.cpu().numpy()


def embed_to_file(encoder, texts, path, batch_size, chunk_rows=20_000, desc=""):
    """Encode texts into a float16 .npy at `path`, resumable at chunk granularity."""
    progress_file = path + ".progress.json"
    n = len(texts)
    if os.path.exists(path):
        shape = np.load(path, mmap_mode="r").shape
        if shape != (n, encoder.dim):  # graph or encoder changed since: start over
            log.info("%s has shape %s, expected %s: recomputing", path, shape, (n, encoder.dim))
            os.remove(path)
            if os.path.exists(progress_file):
                os.remove(progress_file)
        elif not os.path.exists(progress_file):
            log.info("%s already complete", path)
            return
    done = load_json(progress_file)["rows"] if os.path.exists(progress_file) else 0
    out = np.lib.format.open_memmap(path, mode="r+" if done else "w+", dtype=np.float16, shape=(n, encoder.dim))
    with tqdm(total=n, initial=done, desc=desc, unit="doc") as bar:
        for chunk_start in range(done, n, chunk_rows):
            chunk_end = min(chunk_start + chunk_rows, n)
            for s in range(chunk_start, chunk_end, batch_size):
                e = min(s + batch_size, chunk_end)
                out[s:e] = encoder(texts[s:e]).astype(np.float16)
                bar.update(e - s)
            out.flush()
            save_json({"rows": chunk_end}, progress_file)
    del out
    os.remove(progress_file)


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--encoder", choices=["hf", "tfidf"], default="hf")
    ap.add_argument("--model", default=DEFAULT_HF_MODEL, help="HF model name or local path (--encoder hf)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512, help="max tokens per paper text")
    ap.add_argument("--dim", type=int, default=256, help="embedding size for --encoder tfidf")
    args = ap.parse_args()
    setup_logging()
    work = Work(args.work)

    papers = pd.read_csv(work.papers, dtype=str, keep_default_na=False)
    authors = pd.read_csv(work.authors, dtype=str, keep_default_na=False)
    paper_texts = [f"Title: {t}\nAbstract: {x}" for t, x in zip(papers["title"], papers["text"])]
    author_texts = authors["name"].tolist()
    log.info("%d papers, %d authors to embed with %s", len(paper_texts), len(author_texts), args.encoder)

    if args.encoder == "hf":
        device = get_device(args.device)
        enc = HFEncoder(args.model, device, args.max_length)
        embed_to_file(enc, paper_texts, work.paper_emb, args.batch_size, desc="papers")
        enc.max_length = 32
        embed_to_file(enc, author_texts, work.author_emb, args.batch_size * 8, desc="authors")
    else:
        for texts, path, kind in [(paper_texts, work.paper_emb, "papers"), (author_texts, work.author_emb, "authors")]:
            enc = TfidfEncoder(args.dim, *(("word", (1, 2)) if kind == "papers" else ("char_wb", (2, 4)))).fit(texts)
            embed_to_file(enc, texts, path, batch_size=4096, desc=kind)
    save_json({"encoder": args.encoder, "model": args.model if args.encoder == "hf" else "tfidf",
               "dim": int(np.load(work.paper_emb, mmap_mode="r").shape[1])}, work.path("emb", "info.json"))
    log.info("embeddings written to %s", os.path.dirname(work.paper_emb))


if __name__ == "__main__":
    main()
