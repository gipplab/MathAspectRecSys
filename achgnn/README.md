# AchGNN – Aspect-conditioned Heterogeneous GNN

Implementation of the method in *Aspect-Aware Content-Based Recommendations for Mathematical Research Papers*, built on zbMATH Open data harvested with [`../zbmath_hetnet`](../zbmath_hetnet).

Given a query paper and an aspect – **Specialization / Restriction**, **Generalization / Modification**,
**Results**, **Prove / Cases** – or no aspect (**General**), the trained model ranks zbMATH documents as
recommendations. Training uses SilverRiM, testing uses GoldRiM and the SilverRiM test split.

## Pipeline

| Step | Script | Paper | Output (in `--work`) |
|------|--------|-------|----------------------|
| 1 | `dataset.py` – read GoldRiM/SilverRiM, fix Zbl numbers (`1491.6507` → `1491.65070`) | §3.1 | `pairs_raw.csv`, `dataset_ids.txt` |
| 2 | `../zbmath_hetnet/crawl.py` (all of zbMATH) **or** `fetch_ids.py` (dataset docs + cited docs), then `build_graph.py --with-text` | §3.2.1 | zbMATH node/edge tables |
| 3 | `prepare_graph.py` – paper/author nodes, citation + authorship edges, abstracts with in-abstract references removed, pairs → node ids | §3.2.1, §4.2 | `papers.csv.gz`, `authors.csv.gz`, `graph.npz`, `pairs.csv` |
| 4 | `embed.py` – h⁽⁰⁾ = f_text(title + abstract), f_text(author name) with Qwen3-Embedding | Eq. 1 | `emb/paper.npy`, `emb/author.npy` |
| 5 | `train.py` – hard negatives, heterogeneous GraphSAGE, aspect-conditioned BPR + aspect CE, then evaluation | Eqs. 2–10, §4.2 | `runs/<run>/model.pt`, `metrics.json` |
| 6 | `evaluate.py` – P@10, R@10, MRR per aspect and General on GoldRiM / SilverRiM-test | §4.2 | `runs/<run>/metrics.json`, `retrieved_*.json` |
| 7 | `recommend.py` – aspect-specific top-N for any paper | Eq. 11 | printed list |

## Quick start (small sample, CPU)

```bash
pip install -r achgnn/requirements.txt -r zbmath_hetnet/requirements.txt
# download mathRec_dataset.csv from https://drive.google.com/file/d/1npaipBwA606KAUKmxikjZjvjj3VMQV0Q/view
bash achgnn/run_sample.sh path/to/mathRec_dataset.csv tfidf cpu
```

This uses 150 SilverRiM seeds plus all GoldRiM pairs, the documents involved plus up to 3,000 documents they cite,
TF-IDF node features (no model download) and two short epochs. With `hf` instead of `tfidf` it uses the small
`Qwen/Qwen3-Embedding-0.6B` encoder. The numbers from this run are not meaningful; it checks that everything runs.

## Full run (as in the paper)

```bash
# 1. dataset
python achgnn/dataset.py --csv mathRec_dataset.csv --work work

# 2. zbMATH network with review texts (all documents; takes a day or more, resumable)
cd zbmath_hetnet
python crawl.py all --out-dir output --workers 4
python fetch_ids.py --ids-file ../work/dataset_ids.txt --out-dir output   # any dataset docs the crawl missed
python build_graph.py --out-dir output --with-text
cd ..

# 3. AchGNN graph; the paper's retrieval corpus is the English documents
python achgnn/prepare_graph.py --graph-dir zbmath_hetnet/output/graph --work work --english-only

# 4. Qwen3 embeddings (GPU strongly recommended; resumable)
python achgnn/embed.py --work work --encoder hf --model Qwen/Qwen3-Embedding-8B --device cuda --batch-size 16

# 5. train + evaluate, one run per fold (5-fold cross-validation over 75/25 seed splits)
for f in 0 1 2 3 4; do
  python achgnn/train.py --work work --run achgnn_fold$f --fold $f --epochs 10 --device auto
done

# recommendations
python achgnn/recommend.py --work work --run achgnn_fold0 --zbl 0940.14001 --top 10
```

`--device auto` uses CUDA when available and the CPU otherwise; `--device cpu`, `cuda` or `cuda:1` force a device.
Every step works on CPU; for the full corpus, the Qwen3 embedding step needs a GPU in practice, and the GNN
training/inference is much faster on one.

### Hyper-parameters (defaults follow §4.2)

| Option | Default | Paper |
|--------|---------|-------|
| `--layers` | 2 | L = 2 |
| `--paper-neighbors` / `--author-neighbors` | 15 / 5 | 15 paper, 5 author neighbours |
| `--lam` | 0.2 | λ = 0.2 |
| `--fold` | 0 | 75/25 split, 5-fold CV |
| `--hidden-dim`, `--aspect-dim`, `--epochs`, `--lr`, `--batch-size` | 256, 64, 10, 1e-3, 512 | not specified |

Ablations from §4.4: `--drop-aspects "Prove / Cases"`, `--no-authors`, `--random-authors`, and different
`--layers` / `--paper-neighbors` / `--author-neighbors`.

## Model

* **Graph** (§3.2.1): papers and authors; undirected citation edges and authorship edges. Paper features come from
  the title and the zbMATH review/abstract with explicit references to other zbMATH documents removed; author
  features come from the author's name, using the same encoder.
* **Message passing** (Eq. 2): h_v⁽ˡ⁺¹⁾ = σ(W₀ h_v⁽ˡ⁾ + Σ_r mean_{u∈N_r(v)} W_r h_u⁽ˡ⁾), with separate weights per
  node type and relation (paper←paper citations, paper←author, author←paper) and fixed-size neighbour sampling.
  Written in plain PyTorch, so torch_geometric is not needed.
* **Aspect-conditioned score** (Eqs. 3–4): s = f_score([h_q ‖ h_c ‖ h_q⊙h_c ‖ e_as]), an MLP with 2 hidden layers.
* **Aspect prediction** (Eqs. 5–8): f_aspect([h_q ‖ h_c ‖ h_q⊙h_c]) with one hidden layer (Eq. 7),
  cross-entropy on positive pairs only.
* **Loss** (Eqs. 9–10): BPR(s⁺, s⁻) + λ·CE. Each positive pair gets a hard negative: the most similar document
  to the seed by f_text cosine similarity that is neither the seed nor one of its training recommendations. By
  default each positive also gets one random negative, resampled every step (`--easy-negatives`, see below).
* **Inference** (§3.2.6, Eq. 11): embeddings for all papers are computed layer by layer. Per query, the top
  `--pool` (default 200) candidates by f_text (Qwen3) cosine similarity are re-ranked by s^(as)
  (`--pool-source text`). `--pool-source gnn` takes them from cosine similarity of the AchGNN embeddings instead,
  and `--pool 0` scores the whole corpus with the MLP, which is exact but slow. In the General setting the score is
  the maximum over the four aspects.

### Choices where the paper leaves details open

* *Qwen3-7B* is taken to be the Qwen3 embedding model; the default is `Qwen/Qwen3-Embedding-8B` (`--model` takes
  any HF id or local path, e.g. the model directory used by `src/create_indexes/index_Qwen3_7B.py`).
* The 5-fold cross-validation is implemented as five different random 75/25 splits of the SilverRiM seeds
  (`--fold 0..4`). All pairs of a seed stay on the same side of the split.
* The activation σ is applied after every layer, including the last, as Eq. 2 is written
  (`--final-activation 0` drops it after the last layer; `--normalize 1` L2-normalises the final embeddings).
* "Excluding documents that appear in the training data" is read as excluding the seed's own recommendations.
  Excluding *every* dataset document from the negatives lets the model learn a shortcut: positives are always
  dataset documents and negatives never are.
* Random negatives (`--easy-negatives 1`) are an addition to the paper. With hard negatives alone, the model only
  ever compares positives against very similar documents and never learns that unrelated documents are
  irrelevant. `--easy-negatives 0` gives the paper's setting exactly.
* Re-ranking f_text neighbours (`--pool-source text`) uses the same Qwen3 cosine search the paper uses for hard
  negatives. It keeps candidate recall at least at the level of the embedding baseline, whatever the size of the
  GNN training set.
* Hard negatives are mined once and cached in `hard_negatives_fold<f>.npy`.
* The paper does not give an optimiser, learning rate, epochs or hidden size; the defaults above are reasonable
  starting points.

## What the sample run shows (and what it does not)

The sample checks the mechanics: crawling, graph construction, features, training on CPU, inference,
evaluation and recommendation all run end to end. It does **not** reproduce the paper's numbers. With a few hundred
to ~1,500 training pairs, the aspect-conditioned MLP scorer memorises its training pairs (training loss → 0.03) and
ranks unseen seeds' candidates close to random. On a 1,500-seed sample, TF-IDF cosine reaches R@10 ≈ 0.5 on GoldRiM,
AchGNN ≈ 0.06. In a controlled synthetic test (positives = nearest neighbours), the same code generalises to
unseen seeds, so the gap comes from data size, not from a broken training loop. The paper trains on the full
SilverRiM (~200k training pairs) with Qwen3 features over the full zbMATH graph. Judge AchGNN against the
embedding baseline only at that scale, using `evaluate.py`.

Tested here: CPU with torch 1.13 and torch 2.6, the TF-IDF and Hugging Face encoder paths (the latter with a tiny
local Qwen3-architecture model, not the 8B weights). The CUDA path uses the same code with `--device cuda`, but it
has not been run on a GPU.

## Resource notes

The full corpus has ~5M documents (the paper uses the 3.8M English ones). Expect roughly:
* Qwen3-8B features: 4096-dim float16, ~8 KB per node, about 40 GB on disk (memory-mapped, not loaded into RAM).
* Hard-negative mining is a chunked exact cosine search over the corpus; run it on a GPU.
* Inference keeps two layers of 256-dim float16 embeddings for all nodes in RAM (a few GB).
