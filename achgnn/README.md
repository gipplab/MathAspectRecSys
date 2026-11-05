# AchGNN

Implementation of AchGNN, the aspect-conditioned heterogeneous graph neural network of *Aspect-Aware Content-Based Recommendations for Mathematical Research Papers* (SIGIR '26). Given a query paper and an aspect (**Specialization / Restriction**, **Generalization / Modification**, **Results**, **Prove / Cases**) or no aspect (**General**), AchGNN ranks zbMATH Open documents as recommendations.

## Method

| Component | Paper | Module |
|-----------|-------|--------|
| Heterogeneous graph of papers and authors with citation and authorship edges | §3.2.1 | `prepare_graph.py`, `graph.py` |
| Node features from titles and abstracts (in-abstract references removed) and author names, Qwen3 embeddings | Eq. 1, §4.2 | `embed.py` |
| GraphSAGE-style heterogeneous message passing with relation-specific weights and fixed neighbourhood sampling | Eq. 2 | `model.py`, `graph.py` |
| Aspect-conditioned scoring s = f_score([h_q ‖ h_c ‖ h_q⊙h_c ‖ e_as]) | Eqs. 3–4 | `model.py` |
| Auxiliary aspect prediction f_aspect([h_q ‖ h_c ‖ h_q⊙h_c]) | Eqs. 5–8 | `model.py` |
| Objective L = L_rank (BPR) + λ L_aspect with hard negatives | Eqs. 9–10, §4.2 | `model.py`, `train.py` |
| Aspect-specific ranking, Precision/Recall/MRR@k | Eq. 11, §4.2 | `engine.py`, `evaluate.py` |

## Usage

All commands are run from the repository root. `--work` is the directory holding the inputs, features and runs of an experiment.

**1. Dataset.** Normalise the GoldRiM/SilverRiM CSV and list the documents required:

```bash
python achgnn/dataset.py --csv data/mathRec_dataset.csv --work experiments/work
```

**2. zbMATH network.** Harvest zbMATH Open (see [`zbmath_hetnet`](../zbmath_hetnet/README.md)), add any dataset documents missing from the crawl, and export the network with abstracts:

```bash
cd zbmath_hetnet
python crawl.py all --out-dir ../experiments/zbmath
python fetch_ids.py --ids-file ../experiments/work/dataset_ids.txt --out-dir ../experiments/zbmath
python build_graph.py --out-dir ../experiments/zbmath --with-text
cd ..
```

**3. AchGNN graph.** The retrieval corpus is restricted to English documents as in the paper:

```bash
python achgnn/prepare_graph.py --graph-dir experiments/zbmath/graph --work experiments/work --english-only
```

**4. Node features.** Qwen3-Embedding-8B on all visible GPUs (one worker per GPU, resumable):

```bash
python achgnn/embed.py --work experiments/work --encoder hf --model Qwen/Qwen3-Embedding-8B --devices all --batch-size 64
```

**5. Training and evaluation.** One run per cross-validation fold (`--fold 0..4`). With `torchrun`, training, hard-negative mining, inference and evaluation are distributed over all processes:

```bash
torchrun --standalone --nproc_per_node=4 achgnn/train.py --work experiments/work --run-name achgnn_fold0 --fold 0 --amp
python achgnn/train.py --work experiments/work --run-name achgnn_fold0 --fold 0          # single process
```

**6. Results.**

```bash
torchrun --standalone --nproc_per_node=4 achgnn/evaluate.py --work experiments/work --run-name achgnn_fold0
python achgnn/summarize.py --work experiments/work --runs achgnn_fold0 achgnn_fold1 achgnn_fold2 achgnn_fold3 achgnn_fold4
python achgnn/recommend.py --work experiments/work --run-name achgnn_fold0 --zbl 0940.14001 --top 10
```

The SLURM scripts in [`../slurm`](../slurm) run steps 1–6 on a 4-GPU node.

### Small-scale run

`run_sample.sh` executes the complete pipeline on 150 SilverRiM seeds, all GoldRiM pairs and the zbMATH documents they involve (plus up to 3,000 cited documents), using TF-IDF features on the CPU:

```bash
bash achgnn/run_sample.sh data/mathRec_dataset.csv [tfidf|hf] [auto|cpu|cuda]
```

## Outputs

| File (in `--work`) | Produced by | Content |
|--------------------|-------------|---------|
| `pairs_raw.csv`, `dataset_ids.txt` | `dataset.py` | normalised pairs, required documents |
| `papers.csv.gz`, `authors.csv.gz`, `graph.npz`, `candidates.npy`, `pairs.csv`, `graph_stats.json` | `prepare_graph.py` | nodes, edges, retrieval corpus, pairs as node indices |
| `emb/paper.npy`, `emb/author.npy` | `embed.py` | float16 node features h⁽⁰⁾ |
| `hard_negatives_fold<f>.npy` | `train.py` | cached hard negatives |
| `runs/<run>/model.pt`, `history.json` | `train.py` | checkpoint (model, optimiser, configuration), losses |
| `runs/<run>/paper_embeddings.pt` | `train.py`, `evaluate.py` | AchGNN embeddings of all papers |
| `runs/<run>/metrics.json`, `retrieved_<dataset>.json` | `train.py`, `evaluate.py` | P/R/MRR@k per setting, ranked lists |
| `runs/summary.json` | `summarize.py` | mean and standard deviation over runs |

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--layers` | 2 | GNN depth L |
| `--paper-neighbors`, `--author-neighbors` | 15, 5 | neighbours sampled per node and layer |
| `--lam` | 0.2 | weight λ of the aspect loss |
| `--hidden-dim`, `--aspect-dim` | 256, 64 | node embedding and aspect embedding size |
| `--epochs`, `--batch-size`, `--lr` | 10, 512, 1e-3 | optimisation (batch size per process) |
| `--easy-negatives` | 1 | random negatives per positive in addition to the hard negative |
| `--fold` | 0 | random 75/25 SilverRiM split (0–4) |
| `--eval-pool`, `--pool-source` | 200, `text` | candidates re-ranked per query and their origin |
| `--amp` | off | bfloat16 autocast on GPU |
| `--features-on-gpu` | `auto` | keep node features in GPU memory when they fit |
| `--drop-aspects`, `--no-authors`, `--random-authors` | – | ablations (§4.4) |
| `--resume` | off | continue from the last checkpoint of the run |

## Implementation Details

* **Splits.** GoldRiM is used for testing only. SilverRiM seeds are split 75/25 into training and test seeds; all pairs of a seed fall on the same side. `--fold` selects one of five random splits.
* **Negatives.** Each training pair is contrasted with a hard negative, the candidate most similar to the seed under the Qwen3 embeddings that is not one of the seed's training recommendations, and with `--easy-negatives` random candidates resampled at every step.
* **Ranking.** For each query, the `--eval-pool` nearest candidates by Qwen3 cosine similarity are ranked by the aspect-conditioned score s^(as). `--pool-source gnn` takes the candidates from the AchGNN embeddings instead, and `--eval-pool 0` scores the entire corpus. In the General setting, candidates are ranked by their maximum score over all aspects.
* **Scale.** Node features are memory-mapped float16 arrays. On GPUs with enough memory they are kept resident on the device during training (`--features-on-gpu`). Inference is computed layer by layer over the whole graph and split across processes.
