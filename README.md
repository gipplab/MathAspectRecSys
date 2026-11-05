# "Aspect-Aware Content-Based Recommendations for Mathematical Research Papers"

[Ankit Satpute](https://www.fiz-karlsruhe.de/en/bereiche/lebenslauf-und-publikationen-von-ankit-satpute)* , [Andre Greiner-Petter](https://gipplab.uni-goettingen.de/team/dr-andre-greiner-petter/) , [Noah Gießing](https://www.fiz-karlsruhe.de/de/bereiche/lebenslauf-und-publikationen-noah-giessing) , [Olaf Teschke](https://www.fiz-karlsruhe.de/de/bereiche/lebenslauf-und-publikationen-dr-olaf-teschke), [Moritz Schubotz](https://www.fiz-karlsruhe.de/de/bereiche/lebenslauf-und-publikationen-dr-moritz-schubotz), [Akiko Aizawa](https://www.nii.ac.jp/en/faculty/digital_content/aizawa_akiko/) and [Bela Gipp](https://gipplab.uni-goettingen.de/team/prof-dr-bela-gipp/).

*Corresponding Author


## :rocket: Accepted to the Full Papers Track of SIGIR 2026

## Abstract
This repository contains an aspect-aware content-based research paper recommendation system code and dataset designed specifically for mathematics, where relatedness is often conceptual rather than based on textual similarity or citation overlap. Unlike existing approaches that work well in domains such as computer science or biomedicine, this project addresses the unique challenges of mathematical literature by modeling connections through shared proof techniques, logical implications, and theoretical generalizations. The project introduces GoldRiM and SilverRiM, the first datasets for aspect-aware mathematical paper recommendation, and presents AchGNN, an aspect-conditioned heterogeneous graph neural network that integrates textual semantics, citation networks, and author relationships. Experimental results show that AchGNN consistently outperforms prior recommendation methods across multiple aspects and also generalizes effectively to machine learning literature. The system has been deployed on the MaRDI platform [An example document with recommendations](https://portal.mardi4nfdi.de/wiki/Publication:3361952) to support mathematical research discovery.

## Repository Structure

| Path | Content |
|------|---------|
| [`data/`](data/Readme.md) | GoldRiM and SilverRiM: description, download and loading |
| [`zbmath_hetnet/`](zbmath_hetnet/README.md) | Harvesting zbMATH Open and building the author–document citation network |
| [`achgnn/`](achgnn/README.md) | AchGNN: graph construction, node features, training, evaluation and recommendation |
| [`slurm/`](slurm/) | Job scripts for running the full pipeline on a GPU cluster |
| `src/` | Baseline indexers, data analysis and ablation scripts |
| `material/` | Supplementary material: related datasets, aspect definitions, additional plots |

## Installation

Python 3.10 or newer is required. Install PyTorch for your platform first ([pytorch.org](https://pytorch.org/get-started/locally/); CUDA builds for GPU training), then the pipeline dependencies:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch                    # or a CUDA build, e.g. --index-url https://download.pytorch.org/whl/cu124
pip install -r achgnn/requirements.txt -r zbmath_hetnet/requirements.txt
```

The scripts in `src/` use the pinned environment in `requirements.txt`.

## Data

The repository provides two datasets:
- **SilverRiM**: automatically derived aspect-aware recommendations (training and evaluation)
- **GoldRiM**: expert-curated recommendations (evaluation)

Both are distributed as a single CSV file of zbMATH Open document pairs. See [`data/Readme.md`](data/Readme.md) for the download link, the file format and examples. The pipeline below expects the file at `data/mathRec_dataset.csv`.

## Pipeline

AchGNN operates on a heterogeneous graph of zbMATH Open papers and authors. The complete pipeline consists of five steps:

| Step | Command | Output |
|------|---------|--------|
| 1. Harvest zbMATH Open | `python zbmath_hetnet/crawl.py all --out-dir experiments/zbmath` | raw API records |
| 2. Build the networks | `python achgnn/dataset.py`, `python zbmath_hetnet/build_graph.py --with-text`, `python achgnn/prepare_graph.py` | paper–author graph, labelled pairs |
| 3. Node features | `python achgnn/embed.py --encoder hf --devices all` | Qwen3 embeddings of papers and authors |
| 4. Training | `torchrun --nproc_per_node=<GPUs> achgnn/train.py` | trained model, metrics |
| 5. Evaluation and recommendation | `achgnn/evaluate.py`, `achgnn/summarize.py`, `achgnn/recommend.py` | P@10, R@10, MRR; aspect-specific recommendations |

Commands with all arguments are given in [`achgnn/README.md`](achgnn/README.md). Every step runs on CPU or GPU; steps 3 and 4 use all GPUs of a node when available.

A small-scale run (a subset of SilverRiM, all GoldRiM pairs and the zbMATH documents they involve, CPU only) executes the complete pipeline end to end in a few minutes:

```bash
bash achgnn/run_sample.sh data/mathRec_dataset.csv
```

## Running on a SLURM Cluster

The `slurm/` directory contains one job script per pipeline step, configured for a node with 4 × NVIDIA A100 80 GB.

1. Adjust partition, account and paths in [`slurm/config.sh`](slurm/config.sh).
2. On a login node with internet access, create the environment and download the dataset and the Qwen3 encoder weights:
   ```bash
   bash slurm/setup_env.sh
   ```
3. Submit the pipeline from the repository root as a chain of dependent jobs:
   ```bash
   bash slurm/submit_all.sh                # crawl → prepare → embed → train (5 folds) → evaluate
   bash slurm/submit_all.sh --skip-crawl   # when the zbMATH crawl is already available
   ```

| Script | Resources | Step |
|--------|-----------|------|
| `01_crawl.sbatch` | CPU, internet access | harvest zbMATH Open (resumable) |
| `02_prepare.sbatch` | CPU, 192 GB RAM | dataset pairs, zbMATH network, AchGNN graph |
| `03_embed.sbatch` | 4 GPUs | Qwen3-Embedding-8B features, one worker per GPU (resumable) |
| `04_train.sbatch` | 4 GPUs, job array 0–4 | DDP training and evaluation, one task per cross-validation fold (resumable) |
| `05_evaluate.sbatch` | 4 GPUs | evaluation of all folds and mean ± std summary |

Individual steps can be submitted on their own, e.g. `sbatch --partition=<partition> --array=0 slurm/04_train.sbatch`. Logs are written to `slurm/logs/`, results to `experiments/work/runs/`.


## Supplementary Materials

We offer supplementary materials in (`material`), such as a summary table of existing CbRPR datasets, alternative visualizations of the results, definitions of aspect labels, and additional plots.

## Citation
If you're using or referring to our paper in your research or applications, please cite using this BibTeX:
```bibtex
@inproceedings{Satpute2026,
	title     = {Aspect-Aware Content-Based Recommendations for Mathematical Research Papers},
	author    = {Satpute, Ankit and Greiner-Petter, Andre and Giessing, Noah and Teschke, Olaf and Schubotz, Moritz and Aizawa, Akiko and Gipp, Bela},
	year      = 2026,
	month     = {July.},
	booktitle = {Proceedings of 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’26)},
	publisher = {ACM},
	address   = {Melbourne | Naarm, Australia},
	topic     = {rec}
}
```

## License

CC-BY-SA 4.0. The dataset includes non-copyrighted bibliographic metadata and reference data derived from I4OSC (CC0).
