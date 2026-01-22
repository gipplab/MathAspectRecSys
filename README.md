# SIGIR'26 submission repository for paper titled "Aspect-Aware Content-Based Recommendations for Mathematical Research Papers"

### Extra plots Referenced in the Paper

All extra plots referenced in the paper are available in the `plots_/` directory of this repository.

## Source Code

All source code is available in `src`. Any experiments, scripts, or other attempts to reproduce the data require to load the python virtual environment below.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data

This repository includes scripts to obtain and prepare the two datasets:
- **SilverRiM**: Automatically generated aspect-aware recommendations
- **GoldRiM**: Gold standard manually curated recommendations

See `data/README.md` for detailed setup instructions and dataset descriptions.

## Reproducing Results

All results from the paper can be reproduced using the scripts below:

| Figure/Table | Script | Description |
|---|---|---|
| **Figure 3** | `python src/data_stats/n_gram_overlap.py` | N-gram overlap analysis |
| **Figure 4** | `python src/data_stats/cosScores_overlap.py` | Cosine similarity scores (pre-computed; run `src/eval/createIndexes.py` for fresh scores) |
| **Figure 5** | — | Demonstration only (created in PowerPoint) |
| **Figure 6** | — | Demonstration only (created in PowerPoint) |
| **Table 1** | `python src/data_stats/print_data_stats.py` | Dataset statistics (requires loaded dataframes from `data/`) |
| **Table 2** | `bash src/get_eval_goldRiM_silverRiM.sh` | GoldRiM vs SilverRiM evaluation |
| **Table 3** | `bash src/get_eval_pwc.sh` | Papers with Code evaluation |
| **Figures 7–8** | `bash src/ablation_.sh` | Ablation study results |

## License

CC-BY-SA 4.0. The dataset includes non-copyrighted bibliographic metadata and reference data derived from I4OSC (CC0).
