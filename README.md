# "Aspect-Aware Content-Based Recommendations for Mathematical Research Papers"

[Ankit Satpute](https://www.fiz-karlsruhe.de/en/bereiche/lebenslauf-und-publikationen-von-ankit-satpute)* , [Andre Greiner-Petter](https://gipplab.uni-goettingen.de/team/dr-andre-greiner-petter/) , [Noah Gießing](https://www.fiz-karlsruhe.de/de/bereiche/lebenslauf-und-publikationen-noah-giessing) , [Olaf Teschke](https://www.fiz-karlsruhe.de/de/bereiche/lebenslauf-und-publikationen-dr-olaf-teschke), [Moritz Schubotz](https://www.fiz-karlsruhe.de/de/bereiche/lebenslauf-und-publikationen-dr-moritz-schubotz), [Akiko Aizawa](https://www.nii.ac.jp/en/faculty/digital_content/aizawa_akiko/) and [Bela Gipp](https://gipplab.uni-goettingen.de/team/prof-dr-bela-gipp/).

*Corresponding Author


## :rocket: Accepted to the Full Papers Track of SIGIR 2026

## Abstract
This repository contains an aspect-aware content-based research paper recommendation system code and dataset designed specifically for mathematics, where relatedness is often conceptual rather than based on textual similarity or citation overlap. Unlike existing approaches that work well in domains such as computer science or biomedicine, this project addresses the unique challenges of mathematical literature by modeling connections through shared proof techniques, logical implications, and theoretical generalizations. The project introduces GoldRiM and SilverRiM, the first datasets for aspect-aware mathematical paper recommendation, and presents AchGNN, an aspect-conditioned heterogeneous graph neural network that integrates textual semantics, citation networks, and author relationships. Experimental results show that AchGNN consistently outperforms prior recommendation methods across multiple aspects and also generalizes effectively to machine learning literature. The system has been deployed on the MaRDI platform [An example document with recommendations](https://portal.mardi4nfdi.de/wiki/Publication:3361952) to support mathematical research discovery.

This repository includes the complete source code (`src`), datasets (`data`), and supplementary materials (`material`).

## Data

This repository includes scripts to obtain and prepare the two datasets:
- **SilverRiM**: Automatically generated aspect-aware recommendations
- **GoldRiM**: Gold standard manually curated recommendations

See `data/README.md` for detailed setup instructions and dataset descriptions.

## Supplementary Materials

We offer supplementary materials in (`material`), such as a summary table of existing CbRPR datasets, alternative visualizations of the results, definitions of aspect labels, and additional plots.

## Source Code

The source code is in `src`. To run any of the experiments or scripts, first set up the Python virtual environment below.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Code Availability

> **Note:** The repository currently contains the datasets, the model implementations (AchGNN and the baselines) and the data analysis scripts. The full end-to-end experiment pipeline was originally built around our internal zbMATH Open / MaRDI infrastructure. We are refactoring it so that it runs outside that environment, with fixed seeds, configuration files and documented preprocessing. This covers the retrieval and evaluation scripts (`src/retrieval_/`), the remaining index builders and the PwC preprocessing. These parts will be released here soon, once the reproducibility changes are complete. Until then, feel free to open an issue or contact the corresponding author if you need something specific.

## Reproducing Results

The scripts below map to the results in the paper:

| Figure/Table | Script | Description |
|---|---|---|
| **Figure 3** | `python src/data_stats/n_gram_ovelap.py --dataset <dataset.csv> --docs <docs.csv> --top100_dir <dir>` | N-gram overlap analysis |
| **Figure 4** | `python src/data_stats/cosScores_ovelap.py` | Cosine similarity scores (pre-computed; build fresh indexes with the scripts in `src/create_indexes/`) |
| **Table 1** | `python src/data_stats/print_data_stats.py` | Dataset statistics (requires the dataset CSV from `data/`) |
| **Table 2** | `bash src/get_eval_goldRiM_silverRiM.sh` | GoldRiM vs SilverRiM evaluation (requires the retrieval/evaluation scripts, release pending) |
| **Table 3** | `bash src/get_eval_pwc.sh` | Papers with Code evaluation (requires the retrieval/evaluation scripts, release pending) |
| **Figures 7–8** | `bash src/ablation_.sh` | Ablation study results |


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
