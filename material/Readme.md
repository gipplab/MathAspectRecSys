# Supplementary Material

This directory contains the supplementary materials referenced in the paper. The files provide additional dataset descriptions, methodological details, and extended experimental results.

---
## Existing CbRPR Datasets

- **`RelatedDatasets.md`**  
  Provides an overview and comparison of existing CbRPR datasets.
---

## Aspect Descriptions

- **`aspect_description.md`**  
  Describes each high-level *Aspect* category used in the paper, along with definitions.
---

## Additional Plots from Section 4.3

The following files contain visualizations of the evaluation metrics reported in Tables 2 and 3 of the paper.

### GoldRiM / SilverRiM (Table 2)

- **`recall_at_10_goldrim_silverRiM.pdf`**  
  Plotted Recall@10 (R@10) values.

- **`precision_at_10_goldrim_silverRiM.pdf`**  
  Plotted Precision@10 (P@10) values.

- **`MRR_GoldRiM_SilverRiM.pdf`**  
  Plotted Mean Reciprocal Rank (MRR) values.

### PwC (Table 3)

- **`recall_at_10_PwC.pdf`**  
  Plotted Recall@10 (R@10) values.

- **`precision_at_10_PwC.pdf`**  
  Plotted Precision@10 (P@10) values.

- **`MRR_PwC.pdf`**  
  Plotted Mean Reciprocal Rank (MRR) values.

---

## Additional Plots from Section 4.4.1 — Aspect Removal Ablation

- **Normalized performance drop under aspect removal**  
  **`achgnn_normalized_aspect_removal.pdf`**

  This plot shows the normalized Recall@10 performance drop when specific aspects are removed from the training data. The normalization accounts for the proportion of data affected by each removed aspect. The final row corresponds to a configuration where only the data was modified, with no further model changes applied.

---

## Additional Plots from Section 4.4.3 — Neighborhood Size and GNN Depth Ablation

The following plots analyze the impact of sampled neighborhood size and GNN depth (L) on **SilverRiM** performance:

1. **Author neighborhood size = 1**  
   - `ablation_author_1_achGNN.pdf`

2. **Author neighborhood size = 10**  
   - `ablation_author_10_achGNN.pdf`
