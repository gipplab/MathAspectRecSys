# Mathematical Recommendation Datasets (GoldRiM & SilverRiM)

## Overview

This repository provides **two datasets for recommender systems in mathematical literature**, derived from the **zbMATH Open** digital library.

- **GoldRiM**  
  A **high-quality, manually curated** dataset containing **420 expert-reviewed recommendation pairs**.  
  Intended for **evaluation and benchmarking**.

- **SilverRiM**  
  A **large-scale, automatically constructed** dataset containing **532,355 recommendation pairs**.  
  Intended for **training and large-scale experiments**.

Both datasets are based on **zbMATH Open document identifiers (docIDs)**. 
All documents can be accessed directly on zbMATH Open by adding the identifier as suffix to `https://zbmath.org/<identifier>`.
As an example from Figure 1 in the paper: https://zbmath.org/0940.14001 .

In addition, [RelatedDatasets.md] provides an overview table to related datasets which didn't make it into the paper.

## Dataset Access

The GoldRiM and SilverRiM datasets are provided as **docID pairs** in a single CSV file (39.4MB).

📥 **Download dataset:**
https://drive.google.com/file/d/1npaipBwA606KAUKmxikjZjvjj3VMQV0Q/view?usp=sharing

## Dataset Structure

The recommendation dataset CSV contains the following columns:

| Column | Description |
|--------|-------------|
| `Seed` | docID of the seed (query) document |
| `Recommendation` | docID(s) of recommended documents, separated by `;` |
| `Aspect` | Broad aspect category describing the relationship between seed and recommendation |
| `Aspect-label` | Fine-grained label explaining how the seed and recommendation are related |
| `GoldRiM` | `1` if the pair belongs to the GoldRiM dataset, else `0` |
| `SilverRiM` | `1` if the pair belongs to the SilverRiM dataset, else `0` |
| `Recommendation_title` | Title of the recommended document, provided for verification with zbMATH Open |

Please refer to "aspect_description.md" for Aspect description and examples.

## Loading the Dataset in Python

The easiest way to load the data pairs is using `pandas`.

```python
import pandas as pd

df = pd.read_csv("mathRec_dataset.csv") #downloaded file using the link above

# GoldRiM pairs
gold_pairs = df[df["GoldRiM"] == 1][
    ["Seed", "Recommendation", "Recommendation_title"]
]
print(f"GoldRiM pairs: {len(gold_pairs)}")

# SilverRiM pairs
silver_pairs = df[df["SilverRiM"] == 1][
    ["Seed", "Recommendation", "Recommendation_title"]
]
print(f"SilverRiM pairs: {len(silver_pairs)}")
```

To print the list of aspects and aspect-labels, you can use `view_aspect_labels.py`:

```sh
python view_aspect_labels.py
```

## Example

Below is an example of a seed–recommendation pair from SilverRiM with clickable links to zbMATH Open:

| Seed Document | Recommended Document | Aspect | Aspect-Label | GoldRiM | SilverRiM | Recommendation Title |
|--------------|----------------------|---|---|---|---|---|
| [1411.65167](https://zbmath.org/1411.65167) | [1202.65168](https://zbmath.org/1202.65168) | Specialization / Restriction | Focused study of a constrained version of the problem | 0 | 1 | Two-dimensional triangular functions and their applications to nonlinear 2D Volterra-Fredholm integral equations |

Clicking the links opens the corresponding documents on the **zbMATH Open website**.


# Add Metadata to Recommendations

In order to download metadata of the recommendations (or any other zbMATH Open document, for that matter), you can use `zbMATHDocsData.py`. This script either requests a document ID or a CSV file (with `Recommendation` column) and downloads additional metadata for the IDs present in that column.
To do so, it requests the data via the official API:

🔗 **zbMATH Open API:** https://api.zbmath.org/v1/

In order to fetch data from a file, uncomment and adjust lines 72 and 73. Then run:

```sh
python zbMATHDocsData.py
```

📄 **This produces a csv File:** `zbMATHDocsData.csv` containing:

| Column | Description |
|--------|-------------|
| `document_id` | Unique zbMATH Open document identifier |
| `text` | Review text |
| `title` | Document title |
| `msc` | Mathematics Subject Classification (MSC) codes |
| `keywords` | Keywords associated with the document |
| `references` | References cited in the document |


**Sample data from zbMATHDocsData.csv:**

| document_id | text | title | msc | keywords | refrences |
|-------------|------|-------|------|------|------|
| [745.30048](https://zbmath.org/745.30048) | Let D be a g-holed planar domain with . . .  | The geometry of representing measures and their critical values. |  {30H05: Spaces of bounded analytic functions of one complex variable , 46J10: Banach algebras of continuous functions, function algebras,  14H42: Theta functions and curves; Schottky problem} | (measures; critical values; theta functions; Abel-Jacobi theory; multiply connected domains) |  |

### **Step 2: Add Metadata to GoldRiM & SilverRiM**

Once `zbMATHDocsData.csv` is available, use it to add metadata to the correct docIDs in the recommendation datasets.

**Run the following command:**
```sh
python arrangeData.py
```

📌 **This script combines all the obtained data along with recommendstions to give you aresulting dataframe that you can use for experiments or analysis.**
