"""
Combine recommendation pairs with zbMATH document metadata (zbMATHDocsData.csv)
to produce a single enriched DataFrame for analysis.

Output:
    combined_analysis_data.csv
"""
import pandas as pd

# File paths
RECOMMENDATIONS_FILE = "downloaded_dataset.csv"
ZBMATH_METADATA_FILE = "zbMATHDocsData.csv"
OUTPUT_FILE = "combined_analysis_data.csv"

def main():
    # Load recommendation pairs
    print("Loading recommendation pairs...")
    rec_df = pd.read_csv(RECOMMENDATIONS_FILE, dtype=str)  # keep IDs as strings

    # Load zbMATH metadata
    print("Loading zbMATH document metadata...")
    meta_df = pd.read_csv(ZBMATH_METADATA_FILE, dtype=str)

    # Ensure consistent column names and types
    meta_df.rename(columns={"document_id": "Recommendation"}, inplace=True)
    meta_df["Recommendation"] = meta_df["Recommendation"].astype(str)

    # Merge: left join to keep all recommendations (even if metadata missing)
    print("Merging data...")
    combined = pd.merge(
        rec_df,
        meta_df[
            ["Recommendation", "title", "authors", "year", "msc", "language", "source"]
        ],
        on="Recommendation",
        how="left"
    )

    # Rename title to avoid clash with Recommendation_title
    combined.rename(columns={"title": "Recommendation_title_zbMATH"}, inplace=True)

    # Seed-side metadata (disabled)

    """
    print("Loading seed metadata...")
    seed_meta = pd.read_csv(ZBMATH_METADATA_FILE, dtype=str)
    seed_meta.rename(columns={"document_id": "Seed"}, inplace=True)
    seed_meta["Seed"] = seed_meta["Seed"].astype(str)

    combined = pd.merge(
        combined,
        seed_meta[["Seed", "title", "authors", "year", "msc", "language", "source"]],
        on="Seed",
        how="left",
        suffixes=("_rec", "_seed")
    )
    """

    # Save result
    print(f"Saving combined dataset ({len(combined)} rows) to {OUTPUT_FILE}...")
    combined.to_csv(OUTPUT_FILE, index=False)

    # Report missing metadata
    missing_meta = combined["Recommendation_title_zbMATH"].isna().sum()
    print(f"⚠️  {missing_meta} recommendations have no zbMATH metadata.")

if __name__ == "__main__":
    main()