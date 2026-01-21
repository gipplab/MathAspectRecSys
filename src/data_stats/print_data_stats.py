import pandas as pd

# Load data
df = pd.read_csv("final_dataset.csv")

# Split into Gold and Silver
gold = df[df['GoldRiM'] == 1]
silver = df[df['SilverRiM'] == 1]
silver = df[df['PwC'] == 1]

def print_stats(subset, name):
    total_seeds = subset['Seed'].nunique()
    total_pairs = len(subset)
    unique_aspects = subset['Aspect'].nunique()
    aspect_labels = sorted(subset['Aspect'].unique())
    avg_rec_per_seed = total_pairs / total_seeds if total_seeds > 0 else 0

    print(f"\n📊 {name} Statistics")
    print(f"{'─' * 30}")
    print(f"Total Seeds          : {total_seeds:,}")
    print(f"Recommendation Pairs : {total_pairs:,}")
    print(f"Unique Aspects       : {unique_aspects}")
    print(f"Avg. Rec. per Seed   : {avg_rec_per_seed:.2f}")
    print(f"Aspect Categories    : {', '.join(aspect_labels)}")

# Print stats
print_stats(gold, "GoldRiM")
print_stats(silver, "SilverRiM")
print_stats(pwc, "PwC")

# Optional: Overall stats
print_stats(df, "Overall")