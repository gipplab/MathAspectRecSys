import pandas as pd

# Load dataset
df = pd.read_csv("final_dataset.csv")

# GoldRiM aspect labels
print("=== GoldRiM Aspect ===")
gold_aspects = df[df['GoldRiM'] == 1]['Aspect'].tolist()
for label in gold_aspects:
    print(label)

print("\n=== SilverRiM Aspect ===")
silver_aspects = df[df['SilverRiM'] == 1]['Aspect'].tolist()
for label in silver_aspects:
    print(label)

# GoldRiM aspect labels
print("=== GoldRiM Aspect Labels ===")
gold_aspects = df[df['GoldRiM'] == 1]['Aspect-label'].tolist()
for label in gold_aspects:
    print(label)

print("\n=== SilverRiM Aspect Labels ===")
silver_aspects = df[df['SilverRiM'] == 1]['Aspect-label'].tolist()
for label in silver_aspects:
    print(label)
