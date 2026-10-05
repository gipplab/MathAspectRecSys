import re
import torch
import pandas as pd
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel

# -----------------------------
# 1. Data loading function
# -----------------------------

def load_data():
    """
    Load and preprocess document dataset.
    Returns:
        pd.DataFrame with columns: ['document_id', 'title', 'text']
    """
    df = pd.read_csv("data/main_dataset.csv")
    df = df.drop(columns=['id', 'title_language', 'e_identifier', 'j_identifier', 'review_language'])
    df = df.rename(columns={'identifier': 'document_id', 'review_text': 'text'})

    df_aux = pd.read_csv("data/aux_dataset.csv")
    df_aux = df_aux.drop(columns=['id'])
    df_aux = df_aux.rename(columns={'identifier': 'document_id', 'summary': 'text'})

    df = pd.concat([df, df_aux], ignore_index=True)

    # Filter out very short texts
    token_counts = df['text'].str.split().str.len()
    df = df[token_counts >= 30].reset_index(drop=True)

    # Clean bracketed content
    def remove_bracket_content(text):
        if pd.isna(text) or text == '':
            return text
        cleaned = re.sub(r'\[.*?\]', '', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    df['text'] = df['text'].apply(remove_bracket_content)
    return df

# -----------------------------
# 2. Mean pooling function (for SciBERT)
# -----------------------------

def mean_pooling(model_output, attention_mask):
    """Apply mean pooling to get sentence embedding."""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# -----------------------------
# 3. Main indexing pipeline
# -----------------------------

def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} documents.")

    # Load SciBERT tokenizer and model
    model_name = "allenai/scibert_scivocab_uncased"
    # model_name = "models/scibert/"  # local copy

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs.")
    model = model.to(device)
    model.eval()

    # Get embedding dimension
    d = model.module.config.hidden_size if hasattr(model, "module") else model.config.hidden_size

    # Initialize FAISS index
    index_cpu = faiss.IndexFlatIP(d)
    if device == "cuda":
        index = faiss.index_cpu_to_all_gpus(index_cpu)
        print(f"FAISS using {faiss.get_num_gpus()} GPU(s).")
    else:
        index = index_cpu
        print("Using CPU for FAISS.")

    # Process in batches
    batch_size = 1000
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]

        # Combine title and text
        texts = [
            f"{row['title']} {row['text']}"
            for _, row in batch.iterrows()
        ]

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling
            embeddings = mean_pooling(outputs, inputs['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        index.add(embeddings.cpu().numpy())

        if start % 5000 == 0:
            print(f"Processed {start} / {len(df)} rows", flush=True)

    print(f"Total vectors indexed: {index.ntotal}")

    # Save index
    final_index = faiss.index_gpu_to_cpu(index) if device == "cuda" else index
    faiss.write_index(final_index, "faiss_index_scibert.index")
    print("Index saved as 'faiss_index_scibert.index'")

    # Save metadata
    df[['document_id', 'title']].to_csv("doc_metadata_scibert.csv", index=False)
    print("Document metadata saved as 'doc_metadata_scibert.csv'")

# -----------------------------
# Example query
# -----------------------------

def example_query(query_doc_id):
    print("Query functionality requires access to original text. See SPECTER script for pattern.")
    pass

if __name__ == "__main__":
    main()