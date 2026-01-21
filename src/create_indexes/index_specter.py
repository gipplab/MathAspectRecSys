import re
import torch
import pandas as pd
import faiss
from transformers import AutoTokenizer, AutoModel

# -----------------------------
# 1. Data loading function (modular & anonymized)
# -----------------------------

def load_data():
    """
    Load and preprocess the document dataset.
    Returns:
        pd.DataFrame with columns: ['document_id', 'title', 'text']
    """
    # Load main dataset
    df = pd.read_csv("data/main_dataset.csv")
    df = df.drop(columns=['id', 'title_language', 'e_identifier', 'j_identifier', 'review_language'])
    df = df.rename(columns={'identifier': 'document_id', 'review_text': 'text'})

    # Load auxiliary dataset (e.g., arXiv)
    df_aux = pd.read_csv("data/aux_dataset.csv")
    df_aux = df_aux.drop(columns=['id'])
    df_aux = df_aux.rename(columns={'identifier': 'document_id', 'summary': 'text'})

    # Combine
    df = pd.concat([df, df_aux], ignore_index=True)

    # Filter short texts
    token_counts = df['text'].str.split().str.len()
    df = df[token_counts >= 30].reset_index(drop=True)

    # Clean text: remove [ ... ] brackets
    def remove_bracket_content(text):
        if pd.isna(text) or text == '':
            return text
        cleaned = re.sub(r'\[.*?\]', '', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    df['text'] = df['text'].apply(remove_bracket_content)
    return df

# -----------------------------
# 2. Main indexing pipeline
# -----------------------------

def main():
    # Load data
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} documents.")

    # Load model and tokenizer (assumes local model in ./models/specter/)
    model_dir = "models/specter/"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs.")
    model = model.to(device)
    model.eval()

    # Get embedding dimension
    d = model.module.config.hidden_size if hasattr(model, "module") else model.config.hidden_size

    # Initialize FAISS index (GPU if available)
    index_cpu = faiss.IndexFlatIP(d)  # Inner product = cosine similarity (after L2 norm)
    if device == "cuda":
        index = faiss.index_cpu_to_all_gpus(index_cpu)
        print(f"FAISS using {faiss.get_num_gpus()} GPU(s).")
    else:
        index = index_cpu
        print("Using CPU for FAISS.")

    # Process in batches
    batch_size = 5000
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        texts = [
            str(row['title']) + tokenizer.sep_token + str(row['text'])
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
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        index.add(embeddings.cpu().numpy())

        if start % 10000 == 0:
            print(f"Processed {start} / {len(df)} rows", flush=True)

    print(f"Total vectors indexed: {index.ntotal}")

    # Save index
    final_index = faiss.index_gpu_to_cpu(index) if device == "cuda" else index
    faiss.write_index(final_index, "faiss_index_specter.index")
    print("Index saved as 'faiss_index_specter.index'")

    # Optional: Save document metadata for lookup
    df[['document_id', 'title']].to_csv("doc_metadata.csv", index=False)
    print("Document metadata saved as 'doc_metadata.csv'")

# -----------------------------
# 3. Example query (optional)
# -----------------------------

def example_query(query_doc_id=725.35083):
    """Example: retrieve top-k similar documents."""
    df_meta = pd.read_csv("doc_metadata.csv")
    index = faiss.read_index("faiss_index_specter.index")

    model_dir = "models/specter/"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    try:
        query_idx = df_meta.index[df_meta['document_id'] == query_doc_id].item()
        # In practice, you'd need full text for query — this is just illustrative
        print(f"Query index: {query_idx}. Full implementation requires original text.")
    except ValueError:
        print(f"Document ID {query_doc_id} not found in metadata.")

if __name__ == "__main__":
    main()
    # example_query()