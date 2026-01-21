import os
import re
import sys
import argparse
import torch
import faiss
import pandas as pd
import numpy as np
import torch.nn.functional as F
import multiprocessing as mp
from transformers import AutoTokenizer, AutoModel

# -----------------------------
# 1. Data Loading (modular & anonymized)
# -----------------------------

def load_data():
    """Load and preprocess documents."""
    df_main = pd.read_csv("data/main_dataset.csv")
    df_main = df_main.drop(columns=['id', 'title_language', 'e_identifier', 'j_identifier', 'review_language'])
    df_main = df_main.rename(columns={'identifier': 'document_id', 'review_text': 'text'})

    df_aux = pd.read_csv("data/aux_dataset.csv")
    df_aux = df_aux.drop(columns=['id'])
    df_aux = df_aux.rename(columns={'identifier': 'document_id', 'summary': 'text'})

    df = pd.concat([df_main, df_aux], ignore_index=True)

    # Filter short texts
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
# 2. Embedding Utilities
# -----------------------------

def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the last non-padding token."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# -----------------------------
# 3. GPU Worker Function
# -----------------------------

def process_gpu_batch(global_gpu_id, local_gpu_id, node_id, df_slice, model_dir, batch_size, max_length):
    """
    Process a slice of data on a specific GPU.
    """
    # Isolate GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_gpu_id)
    device = "cuda:0"

    print(f"[Node {node_id}, Local GPU {local_gpu_id}, Global GPU {global_gpu_id}] "
          f"Processing {len(df_slice)} rows...", flush=True)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side='left')
    model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float16)  # Use FP16 for large models
    model = model.to(device)
    model.eval()

    # Encode in batches
    embeddings_list = []
    total = len(df_slice)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = df_slice.iloc[start:end]
        texts = [f"Title: {row['title']}. Text: {row['text']}" for _, row in batch.iterrows()]

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model(**inputs)
            embeddings = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)

        embeddings_list.append(embeddings.cpu().numpy())

    # Save embeddings
    all_embeddings = np.vstack(embeddings_list)
    temp_file = f"embeddings_gpu_{global_gpu_id}.npy"
    np.save(temp_file, all_embeddings)
    print(f"[GPU {global_gpu_id}] Saved {len(all_embeddings)} embeddings to {temp_file}", flush=True)

    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()

# -----------------------------
# 4. Main Function
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Qwen embeddings using multiple GPUs.")
    parser.add_argument('--model_dir', type=str, default="models/qwen/", help="Path to Qwen model")
    parser.add_argument('--total_gpus', type=int, required=True, help="Total number of GPUs to use")
    parser.add_argument('--gpus_per_node', type=int, default=4, help="GPUs per node")
    parser.add_argument('--num_nodes', type=int, default=1, help="Total nodes (for cluster)")
    parser.add_argument('--node_id', type=int, default=0, help="Current node ID (0-indexed)")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size per GPU")
    parser.add_argument('--max_length', type=int, default=8192, help="Max token length")
    args = parser.parse_args()

    print(f"Node {args.node_id}: Initializing with {args.total_gpus} total GPUs", flush=True)

    # Load data
    df = load_data()
    print(f"Total documents: {len(df)}", flush=True)

    # Split data across all GPUs
    chunk_size = len(df) // args.total_gpus
    df_chunks = [
        df.iloc[i * chunk_size: (i + 1) * chunk_size if i < args.total_gpus - 1 else len(df)]
        for i in range(args.total_gpus)
    ]

    # Assign GPUs to this node
    gpu_start = args.node_id * args.gpus_per_node
    gpu_end = min(gpu_start + args.gpus_per_node, args.total_gpus)
    num_local_gpus = gpu_end - gpu_start

    print(f"Node {args.node_id}: Processing global GPUs {gpu_start} to {gpu_end - 1}", flush=True)

    # Launch processes
    processes = []
    for local_idx in range(num_local_gpus):
        global_gpu_id = gpu_start + local_idx
        p = mp.Process(
            target=process_gpu_batch,
            args=(
                global_gpu_id,
                local_idx,  # local GPU ID on this node
                args.node_id,
                df_chunks[global_gpu_id],
                args.model_dir,
                args.batch_size,
                args.max_length
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"Node {args.node_id}: All processes completed.", flush=True)

    # Optional: Save document IDs for later alignment
    if args.node_id == 0:
        df[['document_id']].to_csv("doc_ids_order.csv", index=False)
        print("Document ID order saved to 'doc_ids_order.csv'")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()