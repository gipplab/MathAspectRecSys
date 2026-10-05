import os
import re
import argparse
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import multiprocessing as mp
from transformers import AutoTokenizer, AutoModel

# -----------------------------
# 1. Data Loading
# -----------------------------

def load_data():
    """Load and preprocess document dataset."""
    # Load main dataset
    df_main = pd.read_csv("data/main_dataset.csv")
    df_main = df_main.drop(columns=['id', 'title_language', 'e_identifier', 'j_identifier', 'review_language'])
    df_main = df_main.rename(columns={'identifier': 'document_id', 'review_text': 'text'})

    # Load auxiliary dataset (e.g., arXiv)
    df_aux = pd.read_csv("data/aux_dataset.csv")
    df_aux = df_aux.drop(columns=['id'])
    df_aux = df_aux.rename(columns={'identifier': 'document_id', 'summary': 'text'})

    # Combine
    df = pd.concat([df_main, df_aux], ignore_index=True)

    # Filter out very short texts
    token_counts = df['text'].str.split().str.len()
    df = df[token_counts >= 30].reset_index(drop=True)

    # Clean bracketed content [ ... ]
    def remove_bracket_content(text):
        if pd.isna(text) or text == '':
            return text
        cleaned = re.sub(r'\[.*?\]', '', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    df['text'] = df['text'].apply(remove_bracket_content)
    return df

# -----------------------------
# 2. Qwen3-Specific Pooling
# -----------------------------

def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the last non-padding token (works with left or right padding)."""
    # Detect padding side: if last token is attended by all, it's left-padded
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
    Process a data slice on a dedicated GPU using Qwen3-7B.
    """
    # Isolate GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_gpu_id)
    device = "cuda:0"

    print(f"[Node {node_id}, Local GPU {local_gpu_id}, Global GPU {global_gpu_id}] "
          f"Processing {len(df_slice)} documents...", flush=True)

    # Load Qwen3-7B tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        padding_side='left',
        trust_remote_code=True  # Required for Qwen
    )
    model = AutoModel.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,  # Use FP16 to fit in memory
        trust_remote_code=True,
        device_map="auto"
    )
    model = model.to(device)
    model.eval()

    embeddings_list = []
    total = len(df_slice)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = df_slice.iloc[start:end]

        # Format input
        texts = [
            f"Title: {row['title']}\nAbstract: {row['text']}"
            for _, row in batch.iterrows()
        ]

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=True
        ).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(**inputs)
            embeddings = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 norm for cosine similarity

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
# 4. Main Driver
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings using Qwen3-7B.")
    parser.add_argument('--model_dir', type=str, default="models/qwen3_7b/", 
                        help="Path to Qwen3-7B model (Hugging Face format)")
    parser.add_argument('--total_gpus', type=int, required=True, 
                        help="Total number of GPUs across all nodes")
    parser.add_argument('--gpus_per_node', type=int, default=4, 
                        help="Number of GPUs per node")
    parser.add_argument('--node_id', type=int, default=0, 
                        help="Current node ID (0-indexed)")
    parser.add_argument('--batch_size', type=int, default=2, 
                        help="Batch size per GPU (Qwen3-7B is large → keep small)")
    parser.add_argument('--max_length', type=int, default=8192, 
                        help="Maximum sequence length")
    args = parser.parse_args()

    print(f"🚀 Node {args.node_id}: Starting Qwen3-7B embedding generation...", flush=True)

    # Load data
    df = load_data()
    print(f"📊 Total documents loaded: {len(df):,}", flush=True)

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

    if num_local_gpus <= 0:
        print(f"⚠️ Node {args.node_id} has no GPUs assigned. Exiting.")
        return

    print(f"📍 Node {args.node_id}: Processing global GPUs {gpu_start} to {gpu_end - 1}", flush=True)

    # Launch processes
    processes = []
    for local_idx in range(num_local_gpus):
        global_gpu_id = gpu_start + local_idx
        p = mp.Process(
            target=process_gpu_batch,
            args=(
                global_gpu_id,
                local_idx,
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

    print(f"✅ Node {args.node_id}: All GPU processes completed.", flush=True)

    # Save document ID order (only once)
    if args.node_id == 0:
        df[['document_id']].to_csv("doc_ids_order.csv", index=False)
        print("💾 Document ID order saved to 'doc_ids_order.csv'")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()