import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
import faiss
from tqdm import tqdm

# -----------------------------
# 1. Configuration
# -----------------------------

class Config:
    model_name = "allenai/scibert_scivocab_uncased"  # or "models/scibert/"
    metadata_path = "data/doc_metadata.csv"
    edges_path = "data/edges.csv"
    output_dir = "outputs"
    batch_size = 512
    num_epochs = 10
    lr = 1e-4
    hidden_dim = 768
    proj_dim = 128
    temperature = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

# -----------------------------
# 2. Data Loading
# -----------------------------

def load_documents():
    df = pd.read_csv(Config.metadata_path)
    return df[['document_id', 'title', 'text']].reset_index(drop=True)

def load_graph(doc_df):
    """Build graph from edges and align with document order."""
    edges = pd.read_csv(Config.edges_path)
    
    # Map document_id to index
    id_to_idx = {id_: idx for idx, id_ in enumerate(doc_df['document_id'])}
    
    # Filter edges to only include known docs
    edges = edges[
        edges['src_id'].isin(id_to_idx) & 
        edges['dst_id'].isin(id_to_idx)
    ]
    
    edge_index = torch.tensor([
        [id_to_idx[src] for src in edges['src_id']],
        [id_to_idx[dst] for dst in edges['dst_id']]
    ], dtype=torch.long)
    
    return edge_index

def encode_texts(doc_df, model_name, device):
    """Encode all documents using SciBERT (mean pooling)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    embeddings = []
    batch_size = 32
    texts = (doc_df['title'] + " " + doc_df['text']).fillna("").tolist()
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            ).to(device)
            outputs = model(**inputs)
            # Mean pooling
            mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size())
            sum_emb = (outputs.last_hidden_state * mask).sum(1)
            mean_emb = sum_emb / mask.sum(1).clamp(min=1e-9)
            embeddings.append(mean_emb.cpu().numpy())
    
    return np.vstack(embeddings)

# -----------------------------
# 3. GraPHCL Model
# -----------------------------

class GraphCL(nn.Module):
    def __init__(self, input_dim, hidden_dim, proj_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, F.normalize(z, dim=1)

# -----------------------------
# 4. Contrastive Loss (GraphCL)
# -----------------------------

def graphcl_loss(z1, z2, temperature=0.1):
    """Contrastive loss between two augmented views."""
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = torch.mm(z, z.t()) / temperature  # [2B, 2B]
    
    # Mask out self-comparisons
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))
    
    # Positive pairs are (i, i+B)
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)
    
    loss = F.cross_entropy(sim, labels)
    return loss

# -----------------------------
# 5. Training Loop
# -----------------------------

def train():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    
    # Load data
    print("Loading documents...")
    doc_df = load_documents()
    print(f"Loaded {len(doc_df)} documents.")
    
    print("Encoding with SciBERT...")
    initial_embs = encode_texts(doc_df, cfg.model_name, cfg.device)
    x = torch.tensor(initial_embs, dtype=torch.float32).to(cfg.device)
    
    print("Building graph...")
    edge_index = load_graph(doc_df)
    data = Data(x=x, edge_index=edge_index)
    
    # Create augmented views (simple: add noise + dropout)
    def augment(x):
        noise = torch.randn_like(x) * 0.01
        x1 = x + noise
        x2 = F.dropout(x, p=0.2)
        return x1, x2
    
    # Model
    model = GraphCL(cfg.hidden_dim, cfg.hidden_dim, cfg.proj_dim).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    # Training
    model.train()
    for epoch in range(cfg.num_epochs):
        total_loss = 0
        x1, x2 = augment(data.x)
        
        optimizer.zero_grad()
        _, z1 = model(x1)
        _, z2 = model(x2)
        loss = graphcl_loss(z1, z2, cfg.temperature)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{cfg.num_epochs}, Loss: {loss.item():.4f}")
    
    # Save model
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "graphcl_model.pth"))
    print("Model saved.")
    
    # Generate final embeddings
    print("Generating final embeddings...")
    model.eval()
    with torch.no_grad():
        h, _ = model(data.x)
        final_embs = h.cpu().numpy()
    
    # Build FAISS index
    print("Building FAISS index...")
    d = final_embs.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(final_embs)  # for cosine similarity
    index.add(final_embs)
    
    faiss.write_index(index, os.path.join(cfg.output_dir, "graphcl_index.faiss"))
    print("Index saved.")
    
    # Save document IDs for lookup
    doc_df[['document_id']].to_csv(os.path.join(cfg.output_dir, "doc_ids.csv"), index=False)

if __name__ == "__main__":
    train()