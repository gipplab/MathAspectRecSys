import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm

# -----------------------------
# 1. Configuration
# -----------------------------

class Config:
    model_name = "allenai/scibert_scivocab_uncased"  # or "models/scibert/" for local
    data_path = "data/recommendation_pairs.csv"      # must have: Seed, Recommendation
    metadata_path = "data/doc_metadata.csv"          # must have: document_id, title, text
    output_dir = "models/scibert_finetuned"
    batch_size = 128
    num_epochs = 3
    learning_rate = 2e-5
    warmup_steps = 500
    max_length = 512
    temperature = 0.05  # lower = sharper similarity
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

# -----------------------------
# 2. Data Loading
# -----------------------------

def load_document_texts(metadata_path):
    """Load full text for each document_id."""
    meta = pd.read_csv(metadata_path)
    return dict(zip(meta['document_id'], meta['text'].fillna("").astype(str)))

def load_training_pairs(data_path):
    """Load (seed_id, rec_id) pairs."""
    df = pd.read_csv(data_path)
    # Ensure columns exist
    assert 'Seed' in df.columns and 'Recommendation' in df.columns
    return df[['Seed', 'Recommendation']].values.tolist()

# -----------------------------
# 3. Dataset Class
# -----------------------------

class ContrastivePairDataset(Dataset):
    def __init__(self, pairs, doc_texts, tokenizer, max_length=512):
        self.pairs = pairs
        self.doc_texts = doc_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        seed_id, rec_id = self.pairs[idx]
        seed_text = self.doc_texts.get(seed_id, "")
        rec_text = self.doc_texts.get(rec_id, "")
        return seed_text, rec_text

def collate_fn(batch, tokenizer, max_length):
    seeds, recs = zip(*batch)
    seed_enc = tokenizer(
        list(seeds), padding=True, truncation=True,
        max_length=max_length, return_tensors="pt"
    )
    rec_enc = tokenizer(
        list(recs), padding=True, truncation=True,
        max_length=max_length, return_tensors="pt"
    )
    return seed_enc, rec_enc

# -----------------------------
# 4. Contrastive Loss (NT-Xent)
# -----------------------------

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, emb_a, emb_b):
        """
        emb_a, emb_b: [B, D] — aligned positive pairs
        Uses in-batch negatives.
        """
        batch_size = emb_a.size(0)
        # Normalize embeddings
        emb_a = F.normalize(emb_a, dim=1)
        emb_b = F.normalize(emb_b, dim=1)

        # Compute similarity matrix [2B, 2B]
        all_embeddings = torch.cat([emb_a, emb_b], dim=0)  # [2B, D]
        logits = torch.mm(all_embeddings, all_embeddings.t()) / self.temperature  # [2B, 2B]

        # Remove self-comparison
        logits_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=logits.device)
        logits.masked_fill_(logits_mask, float('-inf'))

        # Labels: positive pair is at (i, i+B) and (i+B, i)
        labels = torch.arange(batch_size, device=logits.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)  # [2B]

        loss = self.criterion(logits, labels)
        return loss

# -----------------------------
# 5. Model Wrapper
# -----------------------------

class SciBERTEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling
        last_hidden = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        return self.dropout(mean_pooled)

# -----------------------------
# 6. Training Loop
# -----------------------------

def train():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    # Load data
    print("Loading documents...")
    doc_texts = load_document_texts(cfg.metadata_path)
    print(f"Loaded {len(doc_texts)} documents.")

    print("Loading pairs...")
    pairs = load_training_pairs(cfg.data_path)
    print(f"Loaded {len(pairs)} training pairs.")

    # Split (optional)
    train_pairs, _ = train_test_split(pairs, test_size=0.05, random_state=cfg.seed)

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = SciBERTEncoder(cfg.model_name).to(cfg.device)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    total_steps = (len(train_pairs) // cfg.batch_size + 1) * cfg.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)

    # Loss
    loss_fn = ContrastiveLoss(temperature=cfg.temperature)

    # DataLoader
    train_dataset = ContrastivePairDataset(train_pairs, doc_texts, tokenizer, cfg.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, cfg.max_length),
        num_workers=4,
        pin_memory=True
    )

    # Training
    model.train()
    for epoch in range(cfg.num_epochs):
        total_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
        for seed_enc, rec_enc in progress:
            seed_enc = {k: v.to(cfg.device) for k, v in seed_enc.items()}
            rec_enc = {k: v.to(cfg.device) for k, v in rec_enc.items()}

            optimizer.zero_grad()
            emb_seed = model(**seed_enc)
            emb_rec = model(**rec_enc)
            loss = loss_fn(emb_seed, emb_rec)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

    # Save model
    os.makedirs(cfg.output_dir, exist_ok=True)
    model.bert.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"Model saved to {cfg.output_dir}")

if __name__ == "__main__":
    train()