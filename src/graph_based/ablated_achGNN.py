import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

from achGNN import GraphInitializer, ACHGNN

# -----------------------------
# Data Loading & Preparation
# -----------------------------

def load_data():
    """Load papers, authors, citation edges and aspect labels."""
    # Returns:
    # papers: Dict[str, Dict]  # {paper_id: {'title': ..., 'abstract': ...}}
    # authors: Dict[str, str]   # {author_id: 'name'}
    # citations: List[Tuple]    # [(src, dst, aspect_id), ...]
    # author_papers: List[Tuple] # [(author_id, paper_id), ...]
    # aspect_labels: Dict[Tuple, int] # {(src, dst): aspect_id}
    
    df = pd.read_csv("data/doc_metadata.csv")
    edges = pd.read_csv("data/edges_with_aspects.csv")  # cols: src, dst, aspect
    
    papers = {}
    for _, row in df.iterrows():
        papers[row['document_id']] = {
            'title': row['title'],
            'abstract': row['text']
        }
    
    # Build author mappings (dummy authors)
    authors = {f"a{i}": f"Author_{i}" for i in range(100)}
    author_papers = [(f"a{i % 100}", pid) for i, pid in enumerate(papers.keys())]
    
    citations = []
    aspect_labels = {}
    for _, row in edges.iterrows():
        src, dst, aspect = row['src'], row['dst'], int(row['aspect'])
        citations.append((src, dst, aspect))
        aspect_labels[(src, dst)] = aspect
    
    return papers, authors, citations, author_papers, aspect_labels

# -----------------------------
# Evaluation Metrics
# -----------------------------

def compute_recall_at_k(recommendations: List[Tuple[str, float]], 
                       ground_truth: set, k: int = 10) -> float:
    """Compute Recall@K"""
    top_k = set([rec[0] for rec in recommendations[:k]])
    hits = len(top_k & ground_truth)
    return hits / len(ground_truth) if ground_truth else 0.0

# -----------------------------
# Training & Evaluation Loop
# -----------------------------

def train_and_evaluate(config: Dict, fold: int = 0):
    """Train model and return validation metrics"""
    # Set seed for reproducibility
    torch.manual_seed(42 + fold)
    np.random.seed(42 + fold)
    
    # Load data
    papers, authors, citations, author_papers, aspect_labels = load_data()
    
    # Create graph
    initializer = GraphInitializer(embedding_dim=config['embedding_dim'])
    graph = initializer.initialize_graph(
        papers, authors, citations, author_papers, aspect_labels
    )
    
    # Initialize embeddings
    paper_embeddings = {pid: graph['paper_nodes'][pid]['embedding'] for pid in papers.keys()}
    author_embeddings = {aid: graph['author_nodes'][aid]['embedding'] for aid in authors.keys()}
    
    # Prepare positive/negative pairs
    positive_pairs = list(aspect_labels.keys())
    all_papers = list(papers.keys())
    
    # Split data for this fold (simple random split)
    np.random.shuffle(positive_pairs)
    split_idx = int(0.8 * len(positive_pairs))
    train_pairs = positive_pairs[:split_idx]
    val_pairs = positive_pairs[split_idx:]
    
    # Create model
    model = ACHGNN(
        embedding_dim=config['embedding_dim'],
        aspect_dim=config['aspect_dim'],
        num_aspects=config['num_aspects'],
        num_gnn_layers=config['num_gnn_layers'],
        aspect_loss_weight=config['aspect_loss_weight']
    ).to(config['device'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    # Training loop
    model.train()
    for epoch in range(config['epochs']):
        total_loss = 0
        for query_id, pos_id in train_pairs:
            # Sample negative
            neg_id = np.random.choice(all_papers)
            while neg_id == pos_id or neg_id == query_id:
                neg_id = np.random.choice(all_papers)
            
            optimizer.zero_grad()
            rank_loss, asp_loss, total = model(
                paper_embeddings,
                author_embeddings,
                graph,
                [(query_id, pos_id)],
                [neg_id],
                aspect_labels
            )
            total.backward()
            optimizer.step()
            total_loss += total.item()
    
    # Validation
    model.eval()
    recalls = {k: [] for k in [5, 10, 20]}
    ground_truth_map = {}  # {query: set of relevant papers}
    
    for query_id, pos_id in val_pairs:
        if query_id not in ground_truth_map:
            ground_truth_map[query_id] = set()
        ground_truth_map[query_id].add(pos_id)
    
    for query_id, ground_truth in ground_truth_map.items():
        aspect = aspect_labels.get((query_id, next(iter(ground_truth))), 0)
        try:
            recommendations = model.inference(
                paper_embeddings,
                author_embeddings,
                graph,
                query_id,
                aspect,
                top_k=20
            )
            for k in [5, 10, 20]:
                recall = compute_recall_at_k(recommendations, ground_truth, k)
                recalls[k].append(recall)
        except Exception as e:
            print(f"Error during inference: {e}")
            continue
    
    # Average recalls
    avg_recalls = {k: np.mean(v) if v else 0.0 for k, v in recalls.items()}
    return avg_recalls

# -----------------------------
# Main Ablation Runner
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_authors', type=int, required=True)
    parser.add_argument('--num_neighbors', type=int, required=True)
    parser.add_argument('--num_layers', type=int, required=True)
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='ablation_results')
    args = parser.parse_args()
    
    # Create config
    config = {
        'num_authors': args.num_authors,
        'num_neighbors': args.num_neighbors,
        'num_gnn_layers': args.num_layers,
        'embedding_dim': 256,
        'aspect_dim': 64,
        'num_aspects': 4,
        'aspect_loss_weight': 0.5,
        'lr': 1e-3,
        'epochs': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Run k-fold CV
    results = []
    for fold in range(args.k_folds):
        print(f"Running fold {fold+1}/{args.k_folds}...")
        recalls = train_and_evaluate(config, fold)
        results.append({
            'fold': fold,
            'recall@5': recalls[5],
            'recall@10': recalls[10],
            'recall@20': recalls[20]
        })
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = f"{args.output_dir}/ablation_nauth{args.num_authors}_nneigh{args.num_neighbors}_nlayers{args.num_layers}.json"
    
    summary = {
        'config': {
            'num_authors': args.num_authors,
            'num_neighbors': args.num_neighbors,
            'num_layers': args.num_layers
        },
        'results': results,
        'mean_recall@5': np.mean([r['recall@5'] for r in results]),
        'std_recall@5': np.std([r['recall@5'] for r in results]),
        'mean_recall@10': np.mean([r['recall@10'] for r in results]),
        'std_recall@10': np.std([r['recall@10'] for r in results]),
        'mean_recall@20': np.mean([r['recall@20'] for r in results]),
        'std_recall@20': np.std([r['recall@20'] for r in results])
    }
    
    with open(result_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to {result_file}")

if __name__ == "__main__":
    main()