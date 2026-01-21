import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE
import numpy as np
from typing import Dict, List, Tuple, Optional


# ==================== Graph Initialization ====================
class GraphInitializer:
    """Initializes heterogeneous graph with papers and authors"""
    
    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
        self.text_encoder = nn.Linear(768, embedding_dim)  # Assumes LLM embeddings of dim 768
    
    def initialize_graph(self, 
                        papers: Dict, 
                        authors: Dict,
                        citations: List[Tuple],
                        author_papers: List[Tuple],
                        aspect_labels: Dict) -> Dict:
        """
        Initializes heterogeneous graph G = (V, E) with:
        - Node types: Papers P and Authors A
        - Edge types: Aspect-labeled citation edges and authorship edges
        """
        graph = {
            'paper_nodes': {},
            'author_nodes': {},
            'citation_edges': [],
            'authorship_edges': [],
            'aspect_labels': aspect_labels
        }
        
        # Initialize paper node embeddings h_i^(0) = f_text(p_i)
        for paper_id, paper_data in papers.items():
            title_abstract_embedding = self._encode_text(paper_data['title'], paper_data['abstract'])
            graph['paper_nodes'][paper_id] = {
                'embedding': title_abstract_embedding,
                'title': paper_data['title'],
                'abstract': paper_data['abstract']
            }
        
        # Initialize author node embeddings h_l^(0) = f_text(a_m)
        for author_id, author_name in authors.items():
            author_embedding = self._encode_text(author_name)
            graph['author_nodes'][author_id] = {
                'embedding': author_embedding,
                'name': author_name
            }
        
        # Add aspect-labeled citation edges (p_i, p_j, as_k)
        for source, target, aspect in citations:
            graph['citation_edges'].append({
                'source': source,
                'target': target,
                'aspect': aspect
            })
        
        # Add authorship edges (a_m, p_i)
        for author_id, paper_id in author_papers:
            graph['authorship_edges'].append({
                'author': author_id,
                'paper': paper_id
            })
        
        return graph
    
    def _encode_text(self, *texts) -> torch.Tensor:
        """Encodes text using LLM (simulated here)"""
        combined_text = ' '.join(texts)
        # In practice, use actual LLM embeddings (e.g., from transformers library)
        dummy_embedding = torch.randn(768)
        return self.text_encoder(dummy_embedding.unsqueeze(0)).squeeze(0)


# ==================== Heterogeneous Message Passing ====================
class HeterogeneousMessagePasser(nn.Module):
    """
    GraphSAGE-inspired heterogeneous aggregation scheme
    Stacks L=2 GNN layers with fixed neighborhood sampling
    """
    
    def __init__(self, 
                 input_dim: int = 256,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 paper_neighbors: int = 20,
                 author_neighbors: int = 5):
        super().__init__()
        self.num_layers = num_layers
        self.paper_neighbors = paper_neighbors
        self.author_neighbors = author_neighbors
        
        # Relation-specific transformation matrices W_r^(l)
        self.paper_transform = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.author_transform = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.self_transform = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.activation = nn.ReLU()
    
    def forward(self, 
                paper_embeddings: Dict,
                author_embeddings: Dict,
                graph: Dict) -> Tuple[Dict, Dict]:
        """
        Heterogeneous message passing: h_v^(l+1) = σ(W_0^(l) h_v^(l) + Σ_r Σ_u∈N_r(v) W_r^(l) h_u^(l))
        """
        current_paper_emb = paper_embeddings.copy()
        current_author_emb = author_embeddings.copy()
        
        for layer in range(self.num_layers):
            new_paper_emb = {}
            new_author_emb = {}
            
            # Update paper node embeddings
            for paper_id, emb in current_paper_emb.items():
                # Self embedding with W_0^(l)
                self_update = self.self_transform[layer](emb)
                
                # Aggregate from citation neighbors (paper-to-paper)
                paper_neighbors = self._get_paper_neighbors(paper_id, graph)
                paper_neighbor_embeddings = [current_paper_emb[nid] for nid in paper_neighbors[:self.paper_neighbors]]
                if paper_neighbor_embeddings:
                    neighbor_agg = torch.mean(torch.stack([
                        self.paper_transform[layer](emb) for emb in paper_neighbor_embeddings
                    ]), dim=0)
                else:
                    neighbor_agg = torch.zeros_like(emb)
                
                # Aggregate from author neighbors (authorship edges)
                author_neighbors = self._get_author_neighbors(paper_id, graph)
                author_neighbor_embeddings = [current_author_emb[aid] for aid in author_neighbors[:self.author_neighbors]]
                if author_neighbor_embeddings:
                    author_agg = torch.mean(torch.stack([
                        self.author_transform[layer](emb) for emb in author_neighbor_embeddings
                    ]), dim=0)
                else:
                    author_agg = torch.zeros_like(emb)
                
                # Combined update: h_v^(l+1)
                new_emb = self.activation(self_update + neighbor_agg + author_agg)
                new_paper_emb[paper_id] = new_emb
            
            # Update author node embeddings
            for author_id, emb in current_author_emb.items():
                self_update = self.self_transform[layer](emb)
                
                # Aggregate from connected papers
                connected_papers = self._get_connected_papers(author_id, graph)
                paper_embeddings_list = [current_paper_emb[pid] for pid in connected_papers[:self.paper_neighbors]]
                if paper_embeddings_list:
                    neighbor_agg = torch.mean(torch.stack([
                        self.paper_transform[layer](emb) for emb in paper_embeddings_list
                    ]), dim=0)
                else:
                    neighbor_agg = torch.zeros_like(emb)
                
                new_emb = self.activation(self_update + neighbor_agg)
                new_author_emb[author_id] = new_emb
            
            current_paper_emb = new_paper_emb
            current_author_emb = new_author_emb
        
        return current_paper_emb, current_author_emb
    
    def _get_paper_neighbors(self, paper_id: str, graph: Dict) -> List[str]:
        """Get paper neighbors via citation edges"""
        neighbors = []
        for edge in graph['citation_edges']:
            if edge['source'] == paper_id:
                neighbors.append(edge['target'])
            elif edge['target'] == paper_id:
                neighbors.append(edge['source'])
        return neighbors
    
    def _get_author_neighbors(self, paper_id: str, graph: Dict) -> List[str]:
        """Get author neighbors via authorship edges"""
        authors = []
        for edge in graph['authorship_edges']:
            if edge['paper'] == paper_id:
                authors.append(edge['author'])
        return authors
    
    def _get_connected_papers(self, author_id: str, graph: Dict) -> List[str]:
        """Get papers connected to author"""
        papers = []
        for edge in graph['authorship_edges']:
            if edge['author'] == author_id:
                papers.append(edge['paper'])
        return papers


# ==================== Aspect-Conditioned Similarity Scoring ====================
class AspectConditionedScorer(nn.Module):
    """
    Aspect-conditioned interaction and similarity scoring
    z_{q,c}^(as_k) = [h_q || h_c || h_q ⊙ h_c || e_{as_k}]
    """
    
    def __init__(self, 
                 embedding_dim: int = 256,
                 aspect_dim: int = 64,
                 num_aspects: int = 3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.aspect_dim = aspect_dim
        
        # Learnable aspect embeddings e_{as_k}
        self.aspect_embeddings = nn.Embedding(num_aspects, aspect_dim)
        
        # Interaction vector dimension: 2*d + d + d_{as}
        interaction_dim = 2 * embedding_dim + embedding_dim + aspect_dim
        
        # Similarity scoring network f_score (MLP)
        self.score_network = nn.Sequential(
            nn.Linear(interaction_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, 
                query_emb: torch.Tensor,
                candidate_emb: torch.Tensor,
                aspect_id: int) -> torch.Tensor:
        """
        Constructs aspect-conditioned interaction vector and scores similarity
        s_{q,c}^(as_k) = f_score(z_{q,c}^(as_k))
        """
        # Get aspect embedding e_{as_k}
        aspect_emb = self.aspect_embeddings(torch.tensor(aspect_id))
        
        # Construct interaction vector: [h_q || h_c || h_q ⊙ h_c || e_{as_k}]
        element_wise_product = query_emb * candidate_emb  # h_q ⊙ h_c
        interaction_vector = torch.cat([
            query_emb,
            candidate_emb,
            element_wise_product,
            aspect_emb
        ], dim=0)
        
        # Score: s_{q,c}^(as_k) = f_score(z_{q,c}^(as_k))
        score = self.score_network(interaction_vector)
        
        return score


# ==================== Aspect Prediction Auxiliary Task ====================
class AspectPredictor(nn.Module):
    """
    Auxiliary aspect prediction task
    Encourages aspect-discriminative representations via multi-class classification
    """
    
    def __init__(self,
                 embedding_dim: int = 256,
                 num_aspects: int = 3):
        super().__init__()
        
        # Interaction vector dimension: 2*d + d
        interaction_dim = 3 * embedding_dim
        
        # Aspect classifier f_aspect (MLP)
        self.aspect_classifier = nn.Sequential(
            nn.Linear(interaction_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_aspects)
        )
    
    def forward(self,
                query_emb: torch.Tensor,
                candidate_emb: torch.Tensor) -> torch.Tensor:
        """
        Predicts aspect from paper pair
        ŷ_{q,c} = f_aspect(z_{q,c}) where z_{q,c} = [h_q || h_c || h_q ⊙ h_c]
        """
        # Construct interaction vector without aspect embedding
        element_wise_product = query_emb * candidate_emb
        interaction_vector = torch.cat([
            query_emb,
            candidate_emb,
            element_wise_product
        ], dim=0)
        
        # Predict aspect logits
        aspect_logits = self.aspect_classifier(interaction_vector)
        
        return aspect_logits


# ==================== Complete ACHGNN Model ====================
class ACHGNN(nn.Module):
    """
    Aspect-Conditioned Heterogeneous GNN for mathematical citation-based paper recommendation
    """
    
    def __init__(self,
                 embedding_dim: int = 256,
                 aspect_dim: int = 64,
                 num_aspects: int = 3,
                 num_gnn_layers: int = 2,
                 aspect_loss_weight: float = 0.5):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_aspects = num_aspects
        self.aspect_loss_weight = aspect_loss_weight
        
        # Components
        self.message_passer = HeterogeneousMessagePassing(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            num_layers=num_gnn_layers
        )
        self.aspect_scorer = AspectConditionedScorer(
            embedding_dim=embedding_dim,
            aspect_dim=aspect_dim,
            num_aspects=num_aspects
        )
        self.aspect_predictor = AspectPredictor(
            embedding_dim=embedding_dim,
            num_aspects=num_aspects
        )
    
    def forward(self,
                paper_embeddings: Dict,
                author_embeddings: Dict,
                graph: Dict,
                positive_pairs: List[Tuple],
                negative_pairs: List[Tuple],
                aspect_labels: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: message passing + aspect-conditioned scoring + aspect prediction
        Returns: ranking_loss, aspect_loss, total_loss
        """
        # Step 1: Heterogeneous message passing
        paper_emb, author_emb = self.message_passer(paper_embeddings, author_embeddings, graph)
        
        # Step 2: Compute ranking loss (Bayesian Personalized Ranking)
        ranking_losses = []
        for query_id, pos_id in positive_pairs:
            for neg_id in negative_pairs:
                if pos_id != neg_id:  # Ensure different papers
                    query_emb = paper_emb[query_id]
                    pos_emb = paper_emb[pos_id]
                    neg_emb = paper_emb[neg_id]
                    
                    # Get aspect for this pair
                    aspect = aspect_labels.get((query_id, pos_id), 0)
                    
                    # Score positive and negative
                    score_pos = self.aspect_scorer(query_emb, pos_emb, aspect)
                    score_neg = self.aspect_scorer(query_emb, neg_emb, aspect)
                    
                    # BPR loss: L_rank = -log(σ(s_pos - s_neg))
                    bpr_loss = -torch.log(torch.sigmoid(score_pos - score_neg))
                    ranking_losses.append(bpr_loss)
        
        ranking_loss = torch.mean(torch.stack(ranking_losses)) if ranking_losses else torch.tensor(0.0)
        
        # Step 3: Compute aspect prediction loss
        aspect_losses = []
        for query_id, pos_id in positive_pairs:
            query_emb = paper_emb[query_id]
            pos_emb = paper_emb[pos_id]
            aspect_id = aspect_labels.get((query_id, pos_id), 0)
            
            # Predict aspect
            aspect_logits = self.aspect_predictor(query_emb, pos_emb)
            
            # Cross-entropy loss
            ce_loss = F.cross_entropy(
                aspect_logits.unsqueeze(0),
                torch.tensor([aspect_id])
            )
            aspect_losses.append(ce_loss)
        
        aspect_loss = torch.mean(torch.stack(aspect_losses)) if aspect_losses else torch.tensor(0.0)
        
        # Step 4: Combined objective: L = L_rank + λ * L_aspect
        total_loss = ranking_loss + self.aspect_loss_weight * aspect_loss
        
        return ranking_loss, aspect_loss, total_loss
    
    def inference(self,
                  paper_embeddings: Dict,
                  author_embeddings: Dict,
                  graph: Dict,
                  query_id: str,
                  aspect_id: int,
                  top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Inference: compute contextual embeddings and rank candidates by aspect-conditioned score
        TopN^(as)(p_q) = argsort_c s_{q,c}^(as)
        """
        # Get contextual embeddings
        paper_emb, author_emb = self.message_passer(paper_embeddings, author_embeddings, graph)
        
        query_emb = paper_emb[query_id]
        scores = {}
        
        # Score all candidate papers
        for candidate_id, candidate_emb in paper_emb.items():
            if candidate_id != query_id:
                score = self.aspect_scorer(query_emb, candidate_emb, aspect_id)
                scores[candidate_id] = score.item()
        
        # Rank and return top-k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ==================== Experimental Setup ====================
def setup_experiment():
    """Sets up a simple example with mock data"""
    
    # Mock papers
    papers = {
        'p1': {'title': 'Graph Neural Networks', 'abstract': 'Survey of GNNs'},
        'p2': {'title': 'Citation Networks', 'abstract': 'Citation analysis'},
        'p3': {'title': 'Message Passing', 'abstract': 'GNN message passing'},
        'p4': {'title': 'Authorship Models', 'abstract': 'Author networks'},
    }
    
    # Mock authors
    authors = {
        'a1': 'Alice',
        'a2': 'Bob',
        'a3': 'Charlie',
    }
    
    # Mock citations with aspects (background, proof, motivation)
    citations = [
        ('p1', 'p2', 0),  # background
        ('p2', 'p3', 1),  # proof
        ('p3', 'p4', 2),  # motivation
        ('p1', 'p3', 0),
    ]
    
    # Mock authorship
    author_papers = [
        ('a1', 'p1'),
        ('a1', 'p2'),
        ('a2', 'p2'),
        ('a2', 'p3'),
        ('a3', 'p4'),
    ]
    
    # Aspect labels for paper pairs
    aspect_labels = {
        ('p1', 'p2'): 0,
        ('p2', 'p3'): 1,
        ('p3', 'p4'): 2,
    }
    
    return papers, authors, citations, author_papers, aspect_labels


if __name__ == "__main__":
    # Initialize
    papers, authors, citations, author_papers, aspect_labels = setup_experiment()
    
    # Create graph
    initializer = GraphInitializer(embedding_dim=256)
    graph = initializer.initialize_graph(papers, authors, citations, author_papers, aspect_labels)
    
    # Initialize embeddings
    paper_embeddings = {pid: graph['paper_nodes'][pid]['embedding'] for pid in papers.keys()}
    author_embeddings = {aid: graph['author_nodes'][aid]['embedding'] for aid in authors.keys()}
    
    # Create model
    model = ACHGNN(embedding_dim=256, aspect_dim=64, num_aspects=3)
    
    # Training
    positive_pairs = [('p1', 'p2'), ('p2', 'p3'), ('p3', 'p4')]
    negative_pairs = [pid for pid in papers.keys()]
    
    rank_loss, asp_loss, total = model(
        paper_embeddings,
        author_embeddings,
        graph,
        positive_pairs,
        negative_pairs,
        aspect_labels
    )
    
    print(f"Ranking Loss: {rank_loss.item():.4f}")
    print(f"Aspect Loss: {asp_loss.item():.4f}")
    print(f"Total Loss: {total.item():.4f}")
    
    # Inference: get recommendations for paper p1 with aspect 0 (background)
    recommendations = model.inference(
        paper_embeddings,
        author_embeddings,
        graph,
        query_id='p1',
        aspect_id=0,
        top_k=3
    )
    
    print(f"\nTop-3 recommendations for p1 (aspect: background):")
    for paper_id, score in recommendations:
        print(f"  {paper_id}: {score:.4f}")