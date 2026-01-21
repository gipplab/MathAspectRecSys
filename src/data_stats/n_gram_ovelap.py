import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Set
import re
from tqdm import tqdm
import json
import glob
import os
import warnings
warnings.filterwarnings('ignore')


class NGramOverlapAnalyzer:
    """
    Efficient n-gram overlap analyzer using pre-computed embedding top-100 candidates.
    
    This class handles:
    - Data loading and preparation
    - N-gram extraction and analysis
    - Similarity computation
    - Data aggregation for visualization
    
    All plotting is handled by separate standalone functions.
    """
    
    def __init__(self, df: pd.DataFrame, goldRiM: Dict[str, list], 
                 top100_dir: str, n: int = 1, citDict: Dict[str, list] = None):
        """
        Initialize the analyzer.
        
        Args:
            df: DataFrame with columns ['document_id', 'text', 'title']
            goldRiM: Dict with seed doc_id as key and list of recommendation doc_ids as value
            top100_dir: Directory containing top100 JSON part files
            n: n-gram size (default: 2 for bigrams)
            citDict: Optional dict with seed doc_id as key and list of citation doc_ids as value
        """
        self.df = df.set_index('document_id')
        self.goldRiM = goldRiM
        self.citDict = citDict if citDict is not None else {}
        self.n = n
        self.top100_candidates = {}
        
        print(f"\n{'='*80}")
        print(f"INITIALIZING N-GRAM OVERLAP ANALYZER")
        print(f"{'='*80}")
        print(f"[INFO] Dataset size: {len(self.df):,} documents")
        print(f"[INFO] GoldRiM seeds: {len(self.goldRiM):,}")
        if self.citDict:
            print(f"[INFO] Citation pairs: {len(self.citDict):,}")
        
        self._load_top100_embeddings(top100_dir)
        
        print(f"[SUCCESS] Loaded top-100 candidates for {len(self.top100_candidates):,} seeds")
        print(f"{'='*80}\n")
    
    def _get_text(self, doc_id: str) -> str:
        """
        Get text for a document, handling cases with multiple rows for same doc_id.
        
        Args:
            doc_id: Document ID to retrieve
        
        Returns:
            Text content as string from first row, or empty string if not found
        """
        if doc_id not in self.df.index:
            return ""
        
        text_data = self.df.loc[doc_id, 'text']
        
        if isinstance(text_data, pd.Series):
            text = text_data.iloc[0] if len(text_data) > 0 else ""
        else:
            text = text_data
        
        return text if isinstance(text, str) else ""
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text while preserving content.
        """
        if not isinstance(text, str) or not text or text.isspace():
            return ""
        
        text = re.sub(r'http\S+|www\S+', '', text)
        text = text.lower()
        text = re.sub(r'\\\(|\\\)|\\\[|\\\]', ' ', text)
        text = re.sub(r'\\[a-z]+\*?(?:\{[^}]*\})?', ' ', text)
        text = re.sub(r'[\{\}]', ' ', text)
        text = re.sub(r'[^a-z0-9\s\.\,\-\:\;]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def debug_seed_cleaning(self, seed_id: str, show_length: bool = True):
        """Debug function to see what happens to a seed during cleaning."""
        if seed_id not in self.df.index:
            print(f"[ERROR] Seed '{seed_id}' not found in df")
            return
        
        text = self._get_text(seed_id)
        
        print(f"\n{'='*80}")
        print(f"DEBUG: {seed_id}")
        print(f"{'='*80}\n")
        
        print(f"[1] ORIGINAL TEXT:")
        print(f"    Length: {len(text)} chars")
        print(f"    Preview: {text[:200]}...")
        
        cleaned = self._clean_text(text)
        
        print(f"\n[2] AFTER CLEANING:")
        print(f"    Length: {len(cleaned)} chars")
        print(f"    Preview: {cleaned[:200]}...")
        
        words = cleaned.split()
        print(f"\n[3] WORD EXTRACTION:")
        print(f"    Words: {len(words)}")
        print(f"    Words (first 20): {words[:20]}")
        
        ngrams = self._extract_ngrams(cleaned)
        
        print(f"\n[4] N-GRAM EXTRACTION (n=2):")
        print(f"    Total bigrams: {len(ngrams)}")
        if ngrams:
            print(f"    Sample bigrams: {list(ngrams)[:10]}")
        else:
            print(f"    ⚠️ NO BIGRAMS EXTRACTED!")
            if len(words) < 2:
                print(f"       Reason: Only {len(words)} word(s), need at least 2 for bigrams")
        
        print(f"\n{'='*80}\n")
    
    def _extract_ngrams(self, text: str) -> Set[str]:
        """Extract n-grams from text."""
        if not text:
            return set()
        
        words = text.split()
        
        if len(words) < self.n:
            return set()
        
        ngrams = {' '.join(words[i:i+self.n]) for i in range(len(words) - self.n + 1)}
        return ngrams
    
    def _exact_jaccard_overlap(self, ngrams1: Set[str], ngrams2: Set[str]) -> Tuple[float, float, int]:
        """Calculate exact Jaccard and overlap coefficient."""
        if not ngrams1 or not ngrams2:
            return 0.0, 0.0, 0
        
        common = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        jaccard = common / union if union > 0 else 0.0
        overlap_coeff = common / min(len(ngrams1), len(ngrams2)) if min(len(ngrams1), len(ngrams2)) > 0 else 0.0
        
        return jaccard, overlap_coeff, common
    
    def _load_top100_embeddings(self, input_dir: str):
        """Load top 100 embedding-similar documents from JSON part files."""
        print(f"\n[INFO] Loading top-100 embedding results from {input_dir}...")
        
        part_files = sorted(glob.glob(os.path.join(input_dir, "maRDI_top100_*_part_*.json")))
        
        if not part_files:
            print(f"[WARNING] No part files found in {input_dir}")
            print(f"[WARNING] Searched for: {os.path.join(input_dir, 'maRDI_top100_*_part_*.json')}")
            return
        
        print(f"[INFO] Found {len(part_files)} part files to process")
        
        total_loaded = 0
        for part_file in tqdm(part_files, desc="Loading top-100 files", unit="file"):
            try:
                with open(part_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines[1:-1]:
                    line = line.strip()
                    if not line or line == ',':
                        continue
                    
                    if line.endswith(','):
                        line = line[:-1]
                    
                    try:
                        entry = json.loads(line)
                        
                        for seed_id, candidates_list in entry.items():
                            candidate_ids = [cand[0] for cand in candidates_list]
                            self.top100_candidates[seed_id] = candidate_ids
                            total_loaded += 1
                    except json.JSONDecodeError:
                        continue
            except Exception as e:
                print(f"[WARNING] Error loading {part_file}: {e}")
                continue
        
        print(f"[SUCCESS] Loaded top-100 candidates for {total_loaded:,} documents")
    
    def find_seed_with_text(self, min_word_count: int = 50) -> str:
        """Find a seed document with sufficient text and top-100 candidates."""
        print(f"\n[INFO] Finding seed with at least {min_word_count} words and top-100 candidates...")
        
        for seed_id in self.goldRiM.keys():
            if seed_id not in self.df.index:
                continue
            
            if seed_id not in self.top100_candidates:
                continue
            
            text = self._get_text(seed_id)
            cleaned = self._clean_text(text)
            word_count = len(cleaned.split())
            
            if word_count >= min_word_count:
                print(f"[SUCCESS] Found seed '{seed_id}' with {word_count} words")
                return seed_id
        
        print("[ERROR] No seed with sufficient text found")
        return None
    
    def analyze_single_seed(self, seed_id: str, verbose: bool = True) -> dict:
        """
        Analyze a single seed against its top-100 embedding candidates.
        Compare n-gram overlap between all candidates and recommendations.
        """
        if seed_id not in self.df.index:
            print(f"[ERROR] Seed '{seed_id}' not found in df")
            return None
        
        if seed_id not in self.goldRiM:
            print(f"[ERROR] Seed '{seed_id}' not found in goldRiM")
            return None
        
        if seed_id not in self.top100_candidates:
            print(f"[ERROR] Seed '{seed_id}' not found in top-100 embeddings")
            return None
        
        seed_text = self.df.loc[seed_id, 'text']
        seed_text = self._clean_text(seed_text)
        seed_ngrams = self._extract_ngrams(seed_text)
        
        if not seed_ngrams:
            print(f"[ERROR] Seed '{seed_id}' has no extractable n-grams")
            return None
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"ANALYZING SEED: {seed_id}")
            print(f"{'='*80}")
            print(f"[INFO] Seed has {len(seed_ngrams)} n-grams")
        
        top100_ids = self.top100_candidates[seed_id]
        valid_top100_ids = [did for did in top100_ids if did in self.df.index]
        
        if not valid_top100_ids:
            print(f"[ERROR] No valid top-100 candidates found in df")
            return None
        
        if verbose:
            print(f"\n[INFO] Comparing with {len(valid_top100_ids)} top-100 embedding candidates...")
        
        max_doc_jaccard = 0.0
        max_doc_overlap = 0.0
        max_doc_id = None
        max_doc_common = 0
        
        for doc_id in tqdm(valid_top100_ids, desc="Finding best doc (top-100)", disable=not verbose, unit="doc"):
            if doc_id == seed_id:
                continue
            
            doc_text = self.df.loc[doc_id, 'text']
            doc_text = self._clean_text(doc_text)
            doc_ngrams = self._extract_ngrams(doc_text)
            
            if not doc_ngrams:
                continue
            
            jaccard, overlap_coeff, common = self._exact_jaccard_overlap(seed_ngrams, doc_ngrams)
            
            if jaccard > max_doc_jaccard:
                max_doc_jaccard = jaccard
                max_doc_overlap = overlap_coeff
                max_doc_id = doc_id
                max_doc_common = common
        
        if verbose:
            print(f"[SUCCESS] Found best doc in top-100: {max_doc_id}")
            print(f"  Jaccard: {max_doc_jaccard:.4f} | Overlap Coeff: {max_doc_overlap:.4f}")
        
        rec_ids = self.goldRiM[seed_id]
        valid_rec_ids = [rid for rid in rec_ids if rid in self.df.index]
        
        if not valid_rec_ids:
            print(f"[WARNING] No valid recommendations found for {seed_id}")
            return None
        
        if verbose:
            print(f"\n[INFO] Comparing with {len(valid_rec_ids)} recommendations...")
        
        max_rec_jaccard = 0.0
        max_rec_overlap = 0.0
        max_rec_id = None
        max_rec_common = 0
        
        for rec_id in tqdm(valid_rec_ids, desc="Finding best rec", disable=not verbose, unit="doc"):
            rec_text = self.df.loc[rec_id, 'text']
            rec_text = self._clean_text(rec_text)
            rec_ngrams = self._extract_ngrams(rec_text)
            
            if not rec_ngrams:
                continue
            
            jaccard, overlap_coeff, common = self._exact_jaccard_overlap(seed_ngrams, rec_ngrams)
            
            if jaccard > max_rec_jaccard:
                max_rec_jaccard = jaccard
                max_rec_overlap = overlap_coeff
                max_rec_id = rec_id
                max_rec_common = common
        
        if verbose:
            print(f"[SUCCESS] Found best recommendation: {max_rec_id}")
            print(f"  Jaccard: {max_rec_jaccard:.4f} | Overlap Coeff: {max_rec_overlap:.4f}")
        
        results = {
            'seed_id': seed_id,
            'seed_text_length': len(seed_text.split()),
            'seed_ngrams': len(seed_ngrams),
            'best_doc_id': max_doc_id,
            'best_doc_jaccard': max_doc_jaccard,
            'best_doc_overlap_coeff': max_doc_overlap,
            'best_doc_common': max_doc_common,
            'best_rec_id': max_rec_id,
            'best_rec_jaccard': max_rec_jaccard,
            'best_rec_overlap_coeff': max_rec_overlap,
            'best_rec_common': max_rec_common,
            'jaccard_diff': max_rec_jaccard - max_doc_jaccard,
            'overlap_diff': max_rec_overlap - max_doc_overlap,
            'num_top100_candidates': len(valid_top100_ids),
            'num_recommendations': len(valid_rec_ids)
        }
        
        return results
    
    def analyze_multiple_seeds(self, seed_ids: List[str] = None, limit: int = None) -> List[dict]:
        """Analyze multiple seeds efficiently."""
        if seed_ids is None:
            seed_ids = [sid for sid in self.goldRiM.keys() 
                       if sid in self.top100_candidates and sid in self.df.index]
        
        if limit:
            seed_ids = seed_ids[:limit]
        
        results = []
        for seed_id in tqdm(seed_ids, desc="Analyzing seeds", unit="seed"):
            result = self.analyze_single_seed(seed_id, verbose=False)
            if result is not None:
                results.append(result)
        
        return results
    
    def get_top_k_similar_docs(self, seed_id: str, k: int = 10) -> Tuple[List[str], List[float]]:
        """Get top-k most similar documents by n-gram overlap for a single seed."""
        if seed_id not in self.df.index:
            return [], []
        
        if seed_id not in self.top100_candidates:
            return [], []
        
        seed_text = self._get_text(seed_id)
        seed_text = self._clean_text(seed_text)
        seed_ngrams = self._extract_ngrams(seed_text)
        
        if not seed_ngrams:
            return [], []
        
        top100_ids = self.top100_candidates[seed_id]
        valid_top100_ids = [did for did in top100_ids if did in self.df.index and did != seed_id]
        
        similarities = []
        for doc_id in valid_top100_ids:
            doc_text = self._get_text(doc_id)
            doc_text = self._clean_text(doc_text)
            doc_ngrams = self._extract_ngrams(doc_text)
            
            if not doc_ngrams:
                continue
            
            jaccard, _, _ = self._exact_jaccard_overlap(seed_ngrams, doc_ngrams)
            similarities.append((doc_id, jaccard))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]
        
        top_k_ids = [doc_id for doc_id, _ in top_k]
        top_k_sims = [sim for _, sim in top_k]
        
        return top_k_ids, top_k_sims
    
    def get_top_k_for_all_seeds(self, k: int = 10, limit: int = None) -> Dict[str, List[str]]:
        """Get top-k most n-gram similar documents for all goldRiM seeds."""
        seed_ids = [sid for sid in self.goldRiM.keys() 
                   if sid in self.top100_candidates and sid in self.df.index]
        
        if limit:
            seed_ids = seed_ids[:limit]
        
        print(f"\n[INFO] Computing top-{k} most similar docs for {len(seed_ids):,} seeds...")
        
        top_k_dict = {}
        
        for seed_id in tqdm(seed_ids, desc=f"Computing top-{k} similar docs", unit="seed"):
            top_k_ids, _ = self.get_top_k_similar_docs(seed_id, k=k)
            if top_k_ids:
                top_k_dict[seed_id] = top_k_ids
        
        print(f"[SUCCESS] Computed top-{k} similar docs for {len(top_k_dict):,} seeds")
        
        return top_k_dict
    
    def get_top_k_for_all_seeds_with_scores(self, k: int = 10, limit: int = None) -> Dict[str, List[Tuple[str, float]]]:
        """Get top-k most n-gram similar documents with similarity scores for all goldRiM seeds."""
        seed_ids_with_candidates = [sid for sid in self.goldRiM.keys() 
                                   if sid in self.top100_candidates and sid in self.df.index]
        
        skipped_no_candidates = len(self.goldRiM) - len(seed_ids_with_candidates)
        
        if limit:
            seed_ids_with_candidates = seed_ids_with_candidates[:limit]
        
        print(f"\n[INFO] Computing top-{k} most similar docs (with scores)...")
        print(f"[INFO] Total seeds in goldRiM: {len(self.goldRiM):,}")
        print(f"[INFO] Seeds with top-100 candidates: {len(seed_ids_with_candidates):,}")
        print(f"[INFO] Seeds skipped (no top-100): {skipped_no_candidates:,}")
        
        top_k_dict = {}
        failed_seeds = []
        
        for seed_id in tqdm(seed_ids_with_candidates, desc=f"Computing top-{k} similar docs", unit="seed"):
            top_k_ids, top_k_scores = self.get_top_k_similar_docs(seed_id, k=k)
            if top_k_ids:
                top_k_dict[seed_id] = [[doc_id, float(score)] for doc_id, score in zip(top_k_ids, top_k_scores)]
            else:
                failed_seeds.append(seed_id)
        
        print(f"\n[SUCCESS] Computed top-{k} similar docs for {len(top_k_dict):,} seeds")
        if failed_seeds:
            print(f"[WARNING] {len(failed_seeds)} seeds failed (no valid n-grams or docs)")
            print(f"  Examples: {failed_seeds[:5]}")
        
        return top_k_dict
    
    def save_top_k_results(self, top_k_dict: Dict, output_path: str):
        """Save top-k results to a JSON file."""
        print(f"\n[INFO] Saving results to {output_path}...")
        
        with open(output_path, 'w') as f:
            json.dump(top_k_dict, f, indent=2)
        
        print(f"[SUCCESS] Saved {len(top_k_dict):,} seeds to {output_path}")
    
    def get_top_pairs_by_similarity(self, top_k_with_scores: Dict[str, List], 
                                    num_pairs: int = 420) -> List[Tuple[str, str, float]]:
        """Extract top N pairs with maximum Jaccard similarity scores."""
        print(f"\n[INFO] Extracting top {num_pairs} pairs by Jaccard similarity...")
        
        pairs = []
        
        for seed_id, doc_list in top_k_with_scores.items():
            for doc_id, score in doc_list:
                pairs.append((seed_id, doc_id, float(score)))
        
        pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs = pairs[:num_pairs]
        
        print(f"[SUCCESS] Extracted {len(top_pairs)} top pairs")
        print(f"  Max Jaccard: {top_pairs[0][2]:.4f}")
        print(f"  Min Jaccard: {top_pairs[-1][2]:.4f}")
        print(f"  Mean Jaccard: {np.mean([p[2] for p in top_pairs]):.4f}")
        
        return top_pairs
    
    def get_violin_plot_data(self, top_k_with_scores: Dict[str, List], verbose: bool = True) -> Tuple[List[float], List[float], List[float], dict]:
        """
        Prepare data for violin plot comparing GoldRiM, Citation, and top documents.
        With detailed logging to show where pairs are lost.
        
        Returns:
            Tuple of (goldrim_scores, citation_scores, top_doc_scores, stats_dict)
        """
        print(f"\n[INFO] Computing violin plot data...")
        print(f"[DEBUG] citDict available: {bool(self.citDict)}, size: {len(self.citDict) if self.citDict else 0}")
        
        goldrim_scores = []
        citation_scores = []
        top_doc_scores = []
        
        # ===== GOLDRIM SCORES =====
        print(f"\n[INFO] Computing GoldRiM recommendation overlaps...")
        goldrim_stats = {
            'total_seeds': len(self.goldRiM),
            'seeds_in_df': 0,
            'seeds_with_text': 0,
            'seeds_with_ngrams': 0,
            'total_rec_pairs': 0,
            'rec_pairs_in_df': 0,
            'rec_pairs_with_text': 0,
            'rec_pairs_with_ngrams': 0,
            'final_pairs': 0,
        }
        
        for seed_id, rec_list in tqdm(self.goldRiM.items(), desc="GoldRiM pairs", unit="seed"):
            if seed_id not in self.df.index:
                continue
            goldrim_stats['seeds_in_df'] += 1
            
            seed_text = self._get_text(seed_id)
            if not seed_text:
                continue
            goldrim_stats['seeds_with_text'] += 1
            
            seed_text = self._clean_text(seed_text)
            seed_ngrams = self._extract_ngrams(seed_text)
            
            if not seed_ngrams:
                continue
            goldrim_stats['seeds_with_ngrams'] += 1
            
            goldrim_stats['total_rec_pairs'] += len(rec_list)
            
            for rec_id in rec_list:
                if rec_id not in self.df.index:
                    continue
                goldrim_stats['rec_pairs_in_df'] += 1
                
                rec_text = self._get_text(rec_id)
                if not rec_text:
                    continue
                goldrim_stats['rec_pairs_with_text'] += 1
                
                rec_text = self._clean_text(rec_text)
                rec_ngrams = self._extract_ngrams(rec_text)
                
                if not rec_ngrams:
                    continue
                goldrim_stats['rec_pairs_with_ngrams'] += 1
                
                jaccard, _, _ = self._exact_jaccard_overlap(seed_ngrams, rec_ngrams)
                goldrim_scores.append(jaccard)
                goldrim_stats['final_pairs'] += 1
        
        if verbose:
            print(f"\n  [GOLDRIM LOSS BREAKDOWN]")
            print(f"    Total seeds: {goldrim_stats['total_seeds']}")
            print(f"    Seed in df: {goldrim_stats['seeds_in_df']} ({100*goldrim_stats['seeds_in_df']/goldrim_stats['total_seeds']:.1f}%)")
            print(f"    Seed with text: {goldrim_stats['seeds_with_text']} (loss: {goldrim_stats['seeds_in_df']-goldrim_stats['seeds_with_text']})")
            print(f"    Seed with ngrams: {goldrim_stats['seeds_with_ngrams']} (loss: {goldrim_stats['seeds_with_text']-goldrim_stats['seeds_with_ngrams']})")
            print(f"    Total rec pairs: {goldrim_stats['total_rec_pairs']}")
            print(f"    Rec pairs in df: {goldrim_stats['rec_pairs_in_df']} (loss: {goldrim_stats['total_rec_pairs']-goldrim_stats['rec_pairs_in_df']})")
            print(f"    Rec pairs with text: {goldrim_stats['rec_pairs_with_text']} (loss: {goldrim_stats['rec_pairs_in_df']-goldrim_stats['rec_pairs_with_text']})")
            print(f"    Rec pairs with ngrams: {goldrim_stats['rec_pairs_with_ngrams']} (loss: {goldrim_stats['rec_pairs_with_text']-goldrim_stats['rec_pairs_with_ngrams']})")
            print(f"    Final pairs: {goldrim_stats['final_pairs']}")
        
        print(f"[SUCCESS] Computed {len(goldrim_scores)} GoldRiM pair overlaps")
        
        # ===== CITATION SCORES =====
        if self.citDict and len(self.citDict) > 0:
            print(f"\n[INFO] Computing citation pair overlaps from citDict...")
            citation_stats = {
                'total_seeds': len(self.citDict),
                'seeds_in_df': 0,
                'seeds_with_text': 0,
                'seeds_with_ngrams': 0,
                'total_cit_pairs': 0,
                'cit_pairs_in_df': 0,
                'cit_pairs_with_text': 0,
                'cit_pairs_with_ngrams': 0,
                'final_pairs': 0,
            }
            
            for seed_id, cit_list in tqdm(self.citDict.items(), desc="Citation pairs", unit="seed"):
                if seed_id not in self.df.index:
                    continue
                citation_stats['seeds_in_df'] += 1
                
                seed_text = self._get_text(seed_id)
                if not seed_text:
                    continue
                citation_stats['seeds_with_text'] += 1
                
                seed_text = self._clean_text(seed_text)
                seed_ngrams = self._extract_ngrams(seed_text)
                
                if not seed_ngrams:
                    continue
                citation_stats['seeds_with_ngrams'] += 1
                
                citation_stats['total_cit_pairs'] += len(cit_list)
                
                for cit_id in cit_list:
                    if cit_id not in self.df.index:
                        continue
                    citation_stats['cit_pairs_in_df'] += 1
                    
                    cit_text = self._get_text(cit_id)
                    if not cit_text:
                        continue
                    citation_stats['cit_pairs_with_text'] += 1
                    
                    cit_text = self._clean_text(cit_text)
                    cit_ngrams = self._extract_ngrams(cit_text)
                    
                    if not cit_ngrams:
                        continue
                    citation_stats['cit_pairs_with_ngrams'] += 1
                    
                    jaccard, _, _ = self._exact_jaccard_overlap(seed_ngrams, cit_ngrams)
                    citation_scores.append(jaccard)
                    citation_stats['final_pairs'] += 1
            
            if verbose:
                print(f"\n  [CITATION LOSS BREAKDOWN]")
                print(f"    Total seeds: {citation_stats['total_seeds']}")
                print(f"    Seed in df: {citation_stats['seeds_in_df']} ({100*citation_stats['seeds_in_df']/citation_stats['total_seeds']:.1f}%)")
                print(f"    Seed with text: {citation_stats['seeds_with_text']} (loss: {citation_stats['seeds_in_df']-citation_stats['seeds_with_text']})")
                print(f"    Seed with ngrams: {citation_stats['seeds_with_ngrams']} (loss: {citation_stats['seeds_with_text']-citation_stats['seeds_with_ngrams']})")
                print(f"    Total cit pairs: {citation_stats['total_cit_pairs']}")
                print(f"    Cit pairs in df: {citation_stats['cit_pairs_in_df']} (loss: {citation_stats['total_cit_pairs']-citation_stats['cit_pairs_in_df']})")
                print(f"    Cit pairs with text: {citation_stats['cit_pairs_with_text']} (loss: {citation_stats['cit_pairs_in_df']-citation_stats['cit_pairs_with_text']})")
                print(f"    Cit pairs with ngrams: {citation_stats['cit_pairs_with_ngrams']} (loss: {citation_stats['cit_pairs_with_text']-citation_stats['cit_pairs_with_ngrams']})")
                print(f"    Final pairs: {citation_stats['final_pairs']}")
            
            print(f"[SUCCESS] Computed {len(citation_scores)} citation pair overlaps")
        else:
            print(f"[WARNING] citDict is empty or not provided - citation scores will be empty")
        
        # ===== TOP DOCUMENT SCORES =====
        print(f"\n[INFO] Computing top document overlaps...")
        top_doc_stats = {
            'total_seeds': len(top_k_with_scores),
            'seeds_in_df': 0,
            'seeds_with_text': 0,
            'seeds_with_ngrams': 0,
            'total_doc_pairs': 0,
            'doc_pairs_valid_seed': 0,
            'doc_pairs_in_df': 0,
            'doc_pairs_with_text': 0,
            'doc_pairs_with_ngrams': 0,
            'final_pairs': 0,
        }
        
        for seed_id, doc_list in tqdm(top_k_with_scores.items(), desc="Top docs", unit="seed"):
            if seed_id not in self.df.index:
                continue
            top_doc_stats['seeds_in_df'] += 1
            
            seed_text = self._get_text(seed_id)
            if not seed_text:
                continue
            top_doc_stats['seeds_with_text'] += 1
            
            seed_text = self._clean_text(seed_text)
            seed_ngrams = self._extract_ngrams(seed_text)
            
            if not seed_ngrams:
                continue
            top_doc_stats['seeds_with_ngrams'] += 1
            
            num_recs_for_seed = len(self.goldRiM.get(seed_id, []))
            top_doc_stats['total_doc_pairs'] += num_recs_for_seed
            
            for doc_id, score in doc_list[:num_recs_for_seed]:
                top_doc_stats['doc_pairs_valid_seed'] += 1
                
                if doc_id not in self.df.index or doc_id == seed_id:
                    continue
                top_doc_stats['doc_pairs_in_df'] += 1
                
                doc_text = self._get_text(doc_id)
                if not doc_text:
                    continue
                top_doc_stats['doc_pairs_with_text'] += 1
                
                doc_text = self._clean_text(doc_text)
                doc_ngrams = self._extract_ngrams(doc_text)
                
                if not doc_ngrams:
                    continue
                top_doc_stats['doc_pairs_with_ngrams'] += 1
                
                jaccard, _, _ = self._exact_jaccard_overlap(seed_ngrams, doc_ngrams)
                top_doc_scores.append(jaccard)
                top_doc_stats['final_pairs'] += 1
        
        if verbose:
            print(f"\n  [TOP DOCUMENTS LOSS BREAKDOWN]")
            print(f"    Total seeds: {top_doc_stats['total_seeds']}")
            print(f"    Seed in df: {top_doc_stats['seeds_in_df']} ({100*top_doc_stats['seeds_in_df']/top_doc_stats['total_seeds']:.1f}%)")
            print(f"    Seed with text: {top_doc_stats['seeds_with_text']} (loss: {top_doc_stats['seeds_in_df']-top_doc_stats['seeds_with_text']})")
            print(f"    Seed with ngrams: {top_doc_stats['seeds_with_ngrams']} (loss: {top_doc_stats['seeds_with_text']-top_doc_stats['seeds_with_ngrams']})")
            print(f"    Total doc pairs: {top_doc_stats['total_doc_pairs']}")
            print(f"    Doc pairs valid seed: {top_doc_stats['doc_pairs_valid_seed']} (loss: {top_doc_stats['total_doc_pairs']-top_doc_stats['doc_pairs_valid_seed']})")
            print(f"    Doc pairs in df: {top_doc_stats['doc_pairs_in_df']} (loss: {top_doc_stats['doc_pairs_valid_seed']-top_doc_stats['doc_pairs_in_df']})")
            print(f"    Doc pairs with text: {top_doc_stats['doc_pairs_with_text']} (loss: {top_doc_stats['doc_pairs_in_df']-top_doc_stats['doc_pairs_with_text']})")
            print(f"    Doc pairs with ngrams: {top_doc_stats['doc_pairs_with_ngrams']} (loss: {top_doc_stats['doc_pairs_with_text']-top_doc_stats['doc_pairs_with_ngrams']})")
            print(f"    Final pairs: {top_doc_stats['final_pairs']}")
        
        print(f"[SUCCESS] Computed {len(top_doc_scores)} top document overlaps")
        
        # ===== EQUALIZE AND CREATE STATS =====
        min_len = min(len(goldrim_scores), len(top_doc_scores))
        if citation_scores:
            min_len = min(min_len, len(citation_scores))
        
        print(f"\n[INFO] Equalizing sample sizes to {min_len} samples per group...")
        
        goldrim_scores = goldrim_scores[:min_len]
        top_doc_scores = top_doc_scores[:min_len]
        
        if citation_scores:
            citation_scores = citation_scores[:min_len]
        else:
            print(f"[WARNING] No citation scores computed. Check if citDict is properly loaded and documents exist in df.")
            citation_scores = []
        
        stats = {
            'goldrim_count': len(goldrim_scores),
            'citation_count': len(citation_scores),
            'topdog_count': len(top_doc_scores),
            'goldrim_mean': np.mean(goldrim_scores) if goldrim_scores else 0.0,
            'goldrim_median': np.median(goldrim_scores) if goldrim_scores else 0.0,
            'goldrim_std': np.std(goldrim_scores) if goldrim_scores else 0.0,
            'citation_mean': np.mean(citation_scores) if citation_scores else 0.0,
            'citation_median': np.median(citation_scores) if citation_scores else 0.0,
            'citation_std': np.std(citation_scores) if citation_scores else 0.0,
            'topdog_mean': np.mean(top_doc_scores) if top_doc_scores else 0.0,
            'topdog_median': np.median(top_doc_scores) if top_doc_scores else 0.0,
            'topdog_std': np.std(top_doc_scores) if top_doc_scores else 0.0,
        }
        
        print(f"\n[SUCCESS] Data prepared for violin plot")
        print(f"  GoldRiM pairs: {len(goldrim_scores)} (mean={stats['goldrim_mean']:.4f}, median={stats['goldrim_median']:.4f})")
        print(f"  Citation pairs: {len(citation_scores)} (mean={stats['citation_mean']:.4f}, median={stats['citation_median']:.4f})")
        print(f"  Top docs: {len(top_doc_scores)} (mean={stats['topdog_mean']:.4f}, median={stats['topdog_median']:.4f})")
        
        return goldrim_scores, citation_scores, top_doc_scores, stats
    
    def save_top_pairs(self, top_pairs: List[Tuple[str, str, float]], output_path: str):
        """Save top pairs to JSON file."""
        print(f"\n[INFO] Saving {len(top_pairs)} pairs to {output_path}...")
        
        pairs_list = [
            {
                "seed_id": seed_id,
                "doc_id": doc_id,
                "jaccard_similarity": float(score)
            }
            for seed_id, doc_id, score in top_pairs
        ]
        
        with open(output_path, 'w') as f:
            json.dump(pairs_list, f, indent=2)
        
        print(f"[SUCCESS] Saved {len(top_pairs)} pairs to {output_path}")
    
    def print_results(self, results: dict):
        """Print detailed analysis results."""
        if results is None:
            print("[ERROR] No results to display")
            return
        
        print(f"\n{'='*80}")
        print("ANALYSIS RESULTS")
        print(f"{'='*80}\n")
        
        print(f"Seed Document: {results['seed_id']}")
        print(f"  Text Length: {results['seed_text_length']} words")
        print(f"  N-grams Extracted: {results['seed_ngrams']}")
        
        print(f"\n{'BEST IN TOP-100 EMBEDDING CANDIDATES':^80}")
        print("-" * 80)
        print(f"Document ID: {results['best_doc_id']}")
        print(f"  Jaccard Similarity:    {results['best_doc_jaccard']:.4f}")
        print(f"  Overlap Coefficient:   {results['best_doc_overlap_coeff']:.4f}")
        print(f"  Common N-grams:        {results['best_doc_common']}")
        
        print(f"\n{'BEST RECOMMENDATION':^80}")
        print("-" * 80)
        print(f"Document ID: {results['best_rec_id']}")
        print(f"  Jaccard Similarity:    {results['best_rec_jaccard']:.4f}")
        print(f"  Overlap Coefficient:   {results['best_rec_overlap_coeff']:.4f}")
        print(f"  Common N-grams:        {results['best_rec_common']}")
        
        print(f"\n{'COMPARISON':^80}")
        print("-" * 80)
        print(f"Jaccard Difference (Rec - Top100): {results['jaccard_diff']:+.4f}")
        
        if results['jaccard_diff'] > 0:
            print(f"\n✓ Best recommendation has HIGHER Jaccard similarity than best top-100 doc")
        elif results['jaccard_diff'] < 0:
            print(f"\n✗ Best recommendation has LOWER Jaccard similarity than best top-100 doc")
        else:
            print(f"\n= Best recommendation has EQUAL Jaccard similarity to best top-100 doc")
        
        print(f"{'='*80}\n")



# ============================================
# STANDALONE PLOTTING FUNCTIONS
# Separated from NGramOverlapAnalyzer class
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import List, Dict


def plot_violin_comparison_two_way(goldrim_scores: List[float], 
                                   top_doc_scores: List[float],
                                   stats: dict, 
                                   save_path: str = None):
    """
    Create a two-way violin plot comparing GoldRiM recommendations vs top n-gram documents.
    
    Args:
        goldrim_scores: List of Jaccard scores for GoldRiM pairs
        top_doc_scores: List of Jaccard scores for top documents
        stats: Statistics dictionary from analyzer.get_violin_plot_data()
        save_path: Path to save figure (optional)
    
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    if not goldrim_scores or not top_doc_scores:
        print("[ERROR] No data to plot")
        return None
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    data = [goldrim_scores, top_doc_scores]
    positions = [1, 2]
    labels = ['GoldRiM Recommendations', 'Top N-gram Documents']
    colors = ['#3498db', '#2ecc71']
    
    parts = ax.violinplot(data, positions=positions, widths=0.6, 
                          showmeans=True, showmedians=True, showextrema=False)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(2)
    
    for partname in ('cbars', 'cmaxes', 'cmedians', 'cmeans'):
        if partname in parts:
            parts[partname].set_edgecolor('black')
            parts[partname].set_linewidth(2.5)
    
    parts['cmeans'].set_edgecolor('red')
    parts['cmeans'].set_linewidth(3)
    parts['cmedians'].set_edgecolor('orange')
    parts['cmedians'].set_linewidth(3)
    
    np.random.seed(42)
    jitter_strength = 0.04
    for i, (data_points, pos, color) in enumerate(zip(data, positions, colors)):
        x = np.random.normal(pos, jitter_strength, size=len(data_points))
        ax.scatter(x, data_points, alpha=0.4, s=30, color=color, edgecolors='black', linewidth=0.5)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=13, fontweight='bold')
    ax.set_ylabel('Jaccard Similarity', fontsize=13, fontweight='bold')
    ax.set_title('N-gram Overlap Comparison: GoldRiM vs Top Documents', 
                fontsize=15, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim([-0.08, 1.08])
    ax.set_xlim([0.3, 2.7])
    
    stats_text_1 = (
        f"N = {stats['goldrim_count']}\n"
        f"μ = {stats['goldrim_mean']:.4f}\n"
        f"σ = {stats['goldrim_std']:.4f}\n"
        f"M = {stats['goldrim_median']:.4f}"
    )
    
    stats_text_2 = (
        f"N = {stats['topdog_count']}\n"
        f"μ = {stats['topdog_mean']:.4f}\n"
        f"σ = {stats['topdog_std']:.4f}\n"
        f"M = {stats['topdog_median']:.4f}"
    )
    
    y_pos = 0.93
    ax.text(1, y_pos, stats_text_1, ha='center', va='top', fontsize=11, family='monospace',
           bbox=dict(boxstyle='round,pad=0.7', facecolor='#3498db', alpha=0.15, 
                    edgecolor='#3498db', linewidth=2))
    
    ax.text(2, y_pos, stats_text_2, ha='center', va='top', fontsize=11, family='monospace',
           bbox=dict(boxstyle='round,pad=0.7', facecolor='#2ecc71', alpha=0.15, 
                    edgecolor='#2ecc71', linewidth=2))
    
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=3, label='Mean (μ)'),
        Line2D([0], [0], color='orange', linewidth=3, label='Median (M)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.95, edgecolor='black')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[SUCCESS] Figure saved to {save_path}")
    
    return fig, ax

def plot_violin_comparison_three_way(goldrim_scores: List[float], 
                                     citation_scores: List[float],
                                     top_doc_scores: List[float],
                                     stats: dict, 
                                     save_path: str = None):
    """
    Create three-way violin plot comparing GoldRiM, Citation, and Top Documents.
    Optimized for two-column format (single column width in Overleaf).
    
    Args:
        goldrim_scores: List of Jaccard scores for GoldRiM pairs
        citation_scores: List of Jaccard scores for citation pairs
        top_doc_scores: List of Jaccard scores for top documents
        stats: Statistics dictionary from analyzer.get_violin_plot_data()
        save_path: Path to save figure (optional). Supports .pdf, .png, etc.
    
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    data_lists = []
    labels = []
    colors = []
    
    if goldrim_scores:
        data_lists.append(goldrim_scores)
        labels.append('GoldRiM\nRecommendations')
        colors.append('#3498db')
    
    if citation_scores:
        data_lists.append(citation_scores)
        labels.append('Citation\nPairs')
        colors.append('#e74c3c')
    
    if top_doc_scores:
        data_lists.append(top_doc_scores)
        labels.append('Top N-gram\nDocuments')
        colors.append('#2ecc71')
    
    if not data_lists:
        print("[ERROR] No data to plot")
        return None
    
    # Optimized for two-column Overleaf format with larger contents
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    
    # Use much closer positions to reduce spacing between violins
    positions = [1, 1.6, 2.2]  # Tighter spacing
    
    parts = ax.violinplot(data_lists, positions=positions, widths=0.45, 
                          showmeans=True, showmedians=True, showextrema=False)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(2.5)
    
    for partname in ('cbars', 'cmaxes', 'cmedians', 'cmeans'):
        if partname in parts:
            parts[partname].set_edgecolor('black')
            parts[partname].set_linewidth(3)
    
    parts['cmeans'].set_edgecolor('red')
    parts['cmeans'].set_linewidth(3.5)
    parts['cmedians'].set_edgecolor('orange')
    parts['cmedians'].set_linewidth(3.5)
    
    np.random.seed(42)
    jitter_strength = 0.05
    for i, (data_points, pos, color) in enumerate(zip(data_lists, positions, colors)):
        x = np.random.normal(pos, jitter_strength, size=len(data_points))
        ax.scatter(x, data_points, alpha=0.4, s=35, color=color, edgecolors='black', linewidth=0.7)
    
    # New x-labels (split across multiple lines to avoid occlusion)
    new_labels = [
        'GoldRiM\nrecommendations', 
        'GoldRiM\nseed-citations', 
        'GoldRiM seed-doc\nwith top token\noverlap'
    ]
    
    ax.set_xticks(positions)
    ax.set_xticklabels(new_labels, fontsize=14.5, fontweight='bold')
    ax.set_ylabel('Jaccard Similarity', fontsize=16, fontweight='bold')
    ax.set_title('Three-Way N-gram Overlap Comparison', 
                fontsize=17, fontweight='bold', pad=18)
    
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim([-0.08, 1.08])
    
    # Tight xlim for closer violins
    ax.set_xlim([0.6, 2.8])
    
    
    y_pos = 0.93
    
    if goldrim_scores:
        stats_text = (
            f"N = 420\n"
            f"μ = {stats['goldrim_mean']:.4f}\n"
            f"σ = {stats['goldrim_std']:.4f}\n"
            f"M = {stats['goldrim_median']:.4f}"
        )
        ax.text(positions[0], y_pos, stats_text, ha='center', va='top', fontsize=14, family='monospace', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='#3498db', alpha=0.15, 
                        edgecolor='#3498db', linewidth=2))
    
    if citation_scores:
        idx = 1 if goldrim_scores else 0
        stats_text = (
            f"N = 420\n"
            f"μ = {stats['citation_mean']:.4f}\n"
            f"σ = {stats['citation_std']:.4f}\n"
            f"M = {stats['citation_median']:.4f}"
        )
        ax.text(positions[idx], y_pos, stats_text, ha='center', va='top', fontsize=14, family='monospace', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='#e74c3c', alpha=0.15, 
                        edgecolor='#e74c3c', linewidth=2))
    
    if top_doc_scores:
        idx = len(positions) - 1
        stats_text = (
            f"N = 420\n"
            f"μ = {stats['topdog_mean']:.4f}\n"
            f"σ = {stats['topdog_std']:.4f}\n"
            f"M = {stats['topdog_median']:.4f}"
        )
        ax.text(positions[idx], y_pos, stats_text, ha='center', va='top', fontsize=14, family='monospace', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='#2ecc71', alpha=0.15, 
                        edgecolor='#2ecc71', linewidth=2))
    
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=3, label='Mean (μ)'),
        Line2D([0], [0], color='orange', linewidth=3, label='Median (M)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=14, framealpha=0.95, edgecolor='black', fancybox=True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"\n[SUCCESS] Figure saved to {save_path}")
    
    return fig, ax


def get_summary_stats_three_way(goldrim_scores: List[float],
                                citation_scores: List[float],
                                top_doc_scores: List[float],
                                stats: dict) -> dict:
    """
    Get summary statistics for three-way comparison.
    
    Args:
        goldrim_scores: List of Jaccard scores for GoldRiM pairs
        citation_scores: List of Jaccard scores for citation pairs
        top_doc_scores: List of Jaccard scores for top documents
        stats: Statistics dictionary from analyzer.get_violin_plot_data()
    
    Returns:
        Dictionary with comprehensive comparison statistics
    """
    summary = {
        'goldrim': {
            'count': len(goldrim_scores),
            'mean': np.mean(goldrim_scores) if goldrim_scores else 0.0,
            'median': np.median(goldrim_scores) if goldrim_scores else 0.0,
            'std': np.std(goldrim_scores) if goldrim_scores else 0.0,
            'min': np.min(goldrim_scores) if goldrim_scores else 0.0,
            'max': np.max(goldrim_scores) if goldrim_scores else 0.0,
        },
        'citation': {
            'count': len(citation_scores),
            'mean': np.mean(citation_scores) if citation_scores else 0.0,
            'median': np.median(citation_scores) if citation_scores else 0.0,
            'std': np.std(citation_scores) if citation_scores else 0.0,
            'min': np.min(citation_scores) if citation_scores else 0.0,
            'max': np.max(citation_scores) if citation_scores else 0.0,
        },
        'top_doc': {
            'count': len(top_doc_scores),
            'mean': np.mean(top_doc_scores) if top_doc_scores else 0.0,
            'median': np.median(top_doc_scores) if top_doc_scores else 0.0,
            'std': np.std(top_doc_scores) if top_doc_scores else 0.0,
            'min': np.min(top_doc_scores) if top_doc_scores else 0.0,
            'max': np.max(top_doc_scores) if top_doc_scores else 0.0,
        }
    }
    
    return summary


def print_summary_stats_three_way(summary: dict):
    """Print three-way comparison statistics."""
    if not summary:
        print("[ERROR] No statistics to display")
        return
    
    print(f"\n{'='*80}")
    print("THREE-WAY COMPARISON STATISTICS")
    print(f"{'='*80}\n")
    
    for group_name, group_key in [('GoldRiM Recommendations', 'goldrim'),
                                   ('Citation Pairs', 'citation'),
                                   ('Top N-gram Documents', 'top_doc')]:
        if group_key not in summary:
            continue
        
        data = summary[group_key]
        if data['count'] == 0:
            print(f"{group_name}: NO DATA")
            continue
        
        print(f"{group_name}:")
        print(f"  Count:  {data['count']}")
        print(f"  Mean:   {data['mean']:.4f}")
        print(f"  Median: {data['median']:.4f}")
        print(f"  Std:    {data['std']:.4f}")
        print(f"  Range:  [{data['min']:.4f}, {data['max']:.4f}]")
        print()
    
    print(f"{'PAIRWISE COMPARISONS':^80}")
    print("-" * 80)
    
    if summary['goldrim']['count'] > 0 and summary['citation']['count'] > 0:
        diff = summary['goldrim']['mean'] - summary['citation']['mean']
        print(f"GoldRiM vs Citation (mean diff): {diff:+.4f}")
    
    if summary['goldrim']['count'] > 0 and summary['top_doc']['count'] > 0:
        diff = summary['goldrim']['mean'] - summary['top_doc']['mean']
        print(f"GoldRiM vs Top Doc (mean diff):  {diff:+.4f}")
    
    if summary['citation']['count'] > 0 and summary['top_doc']['count'] > 0:
        diff = summary['citation']['mean'] - summary['top_doc']['mean']
        print(f"Citation vs Top Doc (mean diff): {diff:+.4f}")
    
    print(f"{'='*80}\n")

# load GoldRiM
# load zbRevQual_cit_fw_bw.pickle as citDict
# you need to convert it via rmeoving 0 at start and end to get elements from df

def clean_numeric_id(x):
    if not isinstance(x, str):
        return x
    try:
        # Handle cases like '0' or '0.0' → return '0'
        f = float(x)
        if f == 0:
            return '0'
        # Otherwise, use %g to remove trailing zeros
        return f'{f:.10g}'  # .10g avoids scientific notation for reasonable sizes
    except ValueError:
        return x  # keep non-numeric strings unchanged

# Apply to your dict
cleaned_dict = {
    clean_numeric_id(k): clean_numeric_id(v)
    for k, v in citDict.items()
}

# assuming that you have already downloaded data from "data/" folder in the repository

analyzer = NGramOverlapAnalyzer(
        df=df,
        goldRiM=goldRiM_, 
        citDict=citDict,
        n=2
    )

top10_with_scores = analyzer.get_top_k_for_all_seeds_with_scores(
    k=10,
    limit=None
)

goldrim_scores, citation_scores, top_doc_scores, stats = analyzer.get_violin_plot_data(
    top10_with_scores,
    verbose=True  # ← Shows detailed loss breakdown for each group
)


fig, ax = plot_violin_comparison_three_way(
    goldrim_scores=goldrim_scores,
    citation_scores=citation_scores,
    top_doc_scores=top_doc_scores,
    stats=stats,
    save_path='violin_comparison_three_way_1GRAM.pdf'  # ← Saves as PDF for Overleaf
)