"""
Dataset-Problem Matcher Module

Matches problems with relevant datasets using semantic similarity.
"""

import json
import numpy as np
import logging
# Note: faiss is available but we use numpy for similarity (simpler, no extra dependency needed)
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Try to import sentence-transformers, fallback to simple matching if unavailable
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, Exception) as e:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    logging.warning(f"sentence-transformers not available ({e}). Using simple text matching instead.")

logger = logging.getLogger(__name__)


class DatasetMatcher:
    """Matches problems with datasets using embeddings."""
    
    def __init__(self, config: Dict):
        self.config = config
        # Lower threshold to 0.4 to allow more matches (was 0.6)
        self.similarity_threshold = config.get('similarity_threshold', 0.4)
        self.embedding_model_name = config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        self.top_k = config.get('top_k_matches', 3)
        
        # Load embedding model if available, otherwise use simple matching
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading embedding model: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                self.use_embeddings = True
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}. Using simple text matching.")
                self.embedding_model = None
                self.use_embeddings = False
        else:
            logger.info("Using simple text matching (sentence-transformers not available)")
            self.embedding_model = None
            self.use_embeddings = False
        
        self.embeddings_dir = Path("data/embeddings")
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    def match(self, problem: Dict, datasets: List[Dict]) -> List[Tuple[Dict, float]]:
        """Match problem with datasets and return top matches."""
        if not datasets:
            return []
        
        if self.use_embeddings and self.embedding_model:
            # Use embedding-based matching
            return self._match_with_embeddings(problem, datasets)
        else:
            # Use simple text-based matching
            return self._match_with_text(problem, datasets)
    
    def _match_with_embeddings(self, problem: Dict, datasets: List[Dict]) -> List[Tuple[Dict, float]]:
        """Match using sentence transformers embeddings."""
        # Generate embedding for problem
        problem_text = self._extract_problem_text(problem)
        problem_embedding = self.embedding_model.encode([problem_text])[0]
        
        # Generate embeddings for datasets
        dataset_texts = [self._extract_dataset_text(ds) for ds in datasets]
        dataset_embeddings = self.embedding_model.encode(dataset_texts)
        
        # Compute similarities
        similarities = []
        for dataset, embedding in zip(datasets, dataset_embeddings):
            similarity = float(np.dot(problem_embedding, embedding) / 
                             (np.linalg.norm(problem_embedding) * np.linalg.norm(embedding)))
            similarities.append((dataset, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by threshold and return top k
        matches = [
            (ds, sim) for ds, sim in similarities 
            if sim >= self.similarity_threshold
        ][:self.top_k]
        
        # Fallback: if no matches above threshold, return the best match anyway (with warning)
        if not matches and similarities:
            best_match = similarities[0]
            logger.warning(f"No matches above threshold {self.similarity_threshold}. Using best match with similarity {best_match[1]:.3f}")
            matches = [best_match]
        
        logger.info(f"Found {len(matches)} matching datasets using embeddings (threshold: {self.similarity_threshold})")
        return matches
    
    def _match_with_text(self, problem: Dict, datasets: List[Dict]) -> List[Tuple[Dict, float]]:
        """Match using simple text similarity (fallback when embeddings unavailable)."""
        problem_text = self._extract_problem_text(problem).lower()
        problem_words = set(problem_text.split())
        
        similarities = []
        for dataset in datasets:
            dataset_text = self._extract_dataset_text(dataset).lower()
            dataset_words = set(dataset_text.split())
            
            # Simple Jaccard similarity (word overlap)
            if len(problem_words) == 0 or len(dataset_words) == 0:
                similarity = 0.0
            else:
                intersection = len(problem_words & dataset_words)
                union = len(problem_words | dataset_words)
                similarity = intersection / union if union > 0 else 0.0
            
            # Boost similarity if key terms match
            key_terms = ['data', 'dataset', 'machine learning', 'classification', 'regression', 'model']
            for term in key_terms:
                if term in problem_text and term in dataset_text:
                    similarity += 0.1
            
            similarity = min(1.0, similarity)  # Cap at 1.0
            similarities.append((dataset, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by threshold and return top k
        matches = [
            (ds, sim) for ds, sim in similarities 
            if sim >= self.similarity_threshold
        ][:self.top_k]
        
        # Fallback: if no matches above threshold, return the best match anyway (with warning)
        if not matches and similarities:
            best_match = similarities[0]
            logger.warning(f"No matches above threshold {self.similarity_threshold}. Using best match with similarity {best_match[1]:.3f}")
            matches = [best_match]
        
        logger.info(f"Found {len(matches)} matching datasets using text matching (threshold: {self.similarity_threshold})")
        return matches
    
    def _extract_problem_text(self, problem: Dict) -> str:
        """Extract text representation of problem for embedding."""
        parts = []
        
        if problem.get('title'):
            title = problem['title']
            parts.append(title)
            # For educational/explanation problems, add keywords that might match datasets
            if any(word in title.lower() for word in ['explain', 'explanation', 'tutorial', 'how', 'what']):
                # Extract the ML topic (e.g., "Naive Bayes" from "explanation of Naive Bayes")
                ml_info = problem.get('ml_classification', {})
                task_type = ml_info.get('task_type', 'classification')
                parts.append(f"{task_type} dataset")
                parts.append("machine learning data")
        
        if problem.get('description'):
            parts.append(problem['description'][:500])  # Limit length
        
        # Add ML classification info if available
        ml_info = problem.get('ml_classification', {})
        if ml_info.get('task_type'):
            parts.append(f"Task type: {ml_info['task_type']}")
        if ml_info.get('key_features'):
            parts.append(f"Features: {', '.join(ml_info['key_features'][:5])}")
        
        return ' '.join(parts)
    
    def _extract_dataset_text(self, dataset: Dict) -> str:
        """Extract text representation of dataset for embedding."""
        parts = []
        
        if dataset.get('title'):
            parts.append(dataset['title'])
        
        if dataset.get('description'):
            parts.append(dataset['description'][:500])  # Limit length
        
        if dataset.get('tags'):
            parts.append(' '.join(dataset['tags'][:5]))
        
        return ' '.join(parts)
    
    def select_best_match(self, matches: List[Tuple[Dict, float]]) -> Optional[Dict]:
        """Select the best matching dataset."""
        if not matches:
            return None
        
        # Return the highest similarity match
        best_match, best_similarity = matches[0]
        
        logger.info(f"Selected best match: {best_match.get('id')} (similarity: {best_similarity:.3f})")
        return best_match

