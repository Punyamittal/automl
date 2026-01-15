"""
Problem Registry Module

ML model-based storage system to prevent training models for the same problems.
Uses embeddings to identify duplicate problem statements.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Try to import sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    logging.warning("sentence-transformers not available. Problem registry will use simple text matching.")

# Try to import FAISS for efficient vector search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("faiss-cpu not available. Problem registry will use simple cosine similarity.")

logger = logging.getLogger(__name__)


class ProblemRegistry:
    """Vector-based registry to track and prevent duplicate problem training."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.embedding_model_name = config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        self.similarity_threshold = config.get('problem_similarity_threshold', 0.90)  # Very high threshold for exact duplicates
        self.allow_duplicate_problems = config.get('allow_duplicate_problems', False)  # Allow same problem multiple times
        
        # Initialize embedding model
        self.embedding_model = None
        self.use_embeddings = False
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading embedding model for problem registry: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                self.use_embeddings = True
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}. Using simple text matching.")
                self.embedding_model = None
                self.use_embeddings = False
        else:
            logger.info("Using simple text matching (sentence-transformers not available)")
        
        # Registry storage
        self.registry_dir = Path("data/problem_registry")
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / "problem_registry.json"
        self.embeddings_file = self.registry_dir / "embeddings.npy"
        
        # Load existing registry
        self.registry = self._load_registry()
        self.embeddings = self._load_embeddings()
        
        # Initialize FAISS index if available
        self.faiss_index = None
        if FAISS_AVAILABLE and self.use_embeddings and len(self.embeddings) > 0:
            self._build_faiss_index()
    
    def _load_registry(self) -> List[Dict]:
        """Load existing problem registry from JSON file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    registry = json.load(f)
                    logger.info(f"Loaded {len(registry)} problems from problem registry")
                    return registry
            except Exception as e:
                logger.warning(f"Error loading problem registry: {e}. Starting with empty registry.")
                return []
        return []
    
    def _load_embeddings(self) -> np.ndarray:
        """Load embeddings from numpy file."""
        if self.embeddings_file.exists() and self.use_embeddings:
            try:
                embeddings = np.load(self.embeddings_file)
                logger.info(f"Loaded {len(embeddings)} problem embeddings")
                return embeddings
            except Exception as e:
                logger.warning(f"Error loading problem embeddings: {e}")
                return np.array([])
        return np.array([])
    
    def _build_faiss_index(self):
        """Build FAISS index for fast similarity search."""
        if len(self.embeddings) == 0:
            return
        
        try:
            dimension = self.embeddings.shape[1]
            # Use L2 (Euclidean) distance index
            self.faiss_index = faiss.IndexFlatL2(dimension)
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            self.faiss_index.add(self.embeddings.astype('float32'))
            logger.info(f"Built FAISS index with {self.faiss_index.ntotal} problem vectors")
        except Exception as e:
            logger.warning(f"Error building FAISS index: {e}. Using simple cosine similarity.")
            self.faiss_index = None
    
    def _save_registry(self):
        """Save registry to JSON file."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving problem registry: {e}")
    
    def _save_embeddings(self):
        """Save embeddings to numpy file."""
        if self.use_embeddings and len(self.embeddings) > 0:
            try:
                np.save(self.embeddings_file, self.embeddings)
            except Exception as e:
                logger.error(f"Error saving problem embeddings: {e}")
    
    def _create_problem_signature(self, problem: Dict) -> str:
        """Create a text signature for the problem."""
        title = problem.get('title', '').strip()
        description = problem.get('description', '').strip()
        # Combine title and description, prioritizing title
        if title and description:
            signature = f"{title} {description}"
        elif title:
            signature = title
        elif description:
            signature = description
        else:
            signature = str(problem)  # Fallback
        return signature.strip()
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        if self.use_embeddings and self.embedding_model:
            try:
                embedding = self.embedding_model.encode([text])[0]
                return embedding
            except Exception as e:
                logger.warning(f"Error generating problem embedding: {e}")
                return None
        return None
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if vec1 is None or vec2 is None:
            return 0.0
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text-based similarity (fallback when embeddings unavailable)."""
        # Simple word overlap similarity with normalization
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if len(words1) == 0 and len(words2) == 0:
            return 1.0
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
        
        # Also check for exact substring match (for very similar problems)
        if text1.lower().strip() == text2.lower().strip():
            return 1.0
        
        # Boost similarity if one is substring of another
        if text1.lower() in text2.lower() or text2.lower() in text1.lower():
            jaccard = min(1.0, jaccard + 0.2)
        
        return jaccard
    
    def check_duplicate_problem(self, problem: Dict) -> Tuple[bool, Optional[Dict]]:
        """
        Check if this problem has been seen before.
        
        Returns:
            (is_duplicate, similar_entry): 
            - is_duplicate: True if similar problem exists
            - similar_entry: The most similar registry entry if found
        """
        if self.allow_duplicate_problems:
            return False, None
        
        signature = self._create_problem_signature(problem)
        
        if len(self.registry) == 0:
            return False, None
        
        best_similarity = 0.0
        best_entry = None
        
        if self.use_embeddings:
            # Use embedding-based similarity
            query_embedding = self._get_embedding(signature)
            if query_embedding is not None:
                if self.faiss_index is not None and FAISS_AVAILABLE:
                    # Use FAISS for fast search
                    query_embedding_norm = query_embedding.astype('float32')
                    query_embedding_norm = query_embedding_norm.reshape(1, -1)
                    faiss.normalize_L2(query_embedding_norm)
                    distances, indices = self.faiss_index.search(query_embedding_norm, min(10, len(self.registry)))
                    
                    for idx, dist in zip(indices[0], distances[0]):
                        if idx < len(self.registry):
                            # Convert L2 distance to cosine similarity
                            similarity = 1.0 - (dist / 2.0)  # L2 distance on normalized vectors
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_entry = self.registry[idx]
                else:
                    # Use simple cosine similarity
                    for i, entry in enumerate(self.registry):
                        if i < len(self.embeddings):
                            stored_embedding = self.embeddings[i]
                            similarity = self._cosine_similarity(query_embedding, stored_embedding)
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_entry = entry
        else:
            # Use text-based similarity
            for entry in self.registry:
                stored_signature = entry.get('signature', '')
                similarity = self._text_similarity(signature, stored_signature)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_entry = entry
        
        logger.info(f"Best problem similarity found: {best_similarity:.4f} (threshold: {self.similarity_threshold})")
        
        if best_similarity >= self.similarity_threshold and best_entry:
            logger.warning(f"Duplicate problem detected! Similar problem already processed.")
            logger.warning(f"   Original: {best_entry.get('problem', {}).get('title', 'Unknown')}")
            logger.warning(f"   New: {problem.get('title', 'Unknown')}")
            return True, best_entry
        
        return False, None
    
    def register_problem(self, problem: Dict, task_type: str, dataset_id: Optional[str] = None) -> bool:
        """
        Register a problem in the registry.
        
        Returns:
            True if registered successfully, False otherwise
        """
        signature = self._create_problem_signature(problem)
        
        # Check if exact problem exists
        is_duplicate, similar_entry = self.check_duplicate_problem(problem)
        
        if is_duplicate:
            # Update existing entry with new dataset/task info
            if dataset_id and dataset_id not in similar_entry.get('datasets_used', []):
                similar_entry['datasets_used'].append(dataset_id)
            if task_type not in similar_entry.get('task_types', []):
                similar_entry['task_types'].append(task_type)
            similar_entry['last_seen'] = datetime.now().isoformat()
            similar_entry['seen_count'] = similar_entry.get('seen_count', 0) + 1
            logger.info(f"Updated existing problem entry. Seen count: {similar_entry['seen_count']}")
        else:
            # New entry
            entry = {
                'signature': signature,
                'problem': {
                    'title': problem.get('title', ''),
                    'description': problem.get('description', ''),
                    'source': problem.get('source', 'unknown'),
                    'url': problem.get('url', '')
                },
                'task_type': task_type,
                'datasets_used': [dataset_id] if dataset_id else [],
                'task_types': [task_type],
                'first_seen': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat(),
                'seen_count': 1
            }
            self.registry.append(entry)
            
            # Add embedding if available
            if self.use_embeddings:
                embedding = self._get_embedding(signature)
                if embedding is not None:
                    if len(self.embeddings) == 0:
                        self.embeddings = embedding.reshape(1, -1)
                    else:
                        self.embeddings = np.vstack([self.embeddings, embedding.reshape(1, -1)])
                    
                    # Rebuild FAISS index
                    if FAISS_AVAILABLE:
                        self._build_faiss_index()
            
            logger.info(f"Registered new problem in registry. Total problems: {len(self.registry)}")
        
        # Save registry and embeddings
        self._save_registry()
        self._save_embeddings()
        
        return True
    
    def get_registry_stats(self) -> Dict:
        """Get statistics about the problem registry."""
        total_problems = len(self.registry)
        total_seen = sum(entry.get('seen_count', 0) for entry in self.registry)
        unique_problems = len([e for e in self.registry if e.get('seen_count', 0) == 1])
        duplicate_problems = total_problems - unique_problems
        
        return {
            'total_problems': total_problems,
            'total_seen_count': total_seen,
            'unique_problems': unique_problems,
            'duplicate_problems': duplicate_problems,
            'using_embeddings': self.use_embeddings,
            'using_faiss': self.faiss_index is not None
        }
