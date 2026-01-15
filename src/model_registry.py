"""
Model Registry Module

Vector-based storage system to prevent training duplicate ML models.
Uses embeddings to find similar problems/datasets and track training counts.
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
    logging.warning("sentence-transformers not available. Model registry will use simple text matching.")

# Try to import FAISS for efficient vector search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("faiss-cpu not available. Model registry will use simple cosine similarity.")

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Vector-based registry to track and prevent duplicate model training."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.embedding_model_name = config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        self.similarity_threshold = config.get('model_similarity_threshold', 0.85)  # High threshold for duplicates
        self.max_duplicate_trains = config.get('max_duplicate_trains', 2)  # Allow max 2 similar models
        
        # Initialize embedding model
        self.embedding_model = None
        self.use_embeddings = False
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading embedding model for registry: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                self.use_embeddings = True
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}. Using simple text matching.")
                self.embedding_model = None
                self.use_embeddings = False
        else:
            logger.info("Using simple text matching (sentence-transformers not available)")
        
        # Registry storage
        self.registry_dir = Path("data/model_registry")
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / "model_registry.json"
        self.embeddings_file = self.registry_dir / "embeddings.npy"
        
        # Load existing registry
        self.registry = self._load_registry()
        self.embeddings = self._load_embeddings()
        
        # Initialize FAISS index if available
        self.faiss_index = None
        if FAISS_AVAILABLE and self.use_embeddings and len(self.embeddings) > 0:
            self._build_faiss_index()
    
    def _load_registry(self) -> List[Dict]:
        """Load existing model registry from JSON file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    registry = json.load(f)
                    logger.info(f"Loaded {len(registry)} entries from model registry")
                    return registry
            except Exception as e:
                logger.warning(f"Error loading registry: {e}. Starting with empty registry.")
                return []
        return []
    
    def _load_embeddings(self) -> np.ndarray:
        """Load embeddings from numpy file."""
        if self.embeddings_file.exists() and self.use_embeddings:
            try:
                embeddings = np.load(self.embeddings_file)
                logger.info(f"Loaded {len(embeddings)} embeddings")
                return embeddings
            except Exception as e:
                logger.warning(f"Error loading embeddings: {e}")
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
            logger.info(f"Built FAISS index with {self.faiss_index.ntotal} vectors")
        except Exception as e:
            logger.warning(f"Error building FAISS index: {e}. Using simple cosine similarity.")
            self.faiss_index = None
    
    def _save_registry(self):
        """Save registry to JSON file."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    def _save_embeddings(self):
        """Save embeddings to numpy file."""
        if self.use_embeddings and len(self.embeddings) > 0:
            try:
                np.save(self.embeddings_file, self.embeddings)
            except Exception as e:
                logger.error(f"Error saving embeddings: {e}")
    
    def _create_model_signature(self, problem: Dict, dataset: Dict, task_type: str) -> str:
        """Create a text signature for the model (problem + dataset + task)."""
        problem_text = problem.get('title', '') + ' ' + problem.get('description', '')
        dataset_text = dataset.get('id', '') + ' ' + dataset.get('title', '') + ' ' + dataset.get('description', '')
        signature = f"{problem_text} | {dataset_text} | {task_type}"
        return signature.strip()
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        if self.use_embeddings and self.embedding_model:
            try:
                embedding = self.embedding_model.encode([text])[0]
                return embedding
            except Exception as e:
                logger.warning(f"Error generating embedding: {e}")
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
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if len(words1) == 0 and len(words2) == 0:
            return 1.0
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if len(union) > 0 else 0.0
    
    def check_duplicate(self, problem: Dict, dataset: Dict, task_type: str) -> Tuple[bool, Optional[Dict]]:
        """
        Check if a similar model has been trained before.
        
        Returns:
            (is_duplicate, similar_entry): 
            - is_duplicate: True if similar model exists and has been trained >= max_duplicate_trains times
            - similar_entry: The most similar registry entry if found
        """
        signature = self._create_model_signature(problem, dataset, task_type)
        
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
        
        logger.info(f"Best similarity found: {best_similarity:.4f} (threshold: {self.similarity_threshold})")
        
        if best_similarity >= self.similarity_threshold and best_entry:
            train_count = best_entry.get('train_count', 0)
            logger.info(f"Similar model found with {train_count} previous trainings")
            
            if train_count >= self.max_duplicate_trains:
                logger.warning(f"Duplicate model detected! Similar model already trained {train_count} times (max: {self.max_duplicate_trains})")
                return True, best_entry
            else:
                logger.info(f"Similar model found but only trained {train_count} times. Allowing training.")
                return False, best_entry
        
        return False, None
    
    def register_model(self, problem: Dict, dataset: Dict, task_type: str, 
                      model_path: str, metrics: Dict) -> bool:
        """
        Register a newly trained model in the registry.
        
        Returns:
            True if registered successfully, False otherwise
        """
        signature = self._create_model_signature(problem, dataset, task_type)
        
        # Check if similar entry exists
        is_duplicate, similar_entry = self.check_duplicate(problem, dataset, task_type)
        
        if is_duplicate:
            # This shouldn't happen if check_duplicate is called before training
            logger.warning("Attempting to register duplicate model that should have been blocked!")
            return False
        
        if similar_entry and similar_entry.get('signature') == signature:
            # Exact match - increment count
            similar_entry['train_count'] = similar_entry.get('train_count', 0) + 1
            similar_entry['last_trained'] = datetime.now().isoformat()
            similar_entry['model_paths'].append(model_path)
            similar_entry['metrics'].append(metrics)
            logger.info(f"Updated existing registry entry. Train count: {similar_entry['train_count']}")
        else:
            # New entry
            entry = {
                'signature': signature,
                'problem': {
                    'title': problem.get('title', ''),
                    'description': problem.get('description', '')
                },
                'dataset': {
                    'id': dataset.get('id', ''),
                    'title': dataset.get('title', ''),
                    'description': dataset.get('description', '')
                },
                'task_type': task_type,
                'train_count': 1,
                'first_trained': datetime.now().isoformat(),
                'last_trained': datetime.now().isoformat(),
                'model_paths': [model_path],
                'metrics': [metrics]
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
            
            logger.info(f"Registered new model in registry. Total entries: {len(self.registry)}")
        
        # Save registry and embeddings
        self._save_registry()
        self._save_embeddings()
        
        return True
    
    def get_registry_stats(self) -> Dict:
        """Get statistics about the model registry."""
        total_models = len(self.registry)
        total_trains = sum(entry.get('train_count', 0) for entry in self.registry)
        duplicates = sum(1 for entry in self.registry if entry.get('train_count', 0) >= self.max_duplicate_trains)
        
        return {
            'total_entries': total_models,
            'total_trainings': total_trains,
            'duplicate_models': duplicates,
            'using_embeddings': self.use_embeddings,
            'using_faiss': self.faiss_index is not None
        }
