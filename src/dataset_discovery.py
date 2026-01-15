"""
Dataset Discovery Module

Searches for relevant datasets from Kaggle, HuggingFace, and UCI ML Repository.
"""

import os
import json
import requests
import logging
from typing import List, Dict, Optional
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from huggingface_hub import list_datasets
try:
    from huggingface_hub import DatasetFilter
except ImportError:
    # DatasetFilter may not be available in all versions
    DatasetFilter = None
import pandas as pd

logger = logging.getLogger(__name__)


class DatasetDiscovery:
    """Discovers datasets from multiple sources."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.kaggle_config = config.get('kaggle', {})
        self.huggingface_config = config.get('huggingface', {})
        self.uci_config = config.get('uci', {})
        self.min_size = config.get('min_dataset_size', 100)
        self.max_size = config.get('max_dataset_size', 1000000)
        
        self.datasets_dir = Path("data/datasets")
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Kaggle API if enabled
        self.kaggle_api = None
        if self.kaggle_config.get('enabled', False):
            try:
                self.kaggle_api = KaggleApi()
                # Check for credentials
                kaggle_dir = Path.home() / '.kaggle'
                if (kaggle_dir / 'kaggle.json').exists():
                    self.kaggle_api.authenticate()
                else:
                    logger.warning("Kaggle credentials not found. Set up ~/.kaggle/kaggle.json")
            except Exception as e:
                logger.warning(f"Could not initialize Kaggle API: {e}")
    
    def discover_datasets(self, problem: Dict, task_type: str, keywords: List[str]) -> List[Dict]:
        """Discover datasets relevant to the problem."""
        all_datasets = []
        
        if self.kaggle_config.get('enabled', False) and self.kaggle_api:
            logger.info("Searching Kaggle datasets...")
            kaggle_datasets = self._search_kaggle(keywords, task_type)
            all_datasets.extend(kaggle_datasets)
        
        if self.huggingface_config.get('enabled', False):
            logger.info("Searching HuggingFace datasets...")
            hf_datasets = self._search_huggingface(keywords, task_type)
            all_datasets.extend(hf_datasets)
        
        if self.uci_config.get('enabled', False):
            logger.info("Searching UCI datasets...")
            uci_datasets = self._search_uci(keywords, task_type)
            all_datasets.extend(uci_datasets)
        
        # Filter and rank datasets
        filtered_datasets = self._filter_datasets(all_datasets)
        
        logger.info(f"Discovered {len(filtered_datasets)} relevant datasets")
        return filtered_datasets
    
    def _search_kaggle(self, keywords: List[str], task_type: str) -> List[Dict]:
        """Search Kaggle datasets."""
        datasets = []
        max_results = self.kaggle_config.get('max_results', 20)
        
        if not self.kaggle_api:
            return datasets
        
        try:
            # Search with keywords
            search_query = ' '.join(keywords[:3])  # Use top 3 keywords
            search_results = self.kaggle_api.dataset_list(
                search=search_query,
                max_size=self.max_size,
                min_size=self.min_size
            )
            
            for result in search_results[:max_results]:
                try:
                    # Parse owner_slug and dataset_slug from result.ref (format: "owner/dataset")
                    ref_parts = result.ref.split('/')
                    if len(ref_parts) != 2:
                        logger.warning(f"Invalid Kaggle dataset ref format: {result.ref}")
                        continue
                    
                    owner_slug, dataset_slug = ref_parts
                    
                    # Try to get detailed metadata, but use search result as fallback
                    dataset_info = {}
                    files = []
                    
                    # Try metadata_get if available
                    if hasattr(self.kaggle_api, 'metadata_get'):
                        try:
                            dataset_info = self.kaggle_api.metadata_get(owner_slug, dataset_slug)
                            if not isinstance(dataset_info, dict):
                                dataset_info = {}
                        except Exception as e:
                            logger.debug(f"metadata_get failed for {result.ref}: {e}")
                            dataset_info = {}
                    
                    # Try to get file list
                    if hasattr(self.kaggle_api, 'datasets_list_files'):
                        try:
                            files_list = self.kaggle_api.datasets_list_files(owner_slug, dataset_slug)
                            files = [f.name for f in files_list] if files_list else []
                        except Exception as e:
                            logger.debug(f"datasets_list_files failed for {result.ref}: {e}")
                            files = []
                    
                    # Extract info from search result object (has attributes like ref, title, size, etc.)
                    result_title = getattr(result, 'title', None) or getattr(result, 'ref', result.ref)
                    result_size = getattr(result, 'size', 0)
                    result_usability = getattr(result, 'usabilityRating', 0)
                    result_downloads = getattr(result, 'downloadCount', 0)
                    
                    # Build dataset dict with metadata if available, otherwise use search result
                    dataset = {
                        'source': 'kaggle',
                        'id': result.ref,
                        'title': dataset_info.get('title', result_title) if dataset_info else result_title,
                        'description': dataset_info.get('description', '') if dataset_info else '',
                        'size': dataset_info.get('totalBytes', result_size) if dataset_info else result_size,
                        'files': files,
                        'download_count': dataset_info.get('downloadCount', result_downloads) if dataset_info else result_downloads,
                        'usability_rating': dataset_info.get('usabilityRating', result_usability) if dataset_info else result_usability,
                        'url': f"https://www.kaggle.com/datasets/{result.ref}",
                        'tags': dataset_info.get('tags', []) if dataset_info else []
                    }
                    
                    datasets.append(dataset)
                    
                except Exception as e:
                    logger.warning(f"Error fetching Kaggle dataset {result.ref}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error searching Kaggle: {e}")
        
        return datasets
    
    def _search_huggingface(self, keywords: List[str], task_type: str) -> List[Dict]:
        """Search HuggingFace datasets."""
        datasets = []
        max_results = self.huggingface_config.get('max_results', 20)
        
        try:
            # Map task types to HF task categories
            task_mapping = {
                'classification': 'text-classification',
                'regression': 'tabular-regression',
                'clustering': 'unsupervised'
            }
            
            hf_task = task_mapping.get(task_type, None)
            
            # Search with keywords
            search_query = ' '.join(keywords[:2])
            
            # Use DatasetFilter if available, otherwise just search
            if DatasetFilter and hf_task:
                results = list_datasets(
                    search=search_query,
                    filter=DatasetFilter(task_categories=hf_task)
                )
            else:
                # Fallback: search without filter
                results = list_datasets(search=search_query)
            
            for result in list(results)[:max_results]:
                try:
                    dataset_info = result
                    
                    dataset = {
                        'source': 'huggingface',
                        'id': dataset_info.id,
                        'title': dataset_info.id.split('/')[-1],
                        'description': getattr(dataset_info, 'description', ''),
                        'size': 0,  # HF doesn't provide size easily
                        'files': [],
                        'download_count': getattr(dataset_info, 'downloads', 0),
                        'usability_rating': 0,
                        'url': f"https://huggingface.co/datasets/{dataset_info.id}",
                        'tags': getattr(dataset_info, 'tags', [])
                    }
                    
                    datasets.append(dataset)
                    
                except Exception as e:
                    logger.warning(f"Error processing HF dataset {result.id}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error searching HuggingFace: {e}")
        
        return datasets
    
    def _search_uci(self, keywords: List[str], task_type: str) -> List[Dict]:
        """Search UCI ML Repository datasets."""
        datasets = []
        max_results = self.uci_config.get('max_results', 20)
        
        try:
            # UCI doesn't have a great API, so we'll use a simple search
            # This is a simplified approach - in production, you'd scrape the UCI website
            uci_url = "https://archive.ics.uci.edu/ml/datasets.php"
            
            # For now, return empty list as UCI scraping requires more complex parsing
            # In a full implementation, you would:
            # 1. Scrape the UCI dataset listing page
            # 2. Search for keywords in dataset names/descriptions
            # 3. Extract metadata
            
            logger.info("UCI dataset search not fully implemented (requires web scraping)")
            
        except Exception as e:
            logger.error(f"Error searching UCI: {e}")
        
        return datasets
    
    def _filter_datasets(self, datasets: List[Dict]) -> List[Dict]:
        """Filter datasets by quality criteria."""
        filtered = []
        
        for dataset in datasets:
            # Must have description
            if not dataset.get('description'):
                continue
            
            # Must have reasonable size
            size = dataset.get('size', 0)
            if size > 0 and (size < self.min_size or size > self.max_size):
                continue
            
            # Must have some metadata
            if not dataset.get('title') and not dataset.get('id'):
                continue
            
            filtered.append(dataset)
        
        # Sort by usability/downloads
        filtered.sort(
            key=lambda x: (
                x.get('usability_rating', 0) * 0.7 +
                min(x.get('download_count', 0) / 1000, 1.0) * 0.3
            ),
            reverse=True
        )
        
        return filtered
    
    def save_datasets(self, datasets: List[Dict], filename: Optional[str] = None):
        """Save discovered datasets to JSON."""
        if filename is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"datasets_{timestamp}.json"
        
        filepath = self.datasets_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(datasets, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(datasets)} datasets to {filepath}")
        return filepath

