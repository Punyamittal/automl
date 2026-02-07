"""
Dataset Discovery Module

Searches for relevant datasets from:
- Kaggle, HuggingFace, UCI ML Repository (primary)
- data.gov, World Bank, AWS Open Data (secondary)
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
    DatasetFilter = None
import pandas as pd

logger = logging.getLogger(__name__)

# UCI ML Repository: popular datasets (id -> (name, keywords)) - no account needed, direct HTTP
UCI_DATASETS = {
    53: ("Iris", ["iris", "flower", "classification"]),
    45: ("Heart Disease", ["heart", "disease", "medical", "health", "classification"]),
    186: ("Wine Quality", ["wine", "quality", "regression", "classification"]),
    17: ("Breast Cancer Wisconsin", ["breast", "cancer", "medical", "classification"]),
    222: ("Bank Marketing", ["bank", "marketing", "classification"]),
    2: ("Adult Census", ["adult", "census", "income", "classification"]),
    320: ("Student Performance", ["student", "performance", "education", "regression"]),
    352: ("Online Retail", ["retail", "online", "clustering", "transaction"]),
    109: ("Wine", ["wine", "classification", "chemical"]),
    19: ("Car Evaluation", ["car", "evaluation", "classification"]),
    31: ("Credit Approval", ["credit", "approval", "classification"]),
    296: ("Dermatology", ["dermatology", "skin", "classification"]),
    15: ("Breast Cancer", ["breast", "cancer", "classification"]),
    12: ("Heart Disease (Statlog)", ["heart", "disease", "classification"]),
    334: ("Dry Bean", ["bean", "agriculture", "classification"]),
    148: ("Spambase", ["spam", "email", "classification"]),
    159: ("Letter Recognition", ["letter", "recognition", "classification"]),
    146: ("Mushroom", ["mushroom", "classification"]),
    41: ("Soybean", ["soybean", "agriculture", "classification"]),
    73: ("Mushroom (Agaricus)", ["mushroom", "classification"]),
}


class DatasetDiscovery:
    """Discovers datasets from multiple sources."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.kaggle_config = config.get('kaggle', {})
        self.huggingface_config = config.get('huggingface', {})
        self.uci_config = config.get('uci', {})
        self.datagov_config = config.get('datagov', {})
        self.worldbank_config = config.get('worldbank', {})
        self.aws_config = config.get('aws_open_data', {})
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
        
        logger.info(f"Starting dataset discovery for {len(keywords)} keywords: {keywords}")
        logger.info(f"Kaggle enabled: {self.kaggle_config.get('enabled', False)}, API available: {self.kaggle_api is not None}")
        
        if self.kaggle_config.get('enabled', False) and self.kaggle_api:
            logger.info("Searching Kaggle datasets...")
            kaggle_datasets = self._search_kaggle(keywords, task_type)
            logger.info(f"Found {len(kaggle_datasets)} Kaggle datasets")
            all_datasets.extend(kaggle_datasets)
        else:
            logger.warning("Kaggle search skipped - not enabled or API not available")
        
        if self.huggingface_config.get('enabled', False):
            logger.info("Searching HuggingFace datasets...")
            hf_datasets = self._search_huggingface(keywords, task_type)
            logger.info(f"Found {len(hf_datasets)} HuggingFace datasets")
            all_datasets.extend(hf_datasets)
        
        if self.uci_config.get('enabled', False):
            logger.info("Searching UCI ML Repository datasets...")
            uci_datasets = self._search_uci(keywords, task_type)
            logger.info(f"Found {len(uci_datasets)} UCI datasets")
            all_datasets.extend(uci_datasets)
        
        if self.datagov_config.get('enabled', False):
            logger.info("Searching data.gov datasets...")
            datagov_datasets = self._search_datagov(keywords, task_type)
            logger.info(f"Found {len(datagov_datasets)} data.gov datasets")
            all_datasets.extend(datagov_datasets)
        
        if self.worldbank_config.get('enabled', False):
            logger.info("Searching World Bank Open Data...")
            wb_datasets = self._search_worldbank(keywords, task_type)
            logger.info(f"Found {len(wb_datasets)} World Bank datasets")
            all_datasets.extend(wb_datasets)
        
        if self.aws_config.get('enabled', False):
            logger.info("Searching AWS Open Data Registry...")
            aws_datasets = self._search_aws_opendata(keywords, task_type)
            logger.info(f"Found {len(aws_datasets)} AWS Open Data datasets")
            all_datasets.extend(aws_datasets)
        
        # Filter and rank datasets
        filtered_datasets = self._filter_datasets(all_datasets)
        
        logger.info(f"Discovered {len(filtered_datasets)} relevant datasets (from {len(all_datasets)} total)")
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
        """Search UCI ML Repository - keyword match on curated popular datasets. No account needed."""
        datasets = []
        max_results = self.uci_config.get('max_results', 20)
        keywords_lower = [k.lower() for k in keywords]
        
        try:
            for uci_id, (name, dataset_keywords) in UCI_DATASETS.items():
                if len(datasets) >= max_results:
                    break
                score = sum(1 for kw in keywords_lower if any(dk in kw or kw in dk for dk in dataset_keywords))
                if score > 0 or not keywords:
                    datasets.append({
                        'source': 'uci',
                        'id': str(uci_id),
                        'title': name,
                        'description': f"UCI ML Repository: {name}",
                        'size': 0,
                        'files': [],
                        'download_count': 0,
                        'usability_rating': 0.8,
                        'url': f"https://archive.ics.uci.edu/dataset/{uci_id}/{name.lower().replace(' ', '+')}",
                        'tags': dataset_keywords
                    })
        except Exception as e:
            logger.error(f"Error searching UCI: {e}")
        
        return datasets
    
    def _search_datagov(self, keywords: List[str], task_type: str) -> List[Dict]:
        """Search data.gov via CKAN API - government, economics, health, transport datasets."""
        datasets = []
        max_results = self.datagov_config.get('max_results', 10)
        
        try:
            url = "https://catalog.data.gov/api/3/action/package_search"
            q = ' '.join(keywords[:3]) if keywords else "dataset"
            resp = requests.get(url, params={'q': q, 'rows': max_results}, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            results = data.get('result', {}).get('results', [])
            for pkg in results:
                resources = pkg.get('resources', [])
                csv_resources = [r for r in resources if r.get('format', '').upper() in ('CSV', 'JSON')]
                if not csv_resources and resources:
                    csv_resources = resources[:1]
                download_url = csv_resources[0].get('url') if csv_resources else None
                datasets.append({
                    'source': 'datagov',
                    'id': pkg.get('id', '') or pkg.get('name', ''),
                    'title': pkg.get('title', pkg.get('name', 'Unknown')),
                    'description': (pkg.get('notes') or '')[:500],
                    'size': 0,
                    'files': [r.get('url') for r in csv_resources[:3] if r.get('url')],
                    'download_url': download_url,
                    'download_count': 0,
                    'usability_rating': 0.6,
                    'url': f"https://catalog.data.gov/dataset/{pkg.get('name', '')}",
                    'tags': [t.get('name', '') for t in pkg.get('tags', []) if isinstance(t, dict)]
                })
        except Exception as e:
            logger.error(f"Error searching data.gov: {e}")
        
        return datasets
    
    def _search_worldbank(self, keywords: List[str], task_type: str) -> List[Dict]:
        """Search World Bank Open Data - economy, development, population. Good for economics/finance."""
        economics_keywords = {'economy', 'gdp', 'income', 'population', 'development', 'finance', 'economic'}
        if not any(kw.lower() in economics_keywords for kw in keywords):
            return []
        datasets = []
        max_results = self.worldbank_config.get('max_results', 5)
        
        try:
            # World Bank has a public API - return placeholder entries that downloader can fetch
            datasets.append({
                'source': 'worldbank',
                'id': 'NY.GDP.MKTP.CD',
                'title': 'World Bank GDP Data',
                'description': 'GDP (current US$) from World Bank Open Data',
                'size': 0,
                'files': [],
                'download_count': 0,
                'usability_rating': 0.7,
                'url': 'https://data.worldbank.org',
                'tags': ['gdp', 'economy', 'world bank']
            })
            datasets.append({
                'source': 'worldbank',
                'id': 'SP.POP.TOTL',
                'title': 'World Bank Population Data',
                'description': 'Population total from World Bank Open Data',
                'size': 0,
                'files': [],
                'download_count': 0,
                'usability_rating': 0.7,
                'url': 'https://data.worldbank.org',
                'tags': ['population', 'demographics']
            })
        except Exception as e:
            logger.error(f"Error searching World Bank: {e}")
        
        return datasets[:max_results]
    
    def _search_aws_opendata(self, keywords: List[str], task_type: str) -> List[Dict]:
        """AWS Open Data Registry - big data, satellite, climate. Returns known public buckets."""
        datasets = []
        max_results = self.aws_config.get('max_results', 5)
        bucket_keywords = {
            'climate': ('noaa-gsd-hourly-precipitation', 'NOAA Climate Data'),
            'news': ('commoncrawl', 'Common Crawl News'),
            'genomics': ('1000genomes', '1000 Genomes'),
            'satellite': ('sentinel-2-l2a', 'Sentinel-2 Satellite'),
        }
        kw_lower = ' '.join(k for k in keywords).lower()
        for bk, (bucket, title) in bucket_keywords.items():
            if len(datasets) >= max_results:
                break
            if bk in kw_lower or not keywords:
                datasets.append({
                    'source': 'aws_opendata',
                    'id': f"s3://{bucket}",
                    'title': title,
                    'description': f"AWS Open Data bucket: {bucket}",
                    'size': 0,
                    'files': [],
                    'download_count': 0,
                    'usability_rating': 0.5,
                    'url': f"https://registry.opendata.aws/{bucket}",
                    'tags': [bk]
                })
        return datasets
    
    def _filter_datasets(self, datasets: List[Dict]) -> List[Dict]:
        """Filter datasets by quality criteria."""
        filtered = []
        
        logger.info(f"Filtering {len(datasets)} datasets...")
        
        for i, dataset in enumerate(datasets):
            # Log filtering reasons
            reasons = []
            
            # Description is optional - use title if no description
            if not dataset.get('description') and not dataset.get('title'):
                reasons.append("no description and no title")
            
            # Must have reasonable size (relaxed - size=0 is ok)
            size = dataset.get('size', 0)
            if size > 0 and (size < self.min_size or size > self.max_size):
                reasons.append(f"size {size} out of range")
            
            # Must have some metadata
            if not dataset.get('title') and not dataset.get('id'):
                reasons.append("no title or id")
            
            if reasons:
                logger.debug(f"Filtering dataset {i+1}: {', '.join(reasons)} - {dataset.get('title', dataset.get('id', 'Unknown'))}")
                continue
            
            # Add default description if missing
            if not dataset.get('description'):
                source = dataset.get('source', 'unknown')
                dataset['description'] = f"{source} dataset: {dataset.get('title', dataset.get('id', 'Unknown'))}"
            
            filtered.append(dataset)
            logger.info(f"✅ Passed filter: {dataset.get('title', dataset.get('id', 'Unknown'))}")
        
        logger.info(f"Filtering complete: {len(filtered)}/{len(datasets)} datasets passed")
        
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

