"""
Kaggle Problem Miner Module

Mines ML problems from Kaggle competitions and datasets.
"""

import logging
import time
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    logger.warning("Kaggle API not available. Install with: pip install kaggle")


class KaggleProblemMiner:
    """Mines ML problems from Kaggle competitions and datasets."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_competitions = config.get('max_competitions', 10)
        self.max_datasets = config.get('max_datasets', 20)
        self.problems_dir = Path("data/problems")
        self.problems_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Kaggle API
        self.kaggle_api = None
        if KAGGLE_AVAILABLE:
            try:
                self.kaggle_api = KaggleApi()
                kaggle_dir = Path.home() / '.kaggle'
                if (kaggle_dir / 'kaggle.json').exists():
                    self.kaggle_api.authenticate()
                    logger.info("Kaggle API authenticated successfully")
                else:
                    logger.warning("Kaggle credentials not found. Set up ~/.kaggle/kaggle.json")
            except Exception as e:
                logger.warning(f"Could not initialize Kaggle API: {e}")
    
    def mine_problems(self) -> List[Dict]:
        """Mine problems from Kaggle competitions and datasets."""
        all_problems = []
        
        if not self.kaggle_api:
            logger.warning("Kaggle API not available. Skipping Kaggle problem mining.")
            return all_problems
        
        # Mine from competitions
        logger.info("Mining problems from Kaggle competitions...")
        competition_problems = self._mine_competitions()
        all_problems.extend(competition_problems)
        
        # Mine from datasets
        logger.info("Mining problems from Kaggle datasets...")
        dataset_problems = self._mine_datasets()
        all_problems.extend(dataset_problems)
        
        logger.info(f"Mined {len(all_problems)} problems from Kaggle")
        return all_problems
    
    def _mine_competitions(self) -> List[Dict]:
        """Mine problems from Kaggle competitions."""
        problems = []
        
        if not self.kaggle_api:
            return problems
        
        try:
            # Try to get competitions - competitions API may require special permissions
            # Skip competitions if API call fails and focus on datasets instead
            competitions = []
            
            # Method 1: Try competitions_list with no parameters
            try:
                competitions = list(self.kaggle_api.competitions_list())[:self.max_competitions]
            except Exception as e1:
                logger.debug(f"competitions_list() failed (may require special permissions): {e1}")
                # Competitions API often requires special authentication
                # Skip competitions and focus on datasets which are more accessible
                logger.info("Skipping competitions (API access restricted). Focusing on datasets.")
                return problems  # Return empty - datasets will be mined instead
            
            for comp in competitions:
                try:
                    # Get competition reference
                    comp_ref = getattr(comp, 'ref', None) or getattr(comp, 'slug', None)
                    if not comp_ref:
                        continue
                    
                    # Extract competition description
                    description = getattr(comp, 'description', '') or getattr(comp, 'subtitle', '') or f"Kaggle competition: {comp_ref}"
                    title = getattr(comp, 'title', comp_ref)
                    
                    # Check if it's an ML problem (has target, data, evaluation)
                    if self._is_ml_competition(comp, description):
                        problem = {
                            'id': comp_ref,
                            'source': 'kaggle',
                            'source_type': 'competition',
                            'title': title,
                            'description': description,
                            'full_text': f"{title}\n\n{description}",
                            'url': f"https://www.kaggle.com/competitions/{comp_ref}",
                            'mined_at': datetime.now().isoformat(),
                            'competition_ref': comp_ref,
                            'deadline': getattr(comp, 'deadline', None),
                            'category': getattr(comp, 'category', None),
                            'reward': getattr(comp, 'reward', None)
                        }
                        problems.append(problem)
                    
                    # Rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.debug(f"Error processing competition: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Error mining Kaggle competitions: {e}. Continuing with datasets only.")
        
        return problems
    
    def _mine_datasets(self) -> List[Dict]:
        """Mine problems from Kaggle datasets with ML potential."""
        problems = []
        
        if not self.kaggle_api:
            return problems
        
        try:
            # Search for popular datasets with ML keywords
            ml_keywords = ['prediction', 'classification', 'regression', 'forecast', 'detect', 'recommend', 'machine learning']
            
            datasets_found = 0
            for keyword in ml_keywords[:5]:  # Use top 5 keywords
                if datasets_found >= self.max_datasets:
                    break
                    
                try:
                    datasets = self.kaggle_api.dataset_list(
                        search=keyword,
                        max_size=1000000,
                        min_size=100,
                        sort_by='hottest'
                    )
                    
                    for dataset in datasets:
                        if datasets_found >= self.max_datasets:
                            break
                            
                        try:
                            ref_parts = dataset.ref.split('/')
                            if len(ref_parts) != 2:
                                continue
                            
                            owner_slug, dataset_slug = ref_parts
                            
                            # Get dataset metadata
                            description = ''
                            try:
                                metadata = self.kaggle_api.metadata_get(owner_slug, dataset_slug)
                                if isinstance(metadata, dict):
                                    description = metadata.get('description', '') or metadata.get('subtitle', '')
                            except:
                                description = getattr(dataset, 'description', '') or getattr(dataset, 'subtitle', '')
                            
                            # Check if dataset describes an ML problem
                            if self._is_ml_dataset(dataset, description):
                                title = getattr(dataset, 'title', dataset.ref)
                                problem = {
                                    'id': dataset.ref,
                                    'source': 'kaggle',
                                    'source_type': 'dataset',
                                    'title': title,
                                    'description': description or f"Kaggle dataset: {dataset.ref}",
                                    'full_text': f"{title}\n\n{description or 'ML dataset from Kaggle'}",
                                    'url': f"https://www.kaggle.com/datasets/{dataset.ref}",
                                    'mined_at': datetime.now().isoformat(),
                                    'dataset_ref': dataset.ref,
                                    'usability': getattr(dataset, 'usabilityRating', 0),
                                    'download_count': getattr(dataset, 'downloadCount', 0)
                                }
                                problems.append(problem)
                                datasets_found += 1
                            
                            # Rate limiting
                            time.sleep(0.3)
                            
                        except Exception as e:
                            logger.debug(f"Error processing dataset {dataset.ref}: {e}")
                            continue
                    
                    # Delay between keyword searches
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Error searching Kaggle datasets with keyword '{keyword}': {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Error mining Kaggle datasets: {e}")
        
        return problems
    
    def _is_ml_competition(self, comp, description: str) -> bool:
        """Check if competition is a valid ML problem."""
        # Check for ML indicators in description
        ml_indicators = ['predict', 'classify', 'detect', 'forecast', 'recommend', 'regression', 'classification']
        description_lower = description.lower()
        
        return any(indicator in description_lower for indicator in ml_indicators)
    
    def _is_ml_dataset(self, dataset, description: str) -> bool:
        """Check if dataset describes a valid ML problem."""
        # Check for ML indicators
        ml_indicators = ['predict', 'classify', 'detect', 'forecast', 'recommend', 'target', 'label', 'feature', 'machine learning', 'ml', 'model']
        description_lower = description.lower()
        title = getattr(dataset, 'title', '').lower()
        combined = f"{title} {description_lower}"
        
        # Must have reasonable usability and downloads
        usability = getattr(dataset, 'usabilityRating', 0)
        downloads = getattr(dataset, 'downloadCount', 0)
        
        has_ml_indicators = any(indicator in combined for indicator in ml_indicators)
        # Lower threshold for popularity to get more results
        is_popular = usability >= 5.0 or downloads >= 50
        
        return has_ml_indicators and is_popular
