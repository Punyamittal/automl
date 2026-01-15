"""
AutoML Trainer Module

Automatically trains machine learning models using PyCaret.
"""

import pandas as pd
import numpy as np
import logging
import pickle
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from pycaret.classification import *
    from pycaret.regression import *
    from pycaret.clustering import *
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    logging.warning("PyCaret not available. Install with: pip install pycaret")

logger = logging.getLogger(__name__)


class AutoMLTrainer:
    """Trains ML models using PyCaret."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.framework = config.get('framework', 'pycaret')
        self.max_training_time = config.get('max_training_time', 3600)
        self.test_size = config.get('test_size', 0.2)
        self.random_state = config.get('random_state', 42)
        self.models_to_try = config.get('models_to_try', ['rf', 'gbc', 'lr', 'knn'])
        self.min_model_score = config.get('min_model_score', 0.6)
        
        self.models_dir = Path("outputs/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        if not PYCARET_AVAILABLE:
            raise ImportError("PyCaret is required but not installed")
    
    def train(self, dataset_path: str, task_type: str, target_column: Optional[str] = None) -> Dict:
        """Train models for the given dataset and task type."""
        logger.info(f"Loading dataset from {dataset_path}")
        
        # Load dataset
        try:
            df = self._load_dataset(dataset_path)
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return {
                'success': False,
                'error': f"Failed to load dataset: {e}"
            }
        
        if df.empty:
            return {
                'success': False,
                'error': "Dataset is empty"
            }
        
        # Detect target column if not provided
        if target_column is None:
            target_column = self._detect_target_column(df, task_type)
        
        if target_column is None:
            return {
                'success': False,
                'error': "Could not detect target column"
            }
        
        logger.info(f"Training {task_type} model with target: {target_column}")
        
        # Train based on task type
        if task_type == 'classification':
            result = self._train_classification(df, target_column)
        elif task_type == 'regression':
            result = self._train_regression(df, target_column)
        elif task_type == 'clustering':
            result = self._train_clustering(df)
        else:
            return {
                'success': False,
                'error': f"Unsupported task type: {task_type}"
            }
        
        return result
    
    def _load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load dataset from file."""
        path = Path(dataset_path)
        
        if path.suffix == '.csv':
            df = pd.read_csv(path)
        elif path.suffix == '.json':
            df = pd.read_json(path)
        elif path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(path)
        else:
            # Try CSV as default
            df = pd.read_csv(path)
        
        return df
    
    def _detect_target_column(self, df: pd.DataFrame, task_type: str) -> Optional[str]:
        """Detect target column from dataset."""
        # Common target column names
        common_names = ['target', 'label', 'y', 'class', 'outcome', 'result']
        
        # Check for exact matches
        for col in df.columns:
            if col.lower() in common_names:
                return col
        
        # For classification, look for categorical column with few unique values
        if task_type == 'classification':
            for col in df.columns:
                if df[col].dtype == 'object' or df[col].nunique() < 20:
                    if df[col].nunique() > 1:
                        return col
        
        # For regression, look for numeric column
        if task_type == 'regression':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Return the last numeric column (often the target)
                return numeric_cols[-1]
        
        # Default: return last column
        return df.columns[-1]
    
    def _train_classification(self, df: pd.DataFrame, target: str) -> Dict:
        """Train classification models."""
        try:
            # Setup PyCaret
            setup(
                data=df,
                target=target,
                test_data=None,
                train_size=1 - self.test_size,
                session_id=self.random_state,
                silent=True,
                verbose=False
            )
            
            # Compare models
            logger.info("Comparing classification models...")
            compare_models(
                include=self.models_to_try,
                sort='Accuracy',
                n_select=1,
                verbose=False
            )
            
            # Get best model
            best_model = compare_models(
                include=self.models_to_try,
                sort='Accuracy',
                n_select=1,
                verbose=False
            )
            
            # Evaluate model
            evaluate_model(best_model, verbose=False)
            
            # Finalize model
            final_model = finalize_model(best_model)
            
            # Get metrics
            metrics = pull()
            best_metric = metrics.loc[metrics.index[0]]
            
            # Check if model meets minimum score
            accuracy = best_metric.get('Accuracy', 0)
            if accuracy < self.min_model_score:
                logger.warning(f"Model accuracy {accuracy:.3f} below threshold {self.min_model_score}")
            
            # Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = self.models_dir / f"classification_model_{timestamp}.pkl"
            save_model(final_model, str(model_path))
            
            # Save metrics
            metrics_path = self.models_dir / f"metrics_{timestamp}.json"
            metrics_dict = {
                'task_type': 'classification',
                'target': target,
                'accuracy': float(accuracy),
                'metrics': best_metric.to_dict(),
                'model_path': str(model_path),
                'training_time': None,  # PyCaret doesn't provide this easily
                'timestamp': timestamp
            }
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics_dict, f, indent=2)
            
            logger.info(f"Classification model trained. Accuracy: {accuracy:.3f}")
            
            return {
                'success': True,
                'task_type': 'classification',
                'model_path': str(model_path),
                'metrics_path': str(metrics_path),
                'metrics': metrics_dict,
                'model': final_model
            }
            
        except Exception as e:
            logger.error(f"Error training classification model: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _train_regression(self, df: pd.DataFrame, target: str) -> Dict:
        """Train regression models."""
        try:
            # Setup PyCaret
            setup(
                data=df,
                target=target,
                test_data=None,
                train_size=1 - self.test_size,
                session_id=self.random_state,
                silent=True,
                verbose=False
            )
            
            # Compare models
            logger.info("Comparing regression models...")
            best_model = compare_models(
                include=self.models_to_try,
                sort='R2',
                n_select=1,
                verbose=False
            )
            
            # Evaluate model
            evaluate_model(best_model, verbose=False)
            
            # Finalize model
            final_model = finalize_model(best_model)
            
            # Get metrics
            metrics = pull()
            best_metric = metrics.loc[metrics.index[0]]
            
            # Check if model meets minimum score
            r2 = best_metric.get('R2', 0)
            if r2 < self.min_model_score:
                logger.warning(f"Model R2 {r2:.3f} below threshold {self.min_model_score}")
            
            # Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = self.models_dir / f"regression_model_{timestamp}.pkl"
            save_model(final_model, str(model_path))
            
            # Save metrics
            metrics_path = self.models_dir / f"metrics_{timestamp}.json"
            metrics_dict = {
                'task_type': 'regression',
                'target': target,
                'r2': float(r2),
                'metrics': best_metric.to_dict(),
                'model_path': str(model_path),
                'timestamp': timestamp
            }
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics_dict, f, indent=2)
            
            logger.info(f"Regression model trained. R2: {r2:.3f}")
            
            return {
                'success': True,
                'task_type': 'regression',
                'model_path': str(model_path),
                'metrics_path': str(metrics_path),
                'metrics': metrics_dict,
                'model': final_model
            }
            
        except Exception as e:
            logger.error(f"Error training regression model: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _train_clustering(self, df: pd.DataFrame) -> Dict:
        """Train clustering models."""
        try:
            # Setup PyCaret
            setup(
                data=df,
                session_id=self.random_state,
                silent=True,
                verbose=False
            )
            
            # Create clustering model
            logger.info("Creating clustering model...")
            model = create_model('kmeans', verbose=False)
            
            # Assign clusters
            assigned_df = assign_model(model)
            
            # Evaluate (silhouette score)
            evaluate_model(model, verbose=False)
            
            # Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = self.models_dir / f"clustering_model_{timestamp}.pkl"
            save_model(model, str(model_path))
            
            logger.info("Clustering model trained")
            
            return {
                'success': True,
                'task_type': 'clustering',
                'model_path': str(model_path),
                'metrics': {
                    'task_type': 'clustering',
                    'n_clusters': len(assigned_df['Cluster'].unique()),
                    'timestamp': timestamp
                },
                'model': model
            }
            
        except Exception as e:
            logger.error(f"Error training clustering model: {e}")
            return {
                'success': False,
                'error': str(e)
            }

