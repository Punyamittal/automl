"""
AutoML Trainer Module (scikit-learn version)

Automatically trains machine learning models using scikit-learn.
This version works with Python 3.12+ (unlike PyCaret).
"""

import pandas as pd
import numpy as np
import logging
import pickle
import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, r2_score, silhouette_score, classification_report, mean_squared_error, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Try importing advanced models
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

logger = logging.getLogger(__name__)


class AutoMLTrainer:
    """Trains ML models using scikit-learn (Python 3.12+ compatible)."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.framework = 'sklearn'  # Using scikit-learn instead of PyCaret
        self.max_training_time = config.get('max_training_time', 3600)
        self.test_size = config.get('test_size', 0.2)
        self.random_state = config.get('random_state', 42)
        self.models_to_try = config.get('models_to_try', ['rf', 'gbc', 'lr', 'knn'])
        self.min_model_score = config.get('min_model_score', 0.6)
        self.max_dataset_rows = config.get('max_dataset_rows', 50000)  # Limit dataset size
        self.max_training_samples = config.get('max_training_samples', 50000)  # Limit training samples after SMOTE
        
        self.models_dir = Path("outputs/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Log availability of advanced models
        if XGBOOST_AVAILABLE:
            logger.info("XGBoost available - will use for better accuracy")
        else:
            logger.info("XGBoost not available - install with: pip install xgboost")
        
        if LIGHTGBM_AVAILABLE:
            logger.info("LightGBM available - will use for better accuracy")
        else:
            logger.info("LightGBM not available - install with: pip install lightgbm")
        
        if SMOTE_AVAILABLE:
            logger.info("SMOTE available - will use for imbalanced datasets")
        else:
            logger.info("SMOTE not available - install with: pip install imbalanced-learn")
        
        # Model mapping
        self.model_map = {
            'rf': {
                'classification': RandomForestClassifier,
                'regression': RandomForestRegressor
            },
            'gbc': {
                'classification': GradientBoostingClassifier,
                'regression': GradientBoostingRegressor
            },
            'lr': {
                'classification': LogisticRegression,
                'regression': LinearRegression
            },
            'knn': {
                'classification': KNeighborsClassifier,
                'regression': KNeighborsRegressor
            },
            'svm': {
                'classification': SVC,
                'regression': SVR
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.model_map['xgb'] = {
                'classification': XGBClassifier,
                'regression': XGBRegressor
            }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.model_map['lgbm'] = {
                'classification': LGBMClassifier,
                'regression': LGBMRegressor
            }
    
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
        """Load dataset from file, with sampling for large datasets."""
        path = Path(dataset_path)
        max_rows = self.config.get('max_dataset_rows', 50000)  # Limit to 50k rows by default
        
        if path.suffix == '.csv':
            # For large CSVs, sample rows to avoid memory issues
            try:
                # First, check file size
                file_size_mb = path.stat().st_size / (1024 * 1024)
                
                # If file is large (>100MB), sample rows
                if file_size_mb > 100:
                    logger.info(f"Large dataset detected ({file_size_mb:.1f} MB). Sampling {max_rows} rows for training.")
                    # Read in chunks and sample
                    chunk_list = []
                    sample_frac = min(max_rows / 1000000, 0.1)  # Sample fraction
                    
                    for chunk in pd.read_csv(path, chunksize=10000):
                        if len(chunk_list) * 10000 >= max_rows:
                            break
                        # Sample from chunk
                        if len(chunk) > 0:
                            sampled = chunk.sample(frac=min(sample_frac, 1.0), random_state=self.random_state)
                            chunk_list.append(sampled)
                    
                    df = pd.concat(chunk_list, ignore_index=True)
                    if len(df) > max_rows:
                        df = df.sample(n=max_rows, random_state=self.random_state).reset_index(drop=True)
                else:
                    df = pd.read_csv(path)
                    # Still limit rows if dataset is too large
                    if len(df) > max_rows:
                        logger.info(f"Dataset has {len(df)} rows. Sampling {max_rows} rows for training.")
                        df = df.sample(n=max_rows, random_state=self.random_state).reset_index(drop=True)
            except MemoryError:
                # If still fails, use smaller sample
                logger.warning(f"Memory error loading dataset. Using smaller sample of {max_rows // 2} rows.")
                df = pd.read_csv(path, nrows=max_rows // 2)
        elif path.suffix == '.json':
            df = pd.read_json(path)
            if len(df) > max_rows:
                logger.info(f"Dataset has {len(df)} rows. Sampling {max_rows} rows for training.")
                df = df.sample(n=max_rows, random_state=self.random_state).reset_index(drop=True)
        elif path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(path)
            if len(df) > max_rows:
                logger.info(f"Dataset has {len(df)} rows. Sampling {max_rows} rows for training.")
                df = df.sample(n=max_rows, random_state=self.random_state).reset_index(drop=True)
        else:
            # Try CSV as default
            df = pd.read_csv(path)
            if len(df) > max_rows:
                logger.info(f"Dataset has {len(df)} rows. Sampling {max_rows} rows for training.")
                df = df.sample(n=max_rows, random_state=self.random_state).reset_index(drop=True)
        
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
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
    
    def _prepare_data(self, df: pd.DataFrame, target: str, task_type: str):
        """Prepare data for training with advanced feature engineering."""
        # Separate features and target
        X = df.drop(columns=[target])
        y = df[target]
        
        # Memory-efficient categorical encoding
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        max_categories_per_col = 50
        
        for col in categorical_cols:
            unique_count = X[col].nunique()
            if unique_count > max_categories_per_col:
                logger.warning(f"Column {col} has {unique_count} unique values. Using top {max_categories_per_col} categories.")
                top_categories = X[col].value_counts().head(max_categories_per_col).index
                X[col] = X[col].where(X[col].isin(top_categories), 'other')
        
        # Handle categorical features
        if len(X) > 10000:
            X_encoded = X.copy()
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
        else:
            X_encoded = pd.get_dummies(X, drop_first=True)
            label_encoders = None
        
        # Handle missing values - use median for numeric, mode for categorical
        numeric_cols = X_encoded.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            X_encoded[col].fillna(X_encoded[col].median(), inplace=True)
        for col in X_encoded.columns:
            if X_encoded[col].isna().any():
                X_encoded[col].fillna(X_encoded[col].mode()[0] if len(X_encoded[col].mode()) > 0 else 0, inplace=True)
        
        # Encode target for classification
        if task_type == 'classification' and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.label_encoder = le
        else:
            self.label_encoder = None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=self.test_size, random_state=self.random_state, 
            stratify=y if task_type == 'classification' and len(np.unique(y)) < 100 else None
        )
        
        # Feature selection for classification (select top features)
        if task_type == 'classification' and X_train.shape[1] > 50:
            n_features = min(50, X_train.shape[1])
            logger.info(f"Selecting top {n_features} features from {X_train.shape[1]} features")
            selector = SelectKBest(score_func=f_classif, k=n_features)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            self.feature_selector = selector
        else:
            X_train_selected = X_train
            X_test_selected = X_test
            self.feature_selector = None
        
        # Scale features - use RobustScaler for better outlier handling
        scaler = RobustScaler()  # Changed from StandardScaler
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        self.scaler = scaler
        
        # Handle imbalanced data for classification
        self.use_smote = False
        if task_type == 'classification' and SMOTE_AVAILABLE:
            unique, counts = np.unique(y_train, return_counts=True)
            min_class_count = counts.min()
            max_class_count = counts.max()
            imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else 1
            
            if imbalance_ratio > 2:  # If classes are imbalanced
                logger.info(f"Detected class imbalance (ratio: {imbalance_ratio:.2f}). Applying SMOTE.")
                try:
                    smote = SMOTE(random_state=self.random_state, k_neighbors=min(5, min_class_count - 1))
                    X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
                    self.use_smote = True
                    logger.info(f"After SMOTE: {len(X_train_scaled)} samples")
                    
                    # Limit training samples if too large (for faster training)
                    if len(X_train_scaled) > self.max_training_samples:
                        logger.info(f"Sampling {self.max_training_samples} samples from {len(X_train_scaled)} for faster training")
                        indices = np.random.choice(len(X_train_scaled), size=self.max_training_samples, replace=False, random_state=self.random_state)
                        X_train_scaled = X_train_scaled[indices]
                        y_train = y_train[indices]
                        logger.info(f"Reduced to {len(X_train_scaled)} training samples")
                except Exception as e:
                    logger.warning(f"SMOTE failed: {e}. Continuing without SMOTE.")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def _train_classification(self, df: pd.DataFrame, target: str) -> Dict:
        """Train classification models to achieve >90% accuracy."""
        try:
            # Prepare data
            X_train, X_test, y_train, y_test = self._prepare_data(df, target, 'classification')
            
            # Try different models
            best_model = None
            best_score = 0
            best_model_name = None
            results = {}
            trained_models = []
            
            # Expand model list to include advanced models if available
            models_to_try = list(self.models_to_try)
            if XGBOOST_AVAILABLE and 'xgb' not in models_to_try:
                models_to_try.append('xgb')
            if LIGHTGBM_AVAILABLE and 'lgbm' not in models_to_try:
                models_to_try.append('lgbm')
            if 'svm' not in models_to_try:
                models_to_try.append('svm')
            
            # Get dataset size before training
            n_samples = len(X_train)
            
            # For very large datasets, prioritize faster models
            if n_samples > 100000:
                logger.info(f"Large dataset detected ({n_samples} samples). Prioritizing faster models.")
                # Remove slower models for very large datasets
                models_to_try = [m for m in models_to_try if m in ['lr', 'knn', 'xgb', 'lgbm']]
            
            for model_code in models_to_try:
                if model_code not in self.model_map:
                    continue
                
                model_info = self.model_map[model_code]
                model_class = model_info['classification']
                model_name = model_code.upper()
                
                try:
                    logger.info(f"Training {model_name}...")
                    
                    # Create model with optimized parameters based on dataset size
                    
                    # Set model parameters based on dataset size for faster training
                    if model_code == 'rf':
                        n_estimators = min(100, max(50, n_samples // 500))
                        max_depth = min(20, max(10, int(np.log2(n_samples))))
                        model = model_class(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=self.random_state,
                            n_jobs=-1
                        )
                    elif model_code == 'gbc':
                        n_estimators = min(100, max(50, n_samples // 1000))
                        max_depth = min(7, max(3, int(np.log2(n_samples / 10))))
                        model = model_class(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=0.1,
                            random_state=self.random_state
                        )
                    elif model_code == 'xgb' and XGBOOST_AVAILABLE:
                        n_estimators = min(100, max(50, n_samples // 1000))
                        max_depth = min(7, max(3, int(np.log2(n_samples / 10))))
                        model = model_class(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=0.1,
                            random_state=self.random_state,
                            n_jobs=-1,
                            verbosity=0
                        )
                    elif model_code == 'lgbm' and LIGHTGBM_AVAILABLE:
                        n_estimators = min(100, max(50, n_samples // 1000))
                        max_depth = min(7, max(3, int(np.log2(n_samples / 10))))
                        model = model_class(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=0.1,
                            random_state=self.random_state,
                            n_jobs=-1,
                            verbosity=-1
                        )
                    elif model_code == 'svm':
                        # Skip SVM for very large datasets (too slow)
                        if n_samples > 10000:
                            logger.info(f"Skipping {model_name} for large dataset ({n_samples} samples)")
                            continue
                        model = model_class(random_state=self.random_state, max_iter=1000)
                    else:
                        # Default parameters for other models
                        model = model_class(random_state=self.random_state)
                    
                    model.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    # Cross-validation (use fewer folds for large datasets)
                    cv_folds = 3 if n_samples > 20000 else 5
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy', n_jobs=-1)
                    cv_mean = cv_scores.mean()
                    
                    results[model_name] = {
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1_score': float(f1),
                        'cv_mean': float(cv_mean),
                        'cv_std': float(cv_scores.std())
                    }
                    
                    trained_models.append((model, model_name, accuracy))
                    
                    if accuracy > best_score:
                        best_score = accuracy
                        best_model = model
                        best_model_name = model_name
                    
                    logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, CV: {cv_mean:.4f}")
                    
                except Exception as e:
                    logger.warning(f"Error training {model_name}: {e}")
                    continue
            
            if best_model is None:
                return {
                    'success': False,
                    'error': 'No models could be trained'
                }
            
            # Try ensemble if we have multiple good models (>85% accuracy)
            good_models = [(m, n, s) for m, n, s in trained_models if s > 0.85]
            if len(good_models) >= 2 and best_score < 0.90:
                logger.info("Creating ensemble model from top models...")
                try:
                    # Create voting classifier from top 3 models
                    estimators = [(name, model) for model, name, score in sorted(good_models, key=lambda x: x[2], reverse=True)[:3]]
                    ensemble = VotingClassifier(estimators=estimators, voting='soft' if len(np.unique(y_train)) == 2 else 'hard')
                    ensemble.fit(X_train, y_train)
                    
                    y_pred_ensemble = ensemble.predict(X_test)
                    ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
                    
                    if ensemble_accuracy > best_score:
                        logger.info(f"Ensemble model achieved {ensemble_accuracy:.4f} accuracy (better than single model)")
                        best_model = ensemble
                        best_model_name = 'Ensemble'
                        best_score = ensemble_accuracy
                        results['Ensemble'] = {
                            'accuracy': float(ensemble_accuracy),
                            'precision': float(precision_score(y_test, y_pred_ensemble, average='weighted', zero_division=0)),
                            'recall': float(recall_score(y_test, y_pred_ensemble, average='weighted', zero_division=0)),
                            'f1_score': float(f1_score(y_test, y_pred_ensemble, average='weighted', zero_division=0))
                        }
                except Exception as e:
                    logger.warning(f"Ensemble creation failed: {e}")
            
            # Final check: if still below 90%, try stacking
            if best_score < 0.90 and len(good_models) >= 2:
                logger.info("Trying stacking classifier for better accuracy...")
                try:
                    estimators = [(name, model) for model, name, score in sorted(good_models, key=lambda x: x[2], reverse=True)[:2]]
                    stacker = StackingClassifier(
                        estimators=estimators,
                        final_estimator=LogisticRegression(random_state=self.random_state, max_iter=2000),
                        cv=3
                    )
                    stacker.fit(X_train, y_train)
                    
                    y_pred_stack = stacker.predict(X_test)
                    stack_accuracy = accuracy_score(y_test, y_pred_stack)
                    
                    if stack_accuracy > best_score:
                        logger.info(f"Stacking classifier achieved {stack_accuracy:.4f} accuracy")
                        best_model = stacker
                        best_model_name = 'Stacking'
                        best_score = stack_accuracy
                        results['Stacking'] = {
                            'accuracy': float(stack_accuracy),
                            'precision': float(precision_score(y_test, y_pred_stack, average='weighted', zero_division=0)),
                            'recall': float(recall_score(y_test, y_pred_stack, average='weighted', zero_division=0)),
                            'f1_score': float(f1_score(y_test, y_pred_stack, average='weighted', zero_division=0))
                        }
                except Exception as e:
                    logger.warning(f"Stacking failed: {e}")
            
            # Check if model meets minimum score
            if best_score < self.min_model_score:
                logger.warning(f"Model accuracy {best_score:.3f} below threshold {self.min_model_score}")
            
            if best_score >= 0.90:
                logger.info(f"✓ Achieved target accuracy of {best_score:.4f} (>= 0.90) with {best_model_name}!")
            else:
                logger.warning(f"Accuracy {best_score:.4f} is below 90% target. Consider: more data, better features, or different problem formulation.")
            
            # Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = self.models_dir / f"classification_model_{timestamp}.pkl"
            
            # Save model and scaler
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': best_model,
                    'scaler': self.scaler,
                    'label_encoder': self.label_encoder,
                    'feature_selector': self.feature_selector,
                    'model_name': best_model_name
                }, f)
            
            # Save metrics
            metrics_path = self.models_dir / f"metrics_{timestamp}.json"
            metrics_dict = {
                'task_type': 'classification',
                'target': target,
                'accuracy': float(best_score),
                'best_model': best_model_name,
                'all_results': results,
                'model_path': str(model_path),
                'timestamp': timestamp,
                'target_achieved': best_score >= 0.90
            }
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics_dict, f, indent=2)
            
            logger.info(f"Classification model trained. Accuracy: {best_score:.4f} ({best_model_name})")
            
            return {
                'success': True,
                'task_type': 'classification',
                'model_path': str(model_path),
                'metrics_path': str(metrics_path),
                'metrics': metrics_dict,
                'model': best_model
            }
            
        except Exception as e:
            logger.error(f"Error training classification model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }
    
    def _train_regression(self, df: pd.DataFrame, target: str) -> Dict:
        """Train regression models."""
        try:
            # Prepare data
            X_train, X_test, y_train, y_test = self._prepare_data(df, target, 'regression')
            
            # Try different models
            best_model = None
            best_score = -np.inf
            best_model_name = None
            results = {}
            
            for model_code in self.models_to_try:
                if model_code not in self.model_map:
                    continue
                
                model_class = self.model_map[model_code]['regression']
                model_name = model_code.upper()
                
                try:
                    logger.info(f"Training {model_name}...")
                    model = model_class(random_state=self.random_state)
                    model.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                    cv_mean = cv_scores.mean()
                    
                    results[model_name] = {
                        'r2': float(r2),
                        'mse': float(mse),
                        'cv_mean': float(cv_mean),
                        'cv_std': float(cv_scores.std())
                    }
                    
                    if r2 > best_score:
                        best_score = r2
                        best_model = model
                        best_model_name = model_name
                    
                except Exception as e:
                    logger.warning(f"Error training {model_name}: {e}")
                    continue
            
            if best_model is None:
                return {
                    'success': False,
                    'error': 'No models could be trained'
                }
            
            # Check if model meets minimum score
            if best_score < self.min_model_score:
                logger.warning(f"Model R2 {best_score:.3f} below threshold {self.min_model_score}")
            
            # Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = self.models_dir / f"regression_model_{timestamp}.pkl"
            
            # Save model and scaler
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': best_model,
                    'scaler': self.scaler,
                    'model_name': best_model_name
                }, f)
            
            # Save metrics
            metrics_path = self.models_dir / f"metrics_{timestamp}.json"
            metrics_dict = {
                'task_type': 'regression',
                'target': target,
                'r2': float(best_score),
                'best_model': best_model_name,
                'all_results': results,
                'model_path': str(model_path),
                'timestamp': timestamp
            }
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics_dict, f, indent=2)
            
            logger.info(f"Regression model trained. R2: {best_score:.3f} ({best_model_name})")
            
            return {
                'success': True,
                'task_type': 'regression',
                'model_path': str(model_path),
                'metrics_path': str(metrics_path),
                'metrics': metrics_dict,
                'model': best_model
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
            # Prepare data (no target for clustering)
            X = df.select_dtypes(include=[np.number])
            X = pd.get_dummies(df, drop_first=True)
            X = X.fillna(X.mean())
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Determine optimal number of clusters
            n_clusters = min(10, max(2, len(df) // 10))
            
            logger.info(f"Training KMeans with {n_clusters} clusters...")
            model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            model.fit(X_scaled)
            
            # Evaluate
            silhouette = silhouette_score(X_scaled, model.labels_)
            
            # Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = self.models_dir / f"clustering_model_{timestamp}.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'scaler': scaler,
                    'model_name': 'KMeans'
                }, f)
            
            logger.info(f"Clustering model trained. Silhouette score: {silhouette:.3f}")
            
            return {
                'success': True,
                'task_type': 'clustering',
                'model_path': str(model_path),
                'metrics': {
                    'task_type': 'clustering',
                    'n_clusters': int(n_clusters),
                    'silhouette_score': float(silhouette),
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

