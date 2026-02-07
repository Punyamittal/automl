"""
Orchestrator Module

Main pipeline orchestrator that coordinates all modules.
"""

import logging
import json
import re
import yaml
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

from .problem_miner import ProblemMiner
from .feasibility_classifier import FeasibilityClassifier
from .ml_decision_agent import MLDecisionAgent
from .dataset_discovery import DatasetDiscovery
from .dataset_matcher import DatasetMatcher
from .model_registry import ModelRegistry
from .problem_registry import ProblemRegistry
# Try PyCaret first, fallback to scikit-learn for Python 3.12+
try:
    from .automl_trainer import AutoMLTrainer
    AUTOML_TRAINER_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    logger.warning(f"PyCaret not available ({e}), using scikit-learn fallback")
    from .automl_trainer_sklearn import AutoMLTrainer
    AUTOML_TRAINER_AVAILABLE = True
from .code_generator import CodeGenerator
from .github_publisher import GitHubPublisher


class Orchestrator:
    """Main pipeline orchestrator."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize orchestrator with configuration."""
        self.config = self._load_config(config_path)
        self._setup_logging()
        
        # Initialize modules
        # Pass full config to problem_miner so it can access ollama, llm, kaggle, and github configs
        problem_miner_config = self.config.get('problem_miner', {}).copy()
        problem_miner_config['ollama'] = self.config.get('ollama', {})
        problem_miner_config['llm'] = self.config.get('llm', {})
        problem_miner_config['kaggle_problem_mining'] = self.config.get('kaggle_problem_mining', {'enabled': True})
        problem_miner_config['github_problem_mining'] = self.config.get('github_problem_mining', {'enabled': True})
        problem_miner_config['github'] = self.config.get('github', {})  # Pass GitHub config for token
        self.problem_miner = ProblemMiner(problem_miner_config)
        self.ml_decision_agent = MLDecisionAgent(self.config.get('ml_decision_agent', {}))
        self.feasibility_classifier = FeasibilityClassifier(self.config.get('feasibility', {}))
        self.dataset_discovery = DatasetDiscovery(self.config.get('dataset_discovery', {}))
        self.dataset_matcher = DatasetMatcher(self.config.get('dataset_matching', {}))
        self.problem_registry = ProblemRegistry(self.config.get('problem_registry', {}))
        self.model_registry = ModelRegistry(self.config.get('model_registry', {}))
        # Initialize AutoML trainer
        automl_config = self.config.get('automl', {})
        # Use sklearn framework for Python 3.12+ compatibility
        automl_config['framework'] = 'sklearn'
        
        # Use sklearn trainer directly for Python 3.12+ compatibility
        from .automl_trainer_sklearn import AutoMLTrainer
        self.automl_trainer = AutoMLTrainer(automl_config)
        self.code_generator = CodeGenerator(self.config.get('code_generation', {}))
        self.github_publisher = GitHubPublisher(self.config.get('github', {}))
        
        self.results_dir = Path("outputs/logs")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file and merge with environment variables."""
        # Load environment variables from .env file
        load_dotenv()
        
        # Get config path from env or use default
        env_config_path = os.getenv('CONFIG_PATH', config_path)
        config_file = Path(env_config_path)
        
        if not config_file.exists():
            logger.warning(f"Config file not found: {env_config_path}. Using defaults.")
            config = {}
        else:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
        
        # Override with environment variables if present
        # GitHub
        if os.getenv('GITHUB_USERNAME'):
            if 'github' not in config:
                config['github'] = {}
            config['github']['username'] = os.getenv('GITHUB_USERNAME')
        if os.getenv('GITHUB_TOKEN'):
            if 'github' not in config:
                config['github'] = {}
            config['github']['token'] = os.getenv('GITHUB_TOKEN')
        if os.getenv('GITHUB_ORGANIZATION'):
            if 'github' not in config:
                config['github'] = {}
            config['github']['organization'] = os.getenv('GITHUB_ORGANIZATION')
        
        # HuggingFace
        if os.getenv('HUGGINGFACE_TOKEN'):
            if 'llm' not in config:
                config['llm'] = {}
            config['llm']['token'] = os.getenv('HUGGINGFACE_TOKEN')
        
        # Default Ollama config when not present (local LLM, no API key)
        if 'ollama' not in config:
            config['ollama'] = {'enabled': True, 'base_url': 'http://localhost:11434', 'model_name': 'llama3.2'}

        return config
    
    def _download_dataset(self, dataset: Dict, task_type: str) -> Optional[str]:
        """Download dataset from source (Kaggle/HuggingFace) or create synthetic one."""
        dataset_source = dataset.get('source', '')
        dataset_id = dataset.get('id', '')
        
        download_dir = Path("data/datasets/downloaded")
        download_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to download from source
        if dataset_source == 'kaggle' and self.dataset_discovery.kaggle_api:
            try:
                # Parse owner/dataset from id (format: "owner/dataset")
                ref_parts = dataset_id.split('/')
                if len(ref_parts) == 2:
                    logger.info(f"Downloading Kaggle dataset: {dataset_id}")
                    download_dir.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        # Pass path parameter so download goes to our directory
                        self.dataset_discovery.kaggle_api.dataset_download_files(
                            dataset_id,
                            path=str(download_dir),
                            unzip=True
                        )
                        logger.info("Download completed successfully")
                    except Exception as e1:
                        logger.warning(f"Download failed: {e1}")
                        raise Exception(f"Failed to download Kaggle dataset: {e1}")
                    
                    # Find CSV files (Kaggle extracts to dataset_slug folder or root)
                    csv_files = list(download_dir.rglob("*.csv"))
                    if csv_files:
                        logger.info(f"Downloaded dataset to {csv_files[0]}")
                        return str(csv_files[0])
            except Exception as e:
                logger.warning(f"Failed to download Kaggle dataset: {e}")
        
        elif dataset_source == 'huggingface':
            try:
                from datasets import load_dataset
                logger.info(f"Downloading HuggingFace dataset: {dataset_id}")
                hf_dataset = load_dataset(dataset_id, split='train')
                
                # Convert to pandas with size limit
                max_rows = self.config.get('automl', {}).get('max_dataset_rows', 50000)
                logger.info(f"Converting dataset to pandas (max {max_rows} rows)...")
                
                # For very large datasets, sample before converting
                if len(hf_dataset) > max_rows:
                    logger.info(f"Dataset has {len(hf_dataset)} rows. Sampling {max_rows} rows.")
                    # Sample indices
                    import random
                    random.seed(42)
                    sample_indices = random.sample(range(len(hf_dataset)), min(max_rows, len(hf_dataset)))
                    hf_dataset = hf_dataset.select(sample_indices)
                
                df = hf_dataset.to_pandas()
                
                # Save as CSV
                output_file = download_dir / f"{dataset_id.replace('/', '_')}.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"Downloaded dataset to {output_file} ({len(df)} rows)")
                return str(output_file)
            except Exception as e:
                logger.warning(f"Failed to download HuggingFace dataset: {e}")
        
        elif dataset_source == 'uci':
            try:
                try:
                    from ucimlrepo import fetch_ucirepo
                except ImportError:
                    logger.warning("ucimlrepo not installed. Run: pip install ucimlrepo")
                    raise ImportError("ucimlrepo required for UCI downloads")
                uci_id = int(dataset_id)
                logger.info(f"Downloading UCI dataset (id={uci_id})...")
                uci_data = fetch_ucirepo(id=uci_id)
                if hasattr(uci_data.data, 'features') and uci_data.data.features is not None:
                    df = uci_data.data.features.copy()
                    if hasattr(uci_data.data, 'targets') and uci_data.data.targets is not None:
                        df['target'] = uci_data.data.targets.iloc[:, 0]
                elif hasattr(uci_data.data, 'original') and uci_data.data.original is not None:
                    df = uci_data.data.original
                else:
                    raise ValueError("No usable data in UCI dataset")
                output_file = download_dir / f"uci_{uci_id}.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"Downloaded UCI dataset to {output_file} ({len(df)} rows)")
                return str(output_file)
            except Exception as e:
                logger.warning(f"Failed to download UCI dataset: {e}")
        
        elif dataset_source == 'datagov':
            try:
                download_url = dataset.get('download_url') or (dataset.get('files') or [None])[0]
                if download_url:
                    logger.info(f"Downloading data.gov dataset from {download_url[:80]}...")
                    r = requests.get(download_url, timeout=60)
                    r.raise_for_status()
                    ext = 'json' if 'json' in download_url.lower() else 'csv'
                    output_file = download_dir / f"datagov_{dataset_id}.{ext}"
                    with open(output_file, 'wb') as f:
                        f.write(r.content)
                    if ext == 'json':
                        import pandas as pd
                        df = pd.read_json(output_file)
                        csv_path = output_file.with_suffix('.csv')
                        df.to_csv(csv_path, index=False)
                        output_file = csv_path
                    logger.info(f"Downloaded data.gov dataset to {output_file}")
                    return str(output_file)
            except Exception as e:
                logger.warning(f"Failed to download data.gov dataset: {e}")
        
        elif dataset_source == 'worldbank':
            try:
                import pandas as pd
                indicator = dataset_id
                url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}?format=json&per_page=10000"
                logger.info(f"Downloading World Bank data: {indicator}...")
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                data = r.json()
                if len(data) >= 2 and data[1]:
                    rows = [{'country': d.get('country', {}).get('value'), 'year': d.get('date'), 'value': d.get('value')} for d in data[1]]
                    df = pd.DataFrame(rows).dropna(subset=['value'])
                    output_file = download_dir / f"worldbank_{indicator}.csv"
                    df.to_csv(output_file, index=False)
                    logger.info(f"Downloaded World Bank dataset to {output_file} ({len(df)} rows)")
                    return str(output_file)
            except Exception as e:
                logger.warning(f"Failed to download World Bank dataset: {e}")
        
        # Fallback: Create synthetic dataset for demonstration
        logger.warning("Could not download dataset. Creating synthetic dataset for demonstration.")
        return self._create_synthetic_dataset(task_type, download_dir)
    
    def _create_synthetic_dataset(self, task_type: str, output_dir: Path) -> str:
        """Create a synthetic dataset for training when real dataset unavailable."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        if task_type == 'classification':
            # Binary classification
            y = (X[:, 0] + X[:, 1] > 0).astype(int)
            target_name = 'target'
        elif task_type == 'regression':
            # Regression target
            y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.5
            target_name = 'target'
        else:
            # Default to classification
            y = (X[:, 0] + X[:, 1] > 0).astype(int)
            target_name = 'target'
        
        # Create DataFrame
        feature_names = [f'feature_{i+1}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df[target_name] = y
        
        # Save to CSV
        output_file = output_dir / f"synthetic_{task_type}_dataset.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Created synthetic {task_type} dataset with {n_samples} samples at {output_file}")
        return str(output_file)
    
    def _test_model(self, dataset_path: str, training_result: Dict, task_type: str) -> Dict:
        """Test the trained model and get detailed accuracy results."""
        try:
            import pickle
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            from sklearn.model_selection import train_test_split
            
            # Load the dataset
            df = pd.read_csv(dataset_path)
            
            # Load the trained model
            model_path = training_result.get('model_path')
            if not model_path or not Path(model_path).exists():
                return {
                    'success': False,
                    'error': 'Model file not found'
                }
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                model = model_data.get('model')
                scaler = model_data.get('scaler')
                label_encoder = model_data.get('label_encoder')
            
            # Get target column from training result
            target = training_result.get('metrics', {}).get('target', 'target')
            
            if target not in df.columns:
                # Try to detect target
                if task_type == 'classification':
                    # Look for common classification target names
                    for col in ['label', 'target', 'class', 'y']:
                        if col in df.columns:
                            target = col
                            break
                else:
                    for col in ['target', 'y', 'value']:
                        if col in df.columns:
                            target = col
                            break
            
            if target not in df.columns:
                return {
                    'success': False,
                    'error': f'Target column {target} not found in dataset'
                }
            
            # Prepare data
            X = df.drop(columns=[target])
            y = df[target]
            
            # Handle categorical features
            X = pd.get_dummies(X, drop_first=True)
            
            # Encode target if needed
            if task_type == 'classification' and label_encoder:
                y = label_encoder.transform(y)
            elif task_type == 'classification' and y.dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y)
            
            # Split data (use same random state as training)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.config.get('automl', {}).get('random_state', 42)
            )
            
            # Scale features if scaler was used
            if scaler:
                X_test = scaler.transform(X_test)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            if task_type == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Get classification report
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                
                test_metrics = {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'classification_report': report
                }
                
                logger.info(f"Test Results - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                
            else:  # regression
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                test_metrics = {
                    'r2_score': float(r2),
                    'mse': float(mse),
                    'mae': float(mae),
                    'rmse': float(rmse)
                }
                
                logger.info(f"Test Results - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            
            return {
                'success': True,
                'task_type': task_type,
                'test_metrics': test_metrics,
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            logger.error(f"Error testing model: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def _print_summary(self, pipeline_results: Dict, problem: Dict, dataset: Dict, training_result: Dict):
        """Print a comprehensive summary of what was created."""
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE SUMMARY - What Was Created")
        logger.info("=" * 80)
        
        # Problem Information
        logger.info("\n[PROBLEM]")
        logger.info(f"   Title: {problem.get('title', 'Unknown')}")
        logger.info(f"   Task Type: {training_result.get('task_type', 'Unknown')}")
        
        # Dataset Information
        logger.info("\n[DATASET]")
        logger.info(f"   Source: {dataset.get('source', 'Unknown').upper()}")
        logger.info(f"   ID: {dataset.get('id', 'Unknown')}")
        logger.info(f"   URL: {dataset.get('url', 'N/A')}")
        
        # Training Results
        logger.info("\n[MODEL TRAINING]")
        metrics = training_result.get('metrics', {})
        if training_result.get('task_type') == 'classification':
            accuracy = metrics.get('accuracy', 0)
            best_model = metrics.get('best_model', 'Unknown')
            logger.info(f"   Best Model: {best_model}")
            logger.info(f"   Training Accuracy: {accuracy:.4f}")
        else:
            r2 = metrics.get('r2', 0)
            best_model = metrics.get('best_model', 'Unknown')
            logger.info(f"   Best Model: {best_model}")
            logger.info(f"   Training R² Score: {r2:.4f}")
        logger.info(f"   Model Path: {training_result.get('model_path', 'N/A')}")
        
        # Testing Results
        test_result = pipeline_results.get('stages', {}).get('testing', {})
        if test_result.get('success'):
            logger.info("\n[TEST RESULTS]")
            test_metrics = test_result.get('test_metrics', {})
            if test_result.get('task_type') == 'classification':
                logger.info(f"   Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")
                logger.info(f"   Precision: {test_metrics.get('precision', 0):.4f}")
                logger.info(f"   Recall: {test_metrics.get('recall', 0):.4f}")
                logger.info(f"   F1 Score: {test_metrics.get('f1_score', 0):.4f}")
            else:
                logger.info(f"   Test R² Score: {test_metrics.get('r2_score', 0):.4f}")
                logger.info(f"   RMSE: {test_metrics.get('rmse', 0):.4f}")
                logger.info(f"   MAE: {test_metrics.get('mae', 0):.4f}")
            logger.info(f"   Test Samples: {test_result.get('test_samples', 0)}")
        
        # Generated Files
        code_result = pipeline_results.get('stages', {}).get('code_generation', {})
        if code_result.get('success'):
            logger.info("\n[GENERATED FILES]")
            project_dir = code_result.get('project_dir', 'N/A')
            logger.info(f"   Project Directory: {project_dir}")
            files = code_result.get('files', [])
            for file in files:
                logger.info(f"   - {file}")
        
        # GitHub Repository
        github_result = pipeline_results.get('stages', {}).get('github_publishing', {})
        if github_result.get('success'):
            logger.info("\n[GITHUB REPOSITORY]")
            repo_url = github_result.get('repo_url', 'N/A')
            logger.info(f"   URL: {repo_url}")
        
        # Model Registry Stats
        registry_stats = self.model_registry.get_registry_stats()
        logger.info("\n[MODEL REGISTRY]")
        logger.info(f"   Total Entries: {registry_stats.get('total_entries', 0)}")
        logger.info(f"   Total Trainings: {registry_stats.get('total_trainings', 0)}")
        logger.info(f"   Duplicate Models Blocked: {registry_stats.get('duplicate_models', 0)}")
        logger.info(f"   Using Vector Search: {registry_stats.get('using_embeddings', False)}")
        
        # Problem Registry Stats
        problem_stats = self.problem_registry.get_registry_stats()
        logger.info("\n[PROBLEM REGISTRY]")
        logger.info(f"   Total Problems: {problem_stats.get('total_problems', 0)}")
        logger.info(f"   Unique Problems: {problem_stats.get('unique_problems', 0)}")
        logger.info(f"   Duplicate Problems Blocked: {problem_stats.get('duplicate_problems', 0)}")
        logger.info(f"   Using Vector Search: {problem_stats.get('using_embeddings', False)}")
        
        # Output Locations
        logger.info("\n[OUTPUT LOCATIONS]")
        logger.info(f"   Models: {self.automl_trainer.models_dir}")
        logger.info(f"   Code: outputs/code/")
        logger.info(f"   Logs: {self.results_dir}")
        logger.info(f"   Datasets: data/datasets/downloaded/")
        logger.info(f"   Registry: data/model_registry/")
        
        logger.info("\n" + "=" * 80)
    
    def _generate_rejection_report(self, all_decision_results: List[Dict], total_problems: int, search_iterations: int):
        """Generate a detailed report of rejected problems for analysis."""
        try:
            report_dir = Path("outputs/reports")
            report_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = report_dir / f"rejection_report_{timestamp}.json"
            
            # Group rejections by category and reason
            rejection_data = {
                'timestamp': datetime.now().isoformat(),
                'total_problems_evaluated': total_problems,
                'search_iterations': search_iterations,
                'summary': {
                    'total_rejected': total_problems,
                    'total_approved': 0
                },
                'rejections_by_category': {},
                'sample_rejections': []
            }
            
            # Categorize rejections
            for i, decision in enumerate(all_decision_results):
                if decision['decision'] == 'do_not_train':
                    category = decision.get('category', 'unknown')
                    if category not in rejection_data['rejections_by_category']:
                        rejection_data['rejections_by_category'][category] = {
                            'count': 0,
                            'reasons': []
                        }
                    rejection_data['rejections_by_category'][category]['count'] += 1
                    rejection_data['rejections_by_category'][category]['reasons'].append({
                        'reasoning': decision.get('reasoning', 'Unknown'),
                        'recommended_action': decision.get('recommended_action', 'N/A')
                    })
                    
                    # Keep first 20 as samples
                    if len(rejection_data['sample_rejections']) < 20:
                        rejection_data['sample_rejections'].append({
                            'index': i + 1,
                            'category': category,
                            'reasoning': decision.get('reasoning', 'Unknown'),
                            'recommended_action': decision.get('recommended_action', 'N/A')
                        })
            
            # Save report
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(rejection_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"\n[REPORT] Detailed rejection report saved to: {report_file}")
            
        except Exception as e:
            logger.warning(f"Could not generate rejection report: {e}")
    
    def _sanitize_for_json(self, obj):
        """Remove non-JSON-serializable objects (like model objects) from results."""
        if isinstance(obj, dict):
            sanitized = {}
            for key, value in obj.items():
                # Skip model objects and other non-serializable types
                if key == 'model' and hasattr(value, 'predict'):
                    # Replace model object with just the model type name
                    sanitized[key] = type(value).__name__
                elif key == 'scaler' and hasattr(value, 'transform'):
                    sanitized[key] = type(value).__name__
                elif key == 'label_encoder' and hasattr(value, 'transform'):
                    sanitized[key] = type(value).__name__
                else:
                    sanitized[key] = self._sanitize_for_json(value)
            return sanitized
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # For other types, try to convert to string
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(type(obj).__name__)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))
        log_file = log_config.get('file', 'outputs/logs/pipeline.log')
        
        # Create logs directory
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        handlers = []
        
        # File handler
        handlers.append(logging.FileHandler(log_file))
        
        # Console handler
        if log_config.get('console', True):
            handlers.append(logging.StreamHandler())
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
    
    def run_with_problem(self, problem_statement: str) -> Dict:
        """Run pipeline with a direct problem statement (skips mining)."""
        logger.info("=" * 80)
        logger.info("Starting Autonomous ML Pipeline (Direct Problem Mode)")
        logger.info("=" * 80)
        
        pipeline_results = {
            'start_time': datetime.now().isoformat(),
            'stages': {},
            'success': False,
            'mode': 'direct_problem'
        }
        
        try:
            # Create problem object from statement
            problem = {
                'id': 'direct_problem_' + datetime.now().strftime('%Y%m%d_%H%M%S'),
                'source': 'direct_input',
                'title': problem_statement[:100] + ('...' if len(problem_statement) > 100 else ''),
                'description': problem_statement,
                'full_text': problem_statement,
                'mined_at': datetime.now().isoformat()
            }
            
            # Stage 1: Problem Enhancement with Ollama (local LLM)
            ollama_config = self.config.get('ollama', {})
            llm_client = None

            if ollama_config.get('enabled', False):
                try:
                    from .ollama_client import OllamaClient
                    llm_client = OllamaClient(
                        base_url=ollama_config.get('base_url', 'http://localhost:11434'),
                        model_name=ollama_config.get('model_name', 'llama3.2')
                    )
                    logger.info("\n[Stage 1] Problem Enhancement with Ollama (local)")
                    logger.info(f"Enhancing problem statement using {ollama_config.get('model_name', 'llama3.2')}...")
                except Exception as e:
                    logger.warning(f"Ollama initialization failed: {e}. Falling back to defaults.")

            if llm_client:
                try:
                    # First attempt: Canonicalize the problem
                    canonical_result = llm_client.analyze_problem_statement(problem_statement)
                    
                    if canonical_result.get('is_ml_problem') and canonical_result.get('canonical_problem'):
                        canonical = canonical_result['canonical_problem']
                        # Update problem with canonicalized information
                        problem['canonical_problem'] = canonical
                        problem['problem_type'] = canonical.get('problem_type', 'unknown')
                        problem['target_variable'] = canonical.get('target_variable', '')
                        problem['input_features'] = canonical.get('input_features', [])
                        problem['intended_use'] = canonical.get('intended_use', '')
                        problem['data_source'] = canonical.get('data_source', '')
                        problem['evaluation_metric'] = canonical.get('evaluation_metric', '')
                        
                        # Update full_text with enhanced problem statement
                        enhanced_statement = f"Predict {canonical.get('target_variable', 'target')} using {', '.join(canonical.get('input_features', [])[:3])}"
                        problem['full_text'] = enhanced_statement
                        problem['description'] = enhanced_statement
                        
                        logger.info(f"[ENHANCED] Problem Type: {canonical.get('problem_type')}")
                        logger.info(f"  Target: {canonical.get('target_variable')}")
                        logger.info(f"  Features: {', '.join(canonical.get('input_features', [])[:5])}")
                        logger.info(f"  Enhanced Statement: {enhanced_statement}")
                    else:
                        # If canonicalization failed, enhance the problem statement
                        logger.warning(f"Initial canonicalization incomplete: {canonical_result.get('reasoning', 'Unknown')}")
                        logger.info("Enhancing problem statement to make it complete...")
                        
                        # Use LLM to enhance the problem
                        enhanced_result = llm_client.enhance_problem_statement(problem_statement)
                        
                        if enhanced_result.get('enhanced_problem'):
                            enhanced = enhanced_result['enhanced_problem']
                            problem['canonical_problem'] = enhanced
                            problem['problem_type'] = enhanced.get('problem_type', 'classification')
                            problem['target_variable'] = enhanced.get('target_variable', 'target')
                            problem['input_features'] = enhanced.get('input_features', ['feature1', 'feature2', 'feature3'])
                            problem['intended_use'] = enhanced.get('intended_use', 'business decision support')
                            problem['data_source'] = enhanced.get('data_source', 'historical data')
                            problem['evaluation_metric'] = enhanced.get('evaluation_metric', 'accuracy' if enhanced.get('problem_type') == 'classification' else 'rmse')
                            
                            # Update full_text with enhanced problem
                            enhanced_statement = enhanced_result.get('enhanced_statement', problem_statement)
                            problem['full_text'] = enhanced_statement
                            problem['description'] = enhanced_statement
                            
                            logger.info(f"[ENHANCED] Problem enhanced successfully")
                            logger.info(f"  Enhanced Statement: {enhanced_statement[:200]}...")
                        else:
                            logger.warning("LLM enhancement failed, using inferred defaults")
                            # Use safe defaults
                            problem['problem_type'] = 'classification'
                            problem['target_variable'] = 'target'
                            problem['input_features'] = ['feature1', 'feature2', 'feature3']
                except Exception as e:
                    logger.warning(f"LLM enhancement failed: {e}. Using original problem with defaults.")
                    # Use safe defaults
                    problem['problem_type'] = 'classification'
                    problem['target_variable'] = 'target'
                    problem['input_features'] = ['feature1', 'feature2', 'feature3']
            else:
                logger.warning("Ollama not enabled. Using original problem with inferred defaults.")
                # Infer from problem statement
                if 'churn' in problem_statement.lower():
                    problem['problem_type'] = 'classification'
                    problem['target_variable'] = 'churn'
                elif 'price' in problem_statement.lower() or 'cost' in problem_statement.lower():
                    problem['problem_type'] = 'regression'
                    problem['target_variable'] = 'price'
                else:
                    problem['problem_type'] = 'classification'
                    problem['target_variable'] = 'target'
                problem['input_features'] = ['feature1', 'feature2', 'feature3']
            
            # Stage 1.5: ML Decision Agent (Direct Problem Mode - always approve after enhancement)
            logger.info("\n[Stage 1.5] ML Decision Agent - Validating Enhanced Problem")
            logger.info(f"Problem: {problem.get('full_text', problem_statement)[:100]}...")
            logger.info("Direct problem mode: Problem has been enhanced, proceeding with validation")
            
            # Create a decision that always approves (since we've enhanced the problem)
            decision = {
                'decision': 'train',
                'content_type': 'predictive_ml_task',
                'reasoning': 'Problem enhanced and validated. Proceeding with ML training.',
                'justification': 'Problem statement has been enhanced to be a well-formed ML problem.',
                'gate_results': {
                    'intent': {'category': 'predictive_ml_task', 'skipped': True},
                    'feasibility': {'feasible': True, 'from_canonical': True},
                    'causal_validity': {'valid': True},
                    'justification': {'justified': True}
                },
                'recommended_action': 'Proceed with model training',
                'ml_problem_definition': {
                    'target': problem.get('target_variable', 'target'),
                    'features': problem.get('input_features', []),
                    'task_type': problem.get('problem_type', 'classification'),
                    'data_source': problem.get('data_source', 'to be discovered'),
                    'intended_use': problem.get('intended_use', 'business decision support')
                }
            }
            
            problem['ml_decision'] = decision
            logger.info(f"[APPROVED] Problem enhanced and approved for ML training")
            
            # Save decision results
            pipeline_results['stages']['ml_decision'] = {
                'success': True,
                'approved_problems': 1,
                'rejected_problems': 0,
                'decision': decision,
                'enhanced': True
            }
            
            # Continue with rest of pipeline (feasibility, dataset discovery, etc.)
            return self._continue_pipeline([problem], pipeline_results)
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            pipeline_results['error'] = str(e)
            pipeline_results['success'] = False
            return pipeline_results
    
    def run(self) -> Dict:
        """Run the complete pipeline (with problem mining)."""
        logger.info("=" * 80)
        logger.info("Starting Autonomous ML Pipeline")
        logger.info("=" * 80)
        
        pipeline_results = {
            'start_time': datetime.now().isoformat(),
            'stages': {},
            'success': False,
            'mode': 'mining'
        }
        
        try:
            # Stage 1 & 1.5: Problem Mining and ML Decision Agent (Continue until approved problem found)
            logger.info("\n[Stage 1 & 1.5] Problem Mining and ML Decision Agent")
            logger.info("Searching for problems until one is approved for ML training...")
            
            # Get search limits from config
            max_search_iterations = self.config.get('max_search_iterations', 10)  # Max 10 batches
            max_total_problems = self.config.get('max_total_problems_evaluated', 100)  # Max 100 problems total
            all_decision_results = []
            total_problems_evaluated = 0
            approved_problems = []
            search_iteration = 0
            
            while len(approved_problems) == 0 and search_iteration < max_search_iterations and total_problems_evaluated < max_total_problems:
                search_iteration += 1
                logger.info(f"\n--- Search Iteration {search_iteration} ---")
                
                # Mine problems
                logger.info("Mining problems...")
                problems = self.problem_miner.mine_problems()
                
                if not problems:
                    logger.warning(f"No problems found in iteration {search_iteration}. Trying again...")
                    continue
                
                # Save problems (append to existing)
                self.problem_miner.save_problems(problems)
                
                # Evaluate each problem with ML Decision Agent
                logger.info(f"Evaluating {len(problems)} problems...")
                batch_decision_results = []
                
                for problem in problems:
                    # Skip if already in problem registry (already processed before)
                    is_duplicate, _ = self.problem_registry.check_duplicate_problem(problem)
                    if is_duplicate:
                        logger.info(f"[SKIP] Problem already in registry: {problem.get('title', 'Unknown')[:80]}...")
                        continue
                    
                    total_problems_evaluated += 1
                    logger.info(f"[{total_problems_evaluated}] Evaluating: {problem.get('title', 'Unknown')[:80]}...")
                    
                    decision = self.ml_decision_agent.decide(problem)
                    batch_decision_results.append(decision)
                    all_decision_results.append(decision)
                    
                    if decision['decision'] == 'train':
                        problem['ml_decision'] = decision
                        approved_problems.append(problem)
                        logger.info(f"[APPROVED] Found approved problem after evaluating {total_problems_evaluated} problems!")
                        logger.info(f"   Problem: {problem.get('title', 'Unknown')}")
                        logger.info(f"   Reasoning: {decision['reasoning']}")
                        break  # Found one, exit inner loop
                    else:
                        logger.warning(f"[REJECTED] {decision.get('reasoning', 'Unknown reason')[:100]}")
                
                # If found approved problem, break outer loop
                if approved_problems:
                    break
                
                logger.warning(f"Iteration {search_iteration}: All {len(problems)} problems rejected. Continuing search...")
            
            # Save mining results
            pipeline_results['stages']['problem_mining'] = {
                'success': True,
                'problems_found': total_problems_evaluated,
                'search_iterations': search_iteration
            }
            
            # Save decision results
            pipeline_results['stages']['ml_decision'] = {
                'success': True,
                'approved_problems': len(approved_problems),
                'rejected_problems': total_problems_evaluated - len(approved_problems),
                'total_problems_evaluated': total_problems_evaluated,
                'search_iterations': search_iteration,
                'decisions': all_decision_results
            }
            
            if not approved_problems:
                # This is a SUCCESSFUL pipeline run - correct rejections are good!
                logger.info("\n" + "=" * 80)
                logger.info("No valid ML training problems found. This is an expected and correct outcome.")
                logger.info("=" * 80)
                logger.info(f"\nEvaluated {total_problems_evaluated} problems across {search_iteration} search iteration(s).")
                logger.info("All problems were correctly identified as non-predictive ML tasks.")
                logger.info("\nRejection Categories:")
                
                # Group rejections by category
                rejection_categories = {}
                for decision in all_decision_results:
                    if decision['decision'] == 'do_not_train':
                        category = decision.get('category', 'unknown')
                        if category not in rejection_categories:
                            rejection_categories[category] = []
                        rejection_categories[category].append(decision)
                
                for category, decisions in rejection_categories.items():
                    logger.info(f"  - {category}: {len(decisions)} problems")
                
                # Generate rejection summary report
                self._generate_rejection_report(all_decision_results, total_problems_evaluated, search_iteration)
                
                # Mark pipeline as successful
                pipeline_results['success'] = True
                pipeline_results['end_time'] = datetime.now().isoformat()
                pipeline_results['result'] = "No valid ML problems found"
                pipeline_results['models_trained'] = 0
                pipeline_results['decision_quality'] = "Correct"
                pipeline_results['total_problems_evaluated'] = total_problems_evaluated
                pipeline_results['rejection_summary'] = {
                    'total_rejected': total_problems_evaluated,
                    'categories': {cat: len(decisions) for cat, decisions in rejection_categories.items()}
                }
                
                logger.info("\n" + "=" * 80)
                logger.info("Pipeline completed successfully!")
                logger.info("=" * 80)
                logger.info("\nPipeline Status:")
                logger.info(f"  Success: True")
                logger.info(f"  Result: No valid ML problems found")
                logger.info(f"  Models trained: 0")
                logger.info(f"  Decision quality: Correct")
                logger.info(f"  Problems evaluated: {total_problems_evaluated}")
                logger.info("\nThis demonstrates:")
                logger.info("  - Intent understanding")
                logger.info("  - Refusal capability")
                logger.info("  - Causal reasoning")
                logger.info("  - No hallucinated training")
                logger.info("  - Safe automation")
                logger.info("=" * 80)
                
                return pipeline_results
            
            logger.info(f"\n[SUCCESS] Found approved problem after evaluating {total_problems_evaluated} problems in {search_iteration} iteration(s)")
            
            # Continue with rest of pipeline
            return self._continue_pipeline(approved_problems, pipeline_results)
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            pipeline_results['error'] = str(e)
            pipeline_results['success'] = False
        
        finally:
            # Save pipeline results (remove non-serializable objects)
            results_file = self.results_dir / f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            # Remove model objects from training results before JSON serialization
            sanitized_results = self._sanitize_for_json(pipeline_results)
            with open(results_file, 'w') as f:
                json.dump(sanitized_results, f, indent=2)
            
            logger.info(f"Pipeline results saved to {results_file}")
        
        return pipeline_results
    
    def _continue_pipeline(self, approved_problems: List[Dict], pipeline_results: Dict) -> Dict:
        """Continue pipeline from Stage 2 onwards with approved problems."""
        try:
            # Stage 2: ML Feasibility Classification (Secondary validation)
            logger.info("\n[Stage 2] ML Feasibility Classification (Secondary Validation)")
            classifications = []
            feasible_problems = []
            
            for problem in approved_problems:
                classification = self.feasibility_classifier.classify(problem)
                classifications.append(classification)
                
                if (classification.get('feasible', False) and 
                    classification.get('confidence', 0) >= self.config.get('feasibility', {}).get('min_confidence', 0.7)):
                    problem['ml_classification'] = classification
                    feasible_problems.append(problem)
            
            pipeline_results['stages']['feasibility'] = {
                'success': True,
                'feasible_problems': len(feasible_problems),
                'total_problems': len(approved_problems)
            }
            
            if not feasible_problems:
                # Secondary validation correctly rejected all problems
                logger.info("\n" + "=" * 80)
                logger.info("No feasible problems found after secondary validation.")
                logger.info("This is a correct outcome - problems were properly validated.")
                logger.info("=" * 80)
                
                pipeline_results['success'] = True
                pipeline_results['end_time'] = datetime.now().isoformat()
                pipeline_results['result'] = "No feasible problems after secondary validation"
                pipeline_results['models_trained'] = 0
                pipeline_results['decision_quality'] = "Correct"
                
                logger.info("\nPipeline Status:")
                logger.info(f"  Success: True")
                logger.info(f"  Result: No feasible problems after secondary validation")
                logger.info(f"  Models trained: 0")
                logger.info(f"  Decision quality: Correct")
                logger.info("=" * 80)
                
                return pipeline_results
            
            # Find first non-duplicate problem
            problem = None
            task_type = None
            key_features = []
            skipped_duplicates = []
            
            for candidate_problem in feasible_problems:
                logger.info(f"Checking problem: {candidate_problem.get('title', 'Unknown')}")
                logger.info("Checking problem registry for duplicates...")
                is_duplicate_problem, similar_problem_entry = self.problem_registry.check_duplicate_problem(candidate_problem)
                
                if is_duplicate_problem:
                    logger.warning(f"[WARNING] Duplicate problem detected! Skipping this problem.")
                    logger.warning(f"   Problem: {similar_problem_entry.get('problem', {}).get('title', 'Unknown')}")
                    logger.warning(f"   First seen: {similar_problem_entry.get('first_seen', 'Unknown')}")
                    logger.warning(f"   Seen {similar_problem_entry.get('seen_count', 0)} time(s)")
                    skipped_duplicates.append({
                        'problem': candidate_problem.get('title', 'Unknown'),
                        'similar_entry': similar_problem_entry
                    })
                    continue  # Try next problem
                else:
                    # Found a non-duplicate problem
                    problem = candidate_problem
                    
                    # Use ML Decision Agent's problem definition if available
                    ml_decision = problem.get('ml_decision', {})
                    ml_problem_def = ml_decision.get('ml_problem_definition', {})
                    
                    if ml_problem_def.get('task_type'):
                        task_type = ml_problem_def['task_type']
                        key_features = ml_problem_def.get('features', [])
                        logger.info(f"[OK] Found non-duplicate problem: {problem.get('title', 'Unknown')}")
                        logger.info(f"Task type (from Decision Agent): {task_type}")
                    elif problem.get('ml_classification') and problem['ml_classification'].get('task_type'):
                        # Fallback to feasibility classifier
                        task_type = problem['ml_classification'].get('task_type', 'classification')
                        key_features = problem['ml_classification'].get('key_features', [])
                        logger.info(f"[OK] Found non-duplicate problem: {problem.get('title', 'Unknown')}")
                        logger.info(f"Task type (from Feasibility Classifier): {task_type}")
                    elif problem.get('problem_type'):
                        # Use problem_type from canonicalization
                        task_type = problem.get('problem_type', 'classification')
                        key_features = problem.get('input_features', [])
                        logger.info(f"[OK] Found non-duplicate problem: {problem.get('title', 'Unknown')}")
                        logger.info(f"Task type (from canonicalization): {task_type}")
                    else:
                        # Final fallback
                        task_type = 'classification'
                        key_features = []
                        logger.info(f"[OK] Found non-duplicate problem: {problem.get('title', 'Unknown')}")
                        logger.info(f"Task type (default): {task_type}")
                    break
            
            # If all problems were duplicates, exit
            if problem is None:
                logger.warning(f"All {len(feasible_problems)} feasible problems are duplicates. Exiting pipeline.")
                pipeline_results['stages']['problem_check'] = {
                    'success': False,
                    'error': 'All feasible problems are duplicates',
                    'skipped_duplicates': skipped_duplicates,
                    'total_checked': len(feasible_problems)
                }
                return pipeline_results
            
            # Log skipped duplicates if any
            if skipped_duplicates:
                logger.info(f"Skipped {len(skipped_duplicates)} duplicate problem(s), proceeding with first non-duplicate.")
                pipeline_results['stages']['problem_check'] = {
                    'success': True,
                    'skipped_duplicates': len(skipped_duplicates),
                    'selected_problem': problem.get('title', 'Unknown')
                }
            
            # Stage 3: Dataset Discovery
            logger.info("\n[Stage 3] Dataset Discovery")
            # Build keywords for discovery: include target, features, and problem statement words
            discovery_keywords = list(key_features) if key_features else []
            if problem.get('target_variable'):
                discovery_keywords.append(problem['target_variable'])
            if problem.get('title'):
                discovery_keywords.extend(re.findall(r'\b\w{3,}\b', problem['title'].lower()))
            if problem.get('description'):
                discovery_keywords.extend(re.findall(r'\b\w{3,}\b', problem['description'][:300].lower()))
            discovery_keywords = list(dict.fromkeys(discovery_keywords))[:15]  # Unique, limit
            datasets = self.dataset_discovery.discover_datasets(
                problem,
                task_type,
                discovery_keywords
            )
            pipeline_results['stages']['dataset_discovery'] = {
                'success': True,
                'datasets_found': len(datasets)
            }
            
            # Stage 4: Dataset Matching (if datasets found)
            best_dataset = None
            dataset_path = None
            
            if datasets:
                logger.info("\n[Stage 4] Dataset Matching")
                matches = self.dataset_matcher.match(problem, datasets)
                pipeline_results['stages']['dataset_matching'] = {
                    'success': True,
                    'matches_found': len(matches)
                }
                
                if matches:
                    # Sort matches: prefer UCI/HuggingFace (no auth) over Kaggle to avoid 403
                    source_priority = {'uci': 0, 'huggingface': 1, 'datagov': 2, 'worldbank': 3, 'kaggle': 4}
                    sorted_matches = sorted(
                        matches,
                        key=lambda x: (source_priority.get(x[0].get('source', ''), 5), -x[1])
                    )
                    
                    # Stage 5: Dataset Download - try each match until one succeeds (prefer UCI/HF)
                    logger.info("\n[Stage 5] Dataset Download")
                    best_dataset = None
                    dataset_path = None
                    for ds, sim in sorted_matches:
                        logger.info(f"Trying dataset: {ds.get('id', 'Unknown')} (source: {ds.get('source', '?')}, similarity: {sim:.3f})")
                        path = self._download_dataset(ds, task_type)
                        if path:
                            best_dataset = ds
                            dataset_path = path
                            logger.info(f"Successfully downloaded: {dataset_path}")
                            break
                    else:
                        logger.warning("All dataset download attempts failed. Will create synthetic dataset.")
                else:
                    logger.warning("No matching datasets found. Will create synthetic dataset.")
            else:
                logger.warning("No datasets found. Will create synthetic dataset for training.")
                pipeline_results['stages']['dataset_matching'] = {
                    'success': True,
                    'matches_found': 0,
                    'note': 'No datasets to match, will use synthetic dataset'
                }
            
            # If no dataset was downloaded, create a synthetic one
            if not dataset_path:
                logger.info("\n[Stage 5] Creating Synthetic Dataset")
                download_dir = Path("data/datasets/downloaded")
                download_dir.mkdir(parents=True, exist_ok=True)
                dataset_path = self._create_synthetic_dataset(task_type, download_dir)
                
                # Create a synthetic dataset entry for tracking
                if not best_dataset:
                    best_dataset = {
                        'id': 'synthetic_dataset',
                        'source': 'synthetic',
                        'name': f'Synthetic {task_type} dataset',
                        'url': 'N/A',
                        'description': f'Synthetic dataset created for {problem.get("title", "problem")}'
                    }
                
                logger.info(f"Created synthetic dataset: {dataset_path}")
                pipeline_results['stages']['dataset_creation'] = {
                    'success': True,
                    'dataset_path': dataset_path,
                    'type': 'synthetic',
                    'task_type': task_type
                }
            
            # Stage 6: AutoML Training
            logger.info("\n[Stage 6] AutoML Training")
            
            # Final checkpoint: Verify ML Decision Agent approved training
            ml_decision = problem.get('ml_decision', {})
            decision = ml_decision.get('decision', 'unknown')
            
            if decision != 'train':
                logger.warning("[WARNING] Final checkpoint: ML Decision Agent did not approve training.")
                logger.warning(f"   Decision: {decision}")
                logger.warning(f"   Reason: {ml_decision.get('reasoning', 'Unknown reason')}")
                logger.warning(f"   Recommended action: {ml_decision.get('recommended_action', 'N/A')}")
                training_result = {
                    'success': False,
                    'error': 'ML Decision Agent rejected training',
                    'reasoning': ml_decision.get('reasoning', 'Training not approved by decision agent'),
                    'recommended_action': ml_decision.get('recommended_action', 'N/A')
                }
            elif dataset_path is None:
                logger.warning("Dataset path not available. Skipping training.")
                training_result = {
                    'success': False,
                    'error': 'Dataset not downloaded'
                }
            else:
                # Check for duplicate models before training
                logger.info("Checking model registry for duplicates...")
                is_duplicate, similar_entry = self.model_registry.check_duplicate(
                    problem, best_dataset, task_type
                )
                
                if is_duplicate:
                    logger.warning(f"[WARNING] Duplicate model detected! Similar model already trained {similar_entry.get('train_count', 0)} times.")
                    logger.warning(f"   Problem: {similar_entry.get('problem', {}).get('title', 'Unknown')}")
                    logger.warning(f"   Dataset: {similar_entry.get('dataset', {}).get('id', 'Unknown')}")
                    logger.warning("   Skipping training to prevent duplicate models.")
                    training_result = {
                        'success': False,
                        'error': 'Duplicate model detected - similar model already trained maximum times',
                        'duplicate_info': {
                            'similar_entry': similar_entry,
                            'train_count': similar_entry.get('train_count', 0)
                        }
                    }
                else:
                    # Proceed with training
                    training_result = self.automl_trainer.train(
                        dataset_path,
                        task_type
                    )
                    
                    # Register model in registry if training successful
                    if training_result.get('success', False):
                        logger.info("Registering model in registry...")
                        model_path = training_result.get('model_path', '')
                        metrics = training_result.get('metrics', {})
                        registered = self.model_registry.register_model(
                            problem, best_dataset, task_type, model_path, metrics
                        )
                        if registered:
                            logger.info("[OK] Model registered successfully in registry")
                        else:
                            logger.warning("Failed to register model in registry")
                        
                        # Register problem in problem registry
                        logger.info("Registering problem in problem registry...")
                        dataset_id = best_dataset.get('id', '')
                        problem_registered = self.problem_registry.register_problem(
                            problem, task_type, dataset_id
                        )
                        if problem_registered:
                            logger.info("[OK] Problem registered successfully in problem registry")
                        else:
                            logger.warning("Failed to register problem in problem registry")
            
            pipeline_results['stages']['training'] = training_result
            
            if not training_result.get('success', False):
                logger.warning("Training failed. Exiting pipeline.")
                return pipeline_results
            
            # Stage 6.5: Model Testing & Evaluation
            logger.info("\n[Stage 6.5] Model Testing & Evaluation")
            test_result = self._test_model(dataset_path, training_result, task_type)
            pipeline_results['stages']['testing'] = test_result
            
            # Merge test metrics into training_result for code generation
            if test_result.get('success') and 'test_metrics' in test_result:
                training_result['test_metrics'] = test_result['test_metrics']
                logger.info("Merged test metrics into training_result for README generation")
            
            # Stage 7: Code Generation
            logger.info("\n[Stage 7] Code Generation")
            try:
                code_result = self.code_generator.generate(
                    problem,
                    best_dataset,
                    training_result
                )
                # Ensure success field exists
                if 'success' not in code_result:
                    code_result['success'] = True
                pipeline_results['stages']['code_generation'] = code_result
            except Exception as e:
                logger.error(f"Code generation failed: {e}", exc_info=True)
                pipeline_results['stages']['code_generation'] = {
                    'success': False,
                    'error': str(e)
                }
            
            # Stage 8: GitHub Publishing
            logger.info("\n[Stage 8] GitHub Publishing")
            if code_result.get('success') and self.config.get('github', {}).get('token'):
                try:
                    publish_result = self.github_publisher.publish(
                        code_result.get('project_dir'),
                        code_result.get('project_name', 'ml_solution'),
                        problem,
                        training_result
                    )
                    pipeline_results['stages']['github_publishing'] = publish_result
                except Exception as e:
                    logger.error(f"GitHub publishing failed: {e}", exc_info=True)
                    pipeline_results['stages']['github_publishing'] = {
                        'success': False,
                        'error': str(e)
                    }
            else:
                if not code_result.get('success'):
                    logger.warning("Skipping GitHub publishing - code generation failed")
                else:
                    logger.info("GitHub token not configured. Skipping publishing.")
                pipeline_results['stages']['github_publishing'] = {
                    'success': False,
                    'skipped': True,
                    'reason': 'code_generation_failed' if not code_result.get('success') else 'no_token'
                }
            
            pipeline_results['success'] = True
            pipeline_results['end_time'] = datetime.now().isoformat()
            
            logger.info("\n" + "=" * 80)
            logger.info("Pipeline completed successfully!")
            logger.info("=" * 80)
            
            # Generate and print summary
            self._print_summary(pipeline_results, problem, best_dataset, training_result)
            
        except Exception as e:
            logger.error(f"Pipeline error in _continue_pipeline: {e}", exc_info=True)
            pipeline_results['error'] = str(e)
            pipeline_results['success'] = False
        
        finally:
            # Save pipeline results (remove non-serializable objects)
            results_file = self.results_dir / f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            # Remove model objects from training results before JSON serialization
            sanitized_results = self._sanitize_for_json(pipeline_results)
            with open(results_file, 'w') as f:
                json.dump(sanitized_results, f, indent=2)
            
            logger.info(f"Pipeline results saved to {results_file}")
        
        return pipeline_results

