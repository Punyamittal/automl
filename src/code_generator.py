"""
Code Generator Module

Generates clean, runnable code and documentation for ML solutions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class CodeGenerator:
    """Generates code and documentation for ML solutions."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.include_examples = config.get('include_examples', True)
        self.code_dir = Path("outputs/code")
        self.code_dir.mkdir(parents=True, exist_ok=True)
        
        # Load README generation prompt
        prompt_file = Path("prompts/readme_generation_prompt.txt")
        if prompt_file.exists():
            self.readme_prompt_template = prompt_file.read_text()
        else:
            self.readme_prompt_template = self._default_readme_prompt()
    
    def generate(self, problem: Dict, dataset: Dict, training_result: Dict) -> Dict:
        """Generate complete code package."""
        try:
            project_name = self._sanitize_name(problem.get('title', 'ml_solution')[:50])
            project_dir = self.code_dir / project_name
            project_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Generating code for project: {project_name}")
            
            # Generate files
            files_generated = {}
            
            # Generate train.py
            try:
                train_code = self._generate_train_code(problem, dataset, training_result)
                train_path = project_dir / "train.py"
                train_path.write_text(train_code, encoding='utf-8')
                files_generated['train.py'] = str(train_path)
            except Exception as e:
                logger.error(f"Error generating train.py: {e}")
                raise
            
            # Generate predict.py
            try:
                predict_code = self._generate_predict_code(training_result)
                predict_path = project_dir / "predict.py"
                predict_path.write_text(predict_code, encoding='utf-8')
                files_generated['predict.py'] = str(predict_path)
            except Exception as e:
                logger.error(f"Error generating predict.py: {e}")
                raise
            
            # Generate requirements.txt
            try:
                requirements = self._generate_requirements()
                req_path = project_dir / "requirements.txt"
                req_path.write_text(requirements, encoding='utf-8')
                files_generated['requirements.txt'] = str(req_path)
            except Exception as e:
                logger.error(f"Error generating requirements.txt: {e}")
                raise
            
            # Generate README.md
            try:
                readme = self._generate_readme(problem, dataset, training_result)
                readme_path = project_dir / "README.md"
                readme_path.write_text(readme, encoding='utf-8')
                files_generated['README.md'] = str(readme_path)
            except Exception as e:
                logger.error(f"Error generating README.md: {e}")
                raise
            
            # Copy model file if it exists
            model_path = training_result.get('model_path')
            if model_path and Path(model_path).exists():
                try:
                    import shutil
                    model_filename = Path(model_path).name
                    dest_model_path = project_dir / model_filename
                    shutil.copy(model_path, dest_model_path)
                    files_generated['model'] = str(dest_model_path)
                except Exception as e:
                    logger.warning(f"Could not copy model file: {e}")
                    # Don't fail if model copy fails
            
            logger.info(f"Generated {len(files_generated)} files in {project_dir}")
            
            return {
                'success': True,
                'project_name': project_name,
                'project_dir': str(project_dir),
                'files': list(files_generated.keys()),
                'files_dict': files_generated  # Keep full dict for compatibility
            }
            
        except Exception as e:
            logger.error(f"Error in code generation: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'project_name': None,
                'project_dir': None,
                'files': []
            }
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for use as directory/filename."""
        import re
        # Remove special characters, keep alphanumeric and spaces
        name = re.sub(r'[^\w\s-]', '', name)
        # Replace spaces with underscores
        name = re.sub(r'[-\s]+', '_', name)
        # Lowercase
        name = name.lower()
        # Remove leading/trailing underscores
        name = name.strip('_')
        return name[:50]
    
    def _generate_train_code(self, problem: Dict, dataset: Dict, training_result: Dict) -> str:
        """Generate train.py code."""
        task_type = training_result.get('task_type', 'classification')
        model_path = training_result.get('model_path', 'model.pkl')
        model_filename = Path(model_path).name
        framework = training_result.get('framework', 'sklearn')  # Default to sklearn for Python 3.12+
        
        # Check if PyCaret was used (has 'pycaret' in model path or metrics)
        use_pycaret = 'pycaret' in str(training_result.get('model_path', '')).lower() or framework == 'pycaret'
        
        if use_pycaret:
            code = f'''"""
Training script for {problem.get('title', 'ML Model')}

This script trains a {task_type} model using PyCaret.
"""

import pandas as pd
from pycaret.{task_type} import *

# Load dataset
# TODO: Update this path to your dataset
df = pd.read_csv('data.csv')

# Set target column
# TODO: Update this to your target column name
target_column = '{training_result.get("metrics", {}).get("target", "target")}'

# Setup PyCaret
setup(
    data=df,
    target=target_column,
    session_id=42,
    silent=True
)

# Create and compare models
best_model = compare_models(
    include=['rf', 'gbc', 'lr', 'knn'],
    sort='Accuracy' if task_type == 'classification' else 'R2',
    n_select=1
)

# Finalize model
final_model = finalize_model(best_model)

# Save model
save_model(final_model, '{model_filename}')

print(f"Model trained and saved to {model_filename}")
print("Model metrics:")
print(pull())
'''
        else:
            # scikit-learn version
            code = f'''"""
Training script for {problem.get('title', 'ML Model')}

This script trains a {task_type} model using scikit-learn.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score

# Load dataset
# TODO: Update this path to your dataset
df = pd.read_csv('data.csv')

# Set target column
# TODO: Update this to your target column name
target_column = '{training_result.get("metrics", {}).get("target", "target")}'

# Prepare data
X = df.drop(columns=[target_column])
y = df[target_column]

# Handle categorical features
X = pd.get_dummies(X, drop_first=True)
X = X.fillna(X.mean())

# Encode target for classification
if '{task_type}' == 'classification' and y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
# TODO: Choose your model (RandomForest, GradientBoosting, LogisticRegression, etc.)
if '{task_type}' == 'classification':
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train_scaled, y_train)

# Evaluate
if '{task_type}' == 'classification':
    y_pred = model.predict(X_test_scaled)
    score = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {{score:.3f}}")
else:
    y_pred = model.predict(X_test_scaled)
    score = r2_score(y_test, y_pred)
    print(f"R² Score: {{score:.3f}}")

# Save model and scaler
with open('{model_filename}', 'wb') as f:
    pickle.dump({{
        'model': model,
        'scaler': scaler,
        'label_encoder': le if '{task_type}' == 'classification' and 'le' in locals() else None
    }}, f)

print(f"Model trained and saved to {model_filename}")
'''
        return code
    
    def _generate_predict_code(self, training_result: Dict) -> str:
        """Generate predict.py code."""
        task_type = training_result.get('task_type', 'classification')
        model_path = training_result.get('model_path', 'model.pkl')
        model_filename = Path(model_path).name
        framework = training_result.get('framework', 'sklearn')
        use_pycaret = 'pycaret' in str(model_path).lower() or framework == 'pycaret'
        
        if use_pycaret:
            code = f'''"""
Prediction script for {task_type} model

This script loads a trained model and makes predictions on new data.
"""

import pandas as pd
from pycaret.{task_type} import load_model, predict_model

# Load the trained model
model = load_model('{model_filename}')

# Load new data for prediction
# TODO: Update this path to your new data
new_data = pd.read_csv('new_data.csv')

# Make predictions
predictions = predict_model(model, data=new_data)

# Save predictions
predictions.to_csv('predictions.csv', index=False)

print("Predictions saved to predictions.csv")
print("\\nFirst few predictions:")
print(predictions.head())
'''
        else:
            code = f'''"""
Prediction script for {task_type} model

This script loads a trained model and makes predictions on new data.
"""

import pandas as pd
import pickle

# Load the trained model
with open('{model_filename}', 'rb') as f:
    saved_data = pickle.load(f)
    model = saved_data['model']
    scaler = saved_data['scaler']
    label_encoder = saved_data.get('label_encoder')

# Load new data for prediction
# TODO: Update this path to your new data
new_data = pd.read_csv('new_data.csv')

# Prepare data (same preprocessing as training)
X = pd.get_dummies(new_data, drop_first=True)
X = X.fillna(X.mean())

# Scale features
X_scaled = scaler.transform(X)

# Make predictions
predictions = model.predict(X_scaled)

# Decode if classification with label encoder
if label_encoder is not None:
    predictions = label_encoder.inverse_transform(predictions)

# Save predictions
result_df = pd.DataFrame({{
    'predictions': predictions
}})
result_df.to_csv('predictions.csv', index=False)

print("Predictions saved to predictions.csv")
print("\\nFirst few predictions:")
print(result_df.head())
'''
        return code
    
    def _generate_requirements(self) -> str:
        """Generate requirements.txt."""
        # Check if we should include PyCaret or just scikit-learn
        return '''pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
# Note: PyCaret is optional (only needed for Python 3.9-3.11)
# For Python 3.12+, scikit-learn is used instead
'''
    
    def _generate_readme(self, problem: Dict, dataset: Dict, training_result: Dict) -> str:
        """Generate README.md."""
        problem_title = problem.get('title', 'ML Solution')
        problem_desc = problem.get('description', '')[:500]
        dataset_name = dataset.get('title', dataset.get('id', 'Unknown Dataset'))
        dataset_url = dataset.get('url', '')
        task_type = training_result.get('task_type', 'classification')
        metrics = training_result.get('metrics', {})
        
        # Extract problem understanding from ML classification
        ml_classification = problem.get('ml_classification', {})
        confidence = ml_classification.get('confidence', 0)
        key_features = ml_classification.get('key_features', [])
        task_type_understood = ml_classification.get('task_type', task_type)
        
        # Extract solution details
        best_model = metrics.get('best_model', 'Unknown')
        all_results = metrics.get('all_results', {})
        
        # Format metrics
        metrics_text = ""
        if task_type == 'classification':
            accuracy = metrics.get('accuracy', 0)
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            f1 = metrics.get('f1_score', 0)
            metrics_text = f"""- **Accuracy**: {accuracy:.3f}
- **Precision**: {precision:.3f}
- **Recall**: {recall:.3f}
- **F1 Score**: {f1:.3f}"""
        elif task_type == 'regression':
            r2 = metrics.get('r2', 0)
            mse = metrics.get('mse', 0)
            metrics_text = f"""- **R² Score**: {r2:.3f}
- **Mean Squared Error**: {mse:.3f}"""
        elif task_type == 'clustering':
            n_clusters = metrics.get('n_clusters', 0)
            silhouette = metrics.get('silhouette_score', 0)
            metrics_text = f"""- **Number of Clusters**: {n_clusters}
- **Silhouette Score**: {silhouette:.3f}"""
        
        # Generate problem understanding section
        problem_understanding = f"""The system analyzed the problem statement and identified it as a **{task_type_understood}** task with {confidence:.0%} confidence.

**Problem Analysis:**
- **Task Type**: {task_type_understood.capitalize()}
- **Confidence Level**: {confidence:.0%}
- **Key Features Identified**: {', '.join(key_features[:5]) if key_features else 'Automatically extracted from dataset'}"""

        # Generate solution approach section
        if task_type == 'classification':
            solution_approach = f"""The machine learning model solves this problem by:

1. **Problem Formulation**: Treats this as a {task_type} task where the model learns to predict categories/classes based on input features.

2. **Model Selection**: After evaluating multiple algorithms, **{best_model}** was selected as the best-performing model.

3. **How It Works**:
   - The model learns patterns from historical data
   - It maps input features to output classes/categories
   - Given new data, it predicts the most likely class
   - The model uses learned decision boundaries to make predictions

4. **Solution Capability**: The model can automatically classify new instances into the appropriate category based on the patterns it learned during training."""
        elif task_type == 'regression':
            solution_approach = f"""The machine learning model solves this problem by:

1. **Problem Formulation**: Treats this as a {task_type} task where the model learns to predict continuous numerical values.

2. **Model Selection**: After evaluating multiple algorithms, **{best_model}** was selected as the best-performing model.

3. **How It Works**:
   - The model learns the relationship between input features and target values
   - It finds patterns and correlations in the training data
   - Given new data, it predicts a continuous numerical value
   - The model uses regression techniques to estimate outcomes

4. **Solution Capability**: The model can predict numerical values for new instances based on the relationships it learned during training."""
        else:  # clustering
            solution_approach = f"""The machine learning model solves this problem by:

1. **Problem Formulation**: Treats this as a {task_type} task where the model groups similar data points together.

2. **Model Selection**: After evaluating clustering algorithms, **{best_model}** was selected.

3. **How It Works**:
   - The model identifies patterns and similarities in the data
   - It groups data points into clusters based on feature similarity
   - Each cluster represents a distinct group with common characteristics
   - The model discovers hidden patterns and structures in the data

4. **Solution Capability**: The model can automatically group new data points into appropriate clusters based on similarity."""
        
        readme = f'''# {problem_title}

## Problem Description

{problem_desc}

**Source**: {problem.get('source', 'Unknown')}  
**Original URL**: {problem.get('url', 'N/A')}

## Problem Understanding

{problem_understanding}

## Dataset

- **Name**: {dataset_name}
- **Source**: {dataset.get('source', 'Unknown')}
- **URL**: {dataset_url if dataset_url else 'N/A'}

## ML Solution Approach

{solution_approach}

## Model Performance

{metrics_text}

## Model Details

- **Best Model**: {best_model}
- **Framework**: {training_result.get('framework', 'scikit-learn').upper()}
- **Training Date**: {datetime.now().strftime("%Y-%m-%d")}
- **Model File**: `{Path(training_result.get('model_path', 'model.pkl')).name}`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

1. Place your dataset as `data.csv`
2. Update the target column name in `train.py`
3. Run:
```bash
python train.py
```

### Prediction

1. Place new data as `new_data.csv`
2. Run:
```bash
python predict.py
```

Predictions will be saved to `predictions.csv`.

## Limitations

- This is an automated solution and may require manual tuning
- Model performance depends on data quality
- Additional feature engineering may improve results

## License

This project is generated by an automated ML pipeline. Use at your own discretion.

## Generated By

Autonomous ML Automation Pipeline
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
'''
        return readme
    
    def _default_readme_prompt(self) -> str:
        """Default README prompt template."""
        return "Generate a comprehensive README.md for this ML project."

