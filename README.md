# Autonomous ML Automation Pipeline

An end-to-end automated machine learning system that discovers real-world problems, evaluates ML feasibility, finds datasets, trains models, and publishes solutions to GitHub.

## System Architecture

```
┌─────────────────┐
│  Problem Miner  │ → Scrapes Reddit/StackOverflow for problems
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│ ML Feasibility       │ → Evaluates if ML can solve the problem
│ Classifier           │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ Dataset Discovery    │ → Searches Kaggle/HuggingFace/UCI
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ Dataset-Problem      │ → Matches problems with datasets
│ Matcher              │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ AutoML Trainer       │ → Trains models using PyCaret
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ Code Generator       │ → Generates train.py, predict.py, README
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ GitHub Publisher     │ → Creates repo and pushes code
└──────────────────────┘
```

## Project Structure

```
auto/
├── src/
│   ├── __init__.py
│   ├── problem_miner.py
│   ├── feasibility_classifier.py
│   ├── dataset_discovery.py
│   ├── dataset_matcher.py
│   ├── automl_trainer.py
│   ├── code_generator.py
│   ├── github_publisher.py
│   └── orchestrator.py
├── prompts/
│   ├── feasibility_prompt.txt
│   ├── dataset_matching_prompt.txt
│   └── readme_generation_prompt.txt
├── config/
│   └── config.yaml
├── data/
│   ├── problems/
│   ├── datasets/
│   └── embeddings/
├── outputs/
│   ├── models/
│   ├── code/
│   └── logs/
├── requirements.txt
├── main.py
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy `config/config.yaml.example` to `config/config.yaml`
2. Set your GitHub token (get free token from GitHub Settings > Developer settings)
3. Optionally set HuggingFace token for better API access

## Usage

```bash
python main.py
```

The system will:
1. Discover problems from forums
2. Evaluate ML feasibility
3. Find matching datasets
4. Train models
5. Generate code
6. Publish to GitHub

## Module Details

See individual module documentation in `src/` directory.

# automl
