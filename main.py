"""
Main entry point for Autonomous ML Automation Pipeline
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator import Orchestrator

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Autonomous ML Pipeline')
    parser.add_argument(
        '--problem',
        type=str,
        help='Direct problem statement to solve (skips problem mining)'
    )
    parser.add_argument(
        '--problem-file',
        type=str,
        help='Path to file containing problem statement'
    )
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = Orchestrator()
    
    # Check if Gemini is enabled
    gemini_config = orchestrator.config.get('gemini', {})
    if gemini_config.get('enabled', False):
        print("=" * 80)
        print("Gemini API: ENABLED")
        print(f"  Model: {gemini_config.get('model_name', 'gemini-pro')}")
        print(f"  Min Confidence: {gemini_config.get('min_confidence', 0.6)}")
        print("=" * 80)
        print()
    
    # Check for direct problem input
    problem_statement = None
    if args.problem:
        problem_statement = args.problem
        print("=" * 80)
        print("DIRECT PROBLEM MODE")
        print("=" * 80)
        print(f"Problem: {problem_statement[:100]}...")
        print("=" * 80)
        print()
    elif args.problem_file:
        problem_file = Path(args.problem_file)
        if problem_file.exists():
            problem_statement = problem_file.read_text(encoding='utf-8').strip()
            print("=" * 80)
            print("DIRECT PROBLEM MODE (from file)")
            print("=" * 80)
            print(f"Problem: {problem_statement[:100]}...")
            print("=" * 80)
            print()
        else:
            print(f"Error: Problem file not found: {args.problem_file}")
            sys.exit(1)
    else:
        # Check config for direct problem
        direct_problem = orchestrator.config.get('direct_problem', {})
        if direct_problem.get('enabled', False) and direct_problem.get('statement'):
            problem_statement = direct_problem.get('statement')
            print("=" * 80)
            print("DIRECT PROBLEM MODE (from config)")
            print("=" * 80)
            print(f"Problem: {problem_statement[:100]}...")
            print("=" * 80)
            print()
    
    # Run pipeline with or without direct problem
    if problem_statement:
        results = orchestrator.run_with_problem(problem_statement)
    else:
        results = orchestrator.run()
    
    # Print summary
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    print(f"Success: {results.get('success', False)}")
    
    if results.get('stages'):
        for stage_name, stage_result in results['stages'].items():
            status = "[OK]" if stage_result.get('success', False) else "[FAIL]"
            print(f"{status} {stage_name}")
    
    if results.get('success'):
        print("\nPipeline completed successfully!")
        
        # Show testing results if available
        if 'testing' in results.get('stages', {}):
            test_result = results['stages']['testing']
            if test_result.get('success'):
                test_metrics = test_result.get('test_metrics', {})
                task_type = test_result.get('task_type', 'classification')
                if task_type == 'classification':
                    print(f"\nTest Results:")
                    print(f"  Accuracy: {test_metrics.get('accuracy', 0):.4f}")
                    print(f"  Precision: {test_metrics.get('precision', 0):.4f}")
                    print(f"  Recall: {test_metrics.get('recall', 0):.4f}")
                    print(f"  F1 Score: {test_metrics.get('f1_score', 0):.4f}")
                else:
                    print(f"\nTest Results:")
                    print(f"  R² Score: {test_metrics.get('r2_score', 0):.4f}")
                    print(f"  RMSE: {test_metrics.get('rmse', 0):.4f}")
                    print(f"  MAE: {test_metrics.get('mae', 0):.4f}")
        
        if 'github_publishing' in results.get('stages', {}):
            repo_url = results['stages']['github_publishing'].get('repo_url')
            if repo_url:
                print(f"\nRepository: {repo_url}")
    else:
        print("\nPipeline failed. Check logs for details.")
        sys.exit(1)

