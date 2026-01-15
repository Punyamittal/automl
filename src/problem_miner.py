"""
Problem Miner Module

Discovers real-world ML problems from Kaggle and GitHub.
Prioritizes Kaggle competitions/datasets and GitHub issues.
"""

import json
import hashlib
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Try to import miners
try:
    from .kaggle_problem_miner import KaggleProblemMiner
    KAGGLE_MINER_AVAILABLE = True
except ImportError:
    KAGGLE_MINER_AVAILABLE = False
    logger.warning("Kaggle problem miner not available.")

try:
    from .github_problem_miner import GitHubProblemMiner
    GITHUB_MINER_AVAILABLE = True
except ImportError:
    GITHUB_MINER_AVAILABLE = False
    logger.warning("GitHub problem miner not available.")

# Try to import Gemini client
try:
    from .gemini_client import GeminiClient
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Gemini client not available.")


class ProblemMiner:
    """Mines problems from Kaggle and GitHub (prioritized sources)."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.min_length = config.get('min_problem_length', 50)
        self.max_problems = config.get('max_problems_per_run', 10)
        self.problems_dir = Path("data/problems")
        self.problems_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Kaggle miner (PRIMARY)
        self.kaggle_miner = None
        kaggle_config = config.get('kaggle_problem_mining', {})
        if kaggle_config.get('enabled', True) and KAGGLE_MINER_AVAILABLE:
            try:
                self.kaggle_miner = KaggleProblemMiner(kaggle_config)
                logger.info("Kaggle problem miner initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Kaggle miner: {e}")
        
        # Initialize GitHub miner (SECONDARY)
        self.github_miner = None
        github_config = config.get('github_problem_mining', {})
        if github_config.get('enabled', True) and GITHUB_MINER_AVAILABLE:
            try:
                # Pass GitHub config from main config
                github_config['github'] = config.get('github', {})
                self.github_miner = GitHubProblemMiner(github_config)
                logger.info("GitHub problem miner initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize GitHub miner: {e}")
        
        # Initialize Gemini client if available and configured
        self.gemini_client = None
        gemini_config = config.get('gemini', {})
        if gemini_config.get('enabled', False) and GEMINI_AVAILABLE:
            try:
                api_key = gemini_config.get('api_key') or config.get('llm', {}).get('gemini_api_key')
                model_name = gemini_config.get('model_name', 'gemini-pro')
                self.gemini_client = GeminiClient(api_key=api_key, model_name=model_name)
                logger.info("Gemini client initialized for problem analysis")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}. Continuing without Gemini analysis.")
        
    def mine_problems(self) -> List[Dict]:
        """Mine problems from Kaggle (primary) and GitHub (secondary)."""
        all_problems = []
        
        # Priority 1: Kaggle (PRIMARY)
        if self.kaggle_miner:
            logger.info("Mining problems from Kaggle (PRIMARY)...")
            kaggle_problems = self.kaggle_miner.mine_problems()
            all_problems.extend(kaggle_problems)
        
        # Priority 2: GitHub (SECONDARY)
        if self.github_miner:
            logger.info("Mining problems from GitHub (SECONDARY)...")
            github_problems = self.github_miner.mine_problems()
            all_problems.extend(github_problems)
        
        # Analyze problems with Gemini if enabled
        if self.gemini_client and all_problems:
            logger.info("Analyzing problems with Gemini...")
            analyzed_problems = []
            for problem in all_problems:
                try:
                    title = problem.get('title', '')
                    description = problem.get('description', '')
                    
                    gemini_analysis = self.gemini_client.analyze_reddit_post(title, description)
                    
                    # Only include if Gemini approves
                    if gemini_analysis.get('is_ml_problem', False):
                        confidence = gemini_analysis.get('confidence', 0.0)
                        if confidence >= 0.6:  # Require at least 60% confidence
                            # Use structured ML problem details if available
                            ml_problem_details = gemini_analysis.get('ml_problem_details')
                            if ml_problem_details:
                                problem['ml_problem_details'] = ml_problem_details
                                # Update title and description with Gemini-generated content
                                if ml_problem_details.get('title'):
                                    problem['title'] = ml_problem_details['title']
                                if ml_problem_details.get('problem_statement'):
                                    problem['description'] = ml_problem_details['problem_statement']
                            
                            problem['gemini_analysis'] = gemini_analysis
                            analyzed_problems.append(problem)
                except Exception as e:
                    logger.warning(f"Gemini analysis failed for problem: {e}")
                    # Include problem even if Gemini fails
                    analyzed_problems.append(problem)
            
            all_problems = analyzed_problems
        
        # Deduplicate
        unique_problems = self._deduplicate_problems(all_problems)
        
        # Filter by quality
        filtered_problems = self._filter_problems(unique_problems)
        
        # Limit results
        final_problems = filtered_problems[:self.max_problems]
        
        logger.info(f"Mined {len(final_problems)} unique problems from Kaggle and GitHub")
        return final_problems
    
    def _deduplicate_problems(self, problems: List[Dict]) -> List[Dict]:
        """Remove duplicate problems based on content similarity."""
        seen_hashes = set()
        unique_problems = []
        
        for problem in problems:
            # Create hash from title and first 200 chars of description
            content = f"{problem['title']}{problem.get('description', '')[:200]}"
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_problems.append(problem)
        
        return unique_problems
    
    def _filter_problems(self, problems: List[Dict]) -> List[Dict]:
        """Filter problems by quality criteria."""
        filtered = []
        
        for problem in problems:
            # Must have minimum length
            if len(problem.get('full_text', '')) < self.min_length:
                continue
            
            # Must have title and description
            if not problem.get('title') or not problem.get('description'):
                continue
            
            filtered.append(problem)
        
        # Sort by source priority: Kaggle first, then GitHub
        filtered.sort(key=lambda x: (
            0 if x.get('source') == 'kaggle' else 1,  # Kaggle first
            x.get('usability', 0) if x.get('source') == 'kaggle' else 0,
            x.get('stars', 0) if x.get('source') == 'github' else 0
        ))
        
        return filtered
    
    def save_problems(self, problems: List[Dict], filename: Optional[str] = None):
        """Save problems to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"problems_{timestamp}.json"
        
        filepath = self.problems_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(problems, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(problems)} problems to {filepath}")
        return filepath
