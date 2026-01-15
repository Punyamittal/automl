"""
GitHub Problem Miner Module

Mines ML problems from GitHub issues and repositories.
"""

import logging
import time
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from github import Github
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    logger.warning("PyGithub not available. Install with: pip install PyGithub")


class GitHubProblemMiner:
    """Mines ML problems from GitHub issues and repositories."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.github_config = config.get('github', {})
        self.max_issues = config.get('max_issues', 20)
        self.max_repos = config.get('max_repos', 10)
        
        # ML-related keywords for issue search
        self.ml_keywords = [
            'predict', 'detect', 'classify', 'recommend', 'forecast',
            'anomaly', 'ranking', 'sentiment', 'churn', 'fraud'
        ]
        
        # Repository types to search
        self.repo_types = [
            'saas', 'monitoring', 'analytics', 'devops', 'ml-ops',
            'data-platform', 'recommendation-system'
        ]
        
        self.problems_dir = Path("data/problems")
        self.problems_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize GitHub API
        self.github = None
        token = self.github_config.get('token')
        if token and GITHUB_AVAILABLE:
            try:
                self.github = Github(token)
                # Test authentication
                user = self.github.get_user()
                logger.info(f"GitHub API authenticated as: {user.login}")
            except Exception as e:
                logger.warning(f"Could not authenticate with GitHub API: {e}")
        else:
            if not token:
                logger.warning("GitHub token not provided. GitHub problem mining disabled.")
            if not GITHUB_AVAILABLE:
                logger.warning("PyGithub not installed. GitHub problem mining disabled.")
    
    def mine_problems(self) -> List[Dict]:
        """Mine problems from GitHub issues and repositories."""
        all_problems = []
        
        if not self.github:
            logger.warning("GitHub API not available. Skipping GitHub problem mining.")
            return all_problems
        
        # Mine from issues
        logger.info("Mining problems from GitHub issues...")
        issue_problems = self._mine_issues()
        all_problems.extend(issue_problems)
        
        # Mine from repositories
        logger.info("Mining problems from GitHub repositories...")
        repo_problems = self._mine_repositories()
        all_problems.extend(repo_problems)
        
        logger.info(f"Mined {len(all_problems)} problems from GitHub")
        return all_problems
    
    def _mine_issues(self) -> List[Dict]:
        """Mine problems from GitHub issues."""
        problems = []
        
        if not self.github:
            return problems
        
        try:
            # Search issues with ML keywords
            for keyword in self.ml_keywords[:5]:  # Use top 5 keywords
                try:
                    query = f"{keyword} is:issue is:open"
                    issues = self.github.search_issues(query, sort='updated', order='desc')
                    
                    count = 0
                    for issue in issues:
                        if count >= self.max_issues // len(self.ml_keywords[:5]):
                            break
                        
                        try:
                            # Check if issue describes an ML problem
                            if self._is_ml_issue(issue):
                                problem = {
                                    'id': f"github_issue_{issue.id}",
                                    'source': 'github',
                                    'source_type': 'issue',
                                    'title': issue.title,
                                    'description': issue.body or '',
                                    'full_text': f"{issue.title}\n\n{issue.body or ''}",
                                    'url': issue.html_url,
                                    'mined_at': datetime.now().isoformat(),
                                    'issue_number': issue.number,
                                    'repository': issue.repository.full_name,
                                    'labels': [label.name for label in issue.labels],
                                    'comments_count': issue.comments,
                                    'created_at': issue.created_at.isoformat() if issue.created_at else None
                                }
                                problems.append(problem)
                                count += 1
                            
                            # Rate limiting
                            time.sleep(0.5)
                            
                        except Exception as e:
                            logger.debug(f"Error processing issue {issue.id}: {e}")
                            continue
                    
                    # Delay between keyword searches
                    time.sleep(2)
                    
                except Exception as e:
                    logger.warning(f"Error searching GitHub issues with keyword '{keyword}': {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error mining GitHub issues: {e}")
        
        return problems
    
    def _mine_repositories(self) -> List[Dict]:
        """Mine problems from GitHub repository READMEs and issues."""
        problems = []
        
        if not self.github:
            return problems
        
        try:
            # Search for repositories with ML-related topics
            for repo_type in self.repo_types[:3]:  # Use top 3 types
                try:
                    query = f"topic:{repo_type} language:python"
                    repos = self.github.search_repositories(query, sort='stars', order='desc')
                    
                    count = 0
                    for repo in repos:
                        if count >= self.max_repos // len(self.repo_types[:3]):
                            break
                        
                        try:
                            # Check README for ML problem statements
                            readme_problems = self._extract_from_readme(repo)
                            problems.extend(readme_problems)
                            
                            # Check open issues for ML problems
                            issue_problems = self._extract_from_repo_issues(repo)
                            problems.extend(issue_problems)
                            
                            count += 1
                            
                            # Rate limiting
                            time.sleep(1)
                            
                        except Exception as e:
                            logger.debug(f"Error processing repo {repo.full_name}: {e}")
                            continue
                    
                    # Delay between repo type searches
                    time.sleep(2)
                    
                except Exception as e:
                    logger.warning(f"Error searching repositories with type '{repo_type}': {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error mining GitHub repositories: {e}")
        
        return problems
    
    def _extract_from_readme(self, repo) -> List[Dict]:
        """Extract ML problems from repository README."""
        problems = []
        
        try:
            readme = repo.get_readme()
            readme_content = readme.decoded_content.decode('utf-8')
            
            # Check if README mentions ML problems
            if self._contains_ml_problem(readme_content):
                problem = {
                    'id': f"github_repo_{repo.id}",
                    'source': 'github',
                    'source_type': 'readme',
                    'title': f"{repo.name}: ML Problem from README",
                    'description': readme_content[:500],  # First 500 chars
                    'full_text': readme_content[:1000],  # First 1000 chars
                    'url': repo.html_url,
                    'mined_at': datetime.now().isoformat(),
                    'repository': repo.full_name,
                    'stars': repo.stargazers_count
                }
                problems.append(problem)
        except Exception as e:
            logger.debug(f"Could not extract from README for {repo.full_name}: {e}")
        
        return problems
    
    def _extract_from_repo_issues(self, repo) -> List[Dict]:
        """Extract ML problems from repository issues."""
        problems = []
        
        try:
            issues = repo.get_issues(state='open', sort='updated')
            
            for issue in list(issues)[:5]:  # Limit to 5 issues per repo
                if self._is_ml_issue(issue):
                    problem = {
                        'id': f"github_issue_{issue.id}",
                        'source': 'github',
                        'source_type': 'issue',
                        'title': issue.title,
                        'description': issue.body or '',
                        'full_text': f"{issue.title}\n\n{issue.body or ''}",
                        'url': issue.html_url,
                        'mined_at': datetime.now().isoformat(),
                        'issue_number': issue.number,
                        'repository': repo.full_name,
                        'labels': [label.name for label in issue.labels]
                    }
                    problems.append(problem)
        except Exception as e:
            logger.debug(f"Could not extract issues from {repo.full_name}: {e}")
        
        return problems
    
    def _is_ml_issue(self, issue) -> bool:
        """Check if issue describes an ML problem."""
        title = issue.title.lower()
        body = (issue.body or '').lower()
        combined = f"{title} {body}"
        
        # Must contain ML keywords
        has_ml_keyword = any(keyword in combined for keyword in self.ml_keywords)
        
        # Reject UI-only, refactors, documentation
        reject_keywords = ['ui', 'ux', 'design', 'refactor', 'documentation', 'docs', 'typo', 'bug fix']
        is_rejected = any(keyword in combined for keyword in reject_keywords)
        
        return has_ml_keyword and not is_rejected
    
    def _contains_ml_problem(self, text: str) -> bool:
        """Check if text contains ML problem description."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.ml_keywords)
