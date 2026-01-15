"""
GitHub Publisher Module

Publishes generated code to GitHub repositories.
"""

import logging
from pathlib import Path
from typing import Dict, Optional
from github import Github
from git import Repo
import os

logger = logging.getLogger(__name__)


class GitHubPublisher:
    """Publishes code to GitHub."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.username = config.get('username', '')
        self.token = config.get('token', '')
        self.organization = config.get('organization', '')
        self.create_repo = config.get('create_repo', True)
        self.add_topics = config.get('add_topics', True)
        self.default_topics = config.get('default_topics', ['machine-learning', 'automated-ml'])
        self.make_private = config.get('make_private', False)
        
        if not self.token:
            logger.warning("GitHub token not provided. Publishing will be disabled.")
            self.github = None
            self.user = None
        else:
            try:
                self.github = Github(self.token)
                self.user = self.github.get_user()
                # Test authentication
                self.user.login  # This will raise if token is invalid
                logger.info(f"GitHub authenticated as: {self.user.login}")
            except Exception as e:
                logger.error(f"Error initializing GitHub client: {e}")
                self.github = None
                self.user = None
    
    def publish(self, project_dir: str, project_name: str, problem: Dict, training_result: Dict) -> Dict:
        """Publish project to GitHub."""
        if not self.github:
            return {
                'success': False,
                'error': 'GitHub not initialized. Provide token in config.'
            }
        
        project_path = Path(project_dir)
        if not project_path.exists():
            return {
                'success': False,
                'error': f'Project directory does not exist: {project_dir}'
            }
        
        # Sanitize repository name
        repo_name = self._sanitize_repo_name(project_name)
        
        try:
            # Create repository
            if self.create_repo:
                repo = self._create_repository(repo_name, problem)
            else:
                # Try to get existing repo
                if self.organization:
                    repo = self.github.get_organization(self.organization).get_repo(repo_name)
                else:
                    repo = self.user.get_repo(repo_name)
            
            # Initialize git repo and push
            self._push_code(project_path, repo)
            
            # Add topics
            if self.add_topics:
                self._add_topics(repo, training_result)
            
            repo_url = repo.html_url
            
            logger.info(f"Published to GitHub: {repo_url}")
            
            return {
                'success': True,
                'repo_url': repo_url,
                'repo_name': repo_name
            }
            
        except Exception as e:
            logger.error(f"Error publishing to GitHub: {e}", exc_info=True)
            error_msg = str(e)
            # Provide more helpful error messages
            if "401" in error_msg or "Bad credentials" in error_msg:
                error_msg = "Invalid GitHub token. Please check your token in config.yaml"
            elif "403" in error_msg or "Forbidden" in error_msg:
                error_msg = "GitHub token lacks required permissions. Ensure it has 'repo' scope"
            elif "already exists" in error_msg.lower():
                error_msg = f"Repository already exists. Try a different name or delete the existing repo"
            elif "not found" in error_msg.lower():
                error_msg = "GitHub repository not found. Check repository name and permissions"
            
            return {
                'success': False,
                'error': error_msg
            }
    
    def _sanitize_repo_name(self, name: str) -> str:
        """Sanitize repository name."""
        import re
        # GitHub repo name rules: alphanumeric, hyphens, underscores
        name = re.sub(r'[^\w-]', '', name)
        name = name.lower()
        # Max 100 chars
        name = name[:100]
        # Remove leading/trailing hyphens
        name = name.strip('-')
        return name or 'ml-solution'
    
    def _create_repository(self, repo_name: str, problem: Dict) -> object:
        """Create GitHub repository."""
        description = problem.get('title', 'Automated ML Solution')[:160]
        
        try:
            if self.organization:
                org = self.github.get_organization(self.organization)
                # Check if repo already exists
                try:
                    repo = org.get_repo(repo_name)
                    logger.info(f"Repository {repo_name} already exists, using existing repo")
                    return repo
                except:
                    repo = org.create_repo(
                        name=repo_name,
                        description=description,
                        private=self.make_private,
                        auto_init=False
                    )
            else:
                # Check if repo already exists
                try:
                    repo = self.user.get_repo(repo_name)
                    logger.info(f"Repository {repo_name} already exists, using existing repo")
                    return repo
                except:
                    repo = self.user.create_repo(
                        name=repo_name,
                        description=description,
                        private=self.make_private,
                        auto_init=False
                    )
            
            logger.info(f"Created repository: {repo_name}")
            return repo
        except Exception as e:
            # If repo creation fails, try to get existing repo
            if "already exists" in str(e).lower() or "name already exists" in str(e).lower():
                logger.warning(f"Repository {repo_name} already exists, using existing repo")
                try:
                    if self.organization:
                        org = self.github.get_organization(self.organization)
                        return org.get_repo(repo_name)
                    else:
                        return self.user.get_repo(repo_name)
                except:
                    pass
            raise
    
    def _push_code(self, project_path: Path, repo) -> None:
        """Initialize git repo and push code."""
        # Initialize git repo if not already
        try:
            git_repo = Repo(project_path)
        except:
            git_repo = Repo.init(project_path)
        
        # Configure git user if not set (required for commits)
        try:
            git_repo.config_writer().set_value("user", "name", self.username or "ML Pipeline").release()
            git_repo.config_writer().set_value("user", "email", f"{self.username or 'ml'}@example.com").release()
        except:
            pass  # Ignore if already configured
        
        # Add remote if not exists
        remote_name = 'origin'
        remote_url = repo.clone_url.replace('https://', f'https://{self.token}@')
        
        try:
            remote = git_repo.remote(remote_name)
            remote.set_url(remote_url)
        except:
            git_repo.create_remote(remote_name, remote_url)
        
        # Add all files
        git_repo.git.add(A=True)
        
        # Check if there are changes to commit
        if git_repo.is_dirty() or len(list(git_repo.index.diff(None))) > 0 or len(git_repo.untracked_files) > 0:
            # Commit
            try:
                git_repo.index.commit("Initial commit: Automated ML solution")
            except Exception as e:
                logger.warning(f"Commit failed (might be empty): {e}")
        
        # Get default branch name (try main first, then master)
        default_branch = 'main'
        try:
            # Try to get the default branch from the remote
            refs = git_repo.remote(remote_name).refs
            if refs:
                default_branch = refs[0].remote_head
        except:
            # Check if main branch exists locally
            try:
                git_repo.heads.main
                default_branch = 'main'
            except:
                default_branch = 'master'
        
        # Push - try main first, then master
        try:
            git_repo.git.push(remote_name, default_branch, force=True, set_upstream=True)
            logger.info(f"Code pushed to GitHub on {default_branch} branch")
        except Exception as e:
            # Try master if main failed
            if default_branch == 'main':
                try:
                    git_repo.git.push(remote_name, 'master', force=True, set_upstream=True)
                    logger.info("Code pushed to GitHub on master branch")
                except Exception as e2:
                    logger.error(f"Failed to push to both main and master: {e2}")
                    raise
            else:
                raise
    
    def _add_topics(self, repo, training_result: Dict) -> None:
        """Add topics to repository."""
        topics = list(self.default_topics)
        
        # Add task type as topic
        task_type = training_result.get('task_type', '')
        if task_type:
            topics.append(task_type)
        
        # Add framework
        topics.append('pycaret')
        
        try:
            repo.replace_topics(topics[:20])  # GitHub limit
            logger.info(f"Added topics: {topics}")
        except Exception as e:
            logger.warning(f"Could not add topics: {e}")

