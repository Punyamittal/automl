"""
ML Feasibility Classifier Module

Evaluates whether a problem can be solved using machine learning
and classifies the task type.
"""

import json
import requests
import time
import logging
from typing import Dict, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class FeasibilityClassifier:
    """Classifies ML feasibility of problems."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.provider = config.get('provider', 'huggingface')
        self.model_name = config.get('model_name', 'mistralai/Mistral-7B-Instruct-v0.1')
        self.api_url = config.get('api_url', 'https://api-inference.huggingface.co/models')
        self.token = config.get('token', '')
        self.max_retries = config.get('max_retries', 3)
        self.timeout = config.get('timeout', 60)
        self.min_confidence = config.get('min_confidence', 0.7)
        
        # Flag to track if LLM has failed (to skip subsequent attempts)
        self.llm_unavailable = False
        
        # Load prompt template
        prompt_file = Path("prompts/feasibility_prompt.txt")
        if prompt_file.exists():
            self.prompt_template = prompt_file.read_text()
        else:
            self.prompt_template = self._default_prompt()
    
    def _default_prompt(self) -> str:
        """Default prompt if file doesn't exist."""
        return """Analyze the following problem and determine if it can be solved using machine learning.

CRITICAL: First determine if this is:
- A DISCUSSION/THEORETICAL POST about ML concepts, failures, or frameworks (NOT suitable for training)
- An ACTUAL ML PROBLEM that requires building/training a model (suitable for training)

Discussion posts to REJECT:
- Posts about "why ML fails", "correlation vs causation", "causal inference"
- Theoretical discussions about ML frameworks, paradigms, or concepts
- Research papers, analysis, or explanations
- Questions about ML concepts without asking for implementation
- Posts marked [D] or [Discussion]

Actual ML problems to ACCEPT:
- Requests to build, train, or implement ML models
- Problems that need prediction, classification, or clustering
- Questions asking for help with specific ML tasks
- Problems that can be solved with a dataset and model training

Problem:
{problem_text}

Evaluate:
1. Is this a discussion/theoretical post? (yes/no) - If yes, set feasible=false
2. Is this problem solvable using machine learning? (yes/no)
3. What type of ML task is this? (classification/regression/clustering/other/none)
4. What is the confidence level? (0.0 to 1.0)
5. Is a public dataset likely available? (yes/no/maybe)
6. What are the key features/variables needed?
7. What are potential challenges?

Respond ONLY with valid JSON in this exact format:
{{
    "feasible": true/false,
    "task_type": "classification|regression|clustering|other|none",
    "confidence": 0.0-1.0,
    "dataset_available": true/false/null,
    "key_features": ["feature1", "feature2"],
    "challenges": ["challenge1", "challenge2"],
    "reasoning": "brief explanation",
    "is_discussion": true/false
}}"""
    
    def classify(self, problem: Dict) -> Dict:
        """Classify ML feasibility of a problem."""
        problem_text = problem.get('full_text', '')
        
        if not problem_text:
            return {
                'feasible': False,
                'task_type': 'none',
                'confidence': 0.0,
                'error': 'Empty problem text'
            }
        
        # FIRST: Check if this is a discussion/theoretical post (before LLM call)
        if self._is_discussion_post(problem):
            logger.info("Detected discussion/theoretical post - rejecting before LLM call")
            return {
                'feasible': False,
                'task_type': 'none',
                'confidence': 0.9,
                'reasoning': 'This appears to be a theoretical discussion or research post, not an actual ML problem requiring model training. Discussion posts about ML concepts, failures, or frameworks should not be treated as supervised learning tasks.',
                'is_discussion': True,
                'problem_id': problem.get('id'),
                'problem_title': problem.get('title', '')
            }
        
        # Skip LLM if it's already been determined to be unavailable
        if self.llm_unavailable:
            logger.debug("LLM previously unavailable, using rule-based classification")
            return self._rule_based_classify(problem)
        
        # Try LLM first
        prompt = self.prompt_template.format(problem_text=problem_text[:2000])
        
        # Try LLM only once if it's been consistently failing (check for 410 errors)
        # This avoids wasting time on repeated failures
        try:
            response = self._call_llm(prompt)
            result = self._parse_response(response)
            
            if result and result.get('confidence', 0) >= self.min_confidence:
                result['problem_id'] = problem.get('id')
                result['problem_title'] = problem.get('title', '')
                return result
            else:
                logger.warning(f"Low confidence or invalid response: {result}")
                
        except requests.exceptions.HTTPError as e:
            # If it's a 410 error, mark LLM as unavailable and skip for future problems
            if "410" in str(e) or "Gone" in str(e):
                self.llm_unavailable = True
                logger.info("LLM endpoints unavailable (410), using rule-based classification for all remaining problems")
                return self._rule_based_classify(problem)
            # For other errors, try retries
            logger.warning(f"LLM call failed: {e}, trying rule-based classification")
        except Exception as e:
            # Check if it's a persistent error that should skip future attempts
            if "410" in str(e) or "unavailable" in str(e).lower():
                self.llm_unavailable = True
                logger.info("LLM unavailable, using rule-based classification for all remaining problems")
            logger.warning(f"LLM call failed: {e}, using rule-based classification")
        
        # Fallback to rule-based classification
        return self._rule_based_classify(problem)
    
    def _is_discussion_post(self, problem: Dict) -> bool:
        """Detect if this is a discussion/theoretical post, not an actual ML problem to solve."""
        problem_text = problem.get('full_text', '').lower()
        title = problem.get('title', '').lower()
        combined_text = f"{title} {problem_text}".lower()
        
        # Discussion/theoretical indicators
        discussion_indicators = [
            # Discussion markers
            '[d]', '[discussion]', 'discussion', 'thoughts?', 'what do you think',
            'curious if', 'curious about', 'wondering', 'question about',
            # Theoretical/philosophical markers
            'conceptual', 'theoretical', 'philosophy', 'paradigm', 'framework',
            'causation', 'causal', 'correlation', 'causal inference',
            'judea pearl', 'ladder of causation', 'do-calculus', 'scm',
            'structural causal model', 'counterfactual', 'intervention',
            # Meta-discussion about ML
            'why ml fails', 'ml failure', 'production ml', 'ml in production',
            'silent failure', 'distribution shift', 'robustness',
            'correlation vs causation', 'correlation ≠ causation',
            # Research/analysis markers
            'research', 'analysis', 'study', 'paper', 'article',
            'explain', 'explanation', 'understanding', 'concept',
            # Not asking for implementation
            'won\'t fix', 'can\'t solve', 'limitation', 'problem with',
            'issue with', 'challenge with', 'failure of'
        ]
        
        # Check for discussion markers
        has_discussion_markers = any(indicator in combined_text for indicator in discussion_indicators)
        
        # Check if it's asking a question vs making a statement
        question_indicators = ['?', 'how', 'why', 'what', 'when', 'where']
        is_question = any(q in title[-10:] for q in question_indicators) or '?' in title
        
        # Check for action verbs that indicate actual problem-solving intent
        action_indicators = [
            'build', 'create', 'develop', 'implement', 'train', 'predict',
            'classify', 'solve', 'need', 'want', 'looking for', 'help with'
        ]
        has_action_intent = any(action in combined_text for action in action_indicators)
        
        # If it has discussion markers but lacks action intent, it's likely a discussion
        if has_discussion_markers and not has_action_intent:
            return True
        
        # If it's a question about concepts/theory without asking for implementation
        if is_question and has_discussion_markers and not has_action_intent:
            return True
        
        return False
    
    def _rule_based_classify(self, problem: Dict) -> Dict:
        """Rule-based ML feasibility classification (fallback when LLM unavailable)."""
        problem_text = problem.get('full_text', '').lower()
        title = problem.get('title', '').lower()
        
        # FIRST: Check if this is a discussion/theoretical post
        if self._is_discussion_post(problem):
            logger.info("Detected discussion/theoretical post - not suitable for ML training")
            return {
                'feasible': False,
                'task_type': 'none',
                'confidence': 0.9,
                'reasoning': 'This appears to be a theoretical discussion or research post, not an actual ML problem requiring model training. Discussion posts about ML concepts, failures, or frameworks should not be treated as supervised learning tasks.',
                'is_discussion': True
            }
        
        # ML-related keywords
        ml_keywords = [
            'machine learning', 'ml', 'ai', 'artificial intelligence',
            'neural network', 'deep learning', 'classify', 'classification',
            'regression', 'predict', 'prediction', 'model', 'dataset',
            'training', 'supervised', 'unsupervised', 'clustering'
        ]
        
        # Classification keywords
        classification_keywords = [
            'classify', 'classification', 'category', 'label', 'predict class',
            'spam', 'sentiment', 'image recognition', 'text classification'
        ]
        
        # Regression keywords
        regression_keywords = [
            'predict', 'forecast', 'price', 'cost', 'revenue', 'sales',
            'temperature', 'stock', 'regression', 'continuous'
        ]
        
        # Clustering keywords
        clustering_keywords = [
            'cluster', 'group', 'segment', 'unsupervised', 'pattern',
            'similar', 'grouping'
        ]
        
        # Check if problem mentions ML
        has_ml_keywords = any(keyword in problem_text or keyword in title for keyword in ml_keywords)
        
        if not has_ml_keywords:
            return {
                'feasible': False,
                'task_type': 'none',
                'confidence': 0.3,
                'reasoning': 'No ML-related keywords found',
                'problem_id': problem.get('id'),
                'problem_title': problem.get('title', '')
            }
        
        # Determine task type
        task_type = 'other'
        confidence = 0.6
        
        if any(kw in problem_text or kw in title for kw in classification_keywords):
            task_type = 'classification'
            confidence = 0.7
        elif any(kw in problem_text or kw in title for kw in regression_keywords):
            task_type = 'regression'
            confidence = 0.7
        elif any(kw in problem_text or kw in title for kw in clustering_keywords):
            task_type = 'clustering'
            confidence = 0.65
        
        # Extract potential features (simple keyword extraction)
        feature_keywords = ['feature', 'variable', 'column', 'attribute', 'input']
        key_features = []
        words = problem_text.split()
        for i, word in enumerate(words):
            if word in feature_keywords and i + 1 < len(words):
                key_features.append(words[i + 1][:20])  # Next word as potential feature
        
        return {
            'feasible': True,
            'task_type': task_type,
            'confidence': confidence,
            'dataset_available': True,  # Assume available
            'key_features': key_features[:5] if key_features else ['features', 'data'],
            'challenges': ['Requires dataset', 'May need feature engineering'],
            'reasoning': f'Rule-based classification: {task_type} task detected',
            'problem_id': problem.get('id'),
            'problem_title': problem.get('title', ''),
            'method': 'rule_based'  # Indicate this was rule-based
        }
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM API."""
        if self.provider == 'huggingface':
            return self._call_huggingface(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _call_huggingface(self, prompt: str) -> str:
        """Call HuggingFace Inference API."""
        url = f"{self.api_url}/{self.model_name}"
        
        headers = {}
        if self.token:
            headers['Authorization'] = f"Bearer {self.token}"
        
        payload = {
            'inputs': prompt,
            'parameters': {
                'max_new_tokens': 500,
                'temperature': 0.3,
                'return_full_text': False
            }
        }
        
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        
        # Handle 410 Gone - model endpoint no longer available
        if response.status_code == 410:
            logger.warning(f"Model {self.model_name} endpoint returned 410 (Gone). Trying fallback model...")
            # Try a simpler fallback model
            fallback_models = [
                "microsoft/DialoGPT-medium",  # More reliable fallback
                "distilgpt2"  # Smaller, more available model
            ]
            for fallback_model in fallback_models:
                try:
                    fallback_url = f"{self.api_url}/{fallback_model}"
                    logger.info(f"Trying fallback model: {fallback_model}")
                    response = requests.post(
                        fallback_url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout
                    )
                    if response.status_code not in [410, 404]:
                        logger.info(f"Using fallback model: {fallback_model}")
                        break
                except Exception as e:
                    logger.warning(f"Fallback model {fallback_model} failed: {e}")
                    continue
            else:
                # Don't raise exception, let it fall through to rule-based classification
                logger.warning("All model endpoints unavailable (410). Will use rule-based classification.")
                raise requests.exceptions.HTTPError(f"All model endpoints unavailable (410). Please update model_name in config.")
        
        if response.status_code == 503:
            # Model is loading, wait and retry
            logger.info("Model is loading, waiting...")
            time.sleep(10)
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
        
        response.raise_for_status()
        
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('generated_text', '')
        elif isinstance(result, dict):
            return result.get('generated_text', '')
        else:
            return str(result)
    
    def _parse_response(self, response_text: str) -> Optional[Dict]:
        """Parse LLM response to extract JSON."""
        # Try to extract JSON from response
        response_text = response_text.strip()
        
        # Find JSON object
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            logger.error(f"No JSON found in response: {response_text[:200]}")
            return None
        
        json_str = response_text[start_idx:end_idx]
        
        try:
            result = json.loads(json_str)
            
            # Validate structure
            required_fields = ['feasible', 'task_type', 'confidence']
            if not all(field in result for field in required_fields):
                logger.error(f"Missing required fields in response: {result}")
                return None
            
            # Normalize task_type
            task_type = result.get('task_type', '').lower()
            valid_types = ['classification', 'regression', 'clustering', 'other', 'none']
            if task_type not in valid_types:
                # Try to map common variations
                if 'classif' in task_type:
                    task_type = 'classification'
                elif 'regress' in task_type:
                    task_type = 'regression'
                elif 'cluster' in task_type:
                    task_type = 'clustering'
                else:
                    task_type = 'other'
            
            result['task_type'] = task_type
            
            # Ensure confidence is float
            result['confidence'] = float(result.get('confidence', 0.0))
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}, response: {json_str[:200]}")
            return None
    
    def filter_feasible(self, problems: List[Dict], classifications: List[Dict]) -> List[Dict]:
        """Filter problems that are ML feasible."""
        feasible_problems = []
        
        for problem, classification in zip(problems, classifications):
            if (classification.get('feasible', False) and 
                classification.get('confidence', 0) >= self.min_confidence and
                classification.get('task_type') not in ['none', 'other']):
                problem['ml_classification'] = classification
                feasible_problems.append(problem)
        
        return feasible_problems

