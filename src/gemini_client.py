"""
Gemini API Client Module

Provides interface to Google's Gemini API for analyzing Reddit posts
and determining if they are actual ML problems.
"""

import logging
import json
import os
import re
from typing import Dict, Optional, List
import time

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Gemini API will not be available.")


class GeminiClient:
    """Client for Google Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-pro"):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Gemini API key (or set GEMINI_API_KEY env var)
            model_name: Model to use (default: gemini-pro)
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")
        
        # Get API key from parameter, environment, or config
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. "
                "Set GEMINI_API_KEY env var, pass api_key parameter, "
                "or add to config.yaml under gemini.api_key or llm.gemini_api_key"
            )
        
        self.model_name = model_name
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        
        logger.info(f"Gemini client initialized with model: {model_name}")
    
    def enhance_problem_statement(self, problem_statement: str) -> Dict:
        """
        Enhance an incomplete or vague problem statement into a well-formed ML problem.
        This is used when the initial canonicalization fails or is incomplete.
        
        Args:
            problem_statement: Natural language problem description (may be incomplete)
            
        Returns:
            {
                'enhanced_problem': dict,  # Structured ML problem definition
                'enhanced_statement': str,  # Enhanced natural language statement
                'raw_response': str
            }
        """
        prompt = self._enhancement_prompt()
        full_prompt = prompt.format(raw_problem_text=problem_statement)
        
        try:
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Slightly higher for creativity in enhancement
                    max_output_tokens=2000,
                )
            )
            
            response_text = response.text.strip()
            
            # Parse the enhanced problem
            result = self._parse_canonicalizer_response(response_text)
            
            if result.get('canonical_problem'):
                # Create enhanced statement from canonical problem
                canonical = result['canonical_problem']
                enhanced_statement = (
                    f"Predict {canonical.get('target_variable', 'target')} "
                    f"using {', '.join(canonical.get('input_features', [])[:3])} "
                    f"for {canonical.get('intended_use', 'business decision support')}"
                )
                
                return {
                    'enhanced_problem': canonical,
                    'enhanced_statement': enhanced_statement,
                    'raw_response': response_text
                }
            else:
                # Fallback: create a basic enhanced problem
                enhanced_statement = f"Predict target variable using input features based on: {problem_statement}"
                return {
                    'enhanced_problem': {
                        'problem_type': 'classification',
                        'target_variable': 'target',
                        'input_features': ['feature1', 'feature2', 'feature3'],
                        'intended_use': 'business decision support',
                        'data_source': 'historical data',
                        'evaluation_metric': 'accuracy'
                    },
                    'enhanced_statement': enhanced_statement,
                    'raw_response': response_text
                }
                
        except Exception as e:
            logger.error(f"Error enhancing problem statement: {e}")
            # Return safe defaults
            return {
                'enhanced_problem': {
                    'problem_type': 'classification',
                    'target_variable': 'target',
                    'input_features': ['feature1', 'feature2', 'feature3'],
                    'intended_use': 'business decision support',
                    'data_source': 'historical data',
                    'evaluation_metric': 'accuracy'
                },
                'enhanced_statement': problem_statement,
                'raw_response': ''
            }
    
    def analyze_problem_statement(self, problem_statement: str, prompt_template: Optional[str] = None) -> Dict:
        """
        Analyze a problem statement and canonicalize it into a structured ML problem.
        
        Args:
            problem_statement: Natural language problem description
            prompt_template: Optional custom prompt template
            
        Returns:
            {
                'is_ml_problem': bool,
                'confidence': float,
                'reasoning': str,
                'problem_type': str,
                'canonical_problem': dict,  # Structured ML problem definition
                'raw_response': str
            }
        """
        if prompt_template is None:
            prompt_template = self._default_canonicalizer_prompt()
        
        # Format the prompt with the problem statement
        full_prompt = prompt_template.format(
            raw_problem_text=problem_statement
        )
        
        try:
            # Call Gemini API
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,  # Lower temperature for more structured output
                    max_output_tokens=2000,
                )
            )
            
            response_text = response.text.strip()
            
            # Parse the response
            result = self._parse_canonicalizer_response(response_text)
            
            # Add raw response for debugging
            result['raw_response'] = response_text
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            # Return safe fallback
            return {
                'is_ml_problem': False,
                'confidence': 0.0,
                'reasoning': f"API call failed: {str(e)}",
                'problem_type': 'error',
                'canonical_problem': None,
                'raw_response': ''
            }
    
    def analyze_reddit_post(self, post_title: str, post_text: str, prompt_template: Optional[str] = None) -> Dict:
        """
        Analyze a Reddit post to determine if it's an actual ML problem.
        
        Args:
            post_title: Reddit post title
            post_text: Reddit post body text
            prompt_template: Optional custom prompt template
            
        Returns:
            {
                'is_ml_problem': bool,
                'confidence': float,
                'reasoning': str,
                'problem_type': str,  # 'predictive_ml_task', 'discussion', 'question', etc.
                'extracted_problem': str,  # Cleaned problem statement if it's an ML problem
                'raw_response': str
            }
        """
        if prompt_template is None:
            prompt_template = self._default_analysis_prompt()
        
        # Format the prompt with the Reddit post
        full_prompt = prompt_template.format(
            title=post_title,
            text=post_text
        )
        
        try:
            # Call Gemini API
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=2000,  # Increased for structured ML problem output
                )
            )
            
            response_text = response.text.strip()
            
            # Parse the response
            result = self._parse_response(response_text)
            
            # Add raw response for debugging
            result['raw_response'] = response_text
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            # Return safe fallback
            return {
                'is_ml_problem': False,
                'confidence': 0.0,
                'reasoning': f"API call failed: {str(e)}",
                'problem_type': 'error',
                'extracted_problem': '',
                'raw_response': ''
            }
    
    def _parse_canonicalizer_response(self, response_text: str) -> Dict:
        """Parse canonicalizer response to extract structured ML problem."""
        response_text = response_text.strip()
        
        # Check if rejected
        rejection_phrases = [
            'REJECTED:',
            'not a predictive machine learning task',
            'insufficient information',
            'not a predictive ml task'
        ]
        is_rejected = any(phrase in response_text.upper() for phrase in [p.upper() for p in rejection_phrases])
        
        if is_rejected:
            return {
                'is_ml_problem': False,
                'confidence': 0.0,
                'reasoning': 'Problem rejected: not a predictive machine learning task',
                'problem_type': 'rejected',
                'canonical_problem': None,
                'raw_response': response_text
            }
        
        # Try to extract JSON
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > 0:
            json_str = response_text[start_idx:end_idx]
            try:
                canonical_problem = json.loads(json_str)
                
                # Validate required fields
                required_fields = ['problem_type', 'target_variable', 'input_features']
                if all(field in canonical_problem for field in required_fields):
                    # Calculate confidence based on completeness
                    optional_fields = ['intended_use', 'data_source', 'evaluation_metric']
                    completeness = sum(1 for field in optional_fields if canonical_problem.get(field)) / len(optional_fields)
                    confidence = 0.7 + (completeness * 0.3)  # 0.7 to 1.0
                    
                    return {
                        'is_ml_problem': True,
                        'confidence': confidence,
                        'reasoning': f"Canonicalized ML problem: {canonical_problem.get('problem_type', 'unknown')}",
                        'problem_type': canonical_problem.get('problem_type', 'unknown').lower(),
                        'canonical_problem': canonical_problem,
                        'raw_response': response_text
                    }
                else:
                    return {
                        'is_ml_problem': False,
                        'confidence': 0.0,
                        'reasoning': 'Missing required fields in canonicalized problem',
                        'problem_type': 'invalid',
                        'canonical_problem': canonical_problem,
                        'raw_response': response_text
                    }
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON from canonicalizer response: {response_text[:200]}")
        
        # Fallback: not a valid ML problem
        return {
            'is_ml_problem': False,
            'confidence': 0.0,
            'reasoning': 'Could not parse canonicalized problem from response',
            'problem_type': 'unknown',
            'canonical_problem': None,
            'raw_response': response_text
        }
    
    def _parse_response(self, response_text: str) -> Dict:
        """Parse Gemini response to extract structured ML problem data."""
        response_text = response_text.strip()
        
        # Check if rejected
        if 'REJECTED:' in response_text.upper() or 'not suitable' in response_text.lower():
            return {
                'is_ml_problem': False,
                'confidence': 0.0,
                'reasoning': 'Post rejected: Not suitable for ML problem generation',
                'problem_type': 'rejected',
                'extracted_problem': '',
                'ml_problem_details': None
            }
        
        # Try to parse structured ML problem format
        problem_details = {}
        
        # Extract ML Problem Title
        title_match = self._extract_section(response_text, r'###?\s*ML Problem Title:?\s*\n(.+?)(?=\n###|\n---|$)')
        if title_match:
            problem_details['title'] = title_match.strip()
        
        # Extract Problem Type
        type_match = self._extract_section(response_text, r'###?\s*Problem Type:?\s*\n(.+?)(?=\n###|\n---|$)')
        if type_match:
            problem_details['problem_type'] = type_match.strip()
        
        # Extract Problem Statement
        statement_match = self._extract_section(response_text, r'###?\s*Problem Statement:?\s*\n(.+?)(?=\n###|\n---|$)')
        if statement_match:
            problem_details['problem_statement'] = statement_match.strip()
        
        # Extract Target Variable
        target_match = self._extract_section(response_text, r'###?\s*Target Variable:?\s*\n(.+?)(?=\n###|\n---|$)')
        if target_match:
            problem_details['target_variable'] = target_match.strip()
        
        # Extract Input Features
        features_match = self._extract_section(response_text, r'###?\s*Input Features:?\s*\n(.+?)(?=\n###|\n---|$)')
        if features_match:
            problem_details['input_features'] = features_match.strip()
        
        # Extract Data Source
        data_match = self._extract_section(response_text, r'###?\s*Data Source:?\s*\n(.+?)(?=\n###|\n---|$)')
        if data_match:
            problem_details['data_source'] = data_match.strip()
        
        # Extract Evaluation Metric
        metric_match = self._extract_section(response_text, r'###?\s*Evaluation Metric:?\s*\n(.+?)(?=\n###|\n---|$)')
        if metric_match:
            problem_details['evaluation_metric'] = metric_match.strip()
        
        # Extract Deployment Context
        deployment_match = self._extract_section(response_text, r'###?\s*Deployment Context:?\s*\n(.+?)(?=\n###|\n---|$)')
        if deployment_match:
            problem_details['deployment_context'] = deployment_match.strip()
        
        # If we have at least title and problem statement, consider it a valid ML problem
        if problem_details.get('title') and problem_details.get('problem_statement'):
            # Build extracted problem text
            extracted_problem = f"{problem_details.get('title', '')}\n\n{problem_details.get('problem_statement', '')}"
            
            # Determine confidence based on completeness
            required_fields = ['title', 'problem_statement', 'target_variable', 'input_features']
            completeness = sum(1 for field in required_fields if problem_details.get(field)) / len(required_fields)
            confidence = 0.6 + (completeness * 0.4)  # 0.6 to 1.0 based on completeness
            
            return {
                'is_ml_problem': True,
                'confidence': confidence,
                'reasoning': f"Generated ML problem: {problem_details.get('title', 'Unknown')}",
                'problem_type': problem_details.get('problem_type', 'unknown').lower(),
                'extracted_problem': extracted_problem,
                'ml_problem_details': problem_details
            }
        
        # Fallback: try to detect if it's an ML problem from keywords
        ml_keywords = ['classification', 'regression', 'prediction', 'predict', 'model', 'target variable', 'features']
        has_ml_keywords = any(keyword.lower() in response_text.lower() for keyword in ml_keywords)
        
        if has_ml_keywords and len(response_text) > 100:
            # Try to extract as unstructured problem
            return {
                'is_ml_problem': True,
                'confidence': 0.5,
                'reasoning': 'Detected ML-related content but structure unclear',
                'problem_type': 'unknown',
                'extracted_problem': response_text[:500],
                'ml_problem_details': None
            }
        
        # Default: not an ML problem
        return {
            'is_ml_problem': False,
            'confidence': 0.0,
            'reasoning': 'Could not extract structured ML problem from response',
            'problem_type': 'unknown',
            'extracted_problem': '',
            'ml_problem_details': None
        }
    
    def _extract_section(self, text: str, pattern: str) -> Optional[str]:
        """Extract a section from text using regex pattern."""
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        if match:
            return match.group(1).strip()
        return None
    
    def _default_canonicalizer_prompt(self) -> str:
        """Default prompt for canonicalizing problem statements into structured ML problems."""
        return """You are operating in DIRECT PROBLEM MODE.

IMPORTANT CONTEXT:
- The input is NOT a Reddit post.
- The input is NOT a personal experience.
- The input is NOT an opinion or discussion.
- The input is an INTENTIONAL machine learning task provided by a developer via CLI.

You MUST assume the user intends to define a valid ML problem.

Your task is NOT to judge intent.
Your task is to CANONICALIZE the problem into a well-posed ML specification.

-----------------------------
INPUT:
<raw_problem_text>
{raw_problem_text}
-----------------------------

INSTRUCTIONS (STRICT):

1. DO NOT classify the input as:
   - personal experience
   - opinion
   - discussion
   - narrative
   - research post

2. If the input contains phrases like:
   - "predict"
   - "estimate"
   - "classify"
   - "detect"
   - "forecast"

   then you MUST treat it as a predictive ML task.

3. Infer the following when reasonably implied:
   - Target Variable
   - Input Features
   - Problem Type (classification / regression)

4. Use SAFE, STANDARD assumptions only:
   - Churn → binary classification
   - Price → regression
   - Fraud → anomaly detection / classification
   - Forecast → time-series

5. Intended use MAY be generic:
   - "business decision support"
   - "risk estimation"
   - "customer retention optimization"

6. Data source MAY be generic:
   - "historical transactional data"
   - "customer behavior logs"
   - "public or enterprise datasets"

7. DO NOT reject unless the input is completely non-predictive
   (e.g., "I feel customers are unhappy").

-----------------------------
OUTPUT FORMAT (JSON ONLY):

{{
  "problem_type": "<task_type>",
  "target_variable": "<explicit target>",
  "input_features": ["<feature1>", "<feature2>", "..."],
  "intended_use": "<generic but valid use>",
  "data_source": "<realistic data source>",
  "evaluation_metric": "<appropriate metric>"
}}

-----------------------------
REJECTION RULE:
Only output:

REJECTED: not a predictive machine learning task

IF AND ONLY IF no prediction objective exists.

Your Response:"""
    
    def _enhancement_prompt(self) -> str:
        """Prompt for enhancing incomplete or vague problem statements."""
        return """You are a Problem Enhancement Agent. Your task is to take a vague, incomplete, or unclear problem statement and transform it into a COMPLETE, WELL-FORMED machine learning problem.

IMPORTANT:
- The user wants to solve a REAL ML problem
- Your job is to ENHANCE and COMPLETE the problem, not reject it
- Make reasonable inferences to fill in missing details
- Use standard ML problem patterns

-----------------------------
INPUT PROBLEM:
{raw_problem_text}
-----------------------------

INSTRUCTIONS:

1. Identify what the user wants to predict/classify/detect
   - If unclear, infer from context (e.g., "churn" → predict customer churn)
   - If no target mentioned, create a reasonable one based on the domain

2. Identify or infer input features
   - Extract from the problem statement if mentioned
   - If not mentioned, suggest realistic features for the domain
   - Example: "churn" → ["purchase_history", "engagement_score", "account_age"]

3. Determine problem type
   - "predict X" → classification (if X is categorical) or regression (if X is numeric)
   - "churn" → binary classification
   - "price", "cost", "revenue" → regression
   - "forecast" → time-series

4. Fill in missing details with reasonable defaults:
   - intended_use: "business decision support" or domain-specific use
   - data_source: "historical transactional data" or domain-specific source
   - evaluation_metric: "accuracy" (classification) or "rmse" (regression)

5. Create an enhanced problem statement that is clear and complete

-----------------------------
OUTPUT FORMAT (JSON ONLY):

{{
  "problem_type": "<classification|regression|time-series|clustering|anomaly detection>",
  "target_variable": "<explicit target to predict>",
  "input_features": ["<feature1>", "<feature2>", "<feature3>", "..."],
  "intended_use": "<how predictions will be used>",
  "data_source": "<where data comes from>",
  "evaluation_metric": "<appropriate metric>"
}}

-----------------------------
EXAMPLE:

Input: "predict customer churn"
Output:
{{
  "problem_type": "classification",
  "target_variable": "customer_churn",
  "input_features": ["purchase_history", "engagement_score", "account_age", "last_login_days", "total_spent"],
  "intended_use": "customer retention optimization",
  "data_source": "customer behavior logs and transaction history",
  "evaluation_metric": "f1_score"
}}

Your Response:"""
    
    def _default_analysis_prompt(self) -> str:
        """Default prompt for analyzing Kaggle/GitHub content and generating ML problems."""
        return """You are an ML Problem Discovery Agent with authenticated access to:
- Kaggle API (datasets, competitions, discussions)
- GitHub API (repositories, issues, README files)

Your goal is to discover REAL, TRAINABLE machine learning problems.
You must prioritize Kaggle and GitHub. Reddit is NOT allowed.

---

############################
## SOURCE PRIORITY ORDER ##
############################

1️⃣ Kaggle (PRIMARY)
   - Competitions
   - Dataset descriptions
   - Dataset discussion pages

2️⃣ GitHub (SECONDARY)
   - Issues from production repositories
   - Feature requests describing automation needs
   - README problem statements

---

############################
## STEP 1: KAGGLE MINING ##
############################

Using the Kaggle API:

Search for:
- Active or past competitions
- Popular datasets (high downloads / usability)
- Dataset descriptions that mention prediction, classification, detection, forecasting

For each candidate, ask:
- Is there a clearly defined target?
- Is there structured or semi-structured data?
- Is this solvable with supervised or unsupervised ML?

Reject anything that is:
- Pure visualization
- Static analysis
- One-off scripts
- Toy datasets without learning value

---

############################
## STEP 2: GITHUB MINING ##
############################

Using the GitHub API:

Search repositories with:
- Open issues containing keywords:
  "predict", "detect", "classify", "recommend", "forecast", "anomaly", "ranking"

Focus on:
- SaaS tools
- Monitoring platforms
- Analytics tools
- DevOps / infra tools
- Open-source ML-adjacent products

Reject issues that are:
- UI-only
- Refactors
- Documentation
- Opinion-based discussions

---

############################
## STEP 3: ML PROBLEM GENERATION ##
############################

ONLY after passing steps 1 or 2, generate ONE ML problem in this EXACT format:

### ML Problem Title:
(clear, concrete, production-oriented)

### Source:
(Kaggle Dataset / Kaggle Competition / GitHub Issue)

### Problem Type:
(classification | regression | time-series | clustering | anomaly detection | recommendation | NLP)

### Problem Statement:
Describe the real-world prediction task.

### Target Variable:
Explicitly define what is being predicted.

### Input Features:
List realistic, measurable features.

### Data Source:
Explain how data is obtained.
⚠️ DO NOT invent dataset names.

### Evaluation Metric:
Choose an appropriate metric (F1, ROC-AUC, RMSE, MAE, etc.).

### Deployment Context:
Explain how predictions would be used in production.

---

############################
## STEP 4: HARD VALIDATION ##
############################

Before outputting, ensure ALL are true:
- Target variable is explicit
- Historical data exists or can exist
- ML adds value over rules
- Problem is not descriptive or exploratory

If ANY check fails, output exactly:
REJECTED: Not a valid ML problem.

---

############################
## ABSOLUTE RULES ##
############################

- NEVER hallucinate datasets
- NEVER default to regression
- NEVER output metrics without justification
- NEVER generate problems from opinions or stories
- Prefer rejection over low-quality problems

---

############################
## OUTPUT RULES ##
############################

- Output ONLY the ML problem or REJECTED
- No explanations
- No commentary
- No references to Reddit

---

Content to Analyze:
Title: {title}

Body:
{text}

Your Response:"""
