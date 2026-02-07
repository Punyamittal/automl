"""
Ollama API Client Module

Provides interface to local Ollama for analyzing problem statements
and determining if they are actual ML problems. No API key needed.
"""

import logging
import json
import re
import requests
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for local Ollama API - used for problem analysis and enhancement."""

    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3.2"):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama API base URL (default: http://localhost:11434)
            model_name: Model to use (default: llama3.2). Run `ollama list` to see available models.
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        logger.info(f"Ollama client initialized: {self.base_url}, model={self.model_name}")

    def _generate(self, prompt: str, temperature: float = 0.2) -> str:
        """Call Ollama generate API."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 2000,
            },
        }
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return (data.get("response") or "").strip()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Ensure Ollama is running: ollama serve"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError("Ollama request timed out")
        except Exception as e:
            raise RuntimeError(f"Ollama API call failed: {e}")

    def enhance_problem_statement(self, problem_statement: str) -> Dict:
        """Enhance an incomplete or vague problem statement into a well-formed ML problem."""
        prompt = self._enhancement_prompt().format(raw_problem_text=problem_statement)
        try:
            response_text = self._generate(prompt, temperature=0.3)
            result = self._parse_canonicalizer_response(response_text)
            if result.get("canonical_problem"):
                canonical = result["canonical_problem"]
                enhanced_statement = (
                    f"Predict {canonical.get('target_variable', 'target')} "
                    f"using {', '.join(canonical.get('input_features', [])[:3])} "
                    f"for {canonical.get('intended_use', 'business decision support')}"
                )
                return {
                    "enhanced_problem": canonical,
                    "enhanced_statement": enhanced_statement,
                    "raw_response": response_text,
                }
            else:
                enhanced_statement = f"Predict target variable using input features based on: {problem_statement}"
                return {
                    "enhanced_problem": {
                        "problem_type": "classification",
                        "target_variable": "target",
                        "input_features": ["feature1", "feature2", "feature3"],
                        "intended_use": "business decision support",
                        "data_source": "historical data",
                        "evaluation_metric": "accuracy",
                    },
                    "enhanced_statement": enhanced_statement,
                    "raw_response": response_text,
                }
        except Exception as e:
            logger.error(f"Error enhancing problem statement: {e}")
            return {
                "enhanced_problem": {
                    "problem_type": "classification",
                    "target_variable": "target",
                    "input_features": ["feature1", "feature2", "feature3"],
                    "intended_use": "business decision support",
                    "data_source": "historical data",
                    "evaluation_metric": "accuracy",
                },
                "enhanced_statement": problem_statement,
                "raw_response": "",
            }

    def analyze_problem_statement(self, problem_statement: str, prompt_template: Optional[str] = None) -> Dict:
        """Analyze a problem statement and canonicalize it into a structured ML problem."""
        if prompt_template is None:
            prompt_template = self._default_canonicalizer_prompt()
        full_prompt = prompt_template.format(raw_problem_text=problem_statement)
        try:
            response_text = self._generate(full_prompt, temperature=0.2)
            result = self._parse_canonicalizer_response(response_text)
            result["raw_response"] = response_text
            return result
        except Exception as e:
            logger.error(f"Ollama analyze failed: {e}")
            return {
                "is_ml_problem": False,
                "confidence": 0.0,
                "reasoning": f"API call failed: {str(e)}",
                "problem_type": "error",
                "canonical_problem": None,
                "raw_response": "",
            }

    def analyze_reddit_post(self, post_title: str, post_text: str, prompt_template: Optional[str] = None) -> Dict:
        """Analyze a post to determine if it's an actual ML problem."""
        if prompt_template is None:
            prompt_template = self._default_analysis_prompt()
        full_prompt = prompt_template.format(title=post_title, text=post_text)
        try:
            response_text = self._generate(full_prompt, temperature=0.3)
            result = self._parse_enhancer_response(response_text)
            result["raw_response"] = response_text
            return result
        except Exception as e:
            logger.error(f"Ollama analyze failed: {e}")
            return {
                "is_ml_problem": False,
                "confidence": 0.0,
                "reasoning": f"API call failed: {str(e)}",
                "problem_type": "error",
                "extracted_problem": "",
                "raw_response": "",
            }

    def _parse_canonicalizer_response(self, response_text: str) -> Dict:
        """Parse canonicalizer response to extract structured ML problem."""
        response_text = response_text.strip()
        rejection_phrases = [
            "REJECTED:",
            "not a predictive machine learning task",
            "insufficient information",
            "not a predictive ml task",
        ]
        is_rejected = any(phrase in response_text.upper() for phrase in [p.upper() for p in rejection_phrases])
        if is_rejected:
            return {
                "is_ml_problem": False,
                "confidence": 0.0,
                "reasoning": "Problem rejected: not a predictive machine learning task",
                "problem_type": "rejected",
                "canonical_problem": None,
                "raw_response": response_text,
            }
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        if start_idx != -1 and end_idx > 0:
            json_str = response_text[start_idx:end_idx]
            try:
                canonical_problem = json.loads(json_str)
                required_fields = ["problem_type", "target_variable", "input_features"]
                if all(field in canonical_problem for field in required_fields):
                    optional_fields = ["intended_use", "data_source", "evaluation_metric"]
                    completeness = sum(1 for f in optional_fields if canonical_problem.get(f)) / len(optional_fields)
                    confidence = 0.7 + (completeness * 0.3)
                    return {
                        "is_ml_problem": True,
                        "confidence": confidence,
                        "reasoning": f"Canonicalized ML problem: {canonical_problem.get('problem_type', 'unknown')}",
                        "problem_type": canonical_problem.get("problem_type", "unknown").lower(),
                        "canonical_problem": canonical_problem,
                        "raw_response": response_text,
                    }
                else:
                    return {
                        "is_ml_problem": False,
                        "confidence": 0.0,
                        "reasoning": "Missing required fields in canonicalized problem",
                        "problem_type": "invalid",
                        "canonical_problem": canonical_problem,
                        "raw_response": response_text,
                    }
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON from response: {response_text[:200]}")
        return {
            "is_ml_problem": False,
            "confidence": 0.0,
            "reasoning": "Could not parse canonicalized problem from response",
            "problem_type": "unknown",
            "canonical_problem": None,
            "raw_response": response_text,
        }

    def _parse_enhancer_response(self, response_text: str) -> Dict:
        """Parse response for Reddit/Kaggle/GitHub analysis."""
        response_text = response_text.strip()
        if "REJECTED:" in response_text.upper() or "not suitable" in response_text.lower():
            return {
                "is_ml_problem": False,
                "confidence": 0.0,
                "reasoning": "Post rejected: Not suitable for ML problem generation",
                "problem_type": "rejected",
                "extracted_problem": "",
                "ml_problem_details": None,
            }
        problem_details = {}
        title_match = self._extract_section(response_text, r"###?\s*ML Problem Title:?\s*\n(.+?)(?=\n###|\n---|$)")
        if title_match:
            problem_details["title"] = title_match.strip()
        type_match = self._extract_section(response_text, r"###?\s*Problem Type:?\s*\n(.+?)(?=\n###|\n---|$)")
        if type_match:
            problem_details["problem_type"] = type_match.strip()
        statement_match = self._extract_section(response_text, r"###?\s*Problem Statement:?\s*\n(.+?)(?=\n###|\n---|$)")
        if statement_match:
            problem_details["problem_statement"] = statement_match.strip()
        target_match = self._extract_section(response_text, r"###?\s*Target Variable:?\s*\n(.+?)(?=\n###|\n---|$)")
        if target_match:
            problem_details["target_variable"] = target_match.strip()
        features_match = self._extract_section(response_text, r"###?\s*Input Features:?\s*\n(.+?)(?=\n###|\n---|$)")
        if features_match:
            problem_details["input_features"] = features_match.strip()
        data_match = self._extract_section(response_text, r"###?\s*Data Source:?\s*\n(.+?)(?=\n###|\n---|$)")
        if data_match:
            problem_details["data_source"] = data_match.strip()
        metric_match = self._extract_section(response_text, r"###?\s*Evaluation Metric:?\s*\n(.+?)(?=\n###|\n---|$)")
        if metric_match:
            problem_details["evaluation_metric"] = metric_match.strip()
        deployment_match = self._extract_section(response_text, r"###?\s*Deployment Context:?\s*\n(.+?)(?=\n###|\n---|$)")
        if deployment_match:
            problem_details["deployment_context"] = deployment_match.strip()
        if problem_details.get("title") and problem_details.get("problem_statement"):
            extracted_problem = f"{problem_details.get('title', '')}\n\n{problem_details.get('problem_statement', '')}"
            required_fields = ["title", "problem_statement", "target_variable", "input_features"]
            completeness = sum(1 for f in required_fields if problem_details.get(f)) / len(required_fields)
            confidence = 0.6 + (completeness * 0.4)
            return {
                "is_ml_problem": True,
                "confidence": confidence,
                "reasoning": f"Generated ML problem: {problem_details.get('title', 'Unknown')}",
                "problem_type": problem_details.get("problem_type", "unknown").lower(),
                "extracted_problem": extracted_problem,
                "ml_problem_details": problem_details,
            }
        ml_keywords = ["classification", "regression", "prediction", "predict", "model", "target variable", "features"]
        has_ml_keywords = any(kw.lower() in response_text.lower() for kw in ml_keywords)
        if has_ml_keywords and len(response_text) > 100:
            return {
                "is_ml_problem": True,
                "confidence": 0.5,
                "reasoning": "Detected ML-related content but structure unclear",
                "problem_type": "unknown",
                "extracted_problem": response_text[:500],
                "ml_problem_details": None,
            }
        return {
            "is_ml_problem": False,
            "confidence": 0.0,
            "reasoning": "Could not extract structured ML problem from response",
            "problem_type": "unknown",
            "extracted_problem": "",
            "ml_problem_details": None,
        }

    def _extract_section(self, text: str, pattern: str) -> Optional[str]:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        return match.group(1).strip() if match else None

    def _default_canonicalizer_prompt(self) -> str:
        return """You are operating in DIRECT PROBLEM MODE.

IMPORTANT CONTEXT:
- The input is NOT a Reddit post.
- The input is an INTENTIONAL machine learning task provided by a developer via CLI.
You MUST assume the user intends to define a valid ML problem.
Your task is to CANONICALIZE the problem into a well-posed ML specification.

-----------------------------
INPUT:
<raw_problem_text>
{raw_problem_text}
-----------------------------

INSTRUCTIONS:
1. If the input contains "predict", "estimate", "classify", "detect", "forecast" → treat as predictive ML task.
2. Infer: Target Variable, Input Features, Problem Type (classification/regression).
3. Use SAFE assumptions: Churn→classification, Price→regression, Fraud→classification.

-----------------------------
OUTPUT FORMAT (JSON ONLY):
{{
  "problem_type": "<task_type>",
  "target_variable": "<explicit target>",
  "input_features": ["<feature1>", "<feature2>", "..."],
  "intended_use": "<generic use>",
  "data_source": "<realistic source>",
  "evaluation_metric": "<metric>"
}}

Only output REJECTED: not a predictive machine learning task if NO prediction objective exists.

Your Response:"""

    def _enhancement_prompt(self) -> str:
        return """You are a Problem Enhancement Agent. Take a vague problem and transform it into a COMPLETE, WELL-FORMED ML problem.

INPUT: {raw_problem_text}

INSTRUCTIONS:
1. Identify what to predict/classify/detect. Infer from context if unclear.
2. Identify or infer input features.
3. Determine problem type: classification or regression.
4. Fill in: intended_use, data_source, evaluation_metric.

OUTPUT FORMAT (JSON ONLY):
{{
  "problem_type": "<classification|regression>",
  "target_variable": "<target>",
  "input_features": ["<f1>", "<f2>", "..."],
  "intended_use": "<use>",
  "data_source": "<source>",
  "evaluation_metric": "<metric>"
}}

Your Response:"""

    def _default_analysis_prompt(self) -> str:
        return """Analyze this content and determine if it's a valid ML problem.

Title: {title}
Body: {text}

If it's an ML problem, output in this format:
### ML Problem Title:
(clear title)
### Problem Type:
(classification|regression|...)
### Problem Statement:
(description)
### Target Variable:
(what to predict)
### Input Features:
(features)
### Data Source:
(where data comes from)
### Evaluation Metric:
(metric)

If NOT an ML problem, output: REJECTED: Not a valid ML problem.

Your Response:"""
