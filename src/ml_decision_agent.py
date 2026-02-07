"""
ML Decision Agent Module

Autonomous decision agent that determines whether ML model training is appropriate.
Implements multi-gate validation to prevent inappropriate model training.
"""

import logging
from typing import Dict, Optional, Tuple
import re

logger = logging.getLogger(__name__)


class MLDecisionAgent:
    """
    Autonomous ML Decision Agent - decides whether model training should occur.
    
    Implements a multi-gate validation system:
    - Gate 1: Problem Intent Classification
    - Gate 2: ML Feasibility Check
    - Gate 3: Causal Validity Check
    - Gate 4: Model Justification
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.strict_mode = config.get('strict_mode', True)  # Prefer refusal over incorrect automation
        
    def decide(self, problem: Dict, direct_problem_mode: bool = False) -> Dict:
        """
        Main decision function - runs all gates and returns decision.
        
        Args:
            problem: Problem dictionary
            direct_problem_mode: If True, skip Gate 1 (Intent Classification) since user explicitly provided ML problem
        
        Returns:
            {
                'decision': 'train' | 'do_not_train',
                'reasoning': str,
                'gate_results': dict,
                'recommended_action': str
            }
        """
        gate_results = {}
        
        # Auto-detect direct problem mode from problem source
        is_direct_problem = direct_problem_mode or (problem.get('source') == 'direct_input')
        
        # GATE 1: Problem Intent Classification (SKIP in direct problem mode)
        logger.info(f"ML Decision Agent: direct_problem_mode={direct_problem_mode}, problem.source={problem.get('source')}, is_direct_problem={is_direct_problem}")
        if is_direct_problem:
            # In direct problem mode, assume user provided a valid ML problem
            # Skip intent classification but still validate other aspects
            logger.info("Skipping Gate 1 (Intent Classification) - Direct problem mode")
            gate_results['intent'] = {
                'category': 'predictive_ml_task',
                'confidence': 1.0,
                'reasoning': 'Direct problem mode: User explicitly provided ML problem. Intent classification skipped.',
                'skipped': True
            }
            # Skip to Gate 2 - don't run intent classification
            pass
        else:
            logger.info("Running Gate 1 (Intent Classification)")
            intent_result = self._gate1_intent_classification(problem)
            gate_results['intent'] = intent_result
            
            if intent_result['category'] != 'predictive_ml_task':
                return {
                    'decision': 'do_not_train',
                    'content_type': intent_result['category'],
                    'reasoning': intent_result['reasoning'],
                    'justification': intent_result['reasoning'],
                    'gate_results': gate_results,
                    'recommended_action': intent_result.get('recommended_action', 'No action needed'),
                    'category': intent_result['category']
                }
        
        # GATE 2: ML Feasibility Check (STEP 2: Prediction Necessity)
        feasibility_result = self._gate2_ml_feasibility(problem)
        gate_results['feasibility'] = feasibility_result
        
        if not feasibility_result['feasible']:
            return {
                'decision': 'do_not_train',
                'content_type': 'insufficient_information',
                'reasoning': feasibility_result['reasoning'],
                'justification': 'Problem is not well-posed for ML training. Missing required information.',
                'gate_results': gate_results,
                'recommended_action': 'Clarify target variable, input features, data source, and intended use of predictions',
                'category': 'insufficient_information'
            }
        
        # STEP 3: Research vs Application Distinction
        research_vs_app_result = self._step3_research_vs_application(problem)
        gate_results['research_vs_application'] = research_vs_app_result
        
        if research_vs_app_result['is_research']:
            return {
                'decision': 'do_not_train',
                'content_type': 'research_concept',
                'reasoning': research_vs_app_result['reasoning'],
                'justification': 'This input describes research concepts, not a predictive task. No dataset, model training, or metrics apply.',
                'gate_results': gate_results,
                'recommended_action': research_vs_app_result.get('recommended_action', 'Provide conceptual explanation, key ideas, practical implications, and limitations'),
                'category': 'research_concept',
                'conceptual_content': research_vs_app_result.get('conceptual_content', {})
            }
        
        # GATE 3: Causal Validity Check
        causal_result = self._gate3_causal_validity(problem, feasibility_result)
        gate_results['causal_validity'] = causal_result
        
        if not causal_result['valid']:
            return {
                'decision': 'do_not_train',
                'content_type': 'predictive_ml_task',
                'reasoning': causal_result['reasoning'],
                'justification': 'Causal validity concerns prevent correlation-based model training.',
                'gate_results': gate_results,
                'recommended_action': causal_result.get('recommended_action', 'Consider causal modeling or experimentation'),
                'category': 'causal_concern'
            }
        
        # GATE 4: Model Justification
        justification_result = self._gate4_model_justification(problem, feasibility_result)
        gate_results['justification'] = justification_result
        
        if not justification_result['justified']:
            return {
                'decision': 'do_not_train',
                'content_type': 'predictive_ml_task',
                'reasoning': justification_result['reasoning'],
                'justification': 'ML is not justified for this problem. Simpler alternatives may be more appropriate.',
                'gate_results': gate_results,
                'recommended_action': justification_result.get('recommended_action', 'Consider simpler alternatives'),
                'category': 'unjustified_ml'
            }
        
        # All gates passed - training is appropriate
        return {
            'decision': 'train',
            'content_type': 'predictive_ml_task',
            'reasoning': 'All validation gates passed. ML model training is appropriate.',
            'justification': justification_result.get('reasoning', 'ML is justified for this problem'),
            'gate_results': gate_results,
            'recommended_action': 'Proceed with model training',
            'ml_problem_definition': {
                'target': feasibility_result.get('target'),
                'features': feasibility_result.get('features'),
                'task_type': feasibility_result.get('task_type'),
                'data_source': feasibility_result.get('data_source'),
                'intended_use': feasibility_result.get('intended_use')
            }
        }
    
    def _gate1_intent_classification(self, problem: Dict) -> Dict:
        """
        STEP 1: Content Type Classification (MANDATORY)
        
        Classifies input into:
        - predictive_ml_task (regression/classification/forecasting)
        - ml_research_paper (research paper/technique discussion)
        - system_design (architecture/design explanation)
        - personal_experience (opinion/narrative)
        - news_announcement (news/announcement)
        - unknown (insufficient information)
        """
        problem_text = problem.get('full_text', '').lower()
        title = problem.get('title', '').lower()
        combined = f"{title} {problem_text}".lower()
        
        # ML Research Paper / Technique Discussion indicators
        research_paper_indicators = [
            'paper', 'arxiv', 'research paper', 'publication', 'preprint',
            'technique', 'methodology', 'approach', 'algorithm',
            'test-time training', 'meta-learning', 'few-shot learning',
            'transfer learning', 'domain adaptation', 'continual learning',
            'model internals', 'learning theory', 'theoretical analysis',
            'architectural mechanism', 'fast weights', 'inner loop', 'outer loop',
            'gradient descent', 'backpropagation', 'optimization theory',
            'neural architecture', 'attention mechanism', 'transformer',
            'proposes', 'introduces', 'presents', 'demonstrates'
        ]
        
        # System Design / Architecture Explanation indicators
        system_design_indicators = [
            'system design', 'architecture', 'design pattern', 'framework',
            'pipeline', 'infrastructure', 'deployment', 'production system',
            'mlops', 'ml infrastructure', 'serving', 'inference',
            'scalability', 'performance optimization', 'system architecture',
            'explains', 'explanation', 'how it works', 'mechanism',
            'workflow', 'process', 'methodology'
        ]
        
        # Research discussion / theory indicators
        research_discussion_indicators = [
            'discussion', '[d]', 'curious if', 'wondering', 'thoughts?',
            'theoretical', 'conceptual', 'framework', 'paradigm',
            'causation', 'correlation', 'causal inference', 'judea pearl',
            'ladder of causation', 'do-calculus', 'scm', 'structural causal model',
            'why ml fails', 'ml failure', 'production ml', 'distribution shift',
            'robustness', 'silent failure', 'research', 'analysis', 'study',
            'explain', 'explanation', 'understanding', 'concept',
            'won\'t fix', 'can\'t solve', 'limitation', 'problem with'
        ]
        
        # News / Announcement indicators
        news_indicators = [
            'announcement', 'news', 'release', 'launch', 'update',
            'breaking', 'just announced', 'coming soon', 'new feature'
        ]
        
        # Personal experience / opinion indicators (use word boundaries to avoid false matches)
        personal_indicators = [
            'my experience', 'i think', 'in my opinion', 'i believe',
            'i found', 'i noticed', ' personal ', 'anecdote', ' story ',
            'i tried', 'i used', 'my project', '^story', 'story$'
        ]
        
        # Product / startup / career discussion
        product_indicators = [
            'startup', 'product', 'business', 'career', 'job', 'salary',
            'company', 'team', 'hiring', 'interview', 'resume'
        ]
        
        # Exploratory data analysis
        exploratory_indicators = [
            'explore', 'exploratory', 'analyze data', 'data exploration',
            'visualize', 'understand data', 'data insights'
        ]
        
        # Predictive ML task indicators (action verbs + ML keywords)
        ml_action_indicators = [
            'build', 'create', 'develop', 'implement', 'train', 'predict',
            'classify', 'forecast', 'estimate', 'solve', 'need', 'want',
            'looking for', 'help with', 'how to', 'tutorial', 'guide',
            'i need to', 'i want to', 'can you help', 'how do i'
        ]
        ml_keywords = [
            'machine learning', 'ml model', 'neural network', 'classifier',
            'regression', 'prediction', 'dataset', 'training data',
            'predict', 'forecast', 'classify'
        ]
        
        has_ml_action = any(action in combined for action in ml_action_indicators)
        has_ml_keywords = any(keyword in combined for keyword in ml_keywords)
        
        # Classification logic - Check in priority order
        
        # 1. ML Research Paper / Technique Discussion
        if any(indicator in combined for indicator in research_paper_indicators):
            return {
                'category': 'ml_research_paper',
                'confidence': 0.95,
                'reasoning': 'This appears to be an ML research paper or technique discussion. It describes ML concepts, methodologies, or theoretical work, not a predictive task requiring model training.',
                'recommended_action': 'Provide conceptual explanation, key ideas, practical implications, and limitations. No dataset, model training, or metrics apply.'
            }
        
        # 2. System Design / Architecture Explanation
        if any(indicator in combined for indicator in system_design_indicators):
            return {
                'category': 'system_design',
                'confidence': 0.9,
                'reasoning': 'This appears to be a system design or architecture explanation. It describes how systems work, not a predictive ML task.',
                'recommended_action': 'Provide system explanation and design insights. No model training needed.'
            }
        
        # 3. Research Discussion / Theory
        if any(indicator in combined for indicator in research_discussion_indicators):
            # Check if it's actually asking for implementation despite discussion markers
            if has_ml_action and has_ml_keywords:
                # Might be a predictive task despite discussion markers
                return {
                    'category': 'predictive_ml_task',
                    'confidence': 0.6,
                    'reasoning': 'Contains discussion markers but also has ML action intent'
                }
            return {
                'category': 'research_discussion',
                'confidence': 0.9,
                'reasoning': 'This input describes an ML concept or research work, not a predictive task. No dataset, model training, or metrics apply.',
                'recommended_action': 'Provide conceptual explanation, key ideas, practical implications, and limitations. No model training needed.'
            }
        
        # 4. News / Announcement
        if any(indicator in combined for indicator in news_indicators):
            return {
                'category': 'news_announcement',
                'confidence': 0.85,
                'reasoning': 'This appears to be a news item or announcement, not an ML problem requiring model training.',
                'recommended_action': 'Tag as news/announcement. Extract key information if relevant.'
            }
        
        # 5. Personal Experience (use word boundaries to avoid false matches like "history" matching "story")
        import re
        personal_matched = False
        for indicator in personal_indicators:
            # Use word boundaries for single words, exact phrase matching for multi-word
            if ' ' in indicator.strip():
                # Multi-word phrase - use exact match
                if indicator.strip() in combined:
                    personal_matched = True
                    break
            else:
                # Single word - use word boundary regex
                pattern = r'\b' + re.escape(indicator.strip()) + r'\b'
                if re.search(pattern, combined, re.IGNORECASE):
                    personal_matched = True
                    break
        
        if personal_matched:
            return {
                'category': 'personal_experience',
                'confidence': 0.8,
                'reasoning': 'This appears to be a personal experience or opinion post, not a defined ML problem.',
                'recommended_action': 'Tag as personal experience. Extract insights if valuable.'
            }
        
        if any(indicator in combined for indicator in product_indicators):
            return {
                'category': 'product_discussion',
                'confidence': 0.8,
                'reasoning': 'This appears to be a product, startup, or career discussion, not an ML problem.',
                'recommended_action': 'Tag appropriately. No ML training needed.'
            }
        
        if any(indicator in combined for indicator in exploratory_indicators):
            return {
                'category': 'exploratory_analysis',
                'confidence': 0.7,
                'reasoning': 'This appears to be an exploratory data analysis request, not a predictive ML task.',
                'recommended_action': 'Use data exploration tools. No predictive model needed.'
            }
        
        if has_ml_action and has_ml_keywords:
            return {
                'category': 'predictive_ml_task',
                'confidence': 0.8,
                'reasoning': 'Contains ML action verbs and keywords indicating a predictive ML task.'
            }
        
        # Unknown / insufficient information
        return {
            'category': 'unknown',
            'confidence': 0.5,
            'reasoning': 'Cannot clearly classify content type. Insufficient information to determine if this is a predictive ML task.',
            'recommended_action': 'Request clarification on content type and requirements.'
        }
    
    def _gate2_ml_feasibility(self, problem: Dict) -> Dict:
        """
        STEP 2: Prediction Necessity Check
        
        Only if classified as a predictive ML task, explicitly identify:
        - Target variable (what is being predicted)
        - Input features
        - Dataset source
        - Intended use of predictions
        
        If ANY are missing, ambiguous, or inferred → STOP
        
        NOTE: In direct problem mode, uses canonicalized information if available.
        """
        # Check if canonicalized information is available (from LLM canonicalization)
        canonical_problem = problem.get('canonical_problem', {})
        if canonical_problem:
            # Use canonicalized information - more reliable
            target = canonical_problem.get('target_variable', '')
            features = canonical_problem.get('input_features', [])
            data_source = canonical_problem.get('data_source', '')
            intended_use = canonical_problem.get('intended_use', '')
            problem_type = canonical_problem.get('problem_type', 'classification')
            
            # Validate canonicalized information
            if target and features:
                return {
                    'feasible': True,
                    'target': target,
                    'features': features if isinstance(features, list) else [features],
                    'task_type': problem_type if problem_type != 'unknown' else 'classification',
                    'data_source': data_source if data_source else 'to be discovered',
                    'intended_use': intended_use if intended_use else 'business decision support',
                    'reasoning': 'Problem is well-posed using canonicalized information. Target variable, features, and task type identified.',
                    'from_canonical': True
                }
        
        # Fallback to original text-based extraction
        problem_text = problem.get('full_text', '')
        title = problem.get('title', '')
        combined = f"{title} {problem_text}"
        
        # Extract target variable mentions
        target_patterns = [
            r'predict\s+(\w+)',
            r'forecast\s+(\w+)',
            r'classify\s+(\w+)',
            r'target\s+(?:variable|column|feature)?\s*:?\s*(\w+)',
            r'predicting\s+(\w+)',
            r'outcome\s+(?:variable|is)?\s*:?\s*(\w+)',
            r'dependent\s+variable\s*:?\s*(\w+)'
        ]
        
        targets = []
        for pattern in target_patterns:
            matches = re.findall(pattern, combined, re.IGNORECASE)
            targets.extend(matches)
        
        # Extract feature mentions
        feature_patterns = [
            r'feature[s]?\s+(?:are|include|:)?\s*([^\.]+)',
            r'input[s]?\s+(?:are|include|:)?\s*([^\.]+)',
            r'variable[s]?\s+(?:are|include|:)?\s*([^\.]+)',
            r'based on\s+([^\.]+)',
            r'using\s+([^\.]+)\s+to'
        ]
        
        features = []
        for pattern in feature_patterns:
            matches = re.findall(pattern, combined, re.IGNORECASE)
            features.extend(matches)
        
        # Check for data source mentions
        data_source_patterns = [
            r'dataset[s]?',
            r'data\s+(?:from|source|available)',
            r'csv', r'excel', r'database',
            r'kaggle', r'huggingface', r'uci'
        ]
        
        has_data_source = any(re.search(pattern, combined, re.IGNORECASE) for pattern in data_source_patterns)
        
        # Check for intended use of predictions
        use_patterns = [
            r'use\s+(?:for|to|in)',
            r'predictions?\s+(?:for|to|will)',
            r'deploy', r'production', r'application',
            r'help\s+(?:with|to)', r'goal\s+is', r'purpose'
        ]
        intended_use = None
        for pattern in use_patterns:
            match = re.search(pattern, combined, re.IGNORECASE)
            if match:
                # Try to extract context around the match
                start = max(0, match.start() - 50)
                end = min(len(combined), match.end() + 50)
                intended_use = combined[start:end].strip()
                break
        
        # Determine task type
        task_type = 'unknown'
        if any(word in combined.lower() for word in ['classify', 'classification', 'category', 'label']):
            task_type = 'classification'
        elif any(word in combined.lower() for word in ['predict', 'forecast', 'regression', 'continuous']):
            task_type = 'regression'
        elif any(word in combined.lower() for word in ['cluster', 'group', 'segment']):
            task_type = 'clustering'
        
        # Check if this is a direct problem (user explicitly provided)
        is_direct_problem = problem.get('source') == 'direct_input'
        
        # Validation - STRICT: All must be explicit or clearly inferable
        # BUT: In direct problem mode, be more lenient - user explicitly provided the problem
        missing = []
        ambiguous = []
        
        if not targets and not any(word in combined.lower() for word in ['target', 'predict', 'forecast', 'classify']):
            missing.append('target variable')
        elif targets and len(targets) > 1 and not is_direct_problem:
            ambiguous.append('target variable (multiple candidates)')
        
        if not features and not has_data_source:
            missing.append('input features or data source')
        elif features and len(features) > 3 and not is_direct_problem:
            ambiguous.append('input features (too many candidates, may be ambiguous)')
        
        if not intended_use and not is_direct_problem:
            missing.append('intended use of predictions')
        
        # In direct problem mode, infer missing information from context
        if missing and is_direct_problem:
            # Try to infer from problem text
            inferred_target = None
            if 'predict' in combined.lower():
                # Extract what comes after "predict"
                match = re.search(r'predict\s+(\w+)', combined, re.IGNORECASE)
                if match:
                    inferred_target = match.group(1)
            elif 'churn' in combined.lower():
                inferred_target = 'churn'
            elif 'price' in combined.lower() or 'cost' in combined.lower():
                inferred_target = 'price'
            
            if inferred_target:
                missing = [m for m in missing if 'target' not in m.lower()]
                targets = [inferred_target]
            
            # Infer features from "based on" or "using"
            if 'based on' in combined.lower():
                match = re.search(r'based on\s+([^\.]+)', combined, re.IGNORECASE)
                if match:
                    inferred_features = [f.strip() for f in match.group(1).split(',')]
                    if inferred_features:
                        missing = [m for m in missing if 'feature' not in m.lower() and 'data source' not in m.lower()]
                        features = inferred_features
        
        if missing and not is_direct_problem:
            return {
                'feasible': False,
                'reasoning': f'Problem is not well-posed for ML training. Missing: {", ".join(missing)}. Cannot proceed without explicit target variable, input features, data source, and intended use.',
                'missing': missing
            }
        
        if ambiguous and self.strict_mode and not is_direct_problem:
            return {
                'feasible': False,
                'reasoning': f'Problem definition is ambiguous: {", ".join(ambiguous)}. In strict mode, ambiguous problems are rejected.',
                'ambiguous': ambiguous
            }
        
        # In direct problem mode, if still missing critical info, use safe defaults
        if missing and is_direct_problem:
            logger.warning(f"Direct problem mode: Some information missing ({', '.join(missing)}), using inferred/defaults")
        
        return {
            'feasible': True,
            'target': targets[0] if targets else 'inferred from context',
            'features': features[0] if features else 'to be extracted from dataset',
            'task_type': task_type if task_type != 'unknown' else 'classification',  # Default
            'data_source': 'mentioned' if has_data_source else 'to be discovered',
            'intended_use': intended_use if intended_use else 'not explicitly stated',
            'reasoning': 'Target variable, features, data source, and intended use identified or inferable. Problem is well-posed for ML training.'
        }
    
    def _step3_research_vs_application(self, problem: Dict) -> Dict:
        """
        STEP 3: Research vs Application Distinction
        
        If the input discusses:
        - New training paradigms (test-time training, meta-learning)
        - Model internals or learning theory
        - Architectural mechanisms (fast weights, inner/outer loops)
        
        Then provide conceptual explanation instead of training a model.
        """
        problem_text = problem.get('full_text', '').lower()
        title = problem.get('title', '').lower()
        combined = f"{title} {problem_text}".lower()
        
        # New training paradigms
        training_paradigm_indicators = [
            'test-time training', 'meta-learning', 'few-shot learning',
            'zero-shot learning', 'continual learning', 'lifelong learning',
            'online learning', 'incremental learning', 'transfer learning',
            'domain adaptation', 'adversarial training', 'self-supervised',
            'semi-supervised', 'weakly supervised', 'unsupervised',
            'reinforcement learning', 'imitation learning'
        ]
        
        # Model internals / learning theory
        model_internals_indicators = [
            'model internals', 'learning theory', 'optimization theory',
            'gradient descent', 'backpropagation', 'activation function',
            'loss function', 'objective function', 'regularization',
            'overfitting', 'underfitting', 'bias-variance', 'generalization',
            'neural network architecture', 'layer', 'weights', 'parameters'
        ]
        
        # Architectural mechanisms
        architectural_indicators = [
            'fast weights', 'slow weights', 'inner loop', 'outer loop',
            'attention mechanism', 'self-attention', 'transformer',
            'convolution', 'pooling', 'dropout', 'batch normalization',
            'residual connection', 'skip connection', 'gating mechanism'
        ]
        
        # Check for research concept indicators
        has_training_paradigm = any(indicator in combined for indicator in training_paradigm_indicators)
        has_model_internals = any(indicator in combined for indicator in model_internals_indicators)
        has_architectural = any(indicator in combined for indicator in architectural_indicators)
        
        if has_training_paradigm or has_model_internals or has_architectural:
            # Extract key concepts
            concepts = []
            if has_training_paradigm:
                concepts.append('training paradigm')
            if has_model_internals:
                concepts.append('model internals/learning theory')
            if has_architectural:
                concepts.append('architectural mechanism')
            
            return {
                'is_research': True,
                'reasoning': f'This input discusses {", ".join(concepts)}. These are research concepts, not a predictive task. Do NOT select datasets, choose regression/classification, or output metrics.',
                'recommended_action': 'Provide conceptual explanation, key ideas, practical implications, and limitations instead of training a model.',
                'conceptual_content': {
                    'concepts': concepts,
                    'type': 'research_concept',
                    'should_explain': True,
                    'should_not_train': True
                }
            }
        
        return {
            'is_research': False,
            'reasoning': 'This appears to be an application task, not a research concept discussion.'
        }
    
    def _gate3_causal_validity(self, problem: Dict, feasibility: Dict) -> Dict:
        """
        GATE 3: Causal Validity Check
        
        Asks:
        - Is the goal prediction or decision-making?
        - Would learning correlations cause harm if deployed?
        - Is causal modeling required instead?
        """
        problem_text = problem.get('full_text', '').lower()
        title = problem.get('title', '').lower()
        combined = f"{title} {problem_text}"
        
        # Decision-making indicators (higher risk)
        decision_indicators = [
            'decision', 'action', 'intervene', 'treatment', 'policy',
            'recommend', 'should', 'must', 'need to do', 'take action'
        ]
        
        # Causal concern indicators
        causal_indicators = [
            'causal', 'causation', 'cause', 'effect', 'intervention',
            'counterfactual', 'do-calculus', 'structural model'
        ]
        
        # High-stakes domains (where correlation errors are dangerous)
        high_stakes_domains = [
            'medical', 'health', 'surgery', 'diagnosis', 'treatment',
            'criminal', 'justice', 'sentencing', 'bail',
            'hiring', 'employment', 'credit', 'loan', 'insurance',
            'safety', 'risk', 'critical'
        ]
        
        is_decision_making = any(indicator in combined for indicator in decision_indicators)
        has_causal_indicators = any(indicator in combined for indicator in causal_indicators)
        is_high_stakes = any(domain in combined for domain in high_stakes_domains)
        
        # Check for explicit correlation warnings
        correlation_warnings = [
            'correlation', 'not causation', 'correlation ≠', 'spurious',
            'confounding', 'bias'
        ]
        has_correlation_warning = any(warning in combined for warning in correlation_warnings)
        
        # If it's decision-making in high-stakes domain, require causal reasoning
        if is_decision_making and is_high_stakes:
            if not has_causal_indicators:
                return {
                    'valid': False,
                    'reasoning': 'This is a decision-making problem in a high-stakes domain. Correlation-based ML models could cause harm. Causal modeling or experimentation is required, but not mentioned.',
                    'recommended_action': 'Use causal inference methods (SCMs, do-calculus) or randomized experiments. Do not use observational correlation-based models.'
                }
        
        # If correlation warnings are present, take them seriously
        if has_correlation_warning:
            return {
                'valid': False,
                'reasoning': 'The problem explicitly mentions correlation concerns. Correlation-based ML models are not appropriate here.',
                'recommended_action': 'Address causal structure before modeling. Consider causal inference methods.'
            }
        
        # If it's pure prediction without decision-making, correlation is acceptable
        if not is_decision_making:
            return {
                'valid': True,
                'reasoning': 'This appears to be a pure prediction task without direct decision-making. Correlation-based models may be acceptable.',
                'risk_level': 'low'
            }
        
        # Decision-making but not high-stakes - warn but allow
        return {
            'valid': True,
            'reasoning': 'This involves decision-making. Consider causal validity, but correlation-based models may be acceptable with proper validation.',
            'risk_level': 'medium',
            'warning': 'Monitor for distribution shift and validate causal assumptions if deployed.'
        }
    
    def _gate4_model_justification(self, problem: Dict, feasibility: Dict) -> Dict:
        """
        GATE 4: Model Justification
        
        ABSOLUTE CONSTRAINTS:
        - NEVER hallucinate datasets
        - NEVER assign "regression" by default
        - NEVER output R², MSE, or accuracy without a real prediction task
        - NEVER assume Reddit ML posts imply AutoML execution
        
        Justifies:
        - Why ML is needed over rules/heuristics
        - Model family selection (must be explicit, not default)
        - Dataset choice (no hallucination allowed)
        
        NOTE: In direct problem mode, more permissive about dataset discovery.
        """
        problem_text = problem.get('full_text', '').lower()
        task_type = feasibility.get('task_type', 'unknown')
        
        # In direct problem mode, if canonicalized info exists, be more permissive
        is_direct_problem = problem.get('source') == 'direct_input'
        has_canonical = bool(problem.get('canonical_problem'))
        
        # ABSOLUTE CONSTRAINT: Never assign regression by default
        if task_type == 'regression' and 'regression' not in problem_text and 'predict' not in problem_text and 'forecast' not in problem_text:
            return {
                'justified': False,
                'reasoning': 'Regression was inferred but not explicitly mentioned. In strict mode, we never assign "regression" by default without explicit indication.',
                'recommended_action': 'Clarify task type. Do not default to regression.'
            }
        
        # Check if simple rules would suffice
        simple_rule_indicators = [
            'if-then', 'rule-based', 'heuristic', 'threshold',
            'simple logic', 'straightforward', 'straightforward rule'
        ]
        
        # Check for complexity indicators that justify ML
        complexity_indicators = [
            'complex', 'non-linear', 'pattern', 'learn', 'many features',
            'large dataset', 'thousands', 'millions', 'machine learning',
            'neural network', 'deep learning'
        ]
        
        has_simple_indicators = any(indicator in problem_text for indicator in simple_rule_indicators)
        has_complexity = any(indicator in problem_text for indicator in complexity_indicators)
        
        if has_simple_indicators and not has_complexity:
            return {
                'justified': False,
                'reasoning': 'This problem may be solvable with simple rules or heuristics. ML may be overkill.',
                'recommended_action': 'Consider rule-based solution first. Only use ML if rules are insufficient.'
            }
        
        # ABSOLUTE CONSTRAINT: Never hallucinate datasets
        # Dataset must be explicitly mentioned, discoverable, or provided
        dataset_mentioned = any(word in problem_text for word in ['dataset', 'data', 'csv', 'kaggle', 'huggingface', 'uci'])
        data_source = feasibility.get('data_source', '')
        
        # In direct problem mode with canonicalized info, be more permissive
        if is_direct_problem and has_canonical:
            # User explicitly provided problem, allow dataset discovery
            return {
                'justified': True,
                'reasoning': 'Direct problem mode: User explicitly provided ML problem. Dataset discovery will attempt to find appropriate data.',
                'note': 'CRITICAL: Ensure dataset discovery finds REAL, appropriate datasets. NO hallucination allowed. If no suitable dataset found, reject training.',
                'warning': 'Dataset discovery must validate dataset exists and is appropriate before proceeding.',
                'direct_problem_mode': True
            }
        
        if not dataset_mentioned and data_source == 'to be discovered':
            # In strict mode, require explicit dataset mention
            if self.strict_mode:
                return {
                    'justified': False,
                    'reasoning': 'Dataset not explicitly mentioned. In strict mode, we never hallucinate datasets. Dataset must be explicitly provided or mentioned.',
                    'recommended_action': 'Require explicit dataset mention or provision. Do not invent or assume datasets.'
                }
            else:
                return {
                    'justified': True,
                    'reasoning': 'Dataset not explicitly mentioned, but dataset discovery module will attempt to find appropriate data.',
                    'note': 'CRITICAL: Ensure dataset discovery finds REAL, appropriate datasets. NO hallucination allowed. If no suitable dataset found, reject training.',
                    'warning': 'Dataset discovery must validate dataset exists and is appropriate before proceeding.'
                }
        
        # Validate task type is explicit, not inferred
        if task_type == 'unknown' or (task_type == 'classification' and 'classif' not in problem_text and 'categor' not in problem_text):
            return {
                'justified': False,
                'reasoning': f'Task type "{task_type}" is not explicitly clear. Task type must be explicitly stated or clearly inferable from context.',
                'recommended_action': 'Clarify task type explicitly. Do not infer without strong evidence.'
            }
        
        return {
            'justified': True,
            'reasoning': 'ML is justified. Problem complexity and requirements suggest ML is appropriate. Task type is explicit, dataset is mentioned or discoverable.',
            'model_family': task_type,
            'dataset_validated': dataset_mentioned or data_source != 'to be discovered',
            'note': f'Proceed with {task_type} model family. Ensure dataset is real and appropriate before training.'
        }
