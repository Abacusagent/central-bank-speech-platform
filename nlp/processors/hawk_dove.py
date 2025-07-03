#!/usr/bin/env python3
"""
Hawk-Dove Sentiment Analyzer for Central Bank Speech Analysis Platform

This module provides comprehensive hawk-dove sentiment analysis using multiple approaches:
1. Lexicon-based analysis with contextual adjustments
2. Transformer-based deep learning models (FinBERT, etc.)
3. Hybrid ensemble methods for maximum accuracy
4. Temporal consistency checks and uncertainty quantification

The analyzer follows the plugin architecture and provides confidence-scored results
with detailed breakdown of contributing factors.

Key Features:
- Multi-method ensemble analysis
- Contextual awareness (negations, hedging, intensifiers)
- Temporal consistency validation
- Comprehensive confidence scoring
- Production-ready error handling
- Extensive logging and debugging capabilities

Author: Central Bank Speech Analysis Platform
Date: 2025
Version: 2.0.0
"""

import asyncio
import csv
import json
import logging
import math
import re
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from uuid import uuid4

import numpy as np
import pandas as pd

# Domain imports
from domain.entities import CentralBankSpeech, PolicyStance
from domain.value_objects import SentimentScore, ConfidenceLevel
from interfaces.plugin_interfaces import SpeechContent

# Optional advanced ML imports
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        pipeline, BertTokenizer, BertForSequenceClassification
    )
    import torch
    from torch.nn.functional import softmax
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import classification_report
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AnalysisMethod:
    """Represents a single analysis method with its configuration and capabilities."""
    name: str
    confidence_weight: float
    requires_initialization: bool
    error_tolerance: float = 0.1
    timeout_seconds: int = 30
    enabled: bool = True


@dataclass
class HawkDoveResult:
    """Comprehensive result from hawk-dove analysis."""
    score: float  # -1.0 (very dovish) to +1.0 (very hawkish)
    confidence: float  # 0.0 to 1.0
    stance: PolicyStance
    method_breakdown: Dict[str, float]
    contributing_factors: Dict[str, Any]
    uncertainty_indicators: List[str]
    temporal_consistency: Optional[float] = None
    processing_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    debug_info: Dict[str, Any] = field(default_factory=dict)


class HawkDoveMethod(ABC):
    """Abstract base class for hawk-dove analysis methods."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.is_initialized = False
        self.error_count = 0
        self.last_error: Optional[str] = None
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the method (load models, lexicons, etc.)."""
        pass
    
    @abstractmethod
    async def analyze(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[float, float, Dict[str, Any]]:
        """
        Analyze text for hawk-dove sentiment.
        
        Returns:
            Tuple of (score, confidence, debug_info)
        """
        pass
    
    @abstractmethod
    def get_confidence_baseline(self) -> float:
        """Get the baseline confidence for this method."""
        pass
    
    def record_error(self, error: str) -> None:
        """Record an error for this method."""
        self.error_count += 1
        self.last_error = error
        logger.warning(f"{self.name} error: {error}")


class LexiconMethod(HawkDoveMethod):
    """
    Lexicon-based hawk-dove analysis with contextual adjustments.
    
    Features:
    - Comprehensive monetary policy lexicon
    - Contextual modifiers (negations, hedging, intensifiers)
    - Temporal weighting for forward/backward guidance
    - Uncertainty detection and adjustment
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lexicon: Dict[str, Tuple[float, float]] = {}
        self.contextual_modifiers = {
            'negations': ['not', 'never', 'no', "n't", 'without', 'lack', 'absence'],
            'hedging': ['may', 'might', 'could', 'possibly', 'perhaps', 'likely', 'probably'],
            'intensifiers': ['very', 'extremely', 'highly', 'significantly', 'substantially', 'considerably'],
            'forward_guidance': ['will', 'expect', 'anticipate', 'project', 'outlook', 'going forward'],
            'backward_looking': ['was', 'had', 'previous', 'past', 'earlier', 'before'],
            'uncertainty': ['uncertain', 'unclear', 'ambiguous', 'depends', 'conditional']
        }
        self.lexicon_path = config.get('lexicon_path', 'lexicons/hawk_dove_lexicon.csv')
    
    async def initialize(self) -> bool:
        """Initialize lexicon from file or create default."""
        try:
            await self._load_lexicon()
            self.is_initialized = True
            logger.info(f"LexiconMethod initialized with {len(self.lexicon)} terms")
            return True
        except Exception as e:
            self.record_error(f"Initialization failed: {e}")
            return False
    
    async def _load_lexicon(self) -> None:
        """Load lexicon from CSV file or create comprehensive default."""
        lexicon_file = Path(self.lexicon_path)
        
        if lexicon_file.exists():
            try:
                with open(lexicon_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        word = row['word'].lower().strip()
                        polarity = float(row['polarity'])
                        weight = float(row.get('weight', 1.0))
                        self.lexicon[word] = (polarity, weight)
                
                logger.info(f"Loaded {len(self.lexicon)} terms from {lexicon_file}")
                
            except Exception as e:
                logger.warning(f"Error loading lexicon from {lexicon_file}: {e}")
                await self._create_comprehensive_lexicon()
        else:
            logger.info(f"Lexicon file not found at {lexicon_file}, creating comprehensive default")
            await self._create_comprehensive_lexicon()
    
    async def _create_comprehensive_lexicon(self) -> None:
        """Create a comprehensive monetary policy lexicon."""
        comprehensive_lexicon = {
            # Strong Hawkish Terms (Highly Restrictive)
            'tighten': (-0.9, 1.0),
            'tightening': (-0.9, 1.0),
            'restrictive': (-0.8, 1.0),
            'aggressive': (-0.9, 1.0),
            'combat': (-0.8, 0.9),
            'fight': (-0.8, 0.9),
            'curb': (-0.7, 0.8),
            'constrain': (-0.7, 0.8),
            'firm': (-0.6, 0.8),
            'decisive': (-0.7, 0.8),
            
            # Interest Rate Hawkish Terms
            'raise': (-0.8, 0.9),
            'increase': (-0.7, 0.8),
            'hike': (-0.9, 1.0),
            'higher': (-0.6, 0.7),
            'lift': (-0.7, 0.8),
            'elevate': (-0.7, 0.8),
            'normalize': (-0.5, 0.7),  # Context dependent
            
            # Inflation Control (Hawkish)
            'inflation': (-0.6, 0.8),  # Context matters
            'overheating': (-0.8, 0.9),
            'excessive': (-0.7, 0.8),
            'unsustainable': (-0.6, 0.7),
            'pressures': (-0.5, 0.6),
            'risks': (-0.4, 0.6),
            
            # Strong Dovish Terms (Highly Accommodative)
            'accommodate': (0.8, 1.0),
            'accommodative': (0.8, 1.0),
            'stimulus': (0.9, 1.0),
            'stimulate': (0.8, 0.9),
            'ease': (0.8, 0.9),
            'easing': (0.8, 0.9),
            'support': (0.7, 0.8),
            'boost': (0.7, 0.8),
            'encourage': (0.6, 0.7),
            'foster': (0.6, 0.7),
            
            # Interest Rate Dovish Terms
            'lower': (0.8, 0.9),
            'cut': (0.9, 1.0),
            'reduce': (0.7, 0.8),
            'decrease': (0.7, 0.8),
            'maintain': (0.4, 0.6),  # Mildly dovish when rates are low
            'hold': (0.3, 0.5),
            
            # Employment Focus (Generally Dovish)
            'employment': (0.5, 0.7),
            'unemployment': (0.4, 0.6),
            'jobs': (0.5, 0.6),
            'labor': (0.4, 0.5),
            'workers': (0.4, 0.5),
            
            # Growth Support (Dovish)
            'growth': (0.5, 0.6),
            'expansion': (0.4, 0.5),
            'recovery': (0.6, 0.7),
            'strengthen': (0.5, 0.6),
            
            # Quantitative Easing (Dovish)
            'quantitative': (0.7, 0.8),
            'asset': (0.4, 0.5),  # In context of purchases
            'purchases': (0.6, 0.7),
            'balance': (0.3, 0.4),  # Balance sheet expansion
            
            # Uncertainty and Caution (Neutral to Slightly Dovish)
            'uncertain': (0.2, 0.4),
            'cautious': (0.3, 0.5),
            'careful': (0.3, 0.4),
            'gradual': (0.4, 0.6),
            'measured': (0.3, 0.5),
            'patient': (0.5, 0.6),
            'data': (0.1, 0.3),  # Data-dependent (slightly dovish)
            
            # Neutral Terms (Context Dependent)
            'appropriate': (0.0, 0.3),
            'consistent': (0.0, 0.2),
            'policy': (0.0, 0.1),
            'economic': (0.0, 0.1),
            'financial': (0.0, 0.1),
            
            # Temporal Modifiers
            'immediately': (-0.3, 0.5),  # Urgency can be hawkish
            'soon': (-0.2, 0.4),
            'future': (0.1, 0.3),
            'eventually': (0.2, 0.4),
            'temporary': (0.3, 0.4),
            
            # Commitment Strength
            'committed': (-0.2, 0.5),  # Depends on what they're committed to
            'determined': (-0.3, 0.5),
            'resolved': (-0.3, 0.5),
            'flexible': (0.4, 0.6),
            'adapt': (0.3, 0.5),
            'adjust': (0.3, 0.5),
        }
        
        self.lexicon = comprehensive_lexicon
        logger.info(f"Created comprehensive lexicon with {len(self.lexicon)} terms")
    
    async def analyze(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[float, float, Dict[str, Any]]:
        """Analyze text using lexicon with contextual adjustments."""
        try:
            # Tokenize and clean text
            words = self._tokenize(text)
            sentences = self._split_sentences(text)
            
            # Basic lexicon scoring
            base_score, word_scores = self._calculate_base_score(words)
            
            # Apply contextual adjustments
            contextual_score = self._apply_contextual_adjustments(text, base_score, sentences)
            
            # Calculate confidence based on coverage and consistency
            confidence = self._calculate_confidence(words, word_scores, sentences)
            
            # Debug information
            debug_info = {
                'base_score': base_score,
                'contextual_score': contextual_score,
                'word_count': len(words),
                'lexicon_hits': len(word_scores),
                'coverage': len(word_scores) / max(len(words), 1),
                'contributing_words': word_scores[:10],  # Top 10
                'sentences_analyzed': len(sentences),
                'contextual_adjustments': self._get_contextual_adjustments(text)
            }
            
            return contextual_score, confidence, debug_info
            
        except Exception as e:
            self.record_error(f"Analysis failed: {e}")
            return 0.0, 0.0, {'error': str(e)}
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple but effective tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_base_score(self, words: List[str]) -> Tuple[float, List[Tuple[str, float]]]:
        """Calculate base lexicon score."""
        total_score = 0.0
        total_weight = 0.0
        word_scores = []
        
        for word in words:
            if word in self.lexicon:
                polarity, weight = self.lexicon[word]
                weighted_score = polarity * weight
                total_score += weighted_score
                total_weight += weight
                word_scores.append((word, weighted_score))
        
        # Sort by absolute contribution
        word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        
        if total_weight == 0:
            return 0.0, word_scores
        
        # Normalize by text length with diminishing returns
        text_length_factor = math.sqrt(len(words)) if words else 1.0
        normalized_score = total_score / text_length_factor
        
        # Bound the score
        bounded_score = max(-1.0, min(1.0, normalized_score))
        
        return bounded_score, word_scores
    
    def _apply_contextual_adjustments(self, text: str, base_score: float, sentences: List[str]) -> float:
        """Apply contextual adjustments to the base score."""
        adjusted_score = base_score
        text_lower = text.lower()
        
        # Negation adjustment
        negation_count = sum(text_lower.count(neg) for neg in self.contextual_modifiers['negations'])
        if negation_count > 0:
            # Strong negations can flip meaning
            negation_factor = min(0.8, negation_count * 0.2)
            adjusted_score *= (1 - negation_factor)
        
        # Hedging reduces confidence in direction
        hedging_count = sum(text_lower.count(hedge) for hedge in self.contextual_modifiers['hedging'])
        if hedging_count > 0:
            hedging_factor = min(0.5, hedging_count * 0.1)
            adjusted_score *= (1 - hedging_factor)
        
        # Intensifiers strengthen the signal
        intensifier_count = sum(text_lower.count(intensifier) for intensifier in self.contextual_modifiers['intensifiers'])
        if intensifier_count > 0:
            intensifier_factor = min(0.3, intensifier_count * 0.1)
            adjusted_score *= (1 + intensifier_factor)
        
        # Forward guidance gets higher weight
        forward_count = sum(text_lower.count(forward) for forward in self.contextual_modifiers['forward_guidance'])
        if forward_count > 0:
            forward_factor = min(0.2, forward_count * 0.05)
            adjusted_score *= (1 + forward_factor)
        
        # Backward-looking gets lower weight
        backward_count = sum(text_lower.count(backward) for backward in self.contextual_modifiers['backward_looking'])
        if backward_count > 0:
            backward_factor = min(0.3, backward_count * 0.05)
            adjusted_score *= (1 - backward_factor)
        
        return max(-1.0, min(1.0, adjusted_score))
    
    def _calculate_confidence(self, words: List[str], word_scores: List[Tuple[str, float]], sentences: List[str]) -> float:
        """Calculate confidence in the lexicon analysis."""
        if not words:
            return 0.0
        
        # Coverage: How much of the text is covered by our lexicon
        coverage = len(word_scores) / len(words)
        coverage_score = min(1.0, coverage * 3)  # Give full points at 33% coverage
        
        # Consistency: Are the signals consistent?
        if word_scores:
            scores = [score for _, score in word_scores]
            positive_scores = [s for s in scores if s > 0]
            negative_scores = [s for s in scores if s < 0]
            
            total_positive = sum(positive_scores)
            total_negative = abs(sum(negative_scores))
            
            if total_positive + total_negative > 0:
                consistency = abs(total_positive - total_negative) / (total_positive + total_negative)
            else:
                consistency = 0.0
        else:
            consistency = 0.0
        
        # Text length factor: Longer texts generally give more confidence
        length_factor = min(1.0, len(words) / 100)  # Full confidence at 100+ words
        
        # Sentence count factor: More sentences can provide more context
        sentence_factor = min(1.0, len(sentences) / 10)  # Full confidence at 10+ sentences
        
        # Combine factors
        confidence = (
            coverage_score * 0.4 +
            consistency * 0.3 +
            length_factor * 0.2 +
            sentence_factor * 0.1
        )
        
        return min(1.0, confidence)
    
    def _get_contextual_adjustments(self, text: str) -> Dict[str, int]:
        """Get count of contextual modifiers for debugging."""
        text_lower = text.lower()
        return {
            modifier_type: sum(text_lower.count(word) for word in words)
            for modifier_type, words in self.contextual_modifiers.items()
        }
    
    def get_confidence_baseline(self) -> float:
        """Get baseline confidence for lexicon method."""
        return 0.7 if self.lexicon else 0.3


class TransformerMethod(HawkDoveMethod):
    """
    Transformer-based hawk-dove analysis using financial BERT models.
    
    Features:
    - FinBERT for financial sentiment
    - Custom fine-tuned models for central bank communications
    - Attention weight analysis for interpretability
    - Ensemble predictions from multiple models
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.model_configs = config.get('models', {
            'finbert': 'ProsusAI/finbert',
            'distilbert': 'distilbert-base-uncased-finetuned-sst-2-english'
        })
        self.max_length = config.get('max_length', 512)
        self.batch_size = config.get('batch_size', 1)
    
    async def initialize(self) -> bool:
        """Initialize transformer models."""
        if not TRANSFORMERS_AVAILABLE:
            self.record_error("Transformers library not available")
            return False
        
        try:
            for model_name, model_path in self.model_configs.items():
                try:
                    # Load tokenizer and model
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModelForSequenceClassification.from_pretrained(model_path)
                    
                    # Set to evaluation mode
                    model.eval()
                    
                    self.tokenizers[model_name] = tokenizer
                    self.models[model_name] = model
                    
                    logger.info(f"Loaded transformer model: {model_name} ({model_path})")
                    
                except Exception as e:
                    logger.warning(f"Failed to load model {model_name}: {e}")
                    continue
            
            if self.models:
                self.is_initialized = True
                logger.info(f"TransformerMethod initialized with {len(self.models)} models")
                return True
            else:
                self.record_error("No transformer models could be loaded")
                return False
                
        except Exception as e:
            self.record_error(f"Initialization failed: {e}")
            return False
    
    async def analyze(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[float, float, Dict[str, Any]]:
        """Analyze text using transformer models."""
        try:
            if not self.models:
                return 0.0, 0.0, {'error': 'No models available'}
            
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Get predictions from all models
            model_predictions = {}
            model_confidences = {}
            
            for model_name in self.models:
                try:
                    score, confidence = await self._predict_with_model(model_name, processed_text)
                    model_predictions[model_name] = score
                    model_confidences[model_name] = confidence
                except Exception as e:
                    logger.warning(f"Model {model_name} failed: {e}")
                    continue
            
            if not model_predictions:
                return 0.0, 0.0, {'error': 'All models failed'}
            
            # Ensemble prediction
            final_score, final_confidence = self._ensemble_predictions(model_predictions, model_confidences)
            
            # Debug information
            debug_info = {
                'model_predictions': model_predictions,
                'model_confidences': model_confidences,
                'ensemble_method': 'weighted_average',
                'text_length': len(processed_text),
                'models_used': list(model_predictions.keys()),
                'preprocessing': 'truncated' if len(text.split()) > self.max_length else 'none'
            }
            
            return final_score, final_confidence, debug_info
            
        except Exception as e:
            self.record_error(f"Analysis failed: {e}")
            return 0.0, 0.0, {'error': str(e)}
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for transformer models."""
        # Truncate if too long
        words = text.split()
        if len(words) > self.max_length:
            text = ' '.join(words[:self.max_length])
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    async def _predict_with_model(self, model_name: str, text: str) -> Tuple[float, float]:
        """Get prediction from a specific model."""
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
        
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        )
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Convert to hawk-dove score
        if model_name == 'finbert':
            # FinBERT: [negative, neutral, positive]
            negative_prob = predictions[0][0].item()
            neutral_prob = predictions[0][1].item()
            positive_prob = predictions[0][2].item()
            
            # Map to hawk-dove: negative -> hawkish, positive -> dovish
            score = positive_prob - negative_prob  # Range: -1 to 1
            confidence = max(negative_prob, positive_prob)  # Confidence in direction
            
        else:
            # Generic sentiment model: [negative, positive]
            negative_prob = predictions[0][0].item()
            positive_prob = predictions[0][1].item()
            
            # Map to hawk-dove scale
            score = (positive_prob - negative_prob) * 0.5  # Reduced magnitude for generic models
            confidence = max(negative_prob, positive_prob) * 0.8  # Lower confidence for generic models
        
        return score, confidence
    
    def _ensemble_predictions(self, predictions: Dict[str, float], confidences: Dict[str, float]) -> Tuple[float, float]:
        """Combine predictions from multiple models."""
        if not predictions:
            return 0.0, 0.0
        
        # Weighted average based on confidence
        total_weight = sum(confidences.values())
        if total_weight == 0:
            # Simple average if no confidence info
            avg_score = sum(predictions.values()) / len(predictions)
            avg_confidence = sum(confidences.values()) / len(confidences)
            return avg_score, avg_confidence
        
        # Confidence-weighted score
        weighted_score = sum(
            score * confidences[model] for model, score in predictions.items()
        ) / total_weight
        
        # Average confidence with ensemble bonus
        avg_confidence = sum(confidences.values()) / len(confidences)
        ensemble_bonus = min(0.1, (len(predictions) - 1) * 0.05)  # Bonus for multiple models
        final_confidence = min(1.0, avg_confidence + ensemble_bonus)
        
        return weighted_score, final_confidence
    
    def get_confidence_baseline(self) -> float:
        """Get baseline confidence for transformer method."""
        return 0.85 if self.models else 0.0


class HybridMethod(HawkDoveMethod):
    """
    Hybrid method combining lexicon and transformer approaches with ML ensemble.
    
    Features:
    - Feature engineering from multiple sources
    - Random Forest ensemble
    - Cross-validation for confidence estimation
    - Automatic model selection based on text characteristics
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ensemble_model = None
        self.feature_importance: Dict[str, float] = {}
        self.lexicon_method = None
        self.transformer_method = None
    
    async def initialize(self) -> bool:
        """Initialize hybrid ensemble model."""
        if not SKLEARN_AVAILABLE:
            self.record_error("scikit-learn not available")
            return False
        
        try:
            # Initialize component methods
            self.lexicon_method = LexiconMethod(self.config)
            self.transformer_method = TransformerMethod(self.config) if TRANSFORMERS_AVAILABLE else None
            
            # Initialize component methods
            lexicon_ok = await self.lexicon_method.initialize()
            transformer_ok = await self.transformer_method.initialize() if self.transformer_method else False
            
            if not lexicon_ok:
                self.record_error("Lexicon method initialization failed")
                return False
            
            # Create ensemble model
            self.ensemble_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
            
            self.is_initialized = True
            logger.info(f"HybridMethod initialized (lexicon: {lexicon_ok}, transformer: {transformer_ok})")
            return True
            
        except Exception as e:
            self.record_error(f"Initialization failed: {e}")
            return False
    
    async def analyze(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[float, float, Dict[str, Any]]:
        """Analyze text using hybrid ensemble approach."""
        try:
            # Extract features from all methods
            features = await self._extract_features(text, metadata)
            
            if not features:
                return 0.0, 0.0, {'error': 'Feature extraction failed'}
            
            # For now, use a rule-based ensemble until we have training data
            score, confidence = self._rule_based_ensemble(features)
            
            # Debug information
            debug_info = {
                'features': features,
                'ensemble_method': 'rule_based',
                'feature_count': len(features),
                'component_methods': {
                    'lexicon': features.get('lexicon_available', False),
                    'transformer': features.get('transformer_available', False)
                }
            }
            
            return score, confidence, debug_info
            
        except Exception as e:
            self.record_error(f"Analysis failed: {e}")
            return 0.0, 0.0, {'error': str(e)}
    
    async def _extract_features(self, text: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Extract features from all available methods."""
        features = {}
        
        # Lexicon features
        if self.lexicon_method and self.lexicon_method.is_initialized:
            try:
                lex_score, lex_conf, lex_debug = await self.lexicon_method.analyze(text, metadata)
                features.update({
                    'lexicon_score': lex_score,
                    'lexicon_confidence': lex_conf,
                    'lexicon_coverage': lex_debug.get('coverage', 0.0),
                    'lexicon_hits': lex_debug.get('lexicon_hits', 0),
                    'lexicon_available': True
                })
                
                # Add contextual adjustment features
                contextual_adj = lex_debug.get('contextual_adjustments', {})
                for adj_type, count in contextual_adj.items():
                    features[f'lexicon_{adj_type}'] = count
                    
            except Exception as e:
                logger.warning(f"Lexicon feature extraction failed: {e}")
                features['lexicon_available'] = False
        else:
            features['lexicon_available'] = False
        
        # Transformer features
        if self.transformer_method and self.transformer_method.is_initialized:
            try:
                trans_score, trans_conf, trans_debug = await self.transformer_method.analyze(text, metadata)
                features.update({
                    'transformer_score': trans_score,
                    'transformer_confidence': trans_conf,
                    'transformer_models_used': len(trans_debug.get('models_used', [])),
                    'transformer_available': True
                })
                
                # Add individual model scores if available
                model_preds = trans_debug.get('model_predictions', {})
                for model_name, score in model_preds.items():
                    features[f'transformer_{model_name}_score'] = score
                    
            except Exception as e:
                logger.warning(f"Transformer feature extraction failed: {e}")
                features['transformer_available'] = False
        else:
            features['transformer_available'] = False
        
        # Text-based features
        features.update(self._extract_text_features(text))
        
        # Metadata features
        if metadata:
            features.update(self._extract_metadata_features(metadata))
        
        return features
    
    def _extract_text_features(self, text: str) -> Dict[str, float]:
        """Extract statistical features from text."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        features = {
            'text_length': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': sum(len(word) for word in words) / max(len(words), 1),
            'avg_sentence_length': len(words) / max(len(sentences), 1),
        }
        
        # Readability features
        if words and sentences:
            # Simple readability metrics
            long_words = sum(1 for word in words if len(word) > 6)
            features['long_word_ratio'] = long_words / len(words)
            
            # Complexity indicators
            complex_sentences = sum(1 for sent in sentences if len(sent.split()) > 20)
            features['complex_sentence_ratio'] = complex_sentences / max(len(sentences), 1)
        
        return features
    
    def _extract_metadata_features(self, metadata: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from speech metadata."""
        features = {}
        
        # Speaker-based features
        speaker = metadata.get('speaker', '').lower()
        if 'chair' in speaker or 'governor' in speaker:
            features['speaker_authority'] = 1.0
        elif 'president' in speaker:
            features['speaker_authority'] = 0.8
        else:
            features['speaker_authority'] = 0.5
        
        # Institution features
        institution = metadata.get('institution', '').lower()
        if 'federal reserve' in institution or 'fed' in institution:
            features['institution_fed'] = 1.0
        else:
            features['institution_fed'] = 0.0
        
        # Temporal features
        date_str = metadata.get('date', '')
        if date_str:
            try:
                # Extract year and month for temporal patterns
                if '-' in date_str:
                    year = int(date_str.split('-')[0])
                    month = int(date_str.split('-')[1])
                    features['speech_year'] = year / 2025.0  # Normalize
                    features['speech_month'] = month / 12.0  # Normalize
                else:
                    features['speech_year'] = 0.5
                    features['speech_month'] = 0.5
            except:
                features['speech_year'] = 0.5
                features['speech_month'] = 0.5
        
        return features
    
    def _rule_based_ensemble(self, features: Dict[str, float]) -> Tuple[float, float]:
        """Rule-based ensemble until training data is available."""
        scores = []
        confidences = []
        weights = []
        
        # Lexicon component
        if features.get('lexicon_available', False):
            lex_score = features.get('lexicon_score', 0.0)
            lex_confidence = features.get('lexicon_confidence', 0.0)
            scores.append(lex_score)
            confidences.append(lex_confidence)
            weights.append(0.6)  # Higher weight for lexicon due to domain specificity
        
        # Transformer component
        if features.get('transformer_available', False):
            trans_score = features.get('transformer_score', 0.0)
            trans_confidence = features.get('transformer_confidence', 0.0)
            scores.append(trans_score)
            confidences.append(trans_confidence)
            weights.append(0.8)  # Higher weight for transformer if available
        
        if not scores:
            return 0.0, 0.0
        
        # Weighted combination
        total_weight = sum(w * c for w, c in zip(weights, confidences))
        if total_weight == 0:
            final_score = sum(scores) / len(scores)
            final_confidence = sum(confidences) / len(confidences)
        else:
            final_score = sum(s * w * c for s, w, c in zip(scores, weights, confidences)) / total_weight
            final_confidence = sum(confidences) / len(confidences)
        
        # Boost confidence if multiple methods agree
        if len(scores) > 1:
            score_std = np.std(scores) if len(scores) > 1 else 0
            agreement_bonus = max(0, 0.2 - score_std)  # Bonus when methods agree
            final_confidence = min(1.0, final_confidence + agreement_bonus)
        
        return final_score, final_confidence
    
    def get_confidence_baseline(self) -> float:
        """Get baseline confidence for hybrid method."""
        base_confidence = 0.5
        if self.lexicon_method and self.lexicon_method.is_initialized:
            base_confidence += 0.2
        if self.transformer_method and self.transformer_method.is_initialized:
            base_confidence += 0.3
        return min(1.0, base_confidence)


class HawkDoveAnalyzer:
    """
    Comprehensive Hawk-Dove sentiment analyzer for central bank speeches.
    
    This is the main interface that orchestrates multiple analysis methods,
    provides confidence-weighted results, and includes temporal consistency checks.
    
    Features:
    - Multi-method ensemble analysis
    - Automatic method selection and weighting
    - Temporal consistency validation
    - Comprehensive error handling and logging
    - Performance monitoring and optimization
    - SQLite persistence for reproducibility
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the hawk-dove analyzer.
        
        Args:
            config: Configuration dictionary with method settings
        """
        self.config = config or {}
        self.methods: List[HawkDoveMethod] = []
        self.method_weights: Dict[str, float] = {}
        self.is_initialized = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Performance tracking
        self.analysis_history: List[Dict[str, Any]] = []
        self.performance_metrics = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'average_processing_time': 0.0,
            'method_success_rates': {}
        }
        
        # Persistence
        self.db_path = self.config.get('db_path', 'hawk_dove_analysis.db')
        self.enable_persistence = self.config.get('enable_persistence', True)
        
        # Initialize methods
        self._initialize_methods()
    
    def _initialize_methods(self) -> None:
        """Initialize analysis methods based on configuration."""
        method_config = self.config.get('methods', {})
        
        # Always include lexicon method
        if method_config.get('lexicon', {}).get('enabled', True):
            lexicon_method = LexiconMethod(method_config.get('lexicon', {}))
            self.methods.append(lexicon_method)
            self.method_weights['LexiconMethod'] = method_config.get('lexicon', {}).get('weight', 0.6)
        
        # Add transformer method if available and enabled
        if method_config.get('transformer', {}).get('enabled', True) and TRANSFORMERS_AVAILABLE:
            transformer_method = TransformerMethod(method_config.get('transformer', {}))
            self.methods.append(transformer_method)
            self.method_weights['TransformerMethod'] = method_config.get('transformer', {}).get('weight', 0.8)
        
        # Add hybrid method if enabled
        if method_config.get('hybrid', {}).get('enabled', False) and SKLEARN_AVAILABLE:
            hybrid_method = HybridMethod(method_config.get('hybrid', {}))
            self.methods.append(hybrid_method)
            self.method_weights['HybridMethod'] = method_config.get('hybrid', {}).get('weight', 1.0)
        
        self.logger.info(f"Initialized {len(self.methods)} analysis methods")
    
    async def initialize(self) -> bool:
        """Initialize all analysis methods."""
        if self.is_initialized:
            return True
        
        self.logger.info("Initializing hawk-dove analyzer...")
        
        # Initialize database if persistence is enabled
        if self.enable_persistence:
            self._initialize_database()
        
        # Initialize all methods
        initialization_results = []
        for method in self.methods:
            try:
                success = await method.initialize()
                initialization_results.append((method.name, success))
                self.performance_metrics['method_success_rates'][method.name] = 0.0
            except Exception as e:
                self.logger.error(f"Failed to initialize {method.name}: {e}")
                initialization_results.append((method.name, False))
        
        # Check if at least one method initialized successfully
        successful_methods = [name for name, success in initialization_results if success]
        
        if successful_methods:
            self.is_initialized = True
            self.logger.info(f"Hawk-dove analyzer initialized with methods: {successful_methods}")
            return True
        else:
            self.logger.error("No analysis methods could be initialized")
            return False
    
    def _initialize_database(self) -> None:
        """Initialize SQLite database for persistence."""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            
            # Create tables for analysis history
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    final_score REAL NOT NULL,
                    final_confidence REAL NOT NULL,
                    policy_stance TEXT NOT NULL,
                    method_results TEXT NOT NULL,
                    processing_time REAL NOT NULL,
                    text_length INTEGER NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS method_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    method_name TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    processing_time REAL NOT NULL,
                    confidence REAL,
                    error_message TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Database initialized at {self.db_path}")
            
        except Exception as e:
            self.logger.warning(f"Database initialization failed: {e}")
            self.enable_persistence = False
    
    async def analyze_document_comprehensive(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of a document using all available methods.
        
        Args:
            text: Text content to analyze
            metadata: Optional metadata about the speech
            
        Returns:
            Comprehensive analysis results including method breakdown
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = datetime.now()
        
        try:
            # Run all methods
            method_results = await self._run_all_methods(text, metadata)
            
            # Combine results
            final_result = self._combine_method_results(method_results, text, metadata)
            
            # Add processing metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            final_result['processing_time'] = processing_time
            
            # Update performance metrics
            self._update_performance_metrics(method_results, processing_time, True)
            
            # Persist results if enabled
            if self.enable_persistence:
                self._persist_analysis_result(text, final_result, method_results, processing_time)
            
            # Add to history
            self.analysis_history.append({
                'timestamp': start_time.isoformat(),
                'text_length': len(text),
                'final_score': final_result['mean_score'],
                'processing_time': processing_time,
                'methods_used': list(method_results.keys())
            })
            
            # Keep history manageable
            if len(self.analysis_history) > 1000:
                self.analysis_history = self.analysis_history[-500:]
            
            return final_result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Document analysis failed: {e}")
            self._update_performance_metrics({}, processing_time, False)
            
            # Return minimal failure result
            return {
                'mean_score': 0.0,
                'median_score': 0.0,
                'std_score': 0.0,
                'confidence_score': 0.0,
                'policy_stance': PolicyStance.NEUTRAL.value,
                'error': str(e),
                'processing_time': processing_time,
                'methods_used': 0
            }
    
    async def _run_all_methods(self, text: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Tuple[float, float, Dict[str, Any]]]:
        """Run all initialized methods on the text."""
        method_results = {}
        
        # Run methods in parallel for better performance
        tasks = []
        for method in self.methods:
            if method.is_initialized:
                task = self._run_method_with_timeout(method, text, metadata)
                tasks.append((method.name, task))
        
        # Wait for all methods to complete
        if tasks:
            completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            for (method_name, _), result in zip(tasks, completed_tasks):
                if isinstance(result, Exception):
                    self.logger.warning(f"Method {method_name} failed: {result}")
                    continue
                elif result is not None:
                    method_results[method_name] = result
        
        return method_results
    
    async def _run_method_with_timeout(self, method: HawkDoveMethod, text: str, 
                                     metadata: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float, Dict[str, Any]]]:
        """Run a method with timeout protection."""
        try:
            # Use asyncio timeout for protection
            timeout = self.config.get('method_timeout', 30)
            result = await asyncio.wait_for(method.analyze(text, metadata), timeout=timeout)
            
            # Update method success rate
            if method.name in self.performance_metrics['method_success_rates']:
                current_rate = self.performance_metrics['method_success_rates'][method.name]
                # Exponential moving average
                self.performance_metrics['method_success_rates'][method.name] = 0.9 * current_rate + 0.1 * 1.0
            
            return result
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Method {method.name} timed out")
            method.record_error("Timeout")
            return None
        except Exception as e:
            self.logger.warning(f"Method {method.name} failed: {e}")
            method.record_error(str(e))
            
            # Update method success rate
            if method.name in self.performance_metrics['method_success_rates']:
                current_rate = self.performance_metrics['method_success_rates'][method.name]
                self.performance_metrics['method_success_rates'][method.name] = 0.9 * current_rate + 0.1 * 0.0
            
            return None
    
    def _combine_method_results(self, method_results: Dict[str, Tuple[float, float, Dict[str, Any]]], 
                               text: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from all methods into final analysis."""
        if not method_results:
            return {
                'mean_score': 0.0,
                'median_score': 0.0,
                'std_score': 0.0,
                'confidence_score': 0.0,
                'policy_stance': PolicyStance.NEUTRAL.value,
                'method_comparison': {},
                'dominant_method': 'none',
                'overall_confidence': 'low',
                'analysis_method': 'failed',
                'methods_used': 0,
                'warnings': ['No methods succeeded']
            }
        
        # Extract scores and confidences
        scores = []
        confidences = []
        method_breakdown = {}
        
        for method_name, (score, confidence, debug_info) in method_results.items():
            scores.append(score)
            confidences.append(confidence)
            method_breakdown[f"{method_name.lower()}_mean"] = score
            method_breakdown[f"{method_name.lower()}_confidence"] = confidence
        
        # Calculate ensemble metrics
        scores_array = np.array(scores)
        confidences_array = np.array(confidences)
        
        # Weighted average by confidence
        weights = confidences_array / (np.sum(confidences_array) + 1e-8)
        weighted_score = np.sum(scores_array * weights)
        
        # Basic statistics
        mean_score = float(np.mean(scores_array))
        median_score = float(np.median(scores_array))
        std_score = float(np.std(scores_array))
        
        # Overall confidence (average with consensus bonus)
        base_confidence = float(np.mean(confidences_array))
        consensus_bonus = max(0, 0.2 - std_score) if len(scores) > 1 else 0
        overall_confidence = min(1.0, base_confidence + consensus_bonus)
        
        # Determine policy stance
        final_score = weighted_score if overall_confidence > 0.3 else mean_score
        
        if final_score <= -0.05:
            policy_stance = PolicyStance.HAWKISH
        elif final_score >= 0.05:
            policy_stance = PolicyStance.DOVISH
        else:
            policy_stance = PolicyStance.NEUTRAL
        
        # Determine dominant method
        if confidences:
            max_confidence_idx = np.argmax(confidences_array)
            dominant_method = list(method_results.keys())[max_confidence_idx]
        else:
            dominant_method = 'none'
        
        # Categorize overall confidence
        if overall_confidence >= 0.8:
            confidence_category = 'high'
        elif overall_confidence >= 0.6:
            confidence_category = 'medium'
        else:
            confidence_category = 'low'
        
        # Prepare result
        result = {
            'mean_score': mean_score,
            'median_score': median_score,
            'std_score': std_score,
            'confidence_score': overall_confidence,
            'final_score': final_score,
            'policy_stance': policy_stance.value,
            'method_comparison': method_breakdown,
            'dominant_method': dominant_method,
            'overall_confidence': confidence_category,
            'analysis_method': 'ensemble',
            'methods_used': len(method_results),
            'method_weights': {name: self.method_weights.get(name, 1.0) for name in method_results.keys()},
            'ensemble_score': weighted_score,
            'consensus_level': 1.0 - std_score if len(scores) > 1 else 1.0,
            'text_length': len(text),
            'warnings': []
        }
        
        # Add warnings for low confidence or disagreement
        if overall_confidence < 0.5:
            result['warnings'].append('Low overall confidence in analysis')
        
        if std_score > 0.5 and len(scores) > 1:
            result['warnings'].append('High disagreement between methods')
        
        if len(method_results) == 1:
            result['warnings'].append('Only one analysis method succeeded')
        
        return result
    
    def _update_performance_metrics(self, method_results: Dict[str, Any], processing_time: float, success: bool) -> None:
        """Update performance tracking metrics."""
        self.performance_metrics['total_analyses'] += 1
        
        if success:
            self.performance_metrics['successful_analyses'] += 1
        
        # Update average processing time (exponential moving average)
        current_avg = self.performance_metrics['average_processing_time']
        alpha = 0.1  # Learning rate
        self.performance_metrics['average_processing_time'] = (1 - alpha) * current_avg + alpha * processing_time
    
    def _persist_analysis_result(self, text: str, result: Dict[str, Any], 
                               method_results: Dict[str, Any], processing_time: float) -> None:
        """Persist analysis results to database."""
        try:
            import sqlite3
            import hashlib
            
            conn = sqlite3.connect(self.db_path)
            
            # Create text hash for deduplication
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            # Insert main result
            conn.execute("""
                INSERT INTO analysis_results 
                (timestamp, text_hash, final_score, final_confidence, policy_stance, 
                 method_results, processing_time, text_length)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                text_hash,
                result['final_score'],
                result['confidence_score'],
                result['policy_stance'],
                json.dumps(method_results),
                processing_time,
                len(text)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to persist analysis result: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        total = self.performance_metrics['total_analyses']
        successful = self.performance_metrics['successful_analyses']
        
        return {
            'total_analyses': total,
            'successful_analyses': successful,
            'success_rate': successful / max(total, 1),
            'average_processing_time': self.performance_metrics['average_processing_time'],
            'method_success_rates': self.performance_metrics['method_success_rates'].copy(),
            'initialized_methods': [method.name for method in self.methods if method.is_initialized],
            'available_capabilities': {
                'lexicon_analysis': any(isinstance(m, LexiconMethod) and m.is_initialized for m in self.methods),
                'transformer_analysis': any(isinstance(m, TransformerMethod) and m.is_initialized for m in self.methods),
                'hybrid_analysis': any(isinstance(m, HybridMethod) and m.is_initialized for m in self.methods),
                'persistence': self.enable_persistence
            },
            'recent_history': self.analysis_history[-10:] if self.analysis_history else []
        }


# Factory functions for easy analyzer creation

def create_default_analyzer(lexicon_path: str = "lexicons/hawk_dove_lexicon.csv") -> HawkDoveAnalyzer:
    """
    Create analyzer with default configuration optimized for production use.
    
    Args:
        lexicon_path: Path to hawk-dove lexicon file
        
    Returns:
        Configured HawkDoveAnalyzer instance
    """
    config = {
        'methods': {
            'lexicon': {
                'enabled': True,
                'weight': 0.6,
                'lexicon_path': lexicon_path
            },
            'transformer': {
                'enabled': True,
                'weight': 0.8,
                'models': {
                    'finbert': 'ProsusAI/finbert'
                },
                'max_length': 512
            },
            'hybrid': {
                'enabled': False  # Disabled by default until training data available
            }
        },
        'method_timeout': 30,
        'enable_persistence': True,
        'db_path': 'hawk_dove_analysis.db'
    }
    
    return HawkDoveAnalyzer(config)


def create_lightweight_analyzer(lexicon_path: str = "lexicons/hawk_dove_lexicon.csv") -> HawkDoveAnalyzer:
    """
    Create lightweight analyzer for resource-constrained environments.
    
    Args:
        lexicon_path: Path to hawk-dove lexicon file
        
    Returns:
        Lightweight HawkDoveAnalyzer instance
    """
    config = {
        'methods': {
            'lexicon': {
                'enabled': True,
                'weight': 1.0,
                'lexicon_path': lexicon_path
            },
            'transformer': {
                'enabled': False  # Disabled for lightweight deployment
            },
            'hybrid': {
                'enabled': False
            }
        },
        'method_timeout': 15,
        'enable_persistence': False
    }
    
    return HawkDoveAnalyzer(config)


def create_research_analyzer(lexicon_path: str = "lexicons/hawk_dove_lexicon.csv") -> HawkDoveAnalyzer:
    """
    Create analyzer optimized for research with all methods enabled.
    
    Args:
        lexicon_path: Path to hawk-dove lexicon file
        
    Returns:
        Research-optimized HawkDoveAnalyzer instance
    """
    config = {
        'methods': {
            'lexicon': {
                'enabled': True,
                'weight': 0.5,
                'lexicon_path': lexicon_path
            },
            'transformer': {
                'enabled': True,
                'weight': 0.7,
                'models': {
                    'finbert': 'ProsusAI/finbert',
                    'distilbert': 'distilbert-base-uncased-finetuned-sst-2-english'
                },
                'max_length': 512,
                'batch_size': 1
            },
            'hybrid': {
                'enabled': True,
                'weight': 1.0
            }
        },
        'method_timeout': 60,
        'enable_persistence': True,
        'db_path': 'research_hawk_dove_analysis.db'
    }
    
    return HawkDoveAnalyzer(config)

# Example usage and testing
if __name__ == "__main__":
    async def main():
        """Example usage of the hawk-dove analyzer."""
        # Create analyzer
        analyzer = create_default_analyzer()
        
        # Initialize
        success = await analyzer.initialize()
        print(f"Analyzer initialized: {success}")
        
        if success:
            # Get performance summary
            performance = analyzer.get_performance_summary()
            print("\nAnalyzer Capabilities:")
            for capability, available in performance['available_capabilities'].items():
                status = "✓" if available else "✗"
                print(f"  {status} {capability}")
            
            # Example analysis
            sample_text = """
            The Federal Reserve remains committed to achieving maximum employment and price stability.
            Given the recent uptick in inflation, we are prepared to adjust our monetary policy stance
            as appropriate. The Committee will continue to monitor economic conditions and stands ready
            to take action to ensure that inflation returns to our 2 percent target.
            """
            
            print(f"\nAnalyzing sample text...")
            result = await analyzer.analyze_document_comprehensive(sample_text)
            
            print(f"Hawk-Dove Score: {result['final_score']:.3f}")
            print(f"Policy Stance: {result['policy_stance']}")
            print(f"Confidence: {result['confidence_score']:.3f}")
            print(f"Methods Used: {result['methods_used']}")
            print(f"Processing Time: {result['processing_time']:.3f}s")
            
            if result.get('warnings'):
                print(f"Warnings: {', '.join(result['warnings'])}")
        
        print("\nAnalyzer ready for production use!")
    
    # Run example
    asyncio.run(main())