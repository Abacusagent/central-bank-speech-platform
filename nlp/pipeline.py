#!/usr/bin/env python3
"""
NLP Processing Pipeline for Central Bank Speech Analysis Platform

This module provides a modular, extensible NLP pipeline for analyzing central bank speeches.
The pipeline supports multiple analysis types including sentiment analysis, topic modeling,
uncertainty quantification, and stance detection.

Key Features:
- Modular processor architecture
- Parallel processing capabilities
- Comprehensive analysis results
- Extensible for new analysis types
- Integration with domain entities

Author: Central Bank Speech Analysis Platform
Date: 2025
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from uuid import uuid4

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Domain imports
from domain.entities import CentralBankSpeech, SentimentAnalysis, PolicyStance
from domain.value_objects import SentimentScore, ConfidenceLevel, Version
from interfaces.plugin_interfaces import SpeechContent

# Optional advanced NLP imports
try:
    import spacy
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ProcessorResult:
    """
    Result from an individual NLP processor.
    
    Contains the analysis results, confidence scores, and metadata
    from a specific NLP processing component.
    """
    processor_name: str
    success: bool
    confidence: float
    processing_time: float
    results: Dict[str, Any]
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Convert numeric confidence to categorical level."""
        if self.confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


@dataclass
class NLPAnalysis:
    """
    Comprehensive analysis results from the NLP pipeline.
    
    Aggregates results from all processors into a unified analysis
    with overall confidence scores and combined insights.
    """
    speech_id: str
    analysis_timestamp: datetime
    processor_results: List[ProcessorResult]
    overall_confidence: float
    processing_time: float
    
    # Aggregated results
    hawkish_dovish_score: Optional[float] = None
    policy_stance: Optional[PolicyStance] = None
    uncertainty_score: Optional[float] = None
    complexity_score: Optional[float] = None
    topics: List[str] = field(default_factory=list)
    key_themes: List[str] = field(default_factory=list)
    sentiment_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    pipeline_version: str = "1.0.0"
    processors_used: List[str] = field(default_factory=list)
    
    def get_successful_results(self) -> List[ProcessorResult]:
        """Get only the successful processor results."""
        return [result for result in self.processor_results if result.success]
    
    def get_result_by_processor(self, processor_name: str) -> Optional[ProcessorResult]:
        """Get result from a specific processor."""
        for result in self.processor_results:
            if result.processor_name == processor_name:
                return result
        return None
    
    def to_sentiment_analysis(self) -> SentimentAnalysis:
        """Convert to domain SentimentAnalysis entity."""
        return SentimentAnalysis(
            hawkish_dovish_score=self.hawkish_dovish_score or 0.0,
            policy_stance=self.policy_stance or PolicyStance.NEUTRAL,
            uncertainty_score=self.uncertainty_score or 0.0,
            confidence_score=self.overall_confidence,
            analysis_timestamp=self.analysis_timestamp,
            analyzer_version=self.pipeline_version,
            raw_scores={
                'complexity': self.complexity_score,
                **self.sentiment_breakdown
            },
            topic_classifications=self.topics + self.key_themes
        )


class NLPProcessor(ABC):
    """
    Abstract base class for NLP processors.
    
    All NLP analysis components must implement this interface to be
    compatible with the processing pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the processor with optional configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        self.version = "1.0.0"
        self.is_initialized = False
    
    @abstractmethod
    async def analyze(self, speech: CentralBankSpeech) -> ProcessorResult:
        """
        Analyze a speech and return results.
        
        Args:
            speech: Speech entity to analyze
            
        Returns:
            ProcessorResult with analysis results and confidence
        """
        pass
    
    @abstractmethod
    def get_confidence_score(self) -> float:
        """
        Get the processor's confidence in its analysis capability.
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        pass
    
    async def initialize(self) -> None:
        """Initialize the processor (load models, etc.)."""
        self.is_initialized = True
        logger.info(f"Initialized {self.name} processor")
    
    def _extract_text_for_analysis(self, speech: CentralBankSpeech) -> str:
        """Extract appropriate text content from speech for analysis."""
        if speech.content and speech.content.cleaned_text:
            return speech.content.cleaned_text
        elif speech.content and speech.content.raw_text:
            return speech.content.raw_text
        else:
            return ""


class HawkDoveAnalyzer(NLPProcessor):
    """
    Monetary policy stance analyzer using lexicon-based and ML approaches.
    
    Determines whether a speech is hawkish (supporting tighter monetary policy)
    or dovish (supporting looser monetary policy).
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.lexicon_path = config.get('lexicon_path', 'lexicon.csv') if config else 'lexicon.csv'
        self.lexicon: Dict[str, Tuple[float, float]] = {}
        self.transformer_model = None
        
    async def initialize(self) -> None:
        """Initialize the hawk-dove analyzer."""
        await super().initialize()
        
        # Load lexicon
        self._load_lexicon()
        
        # Initialize transformer model if available
        if ADVANCED_NLP_AVAILABLE:
            try:
                # Use a financial sentiment model if available
                self.transformer_model = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert"
                )
                logger.info("Loaded FinBERT model for hawk-dove analysis")
            except Exception as e:
                logger.warning(f"Could not load FinBERT model: {e}")
                self.transformer_model = None
    
    def _load_lexicon(self) -> None:
        """Load the hawk-dove lexicon."""
        try:
            import csv
            from pathlib import Path
            
            lexicon_file = Path(self.lexicon_path)
            if lexicon_file.exists():
                with open(lexicon_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        word = row['word'].lower()
                        polarity = float(row['polarity'])
                        weight = float(row['weight'])
                        self.lexicon[word] = (polarity, weight)
                
                logger.info(f"Loaded {len(self.lexicon)} terms from lexicon")
            else:
                logger.warning(f"Lexicon file not found: {self.lexicon_path}")
                # Use default basic lexicon
                self._create_default_lexicon()
                
        except Exception as e:
            logger.error(f"Error loading lexicon: {e}")
            self._create_default_lexicon()
    
    def _create_default_lexicon(self) -> None:
        """Create a basic default lexicon for testing."""
        default_terms = {
            # Hawkish terms (negative scores)
            'inflation': (-0.8, 1.0),
            'tighten': (-0.9, 1.0),
            'raise': (-0.7, 0.8),
            'increase': (-0.6, 0.7),
            'restrictive': (-0.8, 0.9),
            'aggressive': (-0.9, 1.0),
            
            # Dovish terms (positive scores)
            'accommodate': (0.8, 1.0),
            'stimulus': (0.9, 1.0),
            'support': (0.6, 0.8),
            'lower': (0.7, 0.8),
            'ease': (0.8, 0.9),
            'gradual': (0.5, 0.7),
        }
        
        self.lexicon = default_terms
        logger.info(f"Created default lexicon with {len(self.lexicon)} terms")
    
    async def analyze(self, speech: CentralBankSpeech) -> ProcessorResult:
        """Analyze hawk-dove sentiment in a speech."""
        start_time = datetime.now()
        
        try:
            text = self._extract_text_for_analysis(speech)
            if not text:
                return ProcessorResult(
                    processor_name=self.name,
                    success=False,
                    confidence=0.0,
                    processing_time=0.0,
                    results={},
                    error_message="No text content available"
                )
            
            # Method 1: Lexicon-based analysis
            lexicon_score = self._analyze_with_lexicon(text)
            
            # Method 2: Transformer-based analysis (if available)
            transformer_score = None
            if self.transformer_model:
                transformer_score = await self._analyze_with_transformer(text)
            
            # Combine scores
            if transformer_score is not None:
                # Weight transformer more heavily if available
                combined_score = 0.3 * lexicon_score + 0.7 * transformer_score
                confidence = 0.85
            else:
                combined_score = lexicon_score
                confidence = 0.65
            
            # Determine policy stance
            if combined_score <= -0.05:
                stance = PolicyStance.HAWKISH
            elif combined_score >= 0.05:
                stance = PolicyStance.DOVISH
            else:
                stance = PolicyStance.NEUTRAL
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessorResult(
                processor_name=self.name,
                success=True,
                confidence=confidence,
                processing_time=processing_time,
                results={
                    'hawkish_dovish_score': combined_score,
                    'policy_stance': stance.value,
                    'lexicon_score': lexicon_score,
                    'transformer_score': transformer_score,
                    'method_used': 'combined' if transformer_score else 'lexicon_only'
                },
                metadata={
                    'lexicon_terms': len(self.lexicon),
                    'transformer_available': self.transformer_model is not None
                }
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return ProcessorResult(
                processor_name=self.name,
                success=False,
                confidence=0.0,
                processing_time=processing_time,
                results={},
                error_message=str(e)
            )
    
    def _analyze_with_lexicon(self, text: str) -> float:
        """Analyze text using lexicon-based approach."""
        import re
        
        # Tokenize text
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not words:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for word in words:
            if word in self.lexicon:
                polarity, weight = self.lexicon[word]
                total_score += polarity * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        # Normalize by text length and weight
        normalized_score = total_score / (len(words) ** 0.5)
        return max(-1.0, min(1.0, normalized_score))
    
    async def _analyze_with_transformer(self, text: str) -> Optional[float]:
        """Analyze text using transformer model."""
        try:
            # Truncate text if too long for model
            max_length = 512
            if len(text.split()) > max_length:
                text = ' '.join(text.split()[:max_length])
            
            # Run inference
            result = self.transformer_model(text)
            
            # Convert to hawk-dove scale
            # FinBERT returns positive/negative/neutral
            if result[0]['label'].lower() == 'positive':
                return 0.3  # Slightly dovish
            elif result[0]['label'].lower() == 'negative':
                return -0.3  # Slightly hawkish
            else:
                return 0.0  # Neutral
                
        except Exception as e:
            logger.warning(f"Transformer analysis failed: {e}")
            return None
    
    def get_confidence_score(self) -> float:
        """Get confidence in hawk-dove analysis capability."""
        base_confidence = 0.7 if self.lexicon else 0.3
        if self.transformer_model:
            base_confidence += 0.2
        return min(1.0, base_confidence)


class TopicModelingProcessor(NLPProcessor):
    """
    Topic modeling processor for identifying key themes in speeches.
    
    Uses Latent Dirichlet Allocation (LDA) and keyword extraction
    to identify the main topics discussed in speeches.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.n_topics = config.get('n_topics', 10) if config else 10
        self.vectorizer = None
        self.lda_model = None
        self.topic_labels = {
            0: "Monetary Policy",
            1: "Economic Outlook",
            2: "Financial Stability",
            3: "Inflation",
            4: "Employment",
            5: "International Economics",
            6: "Banking Regulation",
            7: "Market Conditions",
            8: "Crisis Response",
            9: "Institutional Framework"
        }
    
    async def initialize(self) -> None:
        """Initialize the topic modeling processor."""
        await super().initialize()
        
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            self.lda_model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=42,
                max_iter=20
            )
            logger.info("Initialized topic modeling components")
        else:
            logger.warning("scikit-learn not available, using keyword-based topics only")
    
    async def analyze(self, speech: CentralBankSpeech) -> ProcessorResult:
        """Identify topics in a speech."""
        start_time = datetime.now()
        
        try:
            text = self._extract_text_for_analysis(speech)
            if not text:
                return ProcessorResult(
                    processor_name=self.name,
                    success=False,
                    confidence=0.0,
                    processing_time=0.0,
                    results={},
                    error_message="No text content available"
                )
            
            # Method 1: Keyword-based topic identification
            keyword_topics = self._identify_keyword_topics(text)
            
            # Method 2: LDA-based topic modeling (if available)
            lda_topics = []
            if self.lda_model is not None and SKLEARN_AVAILABLE:
                lda_topics = self._identify_lda_topics(text)
            
            # Combine and rank topics
            all_topics = keyword_topics + lda_topics
            unique_topics = list(set(all_topics))
            
            processing_time = (datetime.now() - start_time).total_seconds()
            confidence = 0.8 if self.lda_model else 0.6
            
            return ProcessorResult(
                processor_name=self.name,
                success=True,
                confidence=confidence,
                processing_time=processing_time,
                results={
                    'topics': unique_topics[:5],  # Top 5 topics
                    'keyword_topics': keyword_topics,
                    'lda_topics': lda_topics,
                    'topic_confidence': self._calculate_topic_confidence(text, unique_topics)
                },
                metadata={
                    'method_used': 'combined' if self.lda_model else 'keyword_only',
                    'total_topics_identified': len(unique_topics)
                }
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return ProcessorResult(
                processor_name=self.name,
                success=False,
                confidence=0.0,
                processing_time=processing_time,
                results={},
                error_message=str(e)
            )
    
    def _identify_keyword_topics(self, text: str) -> List[str]:
        """Identify topics using keyword matching."""
        text_lower = text.lower()
        topics = []
        
        topic_keywords = {
            'Monetary Policy': ['monetary policy', 'interest rate', 'federal funds', 'policy rate'],
            'Inflation': ['inflation', 'price stability', 'deflation', 'cpi', 'pce'],
            'Employment': ['employment', 'unemployment', 'labor market', 'jobs'],
            'Economic Outlook': ['economic outlook', 'forecast', 'projection', 'growth'],
            'Financial Stability': ['financial stability', 'systemic risk', 'banking'],
            'International Economics': ['international', 'global', 'trade', 'foreign'],
            'Crisis Response': ['crisis', 'pandemic', 'emergency', 'covid'],
            'Market Conditions': ['market', 'liquidity', 'volatility', 'credit']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _identify_lda_topics(self, text: str) -> List[str]:
        """Identify topics using LDA (requires training on corpus)."""
        # Note: In a real implementation, this would use a pre-trained LDA model
        # For now, return empty list as we'd need a corpus to train on
        return []
    
    def _calculate_topic_confidence(self, text: str, topics: List[str]) -> float:
        """Calculate confidence in topic identification."""
        if not topics:
            return 0.0
        
        # Simple heuristic: confidence based on text length and topic count
        text_length = len(text.split())
        if text_length < 100:
            return 0.4
        elif text_length < 500:
            return 0.6
        else:
            return 0.8
    
    def get_confidence_score(self) -> float:
        """Get confidence in topic modeling capability."""
        return 0.8 if SKLEARN_AVAILABLE else 0.6


class UncertaintyQuantifier(NLPProcessor):
    """
    Uncertainty quantifier for measuring policy uncertainty in speeches.
    
    Identifies linguistic markers of uncertainty and quantifies the overall
    level of uncertainty expressed in monetary policy communications.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.uncertainty_words = {
            'high': ['uncertain', 'unclear', 'ambiguous', 'unknown', 'unpredictable'],
            'medium': ['may', 'might', 'could', 'possibly', 'perhaps', 'likely'],
            'low': ['depends', 'conditional', 'subject to', 'monitor', 'assess']
        }
    
    async def analyze(self, speech: CentralBankSpeech) -> ProcessorResult:
        """Quantify uncertainty in a speech."""
        start_time = datetime.now()
        
        try:
            text = self._extract_text_for_analysis(speech)
            if not text:
                return ProcessorResult(
                    processor_name=self.name,
                    success=False,
                    confidence=0.0,
                    processing_time=0.0,
                    results={},
                    error_message="No text content available"
                )
            
            # Count uncertainty markers
            uncertainty_score = self._calculate_uncertainty_score(text)
            
            # Classify uncertainty level
            if uncertainty_score >= 0.7:
                uncertainty_level = "High"
            elif uncertainty_score >= 0.4:
                uncertainty_level = "Medium"
            else:
                uncertainty_level = "Low"
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessorResult(
                processor_name=self.name,
                success=True,
                confidence=0.75,
                processing_time=processing_time,
                results={
                    'uncertainty_score': uncertainty_score,
                    'uncertainty_level': uncertainty_level,
                    'uncertainty_indicators': self._find_uncertainty_indicators(text)
                }
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return ProcessorResult(
                processor_name=self.name,
                success=False,
                confidence=0.0,
                processing_time=processing_time,
                results={},
                error_message=str(e)
            )
    
    def _calculate_uncertainty_score(self, text: str) -> float:
        """Calculate uncertainty score based on linguistic markers."""
        import re
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        if not words:
            return 0.0
        
        uncertainty_count = 0
        total_weight = 0
        
        for category, word_list in self.uncertainty_words.items():
            weight = {'high': 1.0, 'medium': 0.6, 'low': 0.3}[category]
            
            for word in word_list:
                count = text_lower.count(word)
                uncertainty_count += count * weight
                total_weight += count
        
        # Normalize by text length
        if len(words) == 0:
            return 0.0
        
        normalized_score = uncertainty_count / len(words) * 100
        return min(1.0, normalized_score)
    
    def _find_uncertainty_indicators(self, text: str) -> List[str]:
        """Find specific uncertainty indicators in text."""
        text_lower = text.lower()
        found_indicators = []
        
        for category, word_list in self.uncertainty_words.items():
            for word in word_list:
                if word in text_lower:
                    found_indicators.append(word)
        
        return list(set(found_indicators))
    
    def get_confidence_score(self) -> float:
        """Get confidence in uncertainty quantification."""
        return 0.75


class StanceDetector(NLPProcessor):
    """
    Stance detector for identifying subtle policy positions.
    
    Detects nuanced policy stances beyond simple hawk-dove classification,
    including forward guidance signals and policy commitment strength.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.stance_indicators = {
            'commitment': ['will', 'committed to', 'determined to', 'pledge'],
            'flexibility': ['flexible', 'adapt', 'adjust', 'respond to'],
            'forward_guidance': ['expect', 'anticipate', 'project', 'outlook'],
            'data_dependency': ['data dependent', 'economic conditions', 'indicators']
        }
    
    async def analyze(self, speech: CentralBankSpeech) -> ProcessorResult:
        """Detect policy stance in a speech."""
        start_time = datetime.now()
        
        try:
            text = self._extract_text_for_analysis(speech)
            if not text:
                return ProcessorResult(
                    processor_name=self.name,
                    success=False,
                    confidence=0.0,
                    processing_time=0.0,
                    results={},
                    error_message="No text content available"
                )
            
            # Analyze different stance dimensions
            stance_scores = {}
            for stance_type, indicators in self.stance_indicators.items():
                stance_scores[stance_type] = self._calculate_stance_score(text, indicators)
            
            # Determine dominant stance
            dominant_stance = max(stance_scores, key=stance_scores.get)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessorResult(
                processor_name=self.name,
                success=True,
                confidence=0.7,
                processing_time=processing_time,
                results={
                    'stance_scores': stance_scores,
                    'dominant_stance': dominant_stance,
                    'stance_strength': stance_scores[dominant_stance]
                }
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return ProcessorResult(
                processor_name=self.name,
                success=False,
                confidence=0.0,
                processing_time=processing_time,
                results={},
                error_message=str(e)
            )
    
    def _calculate_stance_score(self, text: str, indicators: List[str]) -> float:
        """Calculate stance score for specific indicators."""
        text_lower = text.lower()
        total_score = 0
        
        for indicator in indicators:
            count = text_lower.count(indicator)
            total_score += count
        
        # Normalize by text length
        words = len(text.split())
        if words == 0:
            return 0.0
        
        return min(1.0, total_score / words * 100)
    
    def get_confidence_score(self) -> float:
        """Get confidence in stance detection."""
        return 0.7


class ComplexityAnalyzer(NLPProcessor):
    """
    Communication complexity analyzer.
    
    Measures the complexity of central bank communications using
    readability metrics, sentence structure, and vocabulary sophistication.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
    
    async def analyze(self, speech: CentralBankSpeech) -> ProcessorResult:
        """Analyze communication complexity."""
        start_time = datetime.now()
        
        try:
            text = self._extract_text_for_analysis(speech)
            if not text:
                return ProcessorResult(
                    processor_name=self.name,
                    success=False,
                    confidence=0.0,
                    processing_time=0.0,
                    results={},
                    error_message="No text content available"
                )
            
            # Calculate complexity metrics
            metrics = {
                'readability_score': self._calculate_readability(text),
                'avg_sentence_length': self._calculate_avg_sentence_length(text),
                'vocabulary_diversity': self._calculate_vocabulary_diversity(text),
                'complexity_score': 0.0
            }
            
            # Combine into overall complexity score
            metrics['complexity_score'] = (
                metrics['readability_score'] * 0.4 +
                min(1.0, metrics['avg_sentence_length'] / 30) * 0.3 +
                metrics['vocabulary_diversity'] * 0.3
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessorResult(
                processor_name=self.name,
                success=True,
                confidence=0.8,
                processing_time=processing_time,
                results=metrics
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return ProcessorResult(
                processor_name=self.name,
                success=False,
                confidence=0.0,
                processing_time=processing_time,
                results={},
                error_message=str(e)
            )
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (simplified Flesch Reading Ease)."""
        import re
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = re.findall(r'\b\w+\b', text)
        syllables = sum(self._count_syllables(word) for word in words)
        
        if not sentences or not words:
            return 0.5
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)
        
        # Simplified Flesch formula (normalized to 0-1)
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Normalize to 0-1 range (higher = more complex)
        normalized_score = max(0, min(1, (100 - flesch_score) / 100))
        return normalized_score
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)."""
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        
        if word[0] in vowels:
            count += 1
        
        for i in range(1, len(word)):
            if word[i] in vowels and word[i-1] not in vowels:
                count += 1
        
        if word.endswith('e'):
            count -= 1
        
        return max(1, count)
    
    def _calculate_avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length in words."""
        import re
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        total_words = 0
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence)
            total_words += len(words)
        
        return total_words / len(sentences)
    
    def _calculate_vocabulary_diversity(self, text: str) -> float:
        """Calculate vocabulary diversity (Type-Token Ratio)."""
        import re
        
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not words:
            return 0.0
        
        unique_words = set(words)
        return len(unique_words) / len(words)
    
    def get_confidence_score(self) -> float:
        """Get confidence in complexity analysis."""
        return 0.8


class NLPProcessingPipeline:
    """
    Main NLP processing pipeline that orchestrates multiple processors.
    
    This is the primary interface for analyzing central bank speeches using
    the modular processor architecture. Supports parallel processing and
    comprehensive result aggregation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the NLP processing pipeline.
        
        Args:
            config: Optional configuration dictionary for processors
        """
        self.config = config or {}
        self.processors: List[NLPProcessor] = []
        self.is_initialized = False
        self.version = Version.from_string("1.0.0")
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Performance settings
        self.max_workers = self.config.get('max_workers', 4)
        self.timeout_seconds = self.config.get('timeout_seconds', 300)
        self.enable_parallel = self.config.get('enable_parallel', True)
        
        # Initialize default processors
        self._initialize_default_processors()
    
    def _initialize_default_processors(self) -> None:
        """Initialize the default set of processors."""
        processor_configs = self.config.get('processors', {})
        
        # Add core processors
        self.processors = [
            HawkDoveAnalyzer(processor_configs.get('hawk_dove', {})),
            TopicModelingProcessor(processor_configs.get('topic_modeling', {})),
            UncertaintyQuantifier(processor_configs.get('uncertainty', {})),
            StanceDetector(processor_configs.get('stance', {})),
            ComplexityAnalyzer(processor_configs.get('complexity', {}))
        ]
        
        self.logger.info(f"Initialized pipeline with {len(self.processors)} processors")
    
    def add_processor(self, processor: NLPProcessor) -> None:
        """
        Add a custom processor to the pipeline.
        
        Args:
            processor: NLP processor to add
            
        Example:
            >>> pipeline = NLPProcessingPipeline()
            >>> custom_processor = MyCustomProcessor()
            >>> pipeline.add_processor(custom_processor)
        """
        self.processors.append(processor)
        self.logger.info(f"Added processor: {processor.name}")
    
    def remove_processor(self, processor_name: str) -> bool:
        """
        Remove a processor from the pipeline.
        
        Args:
            processor_name: Name of processor to remove
            
        Returns:
            True if processor was removed, False if not found
        """
        for i, processor in enumerate(self.processors):
            if processor.name == processor_name:
                del self.processors[i]
                self.logger.info(f"Removed processor: {processor_name}")
                return True
        
        self.logger.warning(f"Processor not found: {processor_name}")
        return False
    
    async def initialize(self) -> None:
        """Initialize all processors in the pipeline."""
        if self.is_initialized:
            return
        
        self.logger.info("Initializing NLP processing pipeline...")
        
        initialization_tasks = []
        for processor in self.processors:
            initialization_tasks.append(processor.initialize())
        
        try:
            await asyncio.gather(*initialization_tasks)
            self.is_initialized = True
            self.logger.info("Pipeline initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize pipeline: {e}")
    
    async def process_speech(self, speech: CentralBankSpeech) -> NLPAnalysis:
        """
        Process a single speech through the entire pipeline.
        
        Args:
            speech: Speech entity to analyze
            
        Returns:
            Comprehensive NLP analysis results
            
        Example:
            >>> pipeline = NLPProcessingPipeline()
            >>> await pipeline.initialize()
            >>> 
            >>> speech = CentralBankSpeech(...)
            >>> analysis = await pipeline.process_speech(speech)
            >>> print(f"Hawkish-Dovish Score: {analysis.hawkish_dovish_score}")
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = datetime.now()
        speech_id = str(speech.id)
        
        self.logger.info(f"Processing speech {speech_id} with {len(self.processors)} processors")
        
        try:
            # Run processors in parallel or sequentially based on configuration
            if self.enable_parallel and len(self.processors) > 1:
                processor_results = await self._process_parallel(speech)
            else:
                processor_results = await self._process_sequential(speech)
            
            # Aggregate results
            analysis = self._aggregate_results(speech_id, processor_results, start_time)
            
            # Update speech entity with analysis
            if analysis.hawkish_dovish_score is not None and analysis.policy_stance is not None:
                sentiment_analysis = analysis.to_sentiment_analysis()
                speech.set_sentiment_analysis(sentiment_analysis)
            
            self.logger.info(f"Completed analysis for speech {speech_id} in {analysis.processing_time:.2f}s")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error processing speech {speech_id}: {e}")
            # Return a failed analysis result
            processing_time = (datetime.now() - start_time).total_seconds()
            return NLPAnalysis(
                speech_id=speech_id,
                analysis_timestamp=datetime.now(),
                processor_results=[],
                overall_confidence=0.0,
                processing_time=processing_time,
                pipeline_version=self.version.to_string()
            )
    
    async def process_speeches_batch(self, speeches: List[CentralBankSpeech]) -> List[NLPAnalysis]:
        """
        Process multiple speeches in batch with optimized performance.
        
        Args:
            speeches: List of speech entities to analyze
            
        Returns:
            List of NLP analysis results
            
        Example:
            >>> speeches = [speech1, speech2, speech3]
            >>> analyses = await pipeline.process_speeches_batch(speeches)
            >>> print(f"Processed {len(analyses)} speeches")
        """
        if not self.is_initialized:
            await self.initialize()
        
        self.logger.info(f"Starting batch processing of {len(speeches)} speeches")
        start_time = datetime.now()
        
        # Process speeches with concurrency control
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_with_semaphore(speech: CentralBankSpeech) -> NLPAnalysis:
            async with semaphore:
                return await self.process_speech(speech)
        
        try:
            # Create tasks for all speeches
            tasks = [process_with_semaphore(speech) for speech in speeches]
            
            # Execute with timeout
            analyses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.timeout_seconds
            )
            
            # Filter out exceptions and log errors
            successful_analyses = []
            for i, result in enumerate(analyses):
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to process speech {i}: {result}")
                else:
                    successful_analyses.append(result)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            success_rate = len(successful_analyses) / len(speeches) * 100
            
            self.logger.info(
                f"Batch processing completed: {len(successful_analyses)}/{len(speeches)} "
                f"speeches processed successfully ({success_rate:.1f}%) in {processing_time:.2f}s"
            )
            
            return successful_analyses
            
        except asyncio.TimeoutError:
            self.logger.error(f"Batch processing timed out after {self.timeout_seconds}s")
            return []
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return []
    
    async def _process_parallel(self, speech: CentralBankSpeech) -> List[ProcessorResult]:
        """Process speech through all processors in parallel."""
        tasks = []
        for processor in self.processors:
            if processor.is_initialized:
                tasks.append(processor.analyze(speech))
            else:
                self.logger.warning(f"Skipping uninitialized processor: {processor.name}")
        
        if not tasks:
            return []
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to failed ProcessorResult objects
            processor_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processor_name = self.processors[i].name if i < len(self.processors) else "Unknown"
                    processor_results.append(ProcessorResult(
                        processor_name=processor_name,
                        success=False,
                        confidence=0.0,
                        processing_time=0.0,
                        results={},
                        error_message=str(result)
                    ))
                else:
                    processor_results.append(result)
            
            return processor_results
            
        except Exception as e:
            self.logger.error(f"Parallel processing failed: {e}")
            return []
    
    async def _process_sequential(self, speech: CentralBankSpeech) -> List[ProcessorResult]:
        """Process speech through all processors sequentially."""
        processor_results = []
        
        for processor in self.processors:
            if not processor.is_initialized:
                self.logger.warning(f"Skipping uninitialized processor: {processor.name}")
                continue
            
            try:
                result = await processor.analyze(speech)
                processor_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Processor {processor.name} failed: {e}")
                processor_results.append(ProcessorResult(
                    processor_name=processor.name,
                    success=False,
                    confidence=0.0,
                    processing_time=0.0,
                    results={},
                    error_message=str(e)
                ))
        
        return processor_results
    
    def _aggregate_results(self, speech_id: str, processor_results: List[ProcessorResult], 
                          start_time: datetime) -> NLPAnalysis:
        """Aggregate results from all processors into unified analysis."""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Get successful results
        successful_results = [r for r in processor_results if r.success]
        
        if not successful_results:
            return NLPAnalysis(
                speech_id=speech_id,
                analysis_timestamp=datetime.now(),
                processor_results=processor_results,
                overall_confidence=0.0,
                processing_time=processing_time,
                pipeline_version=self.version.to_string(),
                processors_used=[r.processor_name for r in processor_results]
            )
        
        # Calculate overall confidence
        overall_confidence = sum(r.confidence for r in successful_results) / len(successful_results)
        
        # Extract specific results
        hawkish_dovish_score = None
        policy_stance = None
        uncertainty_score = None
        complexity_score = None
        topics = []
        key_themes = []
        sentiment_breakdown = {}
        
        for result in successful_results:
            if result.processor_name == "HawkDoveAnalyzer":
                hawkish_dovish_score = result.results.get('hawkish_dovish_score')
                if result.results.get('policy_stance'):
                    policy_stance = PolicyStance(result.results['policy_stance'])
                sentiment_breakdown.update({
                    'lexicon_score': result.results.get('lexicon_score'),
                    'transformer_score': result.results.get('transformer_score')
                })
            
            elif result.processor_name == "TopicModelingProcessor":
                topics.extend(result.results.get('topics', []))
                key_themes.extend(result.results.get('keyword_topics', []))
            
            elif result.processor_name == "UncertaintyQuantifier":
                uncertainty_score = result.results.get('uncertainty_score')
            
            elif result.processor_name == "ComplexityAnalyzer":
                complexity_score = result.results.get('complexity_score')
                sentiment_breakdown.update({
                    'readability_score': result.results.get('readability_score'),
                    'avg_sentence_length': result.results.get('avg_sentence_length'),
                    'vocabulary_diversity': result.results.get('vocabulary_diversity')
                })
        
        # Remove duplicates from topics and themes
        topics = list(set(topics))
        key_themes = list(set(key_themes))
        
        return NLPAnalysis(
            speech_id=speech_id,
            analysis_timestamp=datetime.now(),
            processor_results=processor_results,
            overall_confidence=overall_confidence,
            processing_time=processing_time,
            hawkish_dovish_score=hawkish_dovish_score,
            policy_stance=policy_stance,
            uncertainty_score=uncertainty_score,
            complexity_score=complexity_score,
            topics=topics,
            key_themes=key_themes,
            sentiment_breakdown=sentiment_breakdown,
            pipeline_version=self.version.to_string(),
            processors_used=[r.processor_name for r in successful_results]
        )
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status and processor information.
        
        Returns:
            Dictionary with pipeline status information
        """
        processor_status = []
        for processor in self.processors:
            processor_status.append({
                'name': processor.name,
                'version': processor.version,
                'initialized': processor.is_initialized,
                'confidence': processor.get_confidence_score()
            })
        
        return {
            'pipeline_version': self.version.to_string(),
            'is_initialized': self.is_initialized,
            'processor_count': len(self.processors),
            'processors': processor_status,
            'configuration': {
                'max_workers': self.max_workers,
                'timeout_seconds': self.timeout_seconds,
                'enable_parallel': self.enable_parallel
            }
        }
    
    def get_capabilities(self) -> Dict[str, bool]:
        """
        Get pipeline capabilities based on available dependencies.
        
        Returns:
            Dictionary indicating available capabilities
        """
        return {
            'advanced_nlp': ADVANCED_NLP_AVAILABLE,
            'sklearn_models': SKLEARN_AVAILABLE,
            'parallel_processing': self.enable_parallel,
            'transformer_models': ADVANCED_NLP_AVAILABLE,
            'topic_modeling': SKLEARN_AVAILABLE,
            'sentiment_analysis': True,
            'uncertainty_quantification': True,
            'complexity_analysis': True
        }


# Factory functions for easy pipeline creation

def create_default_pipeline(config: Optional[Dict[str, Any]] = None) -> NLPProcessingPipeline:
    """
    Create a pipeline with default configuration for typical use cases.
    
    Args:
        config: Optional configuration overrides
        
    Returns:
        Configured NLP processing pipeline
        
    Example:
        >>> pipeline = create_default_pipeline()
        >>> await pipeline.initialize()
        >>> analysis = await pipeline.process_speech(speech)
    """
    default_config = {
        'max_workers': 4,
        'timeout_seconds': 300,
        'enable_parallel': True,
        'processors': {
            'hawk_dove': {'lexicon_path': 'lexicon.csv'},
            'topic_modeling': {'n_topics': 10},
            'uncertainty': {},
            'stance': {},
            'complexity': {}
        }
    }
    
    if config:
        # Merge with provided config
        merged_config = default_config.copy()
        merged_config.update(config)
        config = merged_config
    else:
        config = default_config
    
    return NLPProcessingPipeline(config)


def create_lightweight_pipeline() -> NLPProcessingPipeline:
    """
    Create a lightweight pipeline for resource-constrained environments.
    
    Returns:
        Lightweight NLP processing pipeline
    """
    config = {
        'max_workers': 2,
        'timeout_seconds': 120,
        'enable_parallel': False,
        'processors': {
            'hawk_dove': {'lexicon_path': 'lexicon.csv'},
            'uncertainty': {},
            'complexity': {}
        }
    }
    
    pipeline = NLPProcessingPipeline(config)
    
    # Remove resource-intensive processors
    pipeline.processors = [
        processor for processor in pipeline.processors
        if processor.name not in ['TopicModelingProcessor', 'StanceDetector']
    ]
    
    return pipeline


def create_research_pipeline() -> NLPProcessingPipeline:
    """
    Create a comprehensive pipeline optimized for research applications.
    
    Returns:
        Research-oriented NLP processing pipeline
    """
    config = {
        'max_workers': 8,
        'timeout_seconds': 600,
        'enable_parallel': True,
        'processors': {
            'hawk_dove': {
                'lexicon_path': 'lexicon.csv',
                'use_transformer': True
            },
            'topic_modeling': {
                'n_topics': 20,
                'use_advanced_models': True
            },
            'uncertainty': {'detailed_analysis': True},
            'stance': {'fine_grained': True},
            'complexity': {'comprehensive_metrics': True}
        }
    }
    
    return NLPProcessingPipeline(config)


# Example usage and testing
if __name__ == "__main__":
    async def main():
        """Example usage of the NLP processing pipeline."""
        # Create and initialize pipeline
        pipeline = create_default_pipeline()
        await pipeline.initialize()
        
        # Print pipeline status
        status = pipeline.get_pipeline_status()
        print("Pipeline Status:")
        print(f"  Version: {status['pipeline_version']}")
        print(f"  Initialized: {status['is_initialized']}")
        print(f"  Processors: {status['processor_count']}")
        
        capabilities = pipeline.get_capabilities()
        print("\nCapabilities:")
        for capability, available in capabilities.items():
            status_text = "✓" if available else "✗"
            print(f"  {status_text} {capability}")
        
        # Example speech processing would go here
        # (requires actual speech entity with content)
        
        print("\nPipeline ready for speech analysis!")
    
    # Run example
    asyncio.run(main())