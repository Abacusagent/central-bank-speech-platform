# nlp/processors/topic_modeling_processor.py

"""
Topic Modeling Processor for Central Bank Speech Analysis Platform

Identifies main topics/themes in a speech using both rule-based keyword matching
and, if available, Latent Dirichlet Allocation (LDA) for unsupervised topic modeling.

Author: Central Bank Speech Analysis Platform
Date: 2025
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from nlp.pipeline import NLPProcessor, ProcessorResult
from domain.entities import CentralBankSpeech
from domain.value_objects import ConfidenceLevel

logger = logging.getLogger(__name__)

# Try to import scikit-learn for advanced topic modeling
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class TopicModelingProcessor(NLPProcessor):
    """
    Topic modeling processor for identifying key themes in speeches.

    Uses both keyword-matching and (optionally) LDA topic modeling.
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
        """Initialize the topic modeling processor and any ML models."""
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
            logger.info("Initialized topic modeling components (scikit-learn available)")
        else:
            logger.warning("scikit-learn not available, using keyword-based topics only")

    async def analyze(self, speech: CentralBankSpeech) -> ProcessorResult:
        """
        Identify topics in a speech.
        Returns a ProcessorResult with a list of top topics and diagnostics.
        """
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

            # 1. Keyword-based topics (always runs)
            keyword_topics = self._identify_keyword_topics(text)

            # 2. LDA-based topics (optional, only runs if model is available)
            lda_topics = []
            if self.lda_model and SKLEARN_AVAILABLE:
                lda_topics = self._identify_lda_topics(text)

            # Combine and deduplicate
            all_topics = keyword_topics + lda_topics
            unique_topics = list(dict.fromkeys(all_topics))  # Preserves order

            processing_time = (datetime.now() - start_time).total_seconds()
            confidence = self._calculate_topic_confidence(text, unique_topics)

            return ProcessorResult(
                processor_name=self.name,
                success=True,
                confidence=confidence,
                processing_time=processing_time,
                results={
                    'topics': unique_topics[:5],  # Top 5 topics only
                    'keyword_topics': keyword_topics,
                    'lda_topics': lda_topics,
                    'topic_confidence': confidence,
                },
                metadata={
                    'method_used': 'combined' if self.lda_model else 'keyword_only',
                    'total_topics_identified': len(unique_topics)
                }
            )
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Topic modeling failed: {e}")
            return ProcessorResult(
                processor_name=self.name,
                success=False,
                confidence=0.0,
                processing_time=processing_time,
                results={},
                error_message=str(e)
            )

    def _identify_keyword_topics(self, text: str) -> List[str]:
        """
        Identify topics using curated keyword lists.
        """
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
        """
        Identify topics using LDA (if sklearn available).
        (In a real production setting, the LDA model would be pre-trained on a full corpus.)
        Here, we fit on just this speech for demonstration; not ideal for real use!
        """
        if not self.vectorizer or not self.lda_model:
            return []

        try:
            # Fit vectorizer and LDA model on the text
            docs = [text]
            tfidf = self.vectorizer.fit_transform(docs)
            self.lda_model.fit(tfidf)
            topic_distrib = self.lda_model.transform(tfidf)[0]

            top_indices = topic_distrib.argsort()[-3:][::-1]  # Top 3 topics
            topics = [self.topic_labels.get(idx, f"Topic {idx}") for idx in top_indices if topic_distrib[idx] > 0.05]

            return topics
        except Exception as e:
            logger.warning(f"LDA topic extraction failed: {e}")
            return []

    def _calculate_topic_confidence(self, text: str, topics: List[str]) -> float:
        """
        Heuristic for topic confidence based on length and number of topics.
        """
        if not topics:
            return 0.0
        text_length = len(text.split())
        if text_length < 100:
            return 0.4
        elif text_length < 500:
            return 0.6
        else:
            return 0.8

    def get_confidence_score(self) -> float:
        """Return static confidence in this processor's ability."""
        return 0.8 if SKLEARN_AVAILABLE else 0.6
