# infrastructure/nlp/engines.py

"""
NLP Engines Loader for Central Bank Speech Analysis Platform

Manages centralized loading, caching, and configuration of NLP models
(spaCy pipelines, transformer models, vectorizers, etc.) for use in
all NLP processors and the NLP pipeline.

Author: Central Bank Speech Analysis Platform
Date: 2025
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Singleton containers for engines/models
_loaded_spacy_models: Dict[str, Any] = {}
_loaded_transformers: Dict[str, Any] = {}
_loaded_vectorizers: Dict[str, Any] = {}

# --------------
# spaCy
# --------------

def get_spacy_model(lang: str = "en_core_web_sm") -> Any:
    """
    Loads and caches a spaCy model for the given language.

    Args:
        lang: Name of spaCy model (default: "en_core_web_sm")

    Returns:
        Loaded spaCy Language object
    """
    if lang not in _loaded_spacy_models:
        try:
            import spacy
            logger.info(f"Loading spaCy model: {lang}")
            _loaded_spacy_models[lang] = spacy.load(lang)
        except Exception as e:
            logger.error(f"Failed to load spaCy model '{lang}': {e}")
            raise
    return _loaded_spacy_models[lang]

# --------------
# Transformers (Huggingface)
# --------------

def get_transformer_pipeline(task: str = "sentiment-analysis",
                             model_name: Optional[str] = None,
                             **kwargs) -> Any:
    """
    Loads and caches a Huggingface transformers pipeline.

    Args:
        task: NLP task (e.g., "sentiment-analysis")
        model_name: Model identifier (optional, uses default for task if None)
        kwargs: Extra arguments to transformers.pipeline

    Returns:
        transformers.Pipeline object
    """
    key = (task, model_name or "default")
    if key not in _loaded_transformers:
        try:
            from transformers import pipeline
            logger.info(f"Loading transformers pipeline: task={task}, model={model_name or 'auto'}")
            _loaded_transformers[key] = pipeline(task, model=model_name, **kwargs) if model_name else pipeline(task, **kwargs)
        except Exception as e:
            logger.error(f"Failed to load transformers pipeline for task '{task}': {e}")
            raise
    return _loaded_transformers[key]

# --------------
# Scikit-learn Vectorizers/Models
# --------------

def get_tfidf_vectorizer(config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Loads and caches a TF-IDF vectorizer (scikit-learn).

    Args:
        config: Dictionary of TfidfVectorizer kwargs

    Returns:
        Fitted or new TfidfVectorizer object
    """
    key = str(config) if config else "default"
    if key not in _loaded_vectorizers:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            logger.info("Creating new TfidfVectorizer")
            _loaded_vectorizers[key] = TfidfVectorizer(**(config or {}))
        except Exception as e:
            logger.error(f"Failed to create TfidfVectorizer: {e}")
            raise
    return _loaded_vectorizers[key]

# --------------
# Utility: Clear all loaded models (for testing/reloading)
# --------------

def clear_all_nlp_engines():
    """Clears all loaded/cached NLP models (for test/dev only)."""
    _loaded_spacy_models.clear()
    _loaded_transformers.clear()
    _loaded_vectorizers.clear()
    logger.info("Cleared all NLP engine caches.")

# ----------------
# Example usage:
# ----------------
# spacy_model = get_spacy_model("en_core_web_sm")
# finbert = get_transformer_pipeline("sentiment-analysis", "ProsusAI/finbert")
# vectorizer = get_tfidf_vectorizer({"max_features": 5000})
