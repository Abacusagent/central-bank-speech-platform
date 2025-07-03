#!/usr/bin/env python3
"""
hawk_dove_gauge_4.py – **Self‑Contained Comprehensive Gauge**
==============================================================

A single‑file implementation of a torch‑free, context‑aware Central Bank
Hawk‑Dove sentiment gauge.  No external imports are required beyond NumPy,
Pandas, and PyYAML (and optional scikit‑learn for linear fine‑tune).

The module exposes two public names expected by the rest of the codebase:

```python
from hawk_dove_gauge_4 import CentralBankGauge, HawkDoveGauge
```

Both refer to the same `ComprehensiveHawkDoveGauge` class defined below, which
includes:
* Lexicon‑based scoring
* Contextual adjustments (negations, hedging, intensifiers)
* Semantic boosts (forward guidance, certainty, data‑dependency)
* Simple sequential smoothing
* SQLite persistence for reproducibility
"""
from __future__ import annotations

# ─────────── stdlib ───────────
import argparse
import csv
import logging
import math
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─────────── third‑party (lightweight) ───────────
import numpy as np
import pandas as pd
import yaml

try:
    from sklearn.linear_model import Ridge
    SK_OK = True
except ImportError:
    SK_OK = False

log = logging.getLogger("hawk_dove_gauge_4")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s – %(message)s")

# ─────────── Base gauge ───────────
class HawkDoveGauge:
    """Lexicon‑centric hawk‑dove scorer with optional linear fine‑tune."""

    def __init__(self, *, lexicon_path: str = "lexicon.csv", cfg_path: Optional[str] = None):
        self.lexicon: Dict[str, Tuple[float, float]] = {}
        self.thresholds = (-0.2, 0.2)
        self.model: Optional["Ridge"] = None
        self._load_cfg(cfg_path)
        self._load_lexicon(lexicon_path)

    # ---- cfg ----
    def _load_cfg(self, fp: Optional[str]):
        cfg = {"thresholds": {"hawkish": -0.2, "dovish": 0.2}}
        if fp and Path(fp).exists():
            cfg.update(yaml.safe_load(Path(fp).read_text()))
        lo, hi = cfg["thresholds"].values()
        self.thresholds = (float(os.getenv("HAWK_THRESHOLD", lo)), float(os.getenv("DOVE_THRESHOLD", hi)))

    # ---- lexicon ----
    def _load_lexicon(self, fp: str):
        if not Path(fp).exists():
            log.warning("Lexicon %s not found – neutral scores expected", fp)
            return
        with open(fp, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.lexicon[row["word"].lower()] = (float(row["polarity"]), float(row["weight"]))
        log.info("Loaded %d lexicon terms", len(self.lexicon))

    # ---- helpers ----
    @staticmethod
    def _tok(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _score_lex(self, toks: List[str]) -> Tuple[float, List[Tuple[str, float]]]:
        total = 0.0; arr = []
        for t in toks:
            p, w = self.lexicon.get(t, (0.0, 0.0)); s = p * w; total += s; arr.append((t, s))
        return total / math.sqrt(len(toks) or 1), arr

    def _label(self, s: float) -> str:
        lo, hi = self.thresholds
        return "hawkish" if s <= lo else "dovish" if s >= hi else "neutral"

    # back‑compat alias
    def _get_label(self, s: float) -> str:  # noqa: N802
        return self._label(s)

    # ---- public ----
    def score_sentence(self, text: str) -> Dict[str, Any]:
        val, toks = self._score_lex(self._tok(text))
        if self.model is not None:
            try:
                X = np.array([[self.lexicon.get(t, (0, 0))[0] for t in self._tok(text)]])
                val = (val + float(self.model.predict(X)[0])) / 2
            except Exception as e:
                log.debug("linear model failed: %s", e)
        val = max(-1, min(1, val))
        return {"score": val, "label": self._label(val), "tokens": toks}

    def score_document(self, text: str) -> Dict[str, Any]:
        sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        out = [self.score_sentence(s) for s in sents]
        scores = np.array([o["score"] for o in out]) if out else np.array([0.0])
        return {
            "mean_score": float(scores.mean()),
            "median_score": float(np.median(scores)),
            "std_score": float(scores.std()),
            "sentences": out,
        }

    def fine_tune(self, df: pd.DataFrame):
        if not SK_OK:
            log.warning("sklearn unavailable – skip fine‑tune"); return
        X = np.vstack(df["text"].apply(lambda t: [self.lexicon.get(tok, (0, 0))[0] for tok in self._tok(t)]))
        y = df["score"].values
        self.model = Ridge(alpha=1.0).fit(X, y)
        log.info("Linear fine‑tune complete (R² %.3f)", self.model.score(X, y))

# ─────────── Comprehensive gauge ───────────
from dataclasses import dataclass, asdict

@dataclass
class SentenceStructure:
    main: str
    neg: List[str]
    qual: List[str]
    intens: List[str]
    flags: List[str]

@dataclass
class AnalysisResult:
    timestamp: str
    text: str
    lex: float
    ctx: float
    sem: float
    seq: Optional[float]
    final: float
    label: str
    struct: SentenceStructure

class ComprehensiveHawkDoveGauge(HawkDoveGauge):
    """Adds contextual & semantic heuristics plus simple sequential smoothing."""

    def __init__(self, *, db_path: str = "cb_sentiment.db", **kw):
        super().__init__(**kw)
        self.db = sqlite3.connect(db_path)
        self.db.execute("CREATE TABLE IF NOT EXISTS docs(id INTEGER PRIMARY KEY, mean REAL, ts TEXT)")
        self.hist: List[float] = []
        self._rx()

    def _rx(self):
        self.neg_p = re.compile(r"\b(?:not|never|no|n't)\b", re.I)
        self.qual_p = re.compile(r"\b(?:may|might|could|should|perhaps|likely|possibly)\b", re.I)
        self.int_p = re.compile(r"\b(?:very|extremely|highly|significantly)\b", re.I)
        self.fwd_p = re.compile(r"\bwill\b|\bexpect\b|\boutlook\b", re.I)
        self.bwd_p = re.compile(r"\bwas\b|\bhad\b|\bprevious\b", re.I)
        self.data_p = re.compile(r"\bdata\b|\bindicators\b", re.I)

    # ---- adjustments ----
    def _ctx(self, text: str, s: float):
        if self.neg_p.search(text): s *= -0.8
        if self.qual_p.search(text): s *= 0.7
        if self.int_p.search(text): s *= 1.2 if s < 0 else 1.1
        return s

    def _sem(self, text: str, s: float):
        flags = []
        if self.fwd_p.search(text): s *= 1.2; flags.append("fwd")
        if self.bwd_p.search(text): s *= 0.8; flags.append("bwd")
        if self.data_p.search(text): s *= 0.9; flags.append("data")
        return s, flags

    # ---- sentence ----
    def analyze_sentence(self, text: str) -> AnalysisResult:
        lex, _ = self._score_lex(self._tok(text))
        ctx = self._ctx(text, lex)
        sem, flags = self._sem(text, ctx)
        seq = None
        if self.hist:
            seq = 0.6 * sem + 0.4 * np.mean(self.hist[-5:])
            sem = seq
        sem = max(-1, min(1, sem))
        self.hist.append(sem); self.hist = self.hist[-50:]
        st = SentenceStructure(text, self.neg_p.findall(text), self.qual_p.findall(text), self.int_p.findall(text), flags)
        return AnalysisResult(datetime.utcnow().isoformat(), text, lex, ctx, sem, seq, sem, self._label(sem), st)

    # ---- document ----
    def analyze_document_comprehensive(self, text: str, speech_id: Optional[str] = None):
        sents = [s for s in re.split(r"[.!?]+", text) if len(s) > 10]
        res = [self.analyze_sentence(s) for s in sents]
        arr = np.array([r.final for r in res]) if res else np.array([0.0])
        doc = {
            "mean_score": float(arr.mean()),
            "median_score": float(np.median(arr)),
            "std_score": float(arr.std()),
            "num_sentences": len(res),
            "hawkish_sentences": int((arr <= -0.2).sum()),
            "dovish_sentences": int((arr >= 0.2).sum()),
            "neutral_sentences": int(((-0.2 < arr) & (arr < 0.2)).sum()),
            "sentence_results": [asdict(r) for r in res],
            "dominant_method": "comprehensive",
            "overall_confidence": "medium",
            "method_comparison": {
                "lexicon_mean": float(arr.mean()) if res else None,
                "contextual_mean": None,
                "semantic_mean": None,
                "transformer_mean": None,
                "sequential_mean": None,
            },
        }
        self.db.execute("INSERT INTO docs(mean, ts) VALUES(?, ?)", (doc["mean_score"], datetime.utcnow().isoformat()))
        self.db.commit()
        return doc

# ─────────── public aliases ───────────
CentralBankGauge = ComprehensiveHawkDoveGauge
HawkDoveGauge = ComprehensiveHawkDoveGauge

__all__ = ["CentralBankGauge", "HawkDoveGauge", "ComprehensiveHawkDoveGauge"]

# ─────────── simple CLI test ───────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Quick test for hawk_dove_gauge_4")
    ap.add_argument("file", help="Text file to score")
    ap.add_argument("--lexicon", default="lexicon.csv")
    opt = ap.parse_args()
    txt = Path(opt.file).read_text(encoding="utf-8")
    g = CentralBankGauge(lexicon_path=opt.lexicon)
    print(g.analyze_document_comprehensive(txt)["mean_score"])
