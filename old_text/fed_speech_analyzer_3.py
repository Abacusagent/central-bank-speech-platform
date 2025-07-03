#!/usr/bin/env python3
"""
Fed Speech Sentiment Analysis Pipeline
======================================

Processes Federal Reserve speeches using the CentralBankGauge (comprehensive version).
Outputs a CSV and generates a summary report.

Usage:
~~~~~~
python fed_speech_analyzer.py \
    --data-dir data \
    --lexicon lexicon.csv \
    --output fed_speech_sentiment_analysis.csv

Add --summary-only to skip processing and summarize an existing CSV.
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pandas import DataFrame

try:
    from hawk_dove_gauge_5 import CentralBankGauge
except ImportError:
    print("❌ CentralBankGauge not found. Ensure hawk_dove_gauge_5.py is on the PYTHONPATH.")
    sys.exit(1)

try:
    from textblob import TextBlob
    TEXTBLOB_OK = True
except ImportError:
    TEXTBLOB_OK = False
    logging.warning("TextBlob not available — skipping polarity and subjectivity.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")
log = logging.getLogger("fed_analyzer")

@dataclass
class SpeechMeta:
    directory_name: str
    json_file: str
    txt_file: str
    content: str
    meta: Dict[str, str]

@dataclass
class Row:
    directory_name: str
    date: str
    speaker: str
    speaker_role: str
    institution: str
    speech_title: str
    location: str
    content_length: int
    hd_mean_score: Optional[float] = None
    hd_median_score: Optional[float] = None
    hd_std_score: Optional[float] = None
    hd_label: Optional[str] = None
    hd_num_sentences: Optional[int] = None
    hd_hawkish_sentences: Optional[int] = None
    hd_dovish_sentences: Optional[int] = None
    hd_neutral_sentences: Optional[int] = None
    hd_dominant_method: Optional[str] = None
    hd_overall_confidence: Optional[str] = None
    hd_analysis_method: Optional[str] = None
    hd_methods_used: Optional[int] = None
    hd_lexicon_mean: Optional[float] = None
    hd_contextual_mean: Optional[float] = None
    hd_semantic_mean: Optional[float] = None
    hd_transformer_mean: Optional[float] = None
    hd_sequential_mean: Optional[float] = None
    tb_polarity: Optional[float] = None
    tb_subjectivity: Optional[float] = None
    tb_label: Optional[str] = None
    source_url: str = ""
    analysis_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

class FedSpeechAnalyzer:
    COLS = list(Row.__annotations__.keys())

    def __init__(self, data_dir: str, lexicon_path: str, enable_textblob: bool = True):
        self.data_path = Path(data_dir) / "fed"
        if not self.data_path.exists():
            raise FileNotFoundError(f"Directory not found: {self.data_path}")
        self.gauge = CentralBankGauge(lexicon_path=lexicon_path)
        self.use_textblob = TEXTBLOB_OK and enable_textblob
        self.results: List[Row] = []

    def _clean_speaker_name(self, name: str) -> str:
        if not name or name.lower() == "unknown":
            return name

        name = re.sub(r"[_/]+", " ", name)
        tokens = name.split()

        # Identify all runs of capitalized tokens
        runs = []
        current_run = []
        for t in tokens:
            if t[:1].isupper():
                current_run.append(t)
            elif current_run:
                runs.append(current_run)
                current_run = []
        if current_run:
            runs.append(current_run)

        # Return the longest capitalized run (likely the actual name)
        if runs:
            best = max(runs, key=len)
            return " ".join(best)
        return name.strip()



    def _iter_speech_dirs(self):
        for p in sorted(self.data_path.iterdir(), key=lambda x: x.name):
            if p.is_dir() and list(p.glob("*.json")) and list(p.glob("*.txt")):
                yield p

    def _load_speech(self, folder: Path) -> Optional[SpeechMeta]:
        try:
            meta_fp = next(folder.glob("*.json"))
            txt_fp = next(folder.glob("*.txt"))
            content = txt_fp.read_text(encoding="utf-8").strip()
            if not content:
                log.warning("Empty content: %s", folder)
                return None
            meta = json.loads(meta_fp.read_text(encoding="utf-8"))
            return SpeechMeta(folder.name, meta_fp.name, txt_fp.name, content, meta)
        except (StopIteration, json.JSONDecodeError) as e:
            log.error("Failed loading %s: %s", folder, e)
            return None

    def _analyze_hawk_dove(self, content: str) -> Dict[str, Optional[float]]:
        doc = self.gauge.analyze_document_comprehensive(content)
        label = "hawkish" if doc["mean_score"] <= -0.05 else "dovish" if doc["mean_score"] >= 0.05 else "neutral"
        doc["label"] = label 
        mc = doc.get("method_comparison", {})
        return {
            "hd_mean_score": doc["mean_score"],
            "hd_median_score": doc["median_score"],
            "hd_std_score": doc["std_score"],
            "hd_label": doc.get("label"),
            "hd_num_sentences": doc.get("num_sentences"),
            "hd_hawkish_sentences": doc.get("hawkish_sentences"),
            "hd_dovish_sentences": doc.get("dovish_sentences"),
            "hd_neutral_sentences": doc.get("neutral_sentences"),
            "hd_dominant_method": doc.get("dominant_method"),
            "hd_overall_confidence": doc.get("overall_confidence"),
            "hd_analysis_method": "comprehensive" if mc else "lexicon_only",
            "hd_methods_used": sum(v is not None for k, v in mc.items() if k.endswith("_mean")),
            "hd_lexicon_mean": mc.get("lexicon_mean"),
            "hd_contextual_mean": mc.get("contextual_mean"),
            "hd_semantic_mean": mc.get("semantic_mean"),
            "hd_transformer_mean": mc.get("transformer_mean"),
            "hd_sequential_mean": mc.get("sequential_mean"),
        }

    def _analyze_textblob(self, content: str) -> Dict[str, Optional[float]]:
        if not self.use_textblob:
            return {}
        blob = TextBlob(content)
        pol = blob.sentiment.polarity
        sub = blob.sentiment.subjectivity
        label = "positive" if pol > 0.1 else "negative" if pol < -0.1 else "neutral"
        return {
            "tb_polarity": pol,
            "tb_subjectivity": sub,
            "tb_label": label,
        }

    def run(self):
        for folder in self._iter_speech_dirs():
            meta = self._load_speech(folder)
            if not meta:
                continue
            hd = self._analyze_hawk_dove(meta.content)
            tb = self._analyze_textblob(meta.content)

            speaker_raw = meta.meta.get("speaker", "unknown")
            speaker = self._clean_speaker_name(speaker_raw)

            row = Row(
                directory_name=meta.directory_name,
                date=meta.meta.get("date", "unknown"),
                speaker=speaker,
                speaker_role=meta.meta.get("role", "unknown"),
                institution=meta.meta.get("institution", "Federal Reserve"),
                speech_title=meta.meta.get("title", ""),
                location=meta.meta.get("location", ""),
                content_length=len(meta.content),
                source_url=meta.meta.get("source_url", ""),
                **hd,
                **tb,
            )
            self.results.append(row)
            log.info("Processed %s — mean %.3f", folder.name, row.hd_mean_score or 0.0)

    def to_dataframe(self) -> DataFrame:
        df = pd.DataFrame([asdict(r) for r in self.results])
        for col in self.COLS:
            if col not in df:
                df[col] = None
        return df[self.COLS]

    def save_csv(self, fp: str):
        df = self.to_dataframe()
        df.to_csv(fp, index=False, encoding="utf-8")
        log.info("Saved %d rows → %s", len(df), fp)

    def summary(self) -> str:
        if not self.results:
            return "No results."
        df = self.to_dataframe()
        return f"""
=== FED SPEECH SENTIMENT ANALYSIS SUMMARY ===
Total speeches: {len(df)}
Date range: {df.date.min()} → {df.date.max()}
Unique speakers: {df.speaker.nunique()}

Hawk-Dove (comprehensive):
  Avg score: {df.hd_mean_score.mean():.4f}
  Std dev : {df.hd_mean_score.std():.4f}
  Avg methods used: {df.hd_methods_used.mean():.1f}
Label distribution: {df.hd_label.value_counts().to_dict()}
"""

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--lexicon", default="lexicon.csv")
    p.add_argument("--output", default="fed_speech_sentiment_analysis.csv")
    p.add_argument("--skip-textblob", action="store_true")
    args = p.parse_args()

    analyzer = FedSpeechAnalyzer(
        data_dir=args.data_dir,
        lexicon_path=args.lexicon,
        enable_textblob=not args.skip_textblob,
    )
    analyzer.run()
    analyzer.save_csv(args.output)
    print(analyzer.summary())

if __name__ == "__main__":
    main()
