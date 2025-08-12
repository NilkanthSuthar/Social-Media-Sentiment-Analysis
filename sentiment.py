from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
from readability import Document
from langdetect import detect, LangDetectException

# NLTK tokenizers
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import data as nltk_data, download as nltk_download

# Lightweight ML sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


class AnalyzerType(str, Enum):
    LEXICON = "Lexicon"
    VADER = "VADER"
    HF_TRANSFORMER = "Transformer"
    SKLEARN = "Sklearn"


_PROJECT_ROOT = Path(__file__).resolve().parent
_LEXICON_DIR = _PROJECT_ROOT / "MasterDictionary"
_STOPWORDS_DIR = _PROJECT_ROOT / "StopWords"
_MODELS_DIR = _PROJECT_ROOT / "models"
_MODELS_DIR.mkdir(exist_ok=True)

_HF_MODEL_NAME_DEFAULT = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def ensure_nltk_resources() -> None:
    """Ensure NLTK tokenizers are available at runtime."""
    try:
        nltk_data.find("tokenizers/punkt")
    except LookupError:
        nltk_download("punkt", quiet=True)
    # Newer NLTK versions require punkt_tab resources for sentence tokenization
    try:
        nltk_data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk_download("punkt_tab", quiet=True)
        except Exception:
            pass


def split_sentences(text: str) -> List[str]:
    """Robust sentence split with NLTK fallback to regex."""
    if not text:
        return []
    ensure_nltk_resources()
    try:
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except LookupError:
        # Regex fallback
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f.readlines()]


def _iter_stopword_files() -> Iterable[Path]:
    if not _STOPWORDS_DIR.exists():
        return []
    return sorted(_STOPWORDS_DIR.glob("*.txt"))


def load_lexicons() -> Tuple[set[str], set[str]]:
    """Load positive and negative word lists as lowercase sets."""
    pos_path = _LEXICON_DIR / "positive-words.txt"
    neg_path = _LEXICON_DIR / "negative-words.txt"
    positive_words = {w.lower() for w in _read_lines(pos_path) if w and not w.startswith(";")}
    negative_words = {w.lower() for w in _read_lines(neg_path) if w and not w.startswith(";")}
    return positive_words, negative_words


def load_stopwords(extra: Optional[Iterable[str]] = None) -> set[str]:
    """Load combined stopwords from all files plus optional extras."""
    stopwords: set[str] = set()
    for file in _iter_stopword_files():
        for w in _read_lines(file):
            if w:
                stopwords.add(w.strip().lower())
    # Common social tokens
    stopwords.update({"rt", "via"})
    if extra:
        stopwords.update({w.lower() for w in extra})
    return stopwords


# Simple social text cleaner
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_RE = re.compile(r"<[^>]+>")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")


def clean_social_text(text: str) -> str:
    """Basic cleaning for social content: remove urls/html, normalize mentions/hashtags."""
    if not text:
        return ""
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    # Convert #hashtag to plain token
    text = _HASHTAG_RE.sub(r"\1", text)
    # Drop @mentions
    text = _MENTION_RE.sub(" ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_words(text: str, remove_stopwords: bool, stopwords: Optional[set[str]] = None) -> List[str]:
    ensure_nltk_resources()
    tokens = [t.lower() for t in word_tokenize(text)]
    # Keep alphabetic tokens only; this simplifies lexicon matching
    tokens = [t for t in tokens if t.isalpha()]
    if remove_stopwords and stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    return tokens


def label_from_score(score: float, neutral_threshold: float) -> str:
    if score >= neutral_threshold:
        return "Positive"
    if score <= -neutral_threshold:
        return "Negative"
    return "Neutral"


def score_text_lexicon(
    text: str,
    positive_words: set[str],
    negative_words: set[str],
    remove_stopwords: bool,
    stopwords: Optional[set[str]] = None,
    neutral_threshold: float = 0.05,
) -> Dict[str, object]:
    cleaned = clean_social_text(text)
    tokens = tokenize_words(cleaned, remove_stopwords=remove_stopwords, stopwords=stopwords)

    pos_count = sum(1 for t in tokens if t in positive_words)
    neg_count = sum(1 for t in tokens if t in negative_words)
    total = len(tokens) if tokens else 1

    polarity = (pos_count - neg_count) / max((pos_count + neg_count), 1)
    subjectivity = (pos_count + neg_count) / total
    label = label_from_score(polarity, neutral_threshold)

    return {
        "Sentiment": label,
        "Score": float(polarity),
        "POSITIVE_COUNT": int(pos_count),
        "NEGATIVE_COUNT": int(neg_count),
        "SUBJECTIVITY": float(subjectivity),
        "TOKENS": int(len(tokens)),
        "METHOD": AnalyzerType.LEXICON.value,
    }


_VADER_ANALYZER: Optional[SentimentIntensityAnalyzer] = None
_HF_PIPELINE_CACHE: dict[str, object] = {}


def _get_vader_analyzer() -> SentimentIntensityAnalyzer:
    global _VADER_ANALYZER
    if _VADER_ANALYZER is None:
        _VADER_ANALYZER = SentimentIntensityAnalyzer()
    return _VADER_ANALYZER


def score_text_vader(text: str, neutral_threshold: float = 0.05) -> Dict[str, object]:
    cleaned = clean_social_text(text)
    analyzer = _get_vader_analyzer()
    scores = analyzer.polarity_scores(cleaned)
    compound = float(scores.get("compound", 0.0))
    label = label_from_score(compound, neutral_threshold)
    return {
        "Sentiment": label,
        "Score": compound,
        "V_POS": float(scores.get("pos", 0.0)),
        "V_NEG": float(scores.get("neg", 0.0)),
        "V_NEU": float(scores.get("neu", 0.0)),
        "METHOD": AnalyzerType.VADER.value,
    }


def _get_hf_pipeline(model_name: str = _HF_MODEL_NAME_DEFAULT):
    key = model_name.strip()
    if key not in _HF_PIPELINE_CACHE:
        try:
            device = 0 if torch.cuda.is_available() else -1
        except Exception:
            device = -1
        _HF_PIPELINE_CACHE[key] = pipeline("text-classification", model=key, device=device)
    return _HF_PIPELINE_CACHE[key]


def _extract_triple_scores(entries: list[dict]) -> tuple[float, float, float]:
    neg = neu = pos = 0.0
    for e in entries:
        label = str(e.get("label", "")).lower()
        score = float(e.get("score", 0.0))
        if label in {"negative", "neg", "label_0"}:
            neg = score
        elif label in {"neutral", "neu", "label_1"}:
            neu = score
        elif label in {"positive", "pos", "label_2"}:
            pos = score
    # Normalize if they don't sum to 1
    total = neg + neu + pos
    if total > 0:
        neg, neu, pos = neg / total, neu / total, pos / total
    return neg, neu, pos


def score_text_transformer(
    text: str,
    model_name: str = _HF_MODEL_NAME_DEFAULT,
    neutral_threshold: float = 0.05,
) -> Dict[str, object]:
    cleaned = clean_social_text(text)
    pipe = _get_hf_pipeline(model_name)
    try:
        outputs = pipe(cleaned, top_k=None, truncation=True, max_length=512)
    except TypeError:
        outputs = pipe(cleaned, return_all_scores=True, truncation=True, max_length=512)

    # Normalize return shape to list[dict]
    entries: List[Dict] = []
    if isinstance(outputs, list):
        if outputs and isinstance(outputs[0], dict):
            entries = outputs
        elif outputs and isinstance(outputs[0], list) and outputs[0] and isinstance(outputs[0][0], dict):
            entries = outputs[0]
    if not entries:
        res = pipe(cleaned, top_k=1, truncation=True, max_length=512)
        lbl = (res[0]["label"] if isinstance(res, list) else res["label"]).lower()
        if lbl.startswith("neg"):
            neg, neu, pos = 1.0, 0.0, 0.0
        elif lbl.startswith("neu"):
            neg, neu, pos = 0.0, 1.0, 0.0
        else:
            neg, neu, pos = 0.0, 0.0, 1.0
    else:
        neg, neu, pos = _extract_triple_scores(entries)

    score = pos - neg
    if score >= neutral_threshold:
        label = "Positive"
    elif score <= -neutral_threshold:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "Sentiment": label,
        "Score": float(score),
        "T_POS": float(pos),
        "T_NEU": float(neu),
        "T_NEG": float(neg),
        "METHOD": AnalyzerType.HF_TRANSFORMER.value,
        "MODEL": model_name,
    }


def highlight_sentences(
    text: str,
    model_name: str = _HF_MODEL_NAME_DEFAULT,
    top_k: int = 3,
    neutral_threshold: float = 0.05,
) -> Dict[str, List[Tuple[str, float]]]:
    """Return top positive and negative sentences with scores (pos-neg)."""
    sentences = split_sentences(text)
    if not sentences:
        return {"positive": [], "negative": []}

    pipe = _get_hf_pipeline(model_name)
    scored: List[Tuple[str, float]] = []
    for s in sentences:
        s_clean = clean_social_text(s)
        try:
            outputs = pipe(s_clean, top_k=None, truncation=True, max_length=256)
        except TypeError:
            outputs = pipe(s_clean, return_all_scores=True, truncation=True, max_length=256)
        entries: List[Dict] = []
        if isinstance(outputs, list):
            if outputs and isinstance(outputs[0], dict):
                entries = outputs
            elif outputs and isinstance(outputs[0], list) and outputs[0] and isinstance(outputs[0][0], dict):
                entries = outputs[0]
        if not entries:
            res = pipe(s_clean, top_k=1, truncation=True, max_length=256)
            lbl = (res[0]["label"] if isinstance(res, list) else res["label"]).lower()
            score = 1.0 if lbl.startswith("pos") else (-1.0 if lbl.startswith("neg") else 0.0)
        else:
            neg, neu, pos = _extract_triple_scores(entries)
            score = pos - neg
        scored.append((s, float(score)))

    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    pos_top = [(s, sc) for s, sc in scored_sorted if sc > neutral_threshold][:top_k]
    neg_top = [(s, sc) for s, sc in sorted(scored, key=lambda x: x[1]) if sc < -neutral_threshold][:top_k]
    return {"positive": pos_top, "negative": neg_top}


def scrape_url(url: str, timeout: float = 15.0) -> Dict[str, str]:
    """Fetch and extract main text from a news/blog/article URL.

    Returns keys: title, text, language.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    html = resp.text
    doc = Document(html)
    title = (doc.short_title() or "").strip()
    summary_html = doc.summary(html_partial=True)
    soup = bs(summary_html, "html.parser")
    parts = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    article_text = "\n\n".join([p for p in parts if p])
    if not article_text:
        soup_full = bs(html, "html.parser")
        parts = [p.get_text(" ", strip=True) for p in soup_full.find_all("p")]
        article_text = "\n\n".join([p for p in parts if p])

    language = ""
    try:
        language = detect(article_text) if article_text else ""
    except LangDetectException:
        language = ""

    return {"title": title, "text": article_text, "language": language}


def pick_model_for_language(language: str, default_model: str = _HF_MODEL_NAME_DEFAULT) -> str:
    if not language or language.lower().startswith("en"):
        return default_model
    # Multilingual model for non-English
    return "cardiffnlp/twitter-xlm-roberta-base-sentiment"


def analyze_url(
    url: str,
    model_name: str = _HF_MODEL_NAME_DEFAULT,
    neutral_threshold: float = 0.05,
    auto_model_by_language: bool = True,
    include_highlights: bool = True,
    top_k_highlights: int = 3,
) -> Dict[str, object]:
    scraped = scrape_url(url)
    language = scraped.get("language", "")
    chosen_model = pick_model_for_language(language, default_model=model_name) if auto_model_by_language else model_name
    result = score_text_transformer(scraped.get("text", ""), model_name=chosen_model, neutral_threshold=neutral_threshold)
    out: Dict[str, object] = {
        "url": url,
        "title": scraped.get("title", ""),
        "language": language,
        **result,
    }
    if include_highlights:
        hl = highlight_sentences(scraped.get("text", ""), model_name=chosen_model, top_k=top_k_highlights, neutral_threshold=neutral_threshold)
        out["highlights_positive"] = hl["positive"]
        out["highlights_negative"] = hl["negative"]
    return out


def analyze_urls(
    urls: Iterable[str],
    model_name: str = _HF_MODEL_NAME_DEFAULT,
    neutral_threshold: float = 0.05,
    auto_model_by_language: bool = True,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for u in urls:
        try:
            res = analyze_url(
                u,
                model_name=model_name,
                neutral_threshold=neutral_threshold,
                auto_model_by_language=auto_model_by_language,
                include_highlights=False,
            )
            rows.append(res)
        except Exception as e:
            rows.append({
                "url": u,
                "title": "",
                "language": "",
                "Sentiment": "Error",
                "Score": 0.0,
                "error": str(e),
                "MODEL": model_name,
            })
    return pd.DataFrame(rows)


_SKLEARN_MODEL_PATH = _MODELS_DIR / "sklearn_sentiment.joblib"


def train_sklearn_model(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    model_path: Optional[Path] = None,
) -> dict:
    """Train a simple TF-IDF + Logistic Regression classifier and persist it."""
    model_path = model_path or _SKLEARN_MODEL_PATH
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str).map(clean_social_text)

    X_train, X_val, y_train, y_val = train_test_split(
        df[text_col], df[label_col], test_size=0.2, random_state=42, stratify=df[label_col]
    )

    clf: SkPipeline = SkPipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)),
            (
                "logreg",
                LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", multi_class="auto"),
            ),
        ]
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    report = classification_report(y_val, y_pred, zero_division=0, output_dict=False)

    joblib.dump(clf, model_path)
    return {"model_path": str(model_path), "report": report}


def _load_sklearn_model(model_path: Optional[Path] = None):
    path = model_path or _SKLEARN_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Sklearn model not found at {path}. Train one via train_sklearn_model().")
    return joblib.load(path)


def score_text_sklearn(text: str, model_path: Optional[Path] = None, neutral_threshold: float = 0.05) -> Dict[str, object]:
    clf = _load_sklearn_model(model_path)
    cleaned = clean_social_text(text)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba([cleaned])[0]
        classes = getattr(clf, "classes_", [])
        # Map class probabilities into NEG/NEU/POS if possible
        cls_map = {str(c).lower(): p for c, p in zip(classes, proba)}
        neg = float(cls_map.get("negative", cls_map.get("neg", 0.0)))
        neu = float(cls_map.get("neutral", cls_map.get("neu", 0.0)))
        pos = float(cls_map.get("positive", cls_map.get("pos", 0.0)))
        # If only binary present, infer the missing class as zero
        total = neg + neu + pos
        if total == 0.0:
            # Fallback using decision function or predicted label
            label_pred = str(clf.predict([cleaned])[0]).lower()
            if label_pred.startswith("neg"):
                neg, neu, pos = 1.0, 0.0, 0.0
            elif label_pred.startswith("neu"):
                neg, neu, pos = 0.0, 1.0, 0.0
            else:
                neg, neu, pos = 0.0, 0.0, 1.0
    else:
        label_pred = str(clf.predict([cleaned])[0]).lower()
        if label_pred.startswith("neg"):
            neg, neu, pos = 1.0, 0.0, 0.0
        elif label_pred.startswith("neu"):
            neg, neu, pos = 0.0, 1.0, 0.0
        else:
            neg, neu, pos = 0.0, 0.0, 1.0

    score = pos - neg
    label = "Positive" if score >= neutral_threshold else ("Negative" if score <= -neutral_threshold else "Neutral")

    return {
        "Sentiment": label,
        "Score": float(score),
        "S_POS": float(pos),
        "S_NEU": float(neu),
        "S_NEG": float(neg),
        "METHOD": AnalyzerType.SKLEARN.value,
        "MODEL": str(model_path or _SKLEARN_MODEL_PATH),
    }


def analyze_texts(
    texts: Iterable[str],
    method: AnalyzerType = AnalyzerType.VADER,
    neutral_threshold: float = 0.05,
    remove_stopwords: bool = True,
    hf_model_name: str = _HF_MODEL_NAME_DEFAULT,
    sklearn_model_path: Optional[str] = None,
) -> pd.DataFrame:
    """Analyze an iterable of texts and return a tidy DataFrame."""
    positive_words, negative_words = load_lexicons()
    stopwords = load_stopwords()

    rows: List[Dict[str, object]] = []
    for text in texts:
        if method == AnalyzerType.LEXICON:
            result = score_text_lexicon(
                text=text or "",
                positive_words=positive_words,
                negative_words=negative_words,
                remove_stopwords=remove_stopwords,
                stopwords=stopwords,
                neutral_threshold=neutral_threshold,
            )
        elif method == AnalyzerType.VADER:
            result = score_text_vader(text=text or "", neutral_threshold=neutral_threshold)
        elif method == AnalyzerType.HF_TRANSFORMER:
            result = score_text_transformer(text=text or "", model_name=hf_model_name, neutral_threshold=neutral_threshold)
        elif method == AnalyzerType.SKLEARN:
            result = score_text_sklearn(text=text or "", model_path=Path(sklearn_model_path) if sklearn_model_path else None, neutral_threshold=neutral_threshold)
        else:
            result = score_text_vader(text=text or "", neutral_threshold=neutral_threshold)
        rows.append({"Text": text, **result})

    df = pd.DataFrame(rows)
    # Order columns nicely
    preferred = [
        "Text",
        "Sentiment",
        "Score",
        "METHOD",
        "POSITIVE_COUNT",
        "NEGATIVE_COUNT",
        "SUBJECTIVITY",
        "TOKENS",
        "V_POS",
        "V_NEG",
        "V_NEU",
        "T_POS",
        "T_NEU",
        "T_NEG",
        "S_POS",
        "S_NEU",
        "S_NEG",
        "MODEL",
    ]
    columns = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[columns]


__all__ = [
    "AnalyzerType",
    "analyze_texts",
    "score_text_lexicon",
    "score_text_vader",
    "score_text_transformer",
    "score_text_sklearn",
    "clean_social_text",
    "load_lexicons",
    "load_stopwords",
]


