# Social Media Sentiment Analyzer

Analyze the sentiment of social posts and articles by URL using transformer models. The app extracts readable text from a page, detects language, selects an appropriate model, and returns a sentiment label with calibrated scores. Batch analysis and sentence-level highlights are included.

## Features

- Transformer-based sentiment (default: `cardiffnlp/twitter-roberta-base-sentiment-latest`)
- Optional multilingual auto-pick for non‑English pages (`cardiffnlp/twitter-xlm-roberta-base-sentiment`)
- Readability extraction of main content (reduces boilerplate and ads)
- Language detection and configurable neutral threshold
- Sentence-level highlights (top positive/negative sentences with scores)
- Batch analysis via CSV upload; CSV/JSON export
- Caching for repeat analyses; clean Streamlit UI

## Quickstart

Prerequisites: Python 3.9+ (Windows, macOS, or Linux)

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
./.venv/Scripts/activate        # Windows PowerShell
# source .venv/bin/activate     # macOS/Linux

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

Open the URL shown by Streamlit (e.g., `http://localhost:8501`).

## How to use

### Single URL
1. Paste a post or article URL.
2. Click “Analyze URL”.
3. Review the sentiment label, score, probabilities, detected language, and model used.
4. Inspect top positive/negative sentences. Download the JSON if needed.

### Batch URLs
1. Upload a CSV with a `url` column (header must be `url`).
2. The app analyzes each row and returns a results table.
3. Download the annotated CSV.

### Sidebar options
- Model name (Hugging Face hub identifier)
- Neutral threshold (±)
- Auto-pick model based on detected language

## Output reference

Key fields returned per URL:

| Field                  | Description                                                         |
|------------------------|---------------------------------------------------------------------|
| `Sentiment`            | Final label: Positive, Neutral, or Negative                         |
| `Score`                | Scalar score (approx. `positive_prob - negative_prob`)              |
| `T_POS` / `T_NEU` / `T_NEG` | Class probabilities from the model                         |
| `language`             | ISO language code detected from text                                |
| `MODEL`                | Model used for scoring                                              |
| `url`                  | Source URL                                                          |
| `title`                | Extracted page title (if available)                                 |
| `highlights_positive`  | List of top positive sentences `(sentence, score)`                  |
| `highlights_negative`  | List of top negative sentences `(sentence, score)`                  |

## How it works

1. Fetch and parse the page with requests + BeautifulSoup.
2. Extract the main readable content via `readability-lxml` (fallback to all `<p>` tags).
3. Detect language (for model auto-pick).
4. Score with the configured transformer model using the Transformers pipeline (truncation to 512 tokens).
5. Split into sentences and score each to surface the most positive and negative lines.
6. Cache results for an hour to accelerate repeat analysis.

## Models

- Default: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
- Multilingual (auto mode): https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment

Change the default in `sentiment.py` (`_HF_MODEL_NAME_DEFAULT`) or via the sidebar.

## Configuration

- UI theme and server settings: `.streamlit/config.toml`
- Default model selection and sentence highlight parameters: `sentiment.py`

## Module usage (optional)

```python
from sentiment import analyze_url, analyze_urls

single = analyze_url("https://example.com/post-or-article")
print(single["Sentiment"], single["Score"])  # label and score

batch = analyze_urls(["https://a.com", "https://b.com"])  # returns a DataFrame
print(batch.head())
```

## Troubleshooting

- Port already in use: `streamlit run app.py --server.port 8502`
- Very long pages: scoring truncates to 512 tokens; highlights analyze sentences individually.
- Some sites block scraping: use allowed sources or APIs; respect robots/terms.

## Project structure

```
.
├── app.py                     # Streamlit UI (single/batch, downloads, caching)
├── sentiment.py               # Scoring, scraping, language, highlights
├── MasterDictionary/          # Legacy lexicons (not used by transformer mode)
├── StopWords/                 # Legacy stopwords (not used by transformer mode)
├── requirements.txt
├── requirements.lock.txt      # Pinned environment snapshot
├── .streamlit/config.toml     # Theme / server config
├── .gitignore
├── LICENSE                    # MIT
└── README.md
```

## License

MIT — see `LICENSE`.