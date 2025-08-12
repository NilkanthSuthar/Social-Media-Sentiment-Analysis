<p align="center">
  <a href="https://github.com/NilkanthSuthar/Social-Media-Sentiment-Analysis">
    <img src="https://img.shields.io/badge/Repo-Social--Media--Sentiment--Analysis-blue.svg?style=flat-square" alt="Repo">
  </a>
  <a href="https://img.shields.io/badge/Python-3.7%2B-blue.svg?style=flat-square&logo=python">
    <img src="https://img.shields.io/badge/Python-3.7%2B-blue.svg?style=flat-square&logo=python" alt="Python">
  </a>
  <a href="https://github.com/NilkanthSuthar/Social-Media-Sentiment-Analysis/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square" alt="License: MIT">
  </a>
</p>

# URL Sentiment Analyzer

A professional Streamlit application and Python module for transformer-based sentiment analysis of web articles. Given a URL, the app extracts the main article text, detects language, selects an appropriate model, and returns **Positive**, **Neutral**, or **Negative** with confidence scores.

---

## 🔍 Table of Contents

1. [Key Features](#key-features)  
2. [Tech Stack](#tech-stack)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [Input & Output](#input--output)  
6. [Customization](#customization)  
7. [Project Structure](#project-structure)  
8. [Contributing](#contributing)  
9. [License](#license)  
10. [Contact](#contact)  

---

## Key Features

- **Lexicon-Based Scoring**  
  Uses customizable positive/negative word lists for transparent scoring.  
- **Stop-Word Filtering**  
  Leverages user-defined stop-word lists to eliminate noise.  
- **Bulk Processing**  
  Reads input from `Input.xlsx`, writes results to `Output.xlsx`.  
- **Lightweight & Extensible**  
  Designed for easy integration into larger ML/AI pipelines.

---

## Tech Stack

- **Language**: Python 3.9+
- **Libraries**: Streamlit, Transformers, Torch, Pandas, Readability, BeautifulSoup

---

## Installation

```bash
# From repo root
python -m venv .venv
./.venv/Scripts/activate  # Windows PowerShell
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

---

## Usage

### Run the UI

```bash
streamlit run app.py
```

- Single URL: paste a URL and click Analyze URL
- Batch: upload a CSV with a `url` column and download results
- Sidebar: select a model and threshold; optionally auto-pick model by detected language

---

## Input & Output

- Input: a single URL or a CSV with a `url` column
- Output: sentiment label, score, class probabilities, detected language, model used; downloadable CSV/JSON

---

## Customization

- Change default model in `sentiment.py` (`_HF_MODEL_NAME_DEFAULT`)
- Enable multilingual default in `pick_model_for_language`
- Adjust sentence highlight count and thresholds in `highlight_sentences`

---

## Project Structure

```
.
├── app.py                     # Streamlit UI
├── sentiment.py               # Analysis module (transformer-based, scraping, highlights)
├── MasterDictionary/          # Legacy lexicons (not used in transformer mode)
├── StopWords/                 # Legacy stopwords (not used in transformer mode)
├── requirements.txt
├── .streamlit/config.toml     # Theme / server config
├── .gitignore
├── LICENSE                    # MIT
└── README.md
```

## License

Distributed under the MIT License. See `LICENSE` for details.

---

## Contributing

1. Fork the repository  
2. Create a feature branch: `git checkout -b feature/YourFeature`  
3. Commit your changes: `git commit -m 'Add your message'`  
4. Push to the branch: `git push origin feature/YourFeature`  
5. Open a Pull Request  

Please ensure all tests pass and adhere to the existing style.

---

## License

Distributed under the **MIT License** © 2025 Nilkanth Suthar. See [LICENSE](LICENSE) for details.

---

## Contact

Nilkanth Suthar – [suthar93@uwindsor.ca](mailto:suthar93@uwindsor.ca)  
GitHub: [@NilkanthSuthar](https://github.com/NilkanthSuthar)  
LinkedIn: [nilkanthsuthar](https://linkedin.com/in/nilkanthsuthar)  
