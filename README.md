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

# 📊 Social Media Sentiment Analysis

A **lexicon-based** sentiment analysis tool in Python that classifies social media posts (Twitter, Instagram, YouTube) into **Positive**, **Negative**, or **Neutral**. It cleans and preprocesses text, applies custom sentiment dictionaries and stop-word filters, and outputs results in Excel format for seamless downstream analysis.

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

- **Language**: Python 3.7+  
- **Libraries**:  
  - [pandas](https://pandas.pydata.org/)  
  - [openpyxl](https://openpyxl.readthedocs.io/)  

---

## Installation

```bash
# Clone repository
git clone https://github.com/NilkanthSuthar/Social-Media-Sentiment-Analysis.git
cd Social-Media-Sentiment-Analysis

# (Optional) Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

1. **Prepare your data**  
   - Place messages in `Input.xlsx` under a column named `Text`.  

2. **Run the analysis**  
   ```bash
   python Code.py
   ```

3. **Review output**  
   - `Output.xlsx` contains:
     | Text                             | Sentiment | Score |
     |----------------------------------|-----------|-------|
     | I love this product!             | Positive  |     3 |
     | This was the worst experience.   | Negative  |    -2 |
     | It’s okay, not bad nor great.    | Neutral   |     0 |

---

## Input & Output

| File         | Description                                   |
|--------------|-----------------------------------------------|
| `Input.xlsx` | Excel file with a `Text` column of messages   |
| `Output.xlsx`| Excel file with `Text`, `Sentiment`, `Score`  |

---

## Customization

- **Sentiment Dictionaries**  
  - Edit `MasterDictionary/positive.txt` and `MasterDictionary/negative.txt`.  
- **Stop-Word List**  
  - Modify `StopWords/stopwords.txt`.  
- **Scoring Logic**  
  - Adjust thresholds and weights in `Code.py`.

---

## Project Structure

```
.
├── Code.py
├── Input.xlsx
├── Output.xlsx
├── MasterDictionary/
│   ├── positive.txt
│   └── negative.txt
├── StopWords/
│   └── stopwords.txt
├── requirements.txt
└── README.md
```

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
