# 📊 Social Media Sentiment Analysis

A Python-based sentiment analysis tool that classifies social media content into positive, negative, or neutral categories. Built using a lexicon-based approach, this tool analyzes text from platforms like Twitter, Instagram, and YouTube using custom sentiment dictionaries and stopword filters.

---

## 📌 Overview

This project applies a rule-based sentiment scoring mechanism using predefined positive and negative word lists. It cleans and processes raw text, removes stopwords, and generates sentiment labels for each input message. Results are written to a structured Excel output file, ready for downstream analysis or visualization.

---

## 🧰 Features

- ✅ Lexicon-based sentiment scoring  
- ✅ Stopword filtering using customizable stopword lists  
- ✅ Bulk processing of inputs from Excel (`Input.xlsx`)  
- ✅ Outputs sentiment results to Excel (`Output.xlsx`)  
- ✅ Lightweight and easy to customize or integrate with other ML pipelines

---

## 🛠 Tech Stack

- **Language**: Python  
- **Libraries**:  
  - pandas  
  - openpyxl  
- **Assets**:  
  - MasterDictionary (positive/negative words)  
  - StopWords list

---

## 🚀 How to Run

1. **Install dependencies**

   pip install pandas openpyxl

2. **Place input data**

   Add your text data to a column named `Text` in `Input.xlsx`.

3. **Run the script**

   python Code.py

4. **Check output**

   Results will be saved in `Output.xlsx` with sentiment labels.

---

## 📈 Sample Output

| Text                              | Sentiment |
|----------------------------------|-----------|
| "I love this product!"           | Positive  |
| "This is the worst experience."  | Negative  |
| "It's okay, not great, not bad." | Neutral   |

---

## 📄 Documentation

See `Documentation_Updated.pdf` for a full explanation of the scoring logic, data preprocessing, and customization options.
