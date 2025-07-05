# Al‑Quran Theme & Sentiment Analysis

Apply NLP techniques to the Quran: classifying verses by theme and sentiment using pre-trained models.

---

## 🧠 Project Overview

This project explores:
- **Theme classification** of Quran verses (e.g., Faith, Guidance, Law).
- **Sentiment analysis** (Positive, Neutral, Negative) using NLP pipelines.

All implemented in a single Jupyter Notebook using:
- Text pre-processing (tokenization, stopword removal, lemmatization)
- Pre-trained transformers (e.g., Hugging Face models)
- Visualization of results and model metrics

---

## 📁 Repository Structure

```

Al-Quran-Theme-And-Sentiment-Analysis/
│
├── QuranThemeSentiment.ipynb   # Main notebook with analysis
├── data/
│   └── quran\_verses.csv        # Quran verses + labels (themes & sentiments)
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation (you are here)
└── .gitignore                 # Exclude unnecessary files

````

---

## 🛠️ Tech Stack & Libraries

- **Data**: Pandas, NumPy
- **NLP**: NLTK (tokenization, lemmatization, stopwords), Hugging Face Transformers
- **Modeling**: scikit-learn (classification, evaluation)
- **Visualization**: Matplotlib, Seaborn

---

## 🚀 How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/sarnsrun/Al-Quran-Theme-And-Sentiment-Analysis.git
   cd Al-Quran-Theme-And-Sentiment-Analysis
````

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Launch notebook**

   ```bash
   jupyter notebook QuranThemeSentiment.ipynb
   ```

   or open it in **Google Colab** (link can be added).

---

## 📊 Notebook Sections

1. **Load & Explore Data**

   * Inspect dataset contents, shape, and label distribution

2. **Preprocessing Pipeline**

   * Clean text (remove punctuation, lowercasing)
   * Tokenize, delete stopwords, lemmatize

3. **Feature Engineering**

   * Convert text to embeddings using **Hugging Face Transformers**

4. **Model Building**

   * Train classifier (Logistic Regression / Random Forest)
   * Evaluate accuracy, precision, recall, F1-score

5. **Visualization**

   * Plot label distribution and sklearn metrics

---

## 📈 Results

* **Theme classification**: *X% accuracy*
* **Sentiment analysis**: *Y% F1-score*
* Include sample confusion matrix and performance charts here

---

## 🧾 Future Work

* Explore advanced architectures: CNN / RNN / Transformers
* Expand dataset to include translations or additional themes
* Build a Flask or Streamlit demo for interactive use

---

## ⚙️ Requirements

**requirements.txt sample**:

```txt
pandas
numpy
nltk
scikit-learn
transformers
matplotlib
seaborn
jupyter
```

---

## 📝 License

MIT License – feel free to use and extend!

---

## 👤 Author

**Aisar Nasrun**

* GitHub: [@sarnsrun](https://github.com/sarnsrun)

---
