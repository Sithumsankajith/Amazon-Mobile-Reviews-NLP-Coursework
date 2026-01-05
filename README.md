# Analysis of Amazon Mobile Phone Reviews

A full Natural Language Processing (NLP) pipeline to explore **themes, sentiment, emotion, and aspect-level opinions** in real-world smartphone reviews from Amazon’s Unlocked Mobile dataset. The project combines **classical NLP (TF–IDF, LDA, VADER, PCA, clustering)** with **modern transformer-based methods (BERTopic, SBERT semantic search, summarization, emotion + toxicity classifiers)** to generate interpretable insights for product feedback analysis. :contentReference[oaicite:0]{index=0}

---

## Project Highlights
- ✅ Cleaned + standardized text preprocessing pipeline (tokenization, stopwords, lemmatization)
- ✅ Feature extraction using **TF–IDF**
- ✅ POS tagging + Named Entity Recognition (NER)
- ✅ Topic Modeling with **LDA** + **BERTopic**
- ✅ Sentiment analysis using **VADER** (with rating alignment checks)
- ✅ **PCA** for dimensionality reduction + **Hierarchical clustering (Ward linkage)**
- ✅ Advanced layer:
  - KeyBERT topic naming  
  - Topic-wise summarization (transformers)
  - Emotion detection
  - Aspect-Based Sentiment Analysis (ABSA)
  - Toxicity detection
  - Semantic similarity search using Sentence-BERT  
  - Interactive visualizations (Plotly)

---

## Dataset
- **Source:** Amazon Unlocked Mobile Phone Reviews (Kaggle)
- **Original size:** 400,000+ reviews
- **Subset used:** 2,000 reviews sampled for computational feasibility
- **Final corpus after preprocessing:** **1,765 reviews**
- **Fields used:** `review_text`, `rating` (metadata minimized to reduce bias) :contentReference[oaicite:1]{index=1}

> Dataset file in this repo: `Amazon_Unlocked_Mobile.csv`

---

## Research Questions (What this project answers)
1. What linguistic and thematic structures dominate smartphone reviews?
2. Which product aspects (battery, camera, screen, price, delivery) drive sentiment?
3. How well can transformer models summarize complex user feedback?
4. How do topic, sentiment, emotion, and aspect-level signals interact to describe user experience? :contentReference[oaicite:2]{index=2}

---

## Methods Used (Pipeline Overview)
### 1) Preprocessing
- lowercasing, punctuation/digit removal
- tokenization
- stopword removal (NLTK)
- lemmatization (semantic preservation prioritized over stemming) :contentReference[oaicite:3]{index=3}

### 2) Feature Extraction
- TF–IDF (unigrams, min_df/max_df thresholds)
- Reported matrix shape: **(1765, 1590)** :contentReference[oaicite:4]{index=4}

### 3) Linguistic Analysis
- POS distribution analysis
- NER for product-related entities (brands, models, accessories) :contentReference[oaicite:5]{index=5}

### 4) Topic Modeling
- **LDA** with coherence-based topic count selection
- **BERTopic** using transformer embeddings + clustering
- Auto topic naming using KeyBERT :contentReference[oaicite:6]{index=6}

### 5) Sentiment Analysis
- **VADER** compound scoring
  - ≥ 0.05 positive, ≤ -0.05 negative, else neutral
- Compared with star ratings for consistency validation :contentReference[oaicite:7]{index=7}

### 6) Dimensionality Reduction + Clustering
- PCA variance analysis + 2D projection
- Hierarchical clustering (Ward linkage)
- Cluster vs sentiment relationship inspection :contentReference[oaicite:8]{index=8}

### 7) Advanced Analyses
- Topic summaries (transformer summarization)
- Emotion classification + confidence analysis
- ABSA for key phone features (battery/camera/performance etc.)
- Toxicity detection
- SBERT semantic similarity search
- Interactive Plotly visualizations :contentReference[oaicite:9]{index=9}

---

## Repository Structure (recommended)
```text
.
├── main.ipynb                         # Full analysis notebook (run this)
├── main.html                          # Exported notebook (results view)
├── Analysis of Amazon Mobile Phone Reviews.pdf   # Final report
├── Amazon_Unlocked_Mobile.csv         # Dataset (or sample)
├── outputs/                           # (optional) saved plots/tables
├── figures/                           # (optional) exported figures for report
├── requirements.txt                   # Python dependencies
└── README.md

```
## How to Run

### Option A — Jupyter Notebook

1. Install Python (3.10+ recommended)
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Open and run:

   ```bash
   jupyter notebook main.ipynb
   ```

### Option B — View Results Without Running

* Open `main.html` in a browser to view outputs directly.
* Read the full report PDF for the written analysis. 

---

## Key Outputs You’ll See

* Preprocessing before/after examples
* TF–IDF top terms and feature-space stats
* POS distribution + top entities (NER)
* LDA coherence curve + topic tables
* BERTopic topic summary + KeyBERT labels
* Sentiment distribution + rating vs sentiment alignment
* PCA variance curve + 2D projections
* Dendrogram + cluster-sentiment proportions
* Topic summaries (positive/negative)
* Topic strength vs weakness matrix
* Emotion distribution + confidence scores
* ABSA aspect polarity tables + heatmaps 

---

## Tech Stack

* **Python**, Jupyter
* NLP: NLTK, spaCy-style pipelines
* ML: scikit-learn, gensim
* Topic modeling: BERTopic, KeyBERT
* Transformers: Hugging Face Transformers, SentenceTransformers
* Visualization: matplotlib, seaborn, plotly 

---

## Results Summary (High-level)

* Review content is strongly **feature-centric** (battery, camera, screen, price/value, performance, delivery/service).
* Sentiment distribution is **positively skewed**, with sentiment aligning well with star ratings.
* **BERTopic** produced more semantically coherent topics than LDA, but at higher computational cost.
* Clustering tends to align more strongly with **sentiment tone** than topic identity. 

---
