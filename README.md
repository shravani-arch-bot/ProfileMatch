# Profile-Based Matching Algorithm

> **Intelligent Hybrid Recommendation System** — NLP + MBTI + Adaptive ML

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the dataset
```bash
python src/data_pipeline.py
```
This creates `data/users.csv` (75 profiles) and `data/feedback.csv` (500+ rows).

### 3. Launch the UI
```bash
streamlit run app.py
```

### 4. (Optional) Generate performance analysis
```bash
python analysis/performance_report.py
```
Outputs 6 plots + text report to `analysis/outputs/`.

---

## 📁 Project Structure

```
profile-matcher/
├── data/
│   ├── users.csv               # 75 synthetic user profiles
│   ├── feedback.csv            # 500+ user interactions
│   ├── nlp_cache.pkl           # Precomputed TF-IDF matrix (auto-generated)
│   └── user_weights.json       # Per-user learned weights (auto-generated)
├── src/
│   ├── data_pipeline.py        # Dataset generation
│   ├── nlp_engine.py           # TF-IDF + cosine similarity
│   ├── mbti_engine.py          # 16×16 MBTI compatibility matrix
│   ├── scoring_engine.py       # Weighted hybrid scoring
│   ├── feedback_loop.py        # Adaptive ML weight learning
│   └── utils.py                # Shared helpers
├── analysis/
│   ├── performance_report.py   # Visualization & accuracy analysis
│   └── outputs/                # Generated charts & report
├── app.py                      # Streamlit UI
├── requirements.txt
└── README.md
```

---

## 🔬 Technical Modules

### Module 1 — Data Pipeline (`src/data_pipeline.py`)
Generates realistic synthetic user profiles using industry-specific templates:
- **75 users** across 10 industries (Technology, Healthcare, Finance, etc.)
- Role-correlated MBTI types (analytical → INTJ/INTP, creative → INFP/ENFP)
- Multi-sentence professional summaries and about-me bios
- **500+ feedback interactions** with realistic acceptance probabilities

### Module 2 — NLP Engine (`src/nlp_engine.py`)
Semantic text analysis pipeline:
1. **Preprocessing**: lowercase, remove stopwords, lemmatize (NLTK)
2. **Vectorization**: TF-IDF with bigrams (3,000 features, sublinear TF scaling)
3. **Similarity**: Precomputed N×N cosine similarity matrix
4. **Caching**: Saved to disk (`data/nlp_cache.pkl`) for fast reloads

### Module 3 — MBTI Engine (`src/mbti_engine.py`)
Full 16×16 compatibility matrix:
- Scores based on cognitive function overlap and complementary traits
- Range: [0, 100] (e.g., INTJ↔ENFP = 95, ESFJ↔INTJ = 35)
- Symmetric lookup with fallback

### Module 4 — Scoring Engine (`src/scoring_engine.py`)
**Core Formula:**

$$\text{TotalScore} = (w_1 \times \text{TextSim} \times 100) + (w_2 \times \text{MBTIMatch}) + (w_3 \times \text{LocationScore})$$

| Component | Method | Default Weight |
|---|---|---|
| Text Similarity | Cosine similarity of TF-IDF vectors | w₁ = 0.50 |
| MBTI Match | 16×16 compatibility matrix | w₂ = 0.30 |
| Location | Same city=100, region=60, other=20 | w₃ = 0.20 |

### Module 5 — Feedback Loop (`src/feedback_loop.py`)
Adaptive weight learning using Ridge Regression:
- Builds feature matrix `X = [text_sim, mbti_score, location_score]`
- Target `y = [action (0/1)]` from user feedback
- Normalizes learned coefficients to sum=1
- Persists per-user weights to `data/user_weights.json`
- Minimum 5 interactions required per user

### Module 6 — Performance Analysis (`analysis/performance_report.py`)
Generates 6 visualization charts:
1. **Accuracy improvement** over training rounds
2. **Weight evolution** per user (before vs after learning)
3. **Score distribution** for accepted vs rejected matches
4. **MBTI heatmap** — full 16×16 compatibility matrix
5. **Industry distribution** of user dataset
6. **Match breakdown** — per-component scores for top 5 matches

### Module 7 — Streamlit UI (`app.py`)
Premium dark-themed web interface:
- Profile selector in sidebar
- Top-N match cards with donut charts per component
- Live Accept/Reject feedback buttons
- One-click weight update from feedback
- Algorithm explanation panel

---

## 📊 Dataset Description

> "This dataset was synthetically generated to simulate real-world user profiles for an intelligent hybrid recommendation system."

### `users.csv` — 75 User Profiles
| Column | Description |
|---|---|
| user_id | Unique ID (U001–U075) |
| name | Full name |
| age | Age (22–45) |
| gender | Male/Female/Non-binary |
| location | Indian city |
| industry | One of 10 industries |
| job_title | Role-specific title |
| mbti_type | Valid 16-type MBTI |
| skills | Comma-separated skills |
| professional_summary | 2–4 sentence career bio |
| about_me | 2–4 sentence personal bio |

### `feedback.csv` — 500+ Interactions
| Column | Description |
|---|---|
| user_id | Viewer's user ID |
| matched_user_id | Shown candidate's ID |
| action | 1=Accept, 0=Reject |
| timestamp | Date (YYYY-MM-DD) |

---

## 🛠️ Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
nltk>=3.8.0
matplotlib>=3.6.0
seaborn>=0.12.0
streamlit>=1.28.0
```

---

## 📈 Expected Performance

| Metric | Value |
|---|---|
| Dataset size | 75 users, 500+ feedback rows |
| Initial acceptance rate | ~55% (random weights) |
| Post-learning acceptance rate | ~68%+ (adaptive weights) |
| NLP vocabulary | ~3,000 TF-IDF features |
| MBTI pairs | 256 unique combinations |
