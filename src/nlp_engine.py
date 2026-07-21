"""
nlp_engine.py
=============
Module 2: NLP & Semantic Analysis
- Text preprocessing: lowercase, remove punctuation, stopwords, lemmatize
- TF-IDF vectorization of user bios
- Precomputed cosine similarity matrix for all user pairs

Dependencies: nltk, scikit-learn, pandas
"""

import re
import os
import pickle
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── NLTK setup (download once) ────────────────────────────────────────────────
def _ensure_nltk_data():
    for pkg in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
        try:
            if pkg == "stopwords":
                nltk.data.find("corpora/stopwords")
            elif pkg == "wordnet":
                nltk.data.find("corpora/wordnet")
            elif pkg == "punkt":
                nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download(pkg, quiet=True)

_ensure_nltk_data()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


# ── Text Preprocessing ────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    Clean and normalize raw text:
    1. Lowercase
    2. Remove URLs, punctuation, digits
    3. Tokenize
    4. Remove stopwords
    5. Lemmatize

    Parameters
    ----------
    text : str  Raw bio or summary text

    Returns
    -------
    str : cleaned, space-joined tokens
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove punctuation and digits
    text = re.sub(r"[^a-z\s]", " ", text)
    # Tokenize (simple whitespace split is sufficient post-cleanup)
    tokens = text.split()
    # Remove stopwords + short tokens
    tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]
    # Lemmatize
    tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def build_combined_text(row: pd.Series) -> str:
    """
    Combine professional_summary + about_me + skills into a single text blob.
    Gives professional_summary 2x weight by repeating it.
    """
    prof = str(row.get("professional_summary", ""))
    about = str(row.get("about_me", ""))
    skills = str(row.get("skills", ""))
    # Double-weight professional summary
    return f"{prof} {prof} {about} {skills}"


# ── TF-IDF & Similarity ───────────────────────────────────────────────────────
class NLPEngine:
    """
    Encapsulates TF-IDF vectorization and cosine similarity computation.

    Usage
    -----
    engine = NLPEngine()
    engine.fit(users_df)
    score = engine.get_text_similarity("U001", "U023")  # returns float 0-1
    """

    def __init__(self, cache_path: str = "data/nlp_cache.pkl"):
        self.cache_path = cache_path
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),   # unigrams + bigrams
            sublinear_tf=True,    # log-scale TF
        )
        self.tfidf_matrix = None
        self.user_ids: list = []
        self.id_to_idx: dict = {}
        self.sim_matrix: np.ndarray = None
        self._fitted = False

    def fit(self, users_df: pd.DataFrame) -> None:
        """
        Fit TF-IDF on all user texts and precompute pairwise cosine similarity.

        Parameters
        ----------
        users_df : pd.DataFrame  Must have columns: user_id, professional_summary, about_me, skills
        """
        self.user_ids = users_df["user_id"].tolist()
        self.id_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}

        # Preprocess
        combined_texts = users_df.apply(build_combined_text, axis=1)
        cleaned_texts = combined_texts.apply(preprocess_text).tolist()

        # Fit TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(cleaned_texts)

        # Precompute cosine similarity (NxN dense matrix)
        self.sim_matrix = cosine_similarity(self.tfidf_matrix).astype(np.float32)

        self._fitted = True
        self._save_cache()
        print(f"  [NLP] TF-IDF fitted on {len(self.user_ids)} users. "
              f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path) if os.path.dirname(self.cache_path) else ".", exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump({
                "user_ids": self.user_ids,
                "id_to_idx": self.id_to_idx,
                "sim_matrix": self.sim_matrix,
                "vectorizer": self.vectorizer,
            }, f)

    def load_cache(self) -> bool:
        """Load precomputed matrix from disk. Returns True if successful."""
        if not os.path.exists(self.cache_path):
            return False
        with open(self.cache_path, "rb") as f:
            data = pickle.load(f)
        self.user_ids = data["user_ids"]
        self.id_to_idx = data["id_to_idx"]
        self.sim_matrix = data["sim_matrix"]
        self.vectorizer = data["vectorizer"]
        self._fitted = True
        return True

    def get_text_similarity(self, user_a_id: str, user_b_id: str) -> float:
        """
        Return cosine similarity [0, 1] between two users' text profiles.

        Parameters
        ----------
        user_a_id : str  e.g. 'U001'
        user_b_id : str  e.g. 'U023'

        Returns
        -------
        float : similarity in [0, 1]
        """
        if not self._fitted:
            raise RuntimeError("NLPEngine not fitted. Call fit() or load_cache() first.")
        if user_a_id not in self.id_to_idx or user_b_id not in self.id_to_idx:
            return 0.0
        i = self.id_to_idx[user_a_id]
        j = self.id_to_idx[user_b_id]
        return float(self.sim_matrix[i, j])

    def get_top_text_matches(self, user_id: str, n: int = 10) -> pd.DataFrame:
        """Return top-n most text-similar users (excluding self)."""
        if not self._fitted:
            raise RuntimeError("NLPEngine not fitted.")
        if user_id not in self.id_to_idx:
            return pd.DataFrame()
        i = self.id_to_idx[user_id]
        scores = self.sim_matrix[i].copy()
        scores[i] = -1  # exclude self
        top_indices = np.argsort(scores)[::-1][:n]
        return pd.DataFrame({
            "user_id": [self.user_ids[j] for j in top_indices],
            "text_similarity": [float(scores[j]) for j in top_indices],
        })

    def get_similarity_matrix_df(self) -> pd.DataFrame:
        """Return full similarity matrix as a labeled DataFrame."""
        return pd.DataFrame(self.sim_matrix, index=self.user_ids, columns=self.user_ids)


# ── Standalone usage ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    users_df = pd.read_csv("data/users.csv")
    engine = NLPEngine()
    engine.fit(users_df)

    print("\nTop 5 text matches for U001:")
    print(engine.get_top_text_matches("U001", n=5))

    print("\nText similarity U001 ↔ U002:", engine.get_text_similarity("U001", "U002"))
    print("Text similarity U001 ↔ U001:", engine.get_text_similarity("U001", "U001"))
