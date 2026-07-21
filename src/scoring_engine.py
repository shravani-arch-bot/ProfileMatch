"""
scoring_engine.py
=================
Module 4: Profile Scoring Engine
Synthesizes NLP text similarity, MBTI compatibility, and location
into a single weighted Compatibility Score (0–100%).

Core Formula:
    TotalScore = (w1 × TextSim×100) + (w2 × MBTIMatch) + (w3 × LocationScore)

Usage:
    engine = ScoringEngine()
    engine.load("data/users.csv")
    score = engine.get_compatibility_score("U001", "U010")
    top5  = engine.get_top_matches("U001", n=5)
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure src/ is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nlp_engine import NLPEngine
from mbti_engine import get_mbti_score
from utils import (
    load_users, get_location_score, get_user_weights,
    normalize_weights, format_match_result
)


class ScoringEngine:
    """
    Central matching engine that computes compatibility scores.

    Attributes
    ----------
    users_df   : pd.DataFrame  — loaded user profiles
    nlp_engine : NLPEngine     — fitted TF-IDF engine
    _loaded    : bool          — whether the engine is ready
    """

    def __init__(self):
        self.users_df: pd.DataFrame = pd.DataFrame()
        self.nlp_engine = NLPEngine()
        self._loaded = False
        self._uid_map: dict = {}   # user_id → row dict

    # ── Initialization ────────────────────────────────────────────────────────
    def load(self, users_path: str = "data/users.csv", force_refit: bool = False) -> None:
        """
        Load user profiles and initialise the NLP engine.

        Parameters
        ----------
        users_path  : str   Path to users.csv
        force_refit : bool  If True, refit TF-IDF even if cache exists
        """
        self.users_df = load_users(users_path)
        self._uid_map = {
            row["user_id"]: row.to_dict()
            for _, row in self.users_df.iterrows()
        }

        # Try loading cached NLP, else fit fresh
        if not force_refit and self.nlp_engine.load_cache():
            print("  [NLP] Loaded from cache.")
        else:
            print("  [NLP] Fitting TF-IDF model...")
            self.nlp_engine.fit(self.users_df)

        self._loaded = True
        print(f"  [Scoring] Engine ready — {len(self.users_df)} users loaded.")

    def _require_loaded(self):
        if not self._loaded:
            raise RuntimeError("ScoringEngine not loaded. Call engine.load() first.")

    # ── Score Components ──────────────────────────────────────────────────────
    def _text_score(self, uid_a: str, uid_b: str) -> float:
        """Text similarity [0,100]."""
        raw = self.nlp_engine.get_text_similarity(uid_a, uid_b)  # 0–1
        return round(raw * 100, 2)

    def _mbti_score(self, uid_a: str, uid_b: str) -> float:
        """MBTI compatibility [0,100]."""
        ta = self._uid_map.get(uid_a, {}).get("mbti_type", "")
        tb = self._uid_map.get(uid_b, {}).get("mbti_type", "")
        return get_mbti_score(ta, tb)

    def _location_score(self, uid_a: str, uid_b: str) -> float:
        """Location score [0,100]."""
        ca = self._uid_map.get(uid_a, {}).get("location", "")
        cb = self._uid_map.get(uid_b, {}).get("location", "")
        return get_location_score(ca, cb)

    # ── Main Public API ───────────────────────────────────────────────────────
    def get_compatibility_score(
        self,
        user_a_id: str,
        user_b_id: str,
        weights: dict = None,
        return_breakdown: bool = False,
    ):
        """
        Compute compatibility score between two users.

        Parameters
        ----------
        user_a_id        : str   Source user
        user_b_id        : str   Candidate user
        weights          : dict  {'w1': ..., 'w2': ..., 'w3': ...}
                                  Defaults to per-user learned weights (or global default).
        return_breakdown : bool  If True, return dict with component scores

        Returns
        -------
        float or dict
            If return_breakdown=False → float score in [0, 100]
            If return_breakdown=True  → dict with keys:
                total_score, text_score, mbti_score, location_score,
                w1, w2, w3
        """
        self._require_loaded()

        if user_a_id not in self._uid_map:
            raise ValueError(f"User '{user_a_id}' not found.")
        if user_b_id not in self._uid_map:
            raise ValueError(f"User '{user_b_id}' not found.")

        if weights is None:
            weights = get_user_weights(user_a_id)
        weights = normalize_weights(weights)

        w1, w2, w3 = weights["w1"], weights["w2"], weights["w3"]
        text_s = self._text_score(user_a_id, user_b_id)
        mbti_s = self._mbti_score(user_a_id, user_b_id)
        loc_s  = self._location_score(user_a_id, user_b_id)

        total = round((w1 * text_s) + (w2 * mbti_s) + (w3 * loc_s), 2)
        total = max(0.0, min(100.0, total))

        if return_breakdown:
            return {
                "total_score":     total,
                "text_score":      text_s,
                "mbti_score":      mbti_s,
                "location_score":  loc_s,
                "w1": w1, "w2": w2, "w3": w3,
            }
        return total

    def get_top_matches(
        self,
        user_id: str,
        n: int = 5,
        weights: dict = None,
        exclude_ids: list = None,
    ) -> list[dict]:
        """
        Return top-n compatible users ranked by total score.

        Parameters
        ----------
        user_id     : str   Target user
        n           : int   Number of matches to return
        weights     : dict  Optional custom weights
        exclude_ids : list  User IDs to exclude (e.g. already shown)

        Returns
        -------
        list of dicts, sorted by total_score descending:
            [{'user_id', 'name', 'industry', 'mbti_type', 'location',
              'total_score', 'text_score', 'mbti_score', 'location_score',
              'w1', 'w2', 'w3'}, ...]
        """
        self._require_loaded()

        if user_id not in self._uid_map:
            raise ValueError(f"User '{user_id}' not found.")

        exclude = set(exclude_ids or [])
        exclude.add(user_id)   # never match with self

        results = []
        for other_id, profile in self._uid_map.items():
            if other_id in exclude:
                continue
            breakdown = self.get_compatibility_score(
                user_id, other_id, weights=weights, return_breakdown=True
            )
            results.append({
                "user_id":        other_id,
                "name":           profile.get("name", ""),
                "age":            profile.get("age", ""),
                "industry":       profile.get("industry", ""),
                "job_title":      profile.get("job_title", ""),
                "mbti_type":      profile.get("mbti_type", ""),
                "location":       profile.get("location", ""),
                "skills":         profile.get("skills", ""),
                "professional_summary": profile.get("professional_summary", ""),
                "about_me":       profile.get("about_me", ""),
                **breakdown,
            })

        results.sort(key=lambda x: x["total_score"], reverse=True)
        return results[:n]

    def get_score_matrix(self) -> pd.DataFrame:
        """Return full NxN compatibility score matrix (using default weights)."""
        self._require_loaded()
        user_ids = self.users_df["user_id"].tolist()
        n = len(user_ids)
        matrix = np.zeros((n, n))
        for i, uid_a in enumerate(user_ids):
            for j, uid_b in enumerate(user_ids):
                if i == j:
                    matrix[i, j] = 100.0
                else:
                    matrix[i, j] = self.get_compatibility_score(uid_a, uid_b)
        return pd.DataFrame(matrix, index=user_ids, columns=user_ids)


# ── Singleton (shared across modules) ────────────────────────────────────────
_engine_instance: ScoringEngine = None


def get_engine(users_path: str = "data/users.csv") -> ScoringEngine:
    """Return a loaded singleton ScoringEngine (loads on first call)."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ScoringEngine()
        _engine_instance.load(users_path)
    return _engine_instance


# -- CLI demo ------------------------------------------------------------------
if __name__ == "__main__":
    engine = get_engine()

    print("\n-- Compatibility Score: U001 vs U010 --")
    result = engine.get_compatibility_score("U001", "U010", return_breakdown=True)
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n-- Top 5 Matches for U001 --")
    matches = engine.get_top_matches("U001", n=5)
    for i, m in enumerate(matches, 1):
        print(format_match_result(i, m))
