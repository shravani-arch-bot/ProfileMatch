"""
utils.py
========
Shared helper utilities for the matching system.
"""

import os
import json
import pandas as pd
import numpy as np


# ── Data Loaders ──────────────────────────────────────────────────────────────
def load_users(path: str = "data/users.csv") -> pd.DataFrame:
    """Load and validate users.csv."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"users.csv not found at '{path}'. "
            "Run: python src/data_pipeline.py"
        )
    df = pd.read_csv(path)
    required = ["user_id", "name", "industry", "mbti_type", "location",
                "professional_summary", "about_me", "skills"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"users.csv missing columns: {missing}")
    return df


def load_feedback(path: str = "data/feedback.csv") -> pd.DataFrame:
    """Load and validate feedback.csv."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"feedback.csv not found at '{path}'. "
            "Run: python src/data_pipeline.py"
        )
    df = pd.read_csv(path)
    required = ["user_id", "matched_user_id", "action"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"feedback.csv missing columns: {missing}")
    df["action"] = df["action"].astype(int)
    return df


# ── Location Scoring ───────────────────────────────────────────────────────────
# City → state/region mapping for tier-2 location matching
_CITY_REGION = {
    "Mumbai": "West", "Pune": "West", "Ahmedabad": "West", "Surat": "West",
    "Delhi": "North", "Jaipur": "North", "Lucknow": "North", "Kanpur": "North",
    "Bhopal": "North",
    "Bangalore": "South", "Chennai": "South", "Hyderabad": "South",
    "Visakhapatnam": "South",
    "Kolkata": "East", "Bhubaneswar": "East", "Nagpur": "Central",
}


def get_location_score(city_a: str, city_b: str) -> float:
    """
    Score location compatibility:
      Same city    → 100
      Same region  → 60
      Different    → 20

    Returns
    -------
    float in [0, 100]
    """
    if city_a == city_b:
        return 100.0
    region_a = _CITY_REGION.get(city_a, "Unknown")
    region_b = _CITY_REGION.get(city_b, "Unknown")
    if region_a == region_b and region_a != "Unknown":
        return 60.0
    return 20.0


# ── User Weights Store ─────────────────────────────────────────────────────────
_DEFAULT_WEIGHTS = {"w1": 0.50, "w2": 0.30, "w3": 0.20}
_WEIGHTS_PATH = "data/user_weights.json"


def load_all_weights() -> dict:
    """Load all per-user weights from JSON store."""
    if not os.path.exists(_WEIGHTS_PATH):
        return {}
    with open(_WEIGHTS_PATH, "r") as f:
        return json.load(f)


def save_all_weights(weights_dict: dict) -> None:
    """Persist all per-user weights to disk."""
    os.makedirs("data", exist_ok=True)
    with open(_WEIGHTS_PATH, "w") as f:
        json.dump(weights_dict, f, indent=2)


def get_user_weights(user_id: str) -> dict:
    """Get weights for a specific user (fallback to defaults)."""
    all_weights = load_all_weights()
    return all_weights.get(user_id, dict(_DEFAULT_WEIGHTS))


def set_user_weights(user_id: str, weights: dict) -> None:
    """Update weights for a specific user and persist."""
    all_weights = load_all_weights()
    all_weights[user_id] = weights
    save_all_weights(all_weights)


def normalize_weights(w: dict) -> dict:
    """Ensure weights sum to 1.0."""
    total = w["w1"] + w["w2"] + w["w3"]
    if total == 0:
        return dict(_DEFAULT_WEIGHTS)
    return {k: round(v / total, 4) for k, v in w.items()}


# ── Score Formatting ───────────────────────────────────────────────────────────
def score_to_label(score: float) -> str:
    """Convert numeric score to human-readable label."""
    if score >= 80:
        return "Excellent Match"
    elif score >= 65:
        return "Strong Match"
    elif score >= 50:
        return "Good Match"
    elif score >= 35:
        return "Moderate Match"
    else:
        return "Low Match"


def format_match_result(rank: int, match: dict) -> str:
    """Format a match dict for console display."""
    return (
        f"#{rank}  {match['name']} ({match['user_id']}) — "
        f"{score_to_label(match['total_score'])} "
        f"[{match['total_score']:.1f}%] | "
        f"Text: {match['text_score']:.1f} | "
        f"MBTI: {match['mbti_score']:.1f} | "
        f"Location: {match['location_score']:.1f}"
    )


if __name__ == "__main__":
    print("Location score Mumbai ↔ Pune:", get_location_score("Mumbai", "Pune"))
    print("Location score Delhi ↔ Bangalore:", get_location_score("Delhi", "Bangalore"))
    print("Location score Mumbai ↔ Mumbai:", get_location_score("Mumbai", "Mumbai"))
