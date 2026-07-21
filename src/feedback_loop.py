"""
feedback_loop.py
================
Module 5: Adaptive Feedback Loop (ML Layer)
Learns per-user weights using linear regression on historical feedback.

Goal:
    If user consistently rejects high-MBTI matches but accepts high-text
    matches → lower w2, raise w1 automatically.

Algorithm:
    For each user with ≥5 feedback entries:
      1. Build feature matrix X = [text_sim, mbti_score, location_score]
      2. Build target y = [action (0/1)]
      3. Fit LinearRegression: y ≈ w1*text + w2*mbti + w3*location
      4. Normalize learned weights to sum = 1
      5. Persist updated weights

Usage:
    python src/feedback_loop.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scoring_engine import get_engine
from utils import (
    load_feedback, get_user_weights, set_user_weights, normalize_weights
)

_DEFAULT_WEIGHTS = {"w1": 0.50, "w2": 0.30, "w3": 0.20}
MIN_INTERACTIONS = 5   # minimum feedback rows to attempt learning


def build_feature_matrix(user_id: str, feedback_df: pd.DataFrame, engine) -> tuple:
    """
    Build (X, y) for a specific user's feedback history.

    X columns: [text_sim_score, mbti_score, location_score]
    y column : action (0 or 1)

    Returns
    -------
    X : np.ndarray shape (n, 3)
    y : np.ndarray shape (n,)
    """
    user_fb = feedback_df[feedback_df["user_id"] == user_id].copy()
    if len(user_fb) < MIN_INTERACTIONS:
        return None, None

    rows_X, rows_y = [], []
    for _, row in user_fb.iterrows():
        mid = row["matched_user_id"]
        action = int(row["action"])
        try:
            breakdown = engine.get_compatibility_score(
                user_id, mid, return_breakdown=True
            )
            rows_X.append([
                breakdown["text_score"] / 100.0,
                breakdown["mbti_score"] / 100.0,
                breakdown["location_score"] / 100.0,
            ])
            rows_y.append(action)
        except Exception:
            continue

    if len(rows_X) < MIN_INTERACTIONS:
        return None, None

    return np.array(rows_X, dtype=float), np.array(rows_y, dtype=float)


def learn_weights_for_user(user_id: str, feedback_df: pd.DataFrame, engine) -> dict | None:
    """
    Train Ridge regression on one user's feedback and extract learned weights.

    Returns
    -------
    dict {'w1': ..., 'w2': ..., 'w3': ...}  or None if insufficient data
    """
    X, y = build_feature_matrix(user_id, feedback_df, engine)
    if X is None:
        return None

    # Use Ridge regression (L2) to prevent wild weight values
    model = Ridge(alpha=1.0, positive=True, fit_intercept=False)
    model.fit(X, y)

    coefs = model.coef_
    # Clip negatives to 0 (weights can't be negative)
    coefs = np.clip(coefs, 0, None)

    total = coefs.sum()
    if total < 1e-6:
        return None

    coefs = coefs / total   # normalize

    return {
        "w1": round(float(coefs[0]), 4),
        "w2": round(float(coefs[1]), 4),
        "w3": round(float(coefs[2]), 4),
    }


def run_feedback_learning(
    feedback_path: str = "data/feedback.csv",
    users_path: str = "data/users.csv",
    verbose: bool = True,
) -> dict:
    """
    Run the adaptive weight learning loop for all users.
    Updates data/user_weights.json.

    Returns
    -------
    dict: mapping user_id → new weights
    """
    feedback_df = load_feedback(feedback_path)
    engine = get_engine(users_path)

    user_ids = feedback_df["user_id"].unique().tolist()
    updated = {}
    skipped = 0

    for uid in user_ids:
        new_weights = learn_weights_for_user(uid, feedback_df, engine)
        if new_weights is None:
            skipped += 1
            continue
        set_user_weights(uid, new_weights)
        updated[uid] = new_weights
        if verbose:
            old = _DEFAULT_WEIGHTS
            print(
                f"  {uid}: w1(text)={new_weights['w1']:.3f} "
                f"[was {old['w1']}]  "
                f"w2(mbti)={new_weights['w2']:.3f} "
                f"[was {old['w2']}]  "
                f"w3(loc)={new_weights['w3']:.3f} "
                f"[was {old['w3']}]"
            )

    if verbose:
        print(f"\n  ✓ Updated weights for {len(updated)} users. "
              f"Skipped {skipped} (insufficient feedback).")
    return updated


# ── Performance Simulation ─────────────────────────────────────────────────────
def simulate_performance(
    feedback_df: pd.DataFrame,
    engine,
    rounds: int = 5,
) -> pd.DataFrame:
    """
    Simulate how acceptance-rate prediction accuracy improves over feedback rounds.

    For each round r:
      - Train on first r/rounds fraction of feedback
      - Test on remaining fraction
      - Record predicted_accept vs actual_accept

    Returns
    -------
    pd.DataFrame with columns: round, before_accuracy, after_accuracy
    """
    results = []
    all_users = feedback_df["user_id"].unique()

    feedback_df = feedback_df.sort_values("timestamp") if "timestamp" in feedback_df.columns \
        else feedback_df

    for r in range(1, rounds + 1):
        split = int(len(feedback_df) * (r / rounds))
        train_df = feedback_df.iloc[:split]
        test_df = feedback_df.iloc[split:]

        if len(test_df) == 0:
            continue

        # Before: predict using default weights
        correct_before, correct_after = 0, 0
        total = 0

        for _, row in test_df.iterrows():
            uid = row["user_id"]
            mid = row["matched_user_id"]
            actual = int(row["action"])

            try:
                # Before: default weights
                score_before = engine.get_compatibility_score(uid, mid, weights=_DEFAULT_WEIGHTS)
                pred_before = 1 if score_before >= 55 else 0

                # After: learned weights (from train split)
                train_user = train_df[train_df["user_id"] == uid]
                learned = None
                if len(train_user) >= MIN_INTERACTIONS:
                    learned = learn_weights_for_user(uid, train_user, engine)

                weights_after = learned if learned else _DEFAULT_WEIGHTS
                score_after = engine.get_compatibility_score(uid, mid, weights=weights_after)
                pred_after = 1 if score_after >= 55 else 0

                correct_before += int(pred_before == actual)
                correct_after  += int(pred_after == actual)
                total += 1
            except Exception:
                continue

        if total > 0:
            results.append({
                "round":          r,
                "train_size":     split,
                "test_size":      len(test_df),
                "before_accuracy": round(correct_before / total * 100, 2),
                "after_accuracy":  round(correct_after  / total * 100, 2),
            })

    return pd.DataFrame(results)


def compute_acceptance_rates(
    feedback_df: pd.DataFrame,
    engine,
    use_learned_weights: bool = False,
) -> dict:
    """
    Compute overall acceptance rate and predicted acceptance rate.

    Returns
    -------
    dict: {actual_acceptance_rate, predicted_before, predicted_after}
    """
    actual = feedback_df["action"].mean()

    scores_before, scores_after = [], []
    for _, row in feedback_df.iterrows():
        uid = row["user_id"]
        mid = row["matched_user_id"]
        try:
            s_before = engine.get_compatibility_score(uid, mid, weights=_DEFAULT_WEIGHTS)
            scores_before.append(1 if s_before >= 55 else 0)

            from utils import get_user_weights
            learned = get_user_weights(uid)
            s_after = engine.get_compatibility_score(uid, mid, weights=learned)
            scores_after.append(1 if s_after >= 55 else 0)
        except Exception:
            continue

    return {
        "actual_acceptance_rate": round(actual * 100, 2),
        "predicted_accept_before": round(np.mean(scores_before) * 100, 2) if scores_before else 0,
        "predicted_accept_after":  round(np.mean(scores_after) * 100, 2) if scores_after else 0,
    }


if __name__ == "__main__":
    print("Running Adaptive Feedback Learning...\n")
    updated = run_feedback_learning()
    print(f"\nTotal users updated: {len(updated)}")

    print("\nSimulating performance over 5 rounds...")
    from utils import load_feedback
    fb = load_feedback()
    engine = get_engine()
    perf = simulate_performance(fb, engine, rounds=5)
    print(perf.to_string(index=False))
