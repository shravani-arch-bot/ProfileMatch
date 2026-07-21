"""
mbti_engine.py
==============
Module 3: MBTI Compatibility Matrix
Defines a 16×16 psychological compatibility scoring matrix.

Based on:
 - Cognitive function theory (shared functions = better compatibility)
 - Complementary traits (I/E balance, N/S balance)
 - Popular MBTI relationship research

Scores are in the range [0, 100].
"""

# ── Full 16×16 MBTI Compatibility Matrix ──────────────────────────────────────
# Scores based on cognitive function overlap, complementary strengths,
# and empirical MBTI relationship research.

MBTI_COMPATIBILITY_MATRIX = {
    ("INTJ", "INTJ"): 75, ("INTJ", "INTP"): 85, ("INTJ", "ENTJ"): 80, ("INTJ", "ENTP"): 90,
    ("INTJ", "INFJ"): 70, ("INTJ", "INFP"): 75, ("INTJ", "ENFJ"): 65, ("INTJ", "ENFP"): 95,
    ("INTJ", "ISTJ"): 60, ("INTJ", "ISFJ"): 40, ("INTJ", "ESTJ"): 55, ("INTJ", "ESFJ"): 35,
    ("INTJ", "ISTP"): 65, ("INTJ", "ISFP"): 45, ("INTJ", "ESTP"): 50, ("INTJ", "ESFP"): 30,

    ("INTP", "INTJ"): 85, ("INTP", "INTP"): 70, ("INTP", "ENTJ"): 75, ("INTP", "ENTP"): 85,
    ("INTP", "INFJ"): 90, ("INTP", "INFP"): 70, ("INTP", "ENFJ"): 85, ("INTP", "ENFP"): 80,
    ("INTP", "ISTJ"): 55, ("INTP", "ISFJ"): 45, ("INTP", "ESTJ"): 50, ("INTP", "ESFJ"): 40,
    ("INTP", "ISTP"): 70, ("INTP", "ISFP"): 50, ("INTP", "ESTP"): 55, ("INTP", "ESFP"): 35,

    ("ENTJ", "INTJ"): 80, ("ENTJ", "INTP"): 75, ("ENTJ", "ENTJ"): 65, ("ENTJ", "ENTP"): 80,
    ("ENTJ", "INFJ"): 75, ("ENTJ", "INFP"): 70, ("ENTJ", "ENFJ"): 70, ("ENTJ", "ENFP"): 80,
    ("ENTJ", "ISTJ"): 70, ("ENTJ", "ISFJ"): 55, ("ENTJ", "ESTJ"): 75, ("ENTJ", "ESFJ"): 60,
    ("ENTJ", "ISTP"): 65, ("ENTJ", "ISFP"): 45, ("ENTJ", "ESTP"): 70, ("ENTJ", "ESFP"): 50,

    ("ENTP", "INTJ"): 90, ("ENTP", "INTP"): 85, ("ENTP", "ENTJ"): 80, ("ENTP", "ENTP"): 70,
    ("ENTP", "INFJ"): 95, ("ENTP", "INFP"): 75, ("ENTP", "ENFJ"): 80, ("ENTP", "ENFP"): 75,
    ("ENTP", "ISTJ"): 50, ("ENTP", "ISFJ"): 40, ("ENTP", "ESTJ"): 55, ("ENTP", "ESFJ"): 45,
    ("ENTP", "ISTP"): 65, ("ENTP", "ISFP"): 50, ("ENTP", "ESTP"): 60, ("ENTP", "ESFP"): 40,

    ("INFJ", "INTJ"): 70, ("INFJ", "INTP"): 90, ("INFJ", "ENTJ"): 75, ("INFJ", "ENTP"): 95,
    ("INFJ", "INFJ"): 65, ("INFJ", "INFP"): 75, ("INFJ", "ENFJ"): 70, ("INFJ", "ENFP"): 90,
    ("INFJ", "ISTJ"): 50, ("INFJ", "ISFJ"): 55, ("INFJ", "ESTJ"): 45, ("INFJ", "ESFJ"): 55,
    ("INFJ", "ISTP"): 45, ("INFJ", "ISFP"): 60, ("INFJ", "ESTP"): 40, ("INFJ", "ESFP"): 50,

    ("INFP", "INTJ"): 75, ("INFP", "INTP"): 70, ("INFP", "ENTJ"): 70, ("INFP", "ENTP"): 75,
    ("INFP", "INFJ"): 75, ("INFP", "INFP"): 65, ("INFP", "ENFJ"): 95, ("INFP", "ENFP"): 80,
    ("INFP", "ISTJ"): 40, ("INFP", "ISFJ"): 50, ("INFP", "ESTJ"): 35, ("INFP", "ESFJ"): 50,
    ("INFP", "ISTP"): 45, ("INFP", "ISFP"): 65, ("INFP", "ESTP"): 40, ("INFP", "ESFP"): 60,

    ("ENFJ", "INTJ"): 65, ("ENFJ", "INTP"): 85, ("ENFJ", "ENTJ"): 70, ("ENFJ", "ENTP"): 80,
    ("ENFJ", "INFJ"): 70, ("ENFJ", "INFP"): 95, ("ENFJ", "ENFJ"): 65, ("ENFJ", "ENFP"): 80,
    ("ENFJ", "ISTJ"): 55, ("ENFJ", "ISFJ"): 65, ("ENFJ", "ESTJ"): 60, ("ENFJ", "ESFJ"): 70,
    ("ENFJ", "ISTP"): 40, ("ENFJ", "ISFP"): 65, ("ENFJ", "ESTP"): 50, ("ENFJ", "ESFP"): 65,

    ("ENFP", "INTJ"): 95, ("ENFP", "INTP"): 80, ("ENFP", "ENTJ"): 80, ("ENFP", "ENTP"): 75,
    ("ENFP", "INFJ"): 90, ("ENFP", "INFP"): 80, ("ENFP", "ENFJ"): 80, ("ENFP", "ENFP"): 65,
    ("ENFP", "ISTJ"): 45, ("ENFP", "ISFJ"): 55, ("ENFP", "ESTJ"): 50, ("ENFP", "ESFJ"): 60,
    ("ENFP", "ISTP"): 50, ("ENFP", "ISFP"): 70, ("ENFP", "ESTP"): 55, ("ENFP", "ESFP"): 65,

    ("ISTJ", "INTJ"): 60, ("ISTJ", "INTP"): 55, ("ISTJ", "ENTJ"): 70, ("ISTJ", "ENTP"): 50,
    ("ISTJ", "INFJ"): 50, ("ISTJ", "INFP"): 40, ("ISTJ", "ENFJ"): 55, ("ISTJ", "ENFP"): 45,
    ("ISTJ", "ISTJ"): 75, ("ISTJ", "ISFJ"): 80, ("ISTJ", "ESTJ"): 85, ("ISTJ", "ESFJ"): 75,
    ("ISTJ", "ISTP"): 70, ("ISTJ", "ISFP"): 55, ("ISTJ", "ESTP"): 65, ("ISTJ", "ESFP"): 50,

    ("ISFJ", "INTJ"): 40, ("ISFJ", "INTP"): 45, ("ISFJ", "ENTJ"): 55, ("ISFJ", "ENTP"): 40,
    ("ISFJ", "INFJ"): 55, ("ISFJ", "INFP"): 50, ("ISFJ", "ENFJ"): 65, ("ISFJ", "ENFP"): 55,
    ("ISFJ", "ISTJ"): 80, ("ISFJ", "ISFJ"): 75, ("ISFJ", "ESTJ"): 75, ("ISFJ", "ESFJ"): 85,
    ("ISFJ", "ISTP"): 55, ("ISFJ", "ISFP"): 70, ("ISFJ", "ESTP"): 50, ("ISFJ", "ESFP"): 70,

    ("ESTJ", "INTJ"): 55, ("ESTJ", "INTP"): 50, ("ESTJ", "ENTJ"): 75, ("ESTJ", "ENTP"): 55,
    ("ESTJ", "INFJ"): 45, ("ESTJ", "INFP"): 35, ("ESTJ", "ENFJ"): 60, ("ESTJ", "ENFP"): 50,
    ("ESTJ", "ISTJ"): 85, ("ESTJ", "ISFJ"): 75, ("ESTJ", "ESTJ"): 70, ("ESTJ", "ESFJ"): 80,
    ("ESTJ", "ISTP"): 65, ("ESTJ", "ISFP"): 50, ("ESTJ", "ESTP"): 75, ("ESTJ", "ESFP"): 60,

    ("ESFJ", "INTJ"): 35, ("ESFJ", "INTP"): 40, ("ESFJ", "ENTJ"): 60, ("ESFJ", "ENTP"): 45,
    ("ESFJ", "INFJ"): 55, ("ESFJ", "INFP"): 50, ("ESFJ", "ENFJ"): 70, ("ESFJ", "ENFP"): 60,
    ("ESFJ", "ISTJ"): 75, ("ESFJ", "ISFJ"): 85, ("ESFJ", "ESTJ"): 80, ("ESFJ", "ESFJ"): 70,
    ("ESFJ", "ISTP"): 50, ("ESFJ", "ISFP"): 65, ("ESFJ", "ESTP"): 60, ("ESFJ", "ESFP"): 75,

    ("ISTP", "INTJ"): 65, ("ISTP", "INTP"): 70, ("ISTP", "ENTJ"): 65, ("ISTP", "ENTP"): 65,
    ("ISTP", "INFJ"): 45, ("ISTP", "INFP"): 45, ("ISTP", "ENFJ"): 40, ("ISTP", "ENFP"): 50,
    ("ISTP", "ISTJ"): 70, ("ISTP", "ISFJ"): 55, ("ISTP", "ESTJ"): 65, ("ISTP", "ESFJ"): 50,
    ("ISTP", "ISTP"): 70, ("ISTP", "ISFP"): 65, ("ISTP", "ESTP"): 80, ("ISTP", "ESFP"): 65,

    ("ISFP", "INTJ"): 45, ("ISFP", "INTP"): 50, ("ISFP", "ENTJ"): 45, ("ISFP", "ENTP"): 50,
    ("ISFP", "INFJ"): 60, ("ISFP", "INFP"): 65, ("ISFP", "ENFJ"): 65, ("ISFP", "ENFP"): 70,
    ("ISFP", "ISTJ"): 55, ("ISFP", "ISFJ"): 70, ("ISFP", "ESTJ"): 50, ("ISFP", "ESFJ"): 65,
    ("ISFP", "ISTP"): 65, ("ISFP", "ISFP"): 70, ("ISFP", "ESTP"): 75, ("ISFP", "ESFP"): 80,

    ("ESTP", "INTJ"): 50, ("ESTP", "INTP"): 55, ("ESTP", "ENTJ"): 70, ("ESTP", "ENTP"): 60,
    ("ESTP", "INFJ"): 40, ("ESTP", "INFP"): 40, ("ESTP", "ENFJ"): 50, ("ESTP", "ENFP"): 55,
    ("ESTP", "ISTJ"): 65, ("ESTP", "ISFJ"): 50, ("ESTP", "ESTJ"): 75, ("ESTP", "ESFJ"): 60,
    ("ESTP", "ISTP"): 80, ("ESTP", "ISFP"): 75, ("ESTP", "ESTP"): 70, ("ESTP", "ESFP"): 80,

    ("ESFP", "INTJ"): 30, ("ESFP", "INTP"): 35, ("ESFP", "ENTJ"): 50, ("ESFP", "ENTP"): 40,
    ("ESFP", "INFJ"): 50, ("ESFP", "INFP"): 60, ("ESFP", "ENFJ"): 65, ("ESFP", "ENFP"): 65,
    ("ESFP", "ISTJ"): 50, ("ESFP", "ISFJ"): 70, ("ESFP", "ESTJ"): 60, ("ESFP", "ESFJ"): 75,
    ("ESFP", "ISTP"): 65, ("ESFP", "ISFP"): 80, ("ESFP", "ESTP"): 80, ("ESFP", "ESFP"): 70,
}

VALID_MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP",
]


def get_mbti_score(type_a: str, type_b: str) -> float:
    """
    Return MBTI compatibility score (0-100) for a pair of types.
    The matrix is symmetric; if key not found, returns 50 (neutral).

    Parameters
    ----------
    type_a : str  e.g. 'INTJ'
    type_b : str  e.g. 'ENFP'

    Returns
    -------
    float : score in [0, 100]
    """
    type_a = type_a.strip().upper()
    type_b = type_b.strip().upper()

    if type_a not in VALID_MBTI_TYPES or type_b not in VALID_MBTI_TYPES:
        return 50.0  # Neutral for unknown types

    key = (type_a, type_b)
    rev_key = (type_b, type_a)

    if key in MBTI_COMPATIBILITY_MATRIX:
        return float(MBTI_COMPATIBILITY_MATRIX[key])
    elif rev_key in MBTI_COMPATIBILITY_MATRIX:
        return float(MBTI_COMPATIBILITY_MATRIX[rev_key])
    return 50.0


def get_mbti_matrix_df():
    """Return the compatibility matrix as a DataFrame (for heatmap visualization)."""
    import pandas as pd
    types = VALID_MBTI_TYPES
    data = {}
    for ta in types:
        row = {}
        for tb in types:
            row[tb] = get_mbti_score(ta, tb)
        data[ta] = row
    return pd.DataFrame(data, index=types)


if __name__ == "__main__":
    print("INTJ ↔ ENFP:", get_mbti_score("INTJ", "ENFP"))
    print("INFJ ↔ ENTP:", get_mbti_score("INFJ", "ENTP"))
    print("ISTJ ↔ ISTJ:", get_mbti_score("ISTJ", "ISTJ"))
    print("ENFJ ↔ INFP:", get_mbti_score("ENFJ", "INFP"))
