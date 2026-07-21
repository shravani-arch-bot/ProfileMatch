"""
app.py
======
Streamlit UI Demo — Profile-Based Matching Algorithm
A premium dark-themed interface showing top matches with breakdown charts
and a live feedback system.

Run:
    streamlit run app.py
"""

import os
import sys
import json
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from scoring_engine import ScoringEngine
from utils import (
    load_users, load_feedback, get_user_weights, set_user_weights,
    score_to_label, get_location_score
)
from mbti_engine import get_mbti_score
from feedback_loop import learn_weights_for_user

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Profile Matcher — Intelligent Matching",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0D1117;
    color: #E6EDF3;
}

.main { background-color: #0D1117; }
.block-container { padding: 1.5rem 2rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161B22 0%, #0D1117 100%);
    border-right: 1px solid #30363D;
}
section[data-testid="stSidebar"] .css-1d391kg { padding-top: 1rem; }

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, #6C63FF 0%, #3A86FF 50%, #43D9AD 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(108, 99, 255, 0.3);
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: #fff;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1rem;
    color: rgba(255,255,255,0.85);
    margin: 0.4rem 0 0 0;
}

/* Profile Card */
.profile-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.profile-card:hover { border-color: #6C63FF; }

/* Match Card */
.match-card {
    background: linear-gradient(135deg, #161B22, #1C2128);
    border: 1px solid #30363D;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    transition: all 0.25s ease;
}
.match-card:hover {
    border-color: #6C63FF;
    box-shadow: 0 6px 24px rgba(108,99,255,0.25);
    transform: translateY(-2px);
}

/* Score Badge */
.score-badge {
    display: inline-block;
    font-size: 1.8rem;
    font-weight: 700;
    padding: 0.3rem 1rem;
    border-radius: 10px;
    background: linear-gradient(135deg, #6C63FF, #3A86FF);
    color: white;
    box-shadow: 0 3px 12px rgba(108,99,255,0.4);
}

/* Metric boxes */
.metric-box {
    background: #1C2128;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    border: 1px solid #30363D;
    text-align: center;
}
.metric-label { font-size: 0.75rem; color: #8B949E; margin-bottom: 4px; }
.metric-value { font-size: 1.1rem; font-weight: 600; color: #E6EDF3; }

/* Tags */
.tag {
    display: inline-block;
    background: #21262D;
    border: 1px solid #30363D;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    color: #8B949E;
    margin: 2px;
}
.mbti-tag {
    background: linear-gradient(135deg, #6C63FF22, #3A86FF22);
    border: 1px solid #6C63FF55;
    color: #6C63FF;
    font-weight: 600;
}

/* Feedback buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9rem;
    transition: all 0.2s;
}

/* Section headers */
.section-header {
    font-size: 1.15rem;
    font-weight: 600;
    color: #E6EDF3;
    border-left: 3px solid #6C63FF;
    padding-left: 0.7rem;
    margin: 1.2rem 0 0.8rem 0;
}

/* Weight pills */
.weight-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 2px;
}

/* Divider */
.custom-divider {
    border: none;
    border-top: 1px solid #30363D;
    margin: 1rem 0;
}

/* Info box */
.info-box {
    background: #1C2128;
    border-left: 4px solid #6C63FF;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    color: #8B949E;
}
</style>
""", unsafe_allow_html=True)

# ── Initialization ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Loading matching engine...")
def load_engine():
    engine = ScoringEngine()
    engine.load("data/users.csv")
    return engine


@st.cache_data
def get_users():
    return load_users("data/users.csv")


def get_feedback_df():
    try:
        return load_feedback("data/feedback.csv")
    except Exception:
        return pd.DataFrame(columns=["user_id", "matched_user_id", "action", "timestamp"])


def save_feedback_row(user_id: str, matched_user_id: str, action: int):
    """Append a new feedback row to feedback.csv."""
    path = "data/feedback.csv"
    ts = datetime.datetime.now().strftime("%Y-%m-%d")
    new_row = pd.DataFrame([{
        "user_id": user_id,
        "matched_user_id": matched_user_id,
        "action": action,
        "timestamp": ts,
    }])
    if os.path.exists(path):
        existing = pd.read_csv(path)
        # Remove if same pair already rated in this session
        existing = existing[
            ~((existing["user_id"] == user_id) &
              (existing["matched_user_id"] == matched_user_id))
        ]
        updated = pd.concat([existing, new_row], ignore_index=True)
    else:
        updated = new_row
    updated.to_csv(path, index=False)


def refresh_weights(user_id: str):
    """Re-learn weights for a user after feedback."""
    try:
        fb = get_feedback_df()
        engine = load_engine()
        new_w = learn_weights_for_user(user_id, fb, engine)
        if new_w:
            set_user_weights(user_id, new_w)
            return new_w
    except Exception:
        pass
    return get_user_weights(user_id)


# ── Mini Chart Helpers ─────────────────────────────────────────────────────────
def make_donut(score: float, label: str, color: str):
    """Return a small matplotlib donut chart figure."""
    fig, ax = plt.subplots(figsize=(2.2, 2.2), facecolor="#161B22")
    ax.set_facecolor("#161B22")
    size = 0.35
    vals = [score, 100 - score]
    colors = [color, "#21262D"]
    wedges, _ = ax.pie(vals, colors=colors, startangle=90,
                       wedgeprops={"width": size, "edgecolor": "#0D1117", "linewidth": 2})
    ax.text(0, 0, f"{score:.0f}", ha="center", va="center",
            fontsize=14, fontweight="bold", color="white")
    ax.set_title(label, color="#8B949E", fontsize=8, pad=4)
    fig.tight_layout(pad=0.3)
    return fig


def make_score_bar(matches: list):
    """Return a horizontal bar chart for top matches."""
    names = [f"{m['name'].split()[0]} ({m['user_id']})" for m in matches]
    scores = [m["total_score"] for m in matches]
    colors = ["#6C63FF", "#3A86FF", "#43D9AD", "#FF6584", "#FFD166"]

    fig, ax = plt.subplots(figsize=(7, 2.8), facecolor="#161B22")
    ax.set_facecolor("#161B22")
    bars = ax.barh(names[::-1], scores[::-1], color=colors[::-1], height=0.55,
                   edgecolor="#0D1117", linewidth=0.5)
    for bar, score in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}%", va="center", color="white", fontsize=9, fontweight="600")
    ax.set_xlim(0, 110)
    ax.set_xlabel("Compatibility Score (%)", color="#8B949E", fontsize=9)
    ax.tick_params(colors="white", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#30363D")
    ax.spines["bottom"].set_color("#30363D")
    ax.set_facecolor("#161B22")
    fig.tight_layout()
    return fig


# ── Session State ──────────────────────────────────────────────────────────────
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = {}   # {matched_id: action}
if "weights_updated" not in st.session_state:
    st.session_state.weights_updated = False
if "selected_user" not in st.session_state:
    st.session_state.selected_user = None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem 0;">
        <div style="font-size:2.5rem;">🔗</div>
        <div style="font-size:1.2rem; font-weight:700; color:#6C63FF;">Profile Matcher</div>
        <div style="font-size:0.75rem; color:#8B949E;">Intelligent Hybrid Matching</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 👤 Select Profile")
    users_df = get_users()
    user_options = {
        f"{row['name']} ({row['user_id']})": row["user_id"]
        for _, row in users_df.iterrows()
    }
    selected_label = st.selectbox(
        "Choose your profile",
        list(user_options.keys()),
        label_visibility="collapsed",
    )
    selected_uid = user_options[selected_label]

    if st.session_state.selected_user != selected_uid:
        st.session_state.selected_user = selected_uid
        st.session_state.feedback_log = {}
        st.session_state.weights_updated = False

    st.markdown("---")
    st.markdown("### ⚙️ Match Settings")
    n_matches = st.slider("Number of matches", 3, 10, 5)

    st.markdown("---")
    st.markdown("### 🎛️ Current Weights")
    weights = get_user_weights(selected_uid)
    st.markdown(f"""
    <div class="metric-box" style="margin-bottom:8px;">
        <div class="metric-label">📝 Text Similarity (w1)</div>
        <div class="metric-value" style="color:#6C63FF;">{weights['w1']:.1%}</div>
    </div>
    <div class="metric-box" style="margin-bottom:8px;">
        <div class="metric-label">🧠 MBTI Match (w2)</div>
        <div class="metric-value" style="color:#FF6584;">{weights['w2']:.1%}</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">📍 Location (w3)</div>
        <div class="metric-value" style="color:#43D9AD;">{weights['w3']:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.feedback_log:
        st.markdown("---")
        st.markdown("### 📊 Session Feedback")
        accepted = sum(1 for v in st.session_state.feedback_log.values() if v == 1)
        rejected = sum(1 for v in st.session_state.feedback_log.values() if v == 0)
        st.markdown(f"""
        <div style="display:flex; gap:8px; margin-top:6px;">
            <div class="metric-box" style="flex:1;">
                <div class="metric-label">Accepted</div>
                <div class="metric-value" style="color:#43D9AD;">✅ {accepted}</div>
            </div>
            <div class="metric-box" style="flex:1;">
                <div class="metric-label">Rejected</div>
                <div class="metric-value" style="color:#FF6584;">❌ {rejected}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 Update Weights from Feedback", width='stretch'):
            new_w = refresh_weights(selected_uid)
            st.session_state.weights_updated = True
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.7rem; color:#8B949E; text-align:center; padding-top:0.5rem;">
        Profile-Based Matching Algorithm<br>
        NLP + MBTI + Adaptive ML
    </div>
    """, unsafe_allow_html=True)


# ── Main Content ───────────────────────────────────────────────────────────────
# Hero Banner
user_row = users_df[users_df["user_id"] == selected_uid].iloc[0]

st.markdown(f"""
<div class="hero-banner">
    <div class="hero-title">🔗 Profile Matcher</div>
    <div class="hero-subtitle">
        Intelligent hybrid matching powered by NLP · MBTI Analysis · Adaptive ML
    </div>
</div>
""", unsafe_allow_html=True)

# ── My Profile ─────────────────────────────────────────────────────────────────
with st.expander("👤 Your Profile", expanded=False):
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        st.markdown(f"""
        <div class="profile-card">
            <div style="font-size:1.2rem; font-weight:700; color:#E6EDF3;">{user_row['name']}</div>
            <div style="font-size:0.85rem; color:#8B949E; margin-top:3px;">{user_row['job_title']} · {user_row['industry']}</div>
            <hr class="custom-divider">
            <div style="font-size:0.85rem; color:#C9D1D9;">{user_row['professional_summary']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="profile-card">
            <div style="font-size:0.9rem; font-weight:600; color:#8B949E; margin-bottom:8px;">ABOUT ME</div>
            <div style="font-size:0.85rem; color:#C9D1D9;">{user_row['about_me']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="profile-card">
            <div class="metric-label">Personality</div>
            <div style="font-size:1.5rem; font-weight:700; color:#6C63FF;">{user_row['mbti_type']}</div>
            <hr class="custom-divider">
            <div class="metric-label">Location</div>
            <div class="metric-value">📍 {user_row['location']}</div>
            <hr class="custom-divider">
            <div class="metric-label">Age</div>
            <div class="metric-value">🎂 {user_row['age']}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Weight Update Toast ────────────────────────────────────────────────────────
if st.session_state.weights_updated:
    updated_w = get_user_weights(selected_uid)
    st.success(
        f"✅ Weights updated from your feedback! "
        f"w1={updated_w['w1']:.1%} | w2={updated_w['w2']:.1%} | w3={updated_w['w3']:.1%}"
    )
    st.session_state.weights_updated = False

# ── Get Matches ────────────────────────────────────────────────────────────────
engine = load_engine()
current_weights = get_user_weights(selected_uid)

with st.spinner("🔍 Finding your best matches..."):
    matches = engine.get_top_matches(selected_uid, n=n_matches, weights=current_weights)

# ── Summary Overview ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🏆 Top Matches Overview</div>', unsafe_allow_html=True)

col_chart, col_info = st.columns([1.8, 1])
with col_chart:
    fig = make_score_bar(matches)
    st.pyplot(fig, width='stretch')
    plt.close()
with col_info:
    avg_score = np.mean([m["total_score"] for m in matches])
    best_score = matches[0]["total_score"] if matches else 0
    st.markdown(f"""
    <div class="metric-box" style="margin-bottom:10px;">
        <div class="metric-label">🎯 Best Match Score</div>
        <div style="font-size:1.8rem; font-weight:700; color:#6C63FF;">{best_score:.1f}%</div>
    </div>
    <div class="metric-box" style="margin-bottom:10px;">
        <div class="metric-label">📊 Average Score</div>
        <div style="font-size:1.4rem; font-weight:700; color:#43D9AD;">{avg_score:.1f}%</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">👥 Profiles Scanned</div>
        <div style="font-size:1.4rem; font-weight:700; color:#3A86FF;">{len(users_df) - 1}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Individual Match Cards ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔍 Detailed Match Cards</div>', unsafe_allow_html=True)

rank_colors = ["#FFD700", "#C0C0C0", "#CD7F32", "#6C63FF", "#43D9AD"]
feedback_already = st.session_state.feedback_log

for i, match in enumerate(matches):
    mid = match["user_id"]
    rank_color = rank_colors[min(i, len(rank_colors) - 1)]
    already_rated = mid in feedback_already

    with st.container():
        st.markdown(f"""
        <div class="match-card">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
                <div style="font-size:1.4rem; font-weight:800; color:{rank_color};">#{i+1}</div>
                <div style="flex:1;">
                    <div style="font-size:1.1rem; font-weight:700; color:#E6EDF3;">{match['name']}</div>
                    <div style="font-size:0.82rem; color:#8B949E;">{match['job_title']} · {match['industry']}</div>
                </div>
                <div class="score-badge">{match['total_score']:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Columns: profile details | donuts | feedback
        col_left, col_mid, col_right = st.columns([2.5, 2, 1])

        with col_left:
            st.markdown(f"""
            <div style="background:#1C2128; border-radius:10px; padding:1rem; margin-bottom:8px;">
                <div style="font-size:0.75rem; color:#8B949E; font-weight:600; margin-bottom:6px;">PROFESSIONAL SUMMARY</div>
                <div style="font-size:0.83rem; color:#C9D1D9; line-height:1.5;">{match['professional_summary'][:220]}...</div>
            </div>
            <div style="display:flex; gap:6px; flex-wrap:wrap;">
                <span class="tag mbti-tag">🧠 {match['mbti_type']}</span>
                <span class="tag">📍 {match['location']}</span>
                <span class="tag">🎂 {match['age']}</span>
                <span class="tag">{score_to_label(match['total_score'])}</span>
            </div>
            """, unsafe_allow_html=True)

        with col_mid:
            d1, d2, d3 = st.columns(3)
            with d1:
                fig1 = make_donut(match["text_score"], "Text Sim", "#6C63FF")
                st.pyplot(fig1, width='stretch')
                plt.close()
            with d2:
                fig2 = make_donut(match["mbti_score"], "MBTI", "#FF6584")
                st.pyplot(fig2, width='stretch')
                plt.close()
            with d3:
                fig3 = make_donut(match["location_score"], "Location", "#43D9AD")
                st.pyplot(fig3, width='stretch')
                plt.close()

        with col_right:
            st.markdown("<br>", unsafe_allow_html=True)
            if already_rated:
                action_taken = feedback_already[mid]
                if action_taken == 1:
                    st.markdown("✅ **Accepted**")
                else:
                    st.markdown("❌ **Rejected**")
            else:
                if st.button(f"👍 Accept", key=f"accept_{mid}", width='stretch'):
                    save_feedback_row(selected_uid, mid, 1)
                    st.session_state.feedback_log[mid] = 1
                    st.rerun()
                if st.button(f"👎 Reject", key=f"reject_{mid}", width='stretch'):
                    save_feedback_row(selected_uid, mid, 0)
                    st.session_state.feedback_log[mid] = 0
                    st.rerun()
            st.markdown(f"""
            <div style="margin-top:8px; font-size:0.75rem; color:#8B949E;">
                w1={current_weights['w1']:.0%}<br>
                w2={current_weights['w2']:.0%}<br>
                w3={current_weights['w3']:.0%}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")


# ── About Me Comparison ────────────────────────────────────────────────────────
if matches:
    st.markdown('<div class="section-header">💬 About Me — Top Match vs You</div>', unsafe_allow_html=True)
    best = matches[0]
    ca, cb = st.columns(2)
    with ca:
        st.markdown(f"""
        <div class="profile-card">
            <div style="font-size:0.8rem; color:#6C63FF; font-weight:600; margin-bottom:6px;">YOU — {user_row['name']}</div>
            <div style="font-size:0.85rem; color:#C9D1D9;">{user_row['about_me']}</div>
        </div>
        """, unsafe_allow_html=True)
    with cb:
        st.markdown(f"""
        <div class="profile-card">
            <div style="font-size:0.8rem; color:#43D9AD; font-weight:600; margin-bottom:6px;">#{1} MATCH — {best['name']}</div>
            <div style="font-size:0.85rem; color:#C9D1D9;">{best['about_me']}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Algorithm Info ─────────────────────────────────────────────────────────────
with st.expander("ℹ️ How the Algorithm Works", expanded=False):
    st.markdown("""
    ### 🧮 Core Formula

    $$\\text{Total Score} = (w_1 \\times \\text{TextSim} \\times 100) + (w_2 \\times \\text{MBTIMatch}) + (w_3 \\times \\text{LocationScore})$$

    | Component | Method | Weight |
    |---|---|---|
    | **Text Similarity** | TF-IDF + Cosine Similarity on bios & summaries | w₁ = 50% (default) |
    | **MBTI Compatibility** | 16×16 psychological matrix | w₂ = 30% (default) |
    | **Location Match** | Same city=100, Same region=60, Other=20 | w₃ = 20% (default) |

    ### 🤖 Adaptive Feedback Loop
    - When you **Accept** or **Reject** a match, the system records your preference.
    - After 5+ interactions, **Ridge Regression** learns your personal weights.
    - Click **"Update Weights from Feedback"** in the sidebar to see your personalized weights.

    ### 📊 NLP Processing Pipeline
    1. **Cleaning** — lowercase, remove stopwords, lemmatize
    2. **Vectorization** — TF-IDF with bigrams (vocabulary: 3,000 terms)
    3. **Similarity** — Cosine similarity between user vector pairs
    """)

st.markdown("""
<div style="text-align:center; color:#30363D; font-size:0.75rem; padding:2rem 0 1rem 0;">
    Profile Matcher — Profile-Based Matching Algorithm · Built with NLP + ML · 
    <span style="color:#6C63FF;">Powered by Streamlit</span>
</div>
""", unsafe_allow_html=True)
