import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Analisis Hunger Games",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme Colors ──────────────────────────────────────────────────────────────
COLOR_POS  = "#2ecc71"
COLOR_NEG  = "#e74c3c"
COLOR_NEU  = "#95a5a6"
COLOR_MAP  = {"Positif": COLOR_POS, "Negatif": COLOR_NEG, "Netral": COLOR_NEU}
TOPIC_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        border-left: 4px solid #7c3aed;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #fff; }
    .metric-label { font-size: 0.85rem; color: #a0a0b0; margin-top: 4px; }
    .section-title {
        font-size: 1.15rem; font-weight: 600;
        color: #e2e8f0; margin-bottom: 8px;
        border-bottom: 2px solid #334155; padding-bottom: 6px;
    }
    div[data-testid="stSidebar"] { background: #0f172a; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("hunger_games_reviews_clean.csv")
    return df

df = load_data()

TOPIC_NAMES = {
    1: "Penilaian Akting & Karakter",
    2: "Perbandingan Film dengan Buku",
    3: "Antisipasi & Reaksi Emosional",
    4: "Penilaian Umum Franchise",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 Hunger Games\n### Review Analytics")
    st.markdown("---")

    st.markdown("### 🔍 Filter Data")
    selected_sentiments = st.multiselect(
        "Sentimen",
        options=["Positif", "Negatif", "Netral"],
        default=["Positif", "Negatif", "Netral"],
    )
    selected_topics = st.multiselect(
        "Topik LDA",
        options=list(TOPIC_NAMES.values()),
        default=list(TOPIC_NAMES.values()),
    )
    word_range = st.slider("Jumlah Kata per Review", 1, 113, (1, 113))

    st.markdown("---")
    st.markdown("### 📊 Tentang Dataset")
    st.info(
        "**Total awal:** 1.812 baris\n\n"
        "**Setelah cleaning:** 1.767 baris\n\n"
        "**Metode labeling:** VADER\n\n"
        "**Model:** Naive Bayes (CNB)\n\n"
        "**Akurasi:** 78.85%"
    )

# ── Apply Filters ─────────────────────────────────────────────────────────────
dff = df[
    df["sentiment_label"].isin(selected_sentiments) &
    df["topic_name"].isin(selected_topics) &
    df["word_count"].between(word_range[0], word_range[1])
].copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🎬 Dashboard Analisis Sentimen & Topic Modelling")
st.markdown("**Review Film The Hunger Games** — VADER Labeling · LDA Topic Modelling · Naive Bayes Classification")
st.markdown("---")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

total    = len(dff)
n_pos    = len(dff[dff["sentiment_label"] == "Positif"])
n_neg    = len(dff[dff["sentiment_label"] == "Negatif"])
n_neu    = len(dff[dff["sentiment_label"] == "Netral"])
avg_score = dff["sentiment_score"].mean()

for col, val, label, color in [
    (k1, f"{total:,}",                   "Total Review",         "#7c3aed"),
    (k2, f"{n_pos:,} ({n_pos/total*100:.1f}%)", "Review Positif",  "#2ecc71"),
    (k3, f"{n_neg:,} ({n_neg/total*100:.1f}%)", "Review Negatif",  "#e74c3c"),
    (k4, f"{n_neu:,} ({n_neu/total*100:.1f}%)", "Review Netral",   "#95a5a6"),
    (k5, f"{avg_score:.4f}",             "Rata-rata Skor",       "#f39c12"),
]:
    with col:
        st.markdown(
            f'<div class="metric-card" style="border-left-color:{color};">'
            f'<div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Distribusi Sentimen", "☁️ Word Cloud & Frekuensi", "🗂️ Topic Modelling", "🤖 Model Naive Bayes"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – Distribusi Sentimen
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns(2)

    # Pie chart
    with col_a:
        st.markdown('<div class="section-title">Proporsi Sentimen</div>', unsafe_allow_html=True)
        counts = dff["sentiment_label"].value_counts()
        colors = [COLOR_MAP.get(l, "#ccc") for l in counts.index]
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="none")
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
               colors=colors, startangle=90, explode=[0.05]*len(counts),
               textprops={"color": "white"})
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)

    # Bar chart
    with col_b:
        st.markdown('<div class="section-title">Jumlah Review per Sentimen</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="none")
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.7)
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    str(v), ha="center", fontweight="bold", color="white")
        ax.set_facecolor("none")
        ax.tick_params(colors="white")
        ax.set_ylabel("Jumlah Review", color="white")
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    col_c, col_d = st.columns(2)

    # Histogram word count
    with col_c:
        st.markdown('<div class="section-title">Distribusi Jumlah Kata per Sentimen</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="none")
        for label in ["Positif", "Negatif", "Netral"]:
            subset = dff[dff["sentiment_label"] == label]["word_count"]
            ax.hist(subset, alpha=0.6, label=label, color=COLOR_MAP[label], bins=30)
        ax.set_xlim(0, 60)
        ax.set_xlabel("Jumlah Kata", color="white")
        ax.set_ylabel("Frekuensi", color="white")
        ax.legend()
        ax.set_facecolor("none")
        ax.tick_params(colors="white")
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)

    # Sentiment score distribution
    with col_d:
        st.markdown('<div class="section-title">Distribusi Skor Sentimen (VADER Compound)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="none")
        ax.hist(dff["sentiment_score"], bins=40, color="#3498db", edgecolor="white", linewidth=0.4)
        ax.axvline(x=0.05,  color=COLOR_POS, linestyle="--", label="Batas Positif (0.05)")
        ax.axvline(x=-0.05, color=COLOR_NEG, linestyle="--", label="Batas Negatif (-0.05)")
        ax.set_xlabel("Compound Score", color="white")
        ax.set_ylabel("Frekuensi", color="white")
        ax.legend(fontsize=8)
        ax.set_facecolor("none")
        ax.tick_params(colors="white")
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)

    # Boxplot word count
    st.markdown('<div class="section-title">Boxplot Jumlah Kata per Sentimen</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 3), facecolor="none")
    groups = [dff[dff["sentiment_label"] == l]["word_count"].values for l in ["Positif", "Netral", "Negatif"]]
    bp = ax.boxplot(groups, labels=["Positif", "Netral", "Negatif"], patch_artist=True,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], [COLOR_POS, COLOR_NEU, COLOR_NEG]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylim(0, 60)
    ax.set_facecolor("none")
    ax.tick_params(colors="white")
    ax.set_ylabel("Jumlah Kata", color="white")
    fig.patch.set_alpha(0)
    st.pyplot(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – Word Cloud & Frekuensi
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">25 Kata Paling Sering Muncul</div>', unsafe_allow_html=True)
    all_words = " ".join(dff["review"].dropna().tolist()).split()
    word_freq = Counter(all_words)
    top_words = word_freq.most_common(25)
    words, freqs = zip(*top_words)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="none")
    colors_bar = plt.cm.viridis(np.linspace(0.2, 0.8, len(words)))
    bars = ax.barh(list(reversed(words)), list(reversed(freqs)), color=list(reversed(colors_bar)))
    for bar, freq in zip(bars, list(reversed(freqs))):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(freq), va="center", fontsize=9, color="white")
    ax.set_xlabel("Frekuensi", color="white")
    ax.set_facecolor("none")
    ax.tick_params(colors="white")
    fig.patch.set_alpha(0)
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Word Cloud per Kategori Sentimen</div>', unsafe_allow_html=True)
    wc_cols = st.columns(3)
    wc_config = [("Positif", "Greens"), ("Negatif", "Reds"), ("Netral", "Blues")]

    for col, (label, cmap) in zip(wc_cols, wc_config):
        with col:
            st.markdown(f"**Sentimen {label}**")
            subset_text = " ".join(dff[dff["sentiment_label"] == label]["review"].dropna().tolist())
            if len(subset_text.strip()) > 10:
                wc = WordCloud(width=400, height=280, background_color="white",
                               colormap=cmap, max_words=80, collocations=False).generate(subset_text)
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig, use_container_width=True)
            else:
                st.warning("Tidak cukup data untuk word cloud.")

    st.markdown("---")
    st.markdown('<div class="section-title">Top 15 Kata per Sentimen</div>', unsafe_allow_html=True)
    top_cols = st.columns(3)
    for col, (label, color) in zip(top_cols, [("Positif", COLOR_POS), ("Negatif", COLOR_NEG), ("Netral", COLOR_NEU)]):
        with col:
            st.markdown(f"**Top Kata – {label}**")
            subset_words = " ".join(dff[dff["sentiment_label"] == label]["review"].dropna().tolist()).split()
            top15 = Counter(subset_words).most_common(15)
            if top15:
                w, f = zip(*top15)
                fig, ax = plt.subplots(figsize=(4, 5), facecolor="none")
                ax.barh(list(reversed(w)), list(reversed(f)), color=color, edgecolor="white", linewidth=0.4)
                ax.set_xlabel("Frekuensi", color="white")
                ax.set_facecolor("none")
                ax.tick_params(colors="white")
                fig.patch.set_alpha(0)
                st.pyplot(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – LDA Topic Modelling
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🗂️ LDA Topic Modelling — 4 Topik")
    st.info(
        "LDA mengasumsikan setiap review adalah campuran dari beberapa topik. "
        "Model dilatih dengan `n_components=4`, `max_iter=20`, `learning_method='batch'`.  \n"
        "**Log-likelihood:** -63,728.53 · **Perplexity:** 660.05"
    )

    # Topic summary cards
    t_info = [
        ("1", "🎭", "Penilaian Akting & Karakter",      "book, read, cast, look, give, perfect, bad, feel",      TOPIC_COLORS[0]),
        ("2", "📖", "Perbandingan Film dengan Buku",     "look, scene, excite, arena, year, love, funny, color",  TOPIC_COLORS[1]),
        ("3", "😮", "Antisipasi & Reaksi Emosional",    "book, make, say, thats, comment, reading, kill, laugh", TOPIC_COLORS[2]),
        ("4", "🎬", "Penilaian Umum Franchise",         "game, hunger, book, watch, haymitch, feel, story, movie", TOPIC_COLORS[3]),
    ]
    t_cols = st.columns(4)
    for col, (num, icon, name, keywords, color) in zip(t_cols, t_info):
        with col:
            st.markdown(
                f'<div style="background:#1e1e2e;border-radius:10px;padding:16px;'
                f'border-top:4px solid {color};">'
                f'<div style="font-size:1.5rem">{icon}</div>'
                f'<div style="font-weight:700;color:#fff;margin:6px 0">Topik {num}</div>'
                f'<div style="color:#a0aec0;font-size:0.8rem;margin-bottom:8px">{name}</div>'
                f'<div style="color:#718096;font-size:0.75rem">{keywords}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    # Bar chart distribusi topik
    with col_left:
        st.markdown('<div class="section-title">Jumlah Review per Topik</div>', unsafe_allow_html=True)
        topic_counts = dff["dominant_topic"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="none")
        bars = ax.bar(
            [f"Topik {t}" for t in topic_counts.index],
            topic_counts.values,
            color=TOPIC_COLORS[:len(topic_counts)], edgecolor="white"
        )
        for bar, v in zip(bars, topic_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    str(v), ha="center", fontweight="bold", color="white")
        ax.set_facecolor("none")
        ax.tick_params(colors="white")
        ax.set_ylabel("Jumlah Review", color="white")
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)

    # Stacked bar sentimen per topik
    with col_right:
        st.markdown('<div class="section-title">Komposisi Sentimen per Topik</div>', unsafe_allow_html=True)
        topic_sent = pd.crosstab(dff["dominant_topic"], dff["sentiment_label"], normalize="index").mul(100).round(1)
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="none")
        bottom = np.zeros(len(topic_sent))
        for label, color in [("Positif", COLOR_POS), ("Netral", COLOR_NEU), ("Negatif", COLOR_NEG)]:
            if label in topic_sent.columns:
                vals = topic_sent[label].values
                ax.bar([f"T{t}" for t in topic_sent.index], vals, bottom=bottom, label=label, color=color)
                bottom += vals
        ax.set_ylabel("Proporsi (%)", color="white")
        ax.legend()
        ax.set_facecolor("none")
        ax.tick_params(colors="white")
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)

    # Topic confidence distribution
    st.markdown('<div class="section-title">Distribusi Topic Confidence per Topik</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 3), facecolor="none")
    for i, (t_num, color) in enumerate(zip([1, 2, 3, 4], TOPIC_COLORS)):
        subset = dff[dff["dominant_topic"] == t_num]["topic_confidence"]
        ax.hist(subset, bins=20, alpha=0.6, label=f"Topik {t_num}", color=color)
    ax.set_xlabel("Confidence Score", color="white")
    ax.set_ylabel("Frekuensi", color="white")
    ax.legend()
    ax.set_facecolor("none")
    ax.tick_params(colors="white")
    fig.patch.set_alpha(0)
    st.pyplot(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – Naive Bayes
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 🤖 Naive Bayes Classification")
    st.info(
        "Model terbaik: **Complement NB + TF-IDF Bigram** (α=0.5, threshold=0.45).  \n"
        "Label yang digunakan adalah hasil VADER. Hanya kelas **Positif** dan **Negatif** yang diklasifikasi."
    )

    # Metrics comparison
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("🎯 Akurasi Baseline (MNB)", "77.97%", help="CountVectorizer 500 fitur, α=1.0")
    with m2:
        st.metric("🚀 Akurasi Final (CNB)", "78.85%", delta="+0.88%", help="TF-IDF Bigram, α=0.5, thresh=0.45")
    with m3:
        st.metric("📈 Best CV Score (5-fold)", "78.51%", help="Cross-validation pada TF-IDF + MNB α=0.5")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    # Confusion matrix
    with col_l:
        st.markdown('<div class="section-title">Confusion Matrix — Model Final</div>', unsafe_allow_html=True)
        cm = np.array([[61, 27], [21, 118]])  # from notebook output (Negatif/Positif)
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="none")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Negatif", "Positif"],
                    yticklabels=["Negatif", "Positif"],
                    ax=ax, linewidths=0.5,
                    annot_kws={"size": 14})
        ax.set_xlabel("Prediksi", color="white")
        ax.set_ylabel("Aktual", color="white")
        ax.tick_params(colors="white")
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)

    # Precision / Recall / F1
    with col_r:
        st.markdown('<div class="section-title">Precision, Recall, F1 per Kelas</div>', unsafe_allow_html=True)
        metrics_df = pd.DataFrame({
            "Precision": [0.74, 0.82],
            "Recall":    [0.70, 0.84],
            "F1-Score":  [0.72, 0.83],
        }, index=["Negatif", "Positif"])
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="none")
        x = np.arange(len(metrics_df.index))
        w = 0.25
        for i, (col_name, color) in enumerate(zip(metrics_df.columns, ["#3498db", "#e74c3c", "#2ecc71"])):
            ax.bar(x + i*w, metrics_df[col_name], w, label=col_name, color=color, edgecolor="white")
        ax.set_xticks(x + w)
        ax.set_xticklabels(metrics_df.index, color="white")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score", color="white")
        ax.legend()
        ax.set_facecolor("none")
        ax.tick_params(colors="white")
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)

    # Alpha tuning chart
    st.markdown('<div class="section-title">Grid Search Alpha (5-fold CV Accuracy)</div>', unsafe_allow_html=True)
    alpha_data = {
        "Alpha": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
        "CV Accuracy (%)": [75.60, 77.10, 77.98, 78.42, 78.51, 77.10, 73.56, 65.43],
        "Std (%)": [2.79, 2.63, 2.38, 2.96, 2.77, 1.80, 1.44, 2.11],
    }
    alpha_df = pd.DataFrame(alpha_data)
    fig, ax = plt.subplots(figsize=(8, 3), facecolor="none")
    ax.plot(alpha_df["Alpha"], alpha_df["CV Accuracy (%)"], marker="o", color="#3498db", linewidth=2)
    ax.fill_between(
        alpha_df["Alpha"],
        alpha_df["CV Accuracy (%)"] - alpha_df["Std (%)"],
        alpha_df["CV Accuracy (%)"] + alpha_df["Std (%)"],
        alpha=0.2, color="#3498db"
    )
    ax.axvline(x=0.5, color="#f39c12", linestyle="--", label="Best α=0.5")
    ax.set_xlabel("Alpha", color="white")
    ax.set_ylabel("CV Accuracy (%)", color="white")
    ax.legend()
    ax.set_facecolor("none")
    ax.tick_params(colors="white")
    fig.patch.set_alpha(0)
    st.pyplot(fig, use_container_width=True)

    # Top features
    st.markdown('<div class="section-title">Top Kata Berpengaruh per Kelas (NB)</div>', unsafe_allow_html=True)
    nb_cols = st.columns(2)
    nb_words = {
        "Negatif": (["game", "book", "hunger", "haymitch", "year", "fire", "read", "part", "bad", "end"],
                    [0.95, 0.88, 0.82, 0.79, 0.75, 0.71, 0.68, 0.65, 0.62, 0.59]),
        "Positif": (["book", "love", "game", "look", "hunger", "funny", "cast", "perfect", "amaze", "beautiful"],
                    [0.97, 0.91, 0.87, 0.83, 0.78, 0.74, 0.71, 0.68, 0.65, 0.61]),
    }
    for col, (label, color) in zip(nb_cols, [("Negatif", COLOR_NEG), ("Positif", COLOR_POS)]):
        with col:
            st.markdown(f"**Kelas: {label}**")
            w_list, s_list = nb_words[label]
            fig, ax = plt.subplots(figsize=(5, 4), facecolor="none")
            ax.barh(list(reversed(w_list)), list(reversed(s_list)), color=color, edgecolor="white", linewidth=0.4)
            ax.set_xlabel("Relative Weight", color="white")
            ax.set_facecolor("none")
            ax.tick_params(colors="white")
            fig.patch.set_alpha(0)
            st.pyplot(fig, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#64748b;font-size:0.8rem;">'
    "Analisis Sentimen & Topic Modelling — Review Film Hunger Games · "
    "VADER · LDA · Naive Bayes · Streamlit Dashboard"
    "</p>",
    unsafe_allow_html=True,
)
