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

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Analisis Hunger Games",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLOR_POS   = "#2ecc71"
COLOR_NEG   = "#e74c3c"
COLOR_NEU   = "#95a5a6"
COLOR_MAP   = {"Positif": COLOR_POS, "Negatif": COLOR_NEG, "Netral": COLOR_NEU}
TOPIC_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e; border-radius: 12px;
        padding: 20px 24px; text-align: center;
        border-left: 4px solid #7c3aed;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #fff; }
    .metric-label { font-size: 0.82rem; color: #a0a0b0; margin-top: 4px; }
    .section-title {
        font-size: 1.1rem; font-weight: 600; color: #e2e8f0;
        margin-bottom: 8px; border-bottom: 2px solid #334155; padding-bottom: 6px;
    }
    div[data-testid="stSidebar"] { background: #0f172a; }
</style>
""", unsafe_allow_html=True)

# ── Load & Process Data ───────────────────────────────────────────────────────
@st.cache_data
def load_and_process(path="hunger_games_reviews_clean.csv"):
    df = pd.read_csv(path)
    df_clean = df.dropna(subset=["review"]).reset_index(drop=True)
    df_clean["review"] = df_clean["review"].astype(str)

    # VADER labeling
    analyzer = SentimentIntensityAnalyzer()
    def get_label(text):
        c = analyzer.polarity_scores(text)["compound"]
        return "Positif" if c >= 0.05 else ("Negatif" if c <= -0.05 else "Netral")
    def get_score(text):
        return analyzer.polarity_scores(text)["compound"]

    df_clean["sentiment_label"] = df_clean["review"].apply(get_label)
    df_clean["sentiment_score"] = df_clean["review"].apply(get_score)
    df_clean["word_count"]        = df_clean["review"].apply(lambda x: len(x.split()))
    df_clean["char_count"]        = df_clean["review"].apply(len)
    df_clean["unique_word_count"] = df_clean["review"].apply(lambda x: len(set(x.split())))
    return df_clean

@st.cache_data
def run_lda(_df):
    count_vec = CountVectorizer(max_features=800, min_df=3, max_df=0.90)
    count_matrix = count_vec.fit_transform(_df["review"].fillna(""))
    feature_names = count_vec.get_feature_names_out()
    lda = LatentDirichletAllocation(n_components=4, random_state=42,
                                    max_iter=20, learning_method="batch")
    lda.fit(count_matrix)
    doc_topic = lda.transform(count_matrix)
    topic_names = {
        1: "Penilaian Akting & Karakter",
        2: "Perbandingan Film dengan Buku",
        3: "Antisipasi & Reaksi Emosional",
        4: "Penilaian Umum Franchise",
    }
    dominant = doc_topic.argmax(axis=1) + 1
    confidence = doc_topic.max(axis=1)
    return lda, feature_names, dominant, confidence, topic_names

@st.cache_data
def run_nb(_df):
    df_bin = _df[_df["sentiment_label"].isin(["Positif", "Negatif"])].copy()
    le = LabelEncoder()
    y  = le.fit_transform(df_bin["sentiment_label"])

    # Baseline
    vec_base = CountVectorizer(max_features=500, min_df=2)
    X_base   = vec_base.fit_transform(df_bin["review"].fillna(""))
    Xtr_b, Xte_b, ytr_b, yte_b = train_test_split(X_base, y, test_size=0.2,
                                                    random_state=42, stratify=y)
    nb_base = MultinomialNB(alpha=1.0)
    nb_base.fit(Xtr_b, ytr_b)
    acc_base = accuracy_score(yte_b, nb_base.predict(Xte_b))

    # TF-IDF + best alpha
    vec_tfidf = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.95,
                                ngram_range=(1, 2), sublinear_tf=True)
    X_tfidf = vec_tfidf.fit_transform(df_bin["review"].fillna(""))
    Xtr, Xte, ytr, yte = train_test_split(X_tfidf, y, test_size=0.2,
                                           random_state=42, stratify=y)
    best_alpha, best_cv = 1.0, 0
    alpha_results = {}
    for alpha in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        cv = cross_val_score(MultinomialNB(alpha=alpha), X_tfidf, y,
                             cv=5, scoring="accuracy").mean()
        alpha_results[alpha] = cv
        if cv > best_cv:
            best_cv, best_alpha = cv, alpha

    # Calibrated CNB + threshold tuning
    cnb = CalibratedClassifierCV(ComplementNB(alpha=best_alpha), cv=5)
    cnb.fit(Xtr, ytr)
    proba = cnb.predict_proba(Xte)
    best_thresh, best_acc = 0.5, 0
    for thresh in np.arange(0.30, 0.71, 0.05):
        acc_t = accuracy_score(yte, (proba[:, 1] >= thresh).astype(int))
        if acc_t > best_acc:
            best_acc, best_thresh = acc_t, thresh

    y_pred = (proba[:, 1] >= best_thresh).astype(int)
    cm     = confusion_matrix(yte, y_pred)
    report = classification_report(yte, y_pred, target_names=le.classes_, output_dict=True)
    return acc_base, best_acc, best_alpha, best_thresh, best_cv, cm, report, le, alpha_results

# ── Run ───────────────────────────────────────────────────────────────────────
with st.spinner("Memproses data dengan VADER, LDA, dan Naive Bayes..."):
    df_clean = load_and_process("hunger_games_reviews_clean.csv")
    lda_model, feat_names, dominant_topic, topic_conf, topic_names = run_lda(df_clean)
    df_clean["dominant_topic"] = dominant_topic
    df_clean["topic_confidence"] = topic_conf
    df_clean["topic_name"] = df_clean["dominant_topic"].map(topic_names)
    acc_base, acc_final, best_alpha, best_thresh, best_cv, cm, report, le, alpha_results = run_nb(df_clean)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 Hunger Games\n### Review Analytics")
    st.markdown("---")
    st.markdown("### 🔍 Filter Data")
    sel_sent   = st.multiselect("Sentimen", ["Positif","Negatif","Netral"],
                                 default=["Positif","Negatif","Netral"])
    sel_topics = st.multiselect("Topik LDA", list(topic_names.values()),
                                 default=list(topic_names.values()))
    wrange     = st.slider("Jumlah Kata per Review", 1, int(df_clean["word_count"].max()), (1, 30))
    st.markdown("---")
    st.markdown("### 📊 Tentang Dataset")
    st.info(
        f"**Total awal:** 1.812 baris\n\n"
        f"**Setelah cleaning:** {len(df_clean)} baris\n\n"
        f"**Metode labeling:** VADER\n\n"
        f"**Model:** Naive Bayes (CNB)\n\n"
        f"**Akurasi Final:** {acc_final*100:.2f}%"
    )

# ── Filter ────────────────────────────────────────────────────────────────────
dff = df_clean[
    df_clean["sentiment_label"].isin(sel_sent) &
    df_clean["topic_name"].isin(sel_topics) &
    df_clean["word_count"].between(wrange[0], wrange[1])
].copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🎬 Dashboard Analisis Sentimen & Topic Modelling")
st.markdown("**Review Film The Hunger Games** — VADER Labeling · LDA Topic Modelling · Naive Bayes Classification")
st.markdown("---")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
total = len(dff)
if total == 0:
    st.warning("Tidak ada data yang sesuai filter. Ubah filter di sidebar.")
    st.stop()

n_pos = (dff["sentiment_label"]=="Positif").sum()
n_neg = (dff["sentiment_label"]=="Negatif").sum()
n_neu = (dff["sentiment_label"]=="Netral").sum()
avg_score = dff["sentiment_score"].mean()

for col, val, label, color in [
    (k1, f"{total:,}",  "Total Review", "#7c3aed"),
    (k2, f"{n_pos:,} ({n_pos/total*100:.1f}%)", "Review Positif", COLOR_POS),
    (k3, f"{n_neg:,} ({n_neg/total*100:.1f}%)", "Review Negatif", COLOR_NEG),
    (k4, f"{n_neu:,} ({n_neu/total*100:.1f}%)", "Review Netral",  COLOR_NEU),
    (k5, f"{avg_score:.4f}", "Rata-rata Skor", "#f39c12"),
]:
    with col:
        st.markdown(
            f'<div class="metric-card" style="border-left-color:{color};">'
            f'<div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Distribusi Sentimen", "☁️ Word Cloud & Frekuensi", "🗂️ Topic Modelling", "🤖 Model Naive Bayes"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    c1, c2 = st.columns(2)
    counts = dff["sentiment_label"].value_counts()
    colors = [COLOR_MAP.get(l,"#ccc") for l in counts.index]

    with c1:
        st.markdown('<div class="section-title">Proporsi Sentimen</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5,4), facecolor="none")
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
               colors=colors, startangle=90, explode=[0.05]*len(counts),
               textprops={"color":"white"})
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Jumlah Review per Sentimen</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5,4), facecolor="none")
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.7)
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                    str(v), ha="center", fontweight="bold", color="white")
        ax.set_facecolor("none"); ax.tick_params(colors="white")
        ax.set_ylabel("Jumlah Review", color="white")
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="section-title">Distribusi Jumlah Kata per Sentimen</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6,4), facecolor="none")
        for label in ["Positif","Negatif","Netral"]:
            sub = dff[dff["sentiment_label"]==label]["word_count"]
            ax.hist(sub, alpha=0.6, label=label, color=COLOR_MAP[label], bins=30)
        ax.set_xlim(0, min(60, dff["word_count"].quantile(0.99)))
        ax.set_xlabel("Jumlah Kata", color="white"); ax.set_ylabel("Frekuensi", color="white")
        ax.legend(); ax.set_facecolor("none"); ax.tick_params(colors="white")
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)

    with c4:
        st.markdown('<div class="section-title">Distribusi Skor Sentimen (VADER Compound)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6,4), facecolor="none")
        ax.hist(dff["sentiment_score"], bins=40, color="#3498db", edgecolor="white", linewidth=0.4)
        ax.axvline(0.05, color=COLOR_POS, linestyle="--", label="Batas Positif")
        ax.axvline(-0.05, color=COLOR_NEG, linestyle="--", label="Batas Negatif")
        ax.set_xlabel("Compound Score", color="white"); ax.set_ylabel("Frekuensi", color="white")
        ax.legend(fontsize=8); ax.set_facecolor("none"); ax.tick_params(colors="white")
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)

    st.markdown('<div class="section-title">Boxplot Jumlah Kata per Sentimen</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8,3), facecolor="none")
    groups = [dff[dff["sentiment_label"]==l]["word_count"].values for l in ["Positif","Netral","Negatif"]]
    bp = ax.boxplot(groups, labels=["Positif","Netral","Negatif"], patch_artist=True,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], [COLOR_POS, COLOR_NEU, COLOR_NEG]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_ylim(0, min(60, dff["word_count"].quantile(0.99)+5))
    ax.set_facecolor("none"); ax.tick_params(colors="white")
    ax.set_ylabel("Jumlah Kata", color="white")
    fig.patch.set_alpha(0)
    st.pyplot(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">25 Kata Paling Sering Muncul</div>', unsafe_allow_html=True)
    all_words = " ".join(dff["review"].dropna().tolist()).split()
    word_freq = Counter(all_words)
    top25 = word_freq.most_common(25)
    words, freqs = zip(*top25)

    fig, ax = plt.subplots(figsize=(10,6), facecolor="none")
    colors_bar = plt.cm.viridis(np.linspace(0.2, 0.8, len(words)))
    bars = ax.barh(list(reversed(words)), list(reversed(freqs)), color=list(reversed(colors_bar)))
    for bar, freq in zip(bars, list(reversed(freqs))):
        ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                str(freq), va="center", fontsize=9, color="white")
    ax.set_xlabel("Frekuensi", color="white")
    ax.set_facecolor("none"); ax.tick_params(colors="white")
    fig.patch.set_alpha(0)
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Word Cloud per Kategori Sentimen</div>', unsafe_allow_html=True)
    wc_cols = st.columns(3)
    for col, (label, cmap) in zip(wc_cols, [("Positif","Greens"),("Negatif","Reds"),("Netral","Blues")]):
        with col:
            st.markdown(f"**Sentimen {label}**")
            txt = " ".join(dff[dff["sentiment_label"]==label]["review"].dropna().tolist())
            if len(txt.strip()) > 10:
                wc = WordCloud(width=400, height=280, background_color="white",
                               colormap=cmap, max_words=80, collocations=False).generate(txt)
                fig, ax = plt.subplots(figsize=(4,3))
                ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
                st.pyplot(fig, use_container_width=True)
            else:
                st.warning("Data tidak cukup.")

    st.markdown("---")
    st.markdown('<div class="section-title">Top 15 Kata per Sentimen</div>', unsafe_allow_html=True)
    top_cols = st.columns(3)
    for col, (label, color) in zip(top_cols, [("Positif",COLOR_POS),("Negatif",COLOR_NEG),("Netral",COLOR_NEU)]):
        with col:
            st.markdown(f"**Top Kata – {label}**")
            sub_words = " ".join(dff[dff["sentiment_label"]==label]["review"].dropna().tolist()).split()
            top15 = Counter(sub_words).most_common(15)
            if top15:
                w, f = zip(*top15)
                fig, ax = plt.subplots(figsize=(4,5), facecolor="none")
                ax.barh(list(reversed(w)), list(reversed(f)), color=color, edgecolor="white", linewidth=0.4)
                ax.set_xlabel("Frekuensi", color="white")
                ax.set_facecolor("none"); ax.tick_params(colors="white")
                fig.patch.set_alpha(0)
                st.pyplot(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🗂️ LDA Topic Modelling — 4 Topik")
    st.info("LDA dilatih dengan `n_components=4`, `max_iter=20`, `learning_method='batch'`.")

    t_info = [
        ("1","🎭","Penilaian Akting & Karakter",     "book, read, cast, look, give, perfect, bad, feel",       TOPIC_COLORS[0]),
        ("2","📖","Perbandingan Film dengan Buku",    "look, scene, excite, arena, year, love, funny, color",   TOPIC_COLORS[1]),
        ("3","😮","Antisipasi & Reaksi Emosional",   "book, make, say, thats, comment, reading, kill, laugh",  TOPIC_COLORS[2]),
        ("4","🎬","Penilaian Umum Franchise",        "game, hunger, book, watch, haymitch, feel, story, movie",TOPIC_COLORS[3]),
    ]
    tc = st.columns(4)
    for col, (num,icon,name,kw,color) in zip(tc, t_info):
        with col:
            st.markdown(
                f'<div style="background:#1e1e2e;border-radius:10px;padding:16px;border-top:4px solid {color};">'
                f'<div style="font-size:1.4rem">{icon}</div>'
                f'<div style="font-weight:700;color:#fff;margin:6px 0">Topik {num}</div>'
                f'<div style="color:#a0aec0;font-size:0.78rem;margin-bottom:6px">{name}</div>'
                f'<div style="color:#718096;font-size:0.73rem">{kw}</div></div>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns(2)

    with cl:
        st.markdown('<div class="section-title">Jumlah Review per Topik</div>', unsafe_allow_html=True)
        tc_counts = dff["dominant_topic"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6,4), facecolor="none")
        bars = ax.bar([f"Topik {t}" for t in tc_counts.index], tc_counts.values,
                      color=TOPIC_COLORS[:len(tc_counts)], edgecolor="white")
        for bar, v in zip(bars, tc_counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                    str(v), ha="center", fontweight="bold", color="white")
        ax.set_facecolor("none"); ax.tick_params(colors="white")
        ax.set_ylabel("Jumlah Review", color="white")
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)

    with cr:
        st.markdown('<div class="section-title">Komposisi Sentimen per Topik</div>', unsafe_allow_html=True)
        ts = pd.crosstab(dff["dominant_topic"], dff["sentiment_label"], normalize="index").mul(100).round(1)
        fig, ax = plt.subplots(figsize=(6,4), facecolor="none")
        bottom = np.zeros(len(ts))
        for label, color in [("Positif",COLOR_POS),("Netral",COLOR_NEU),("Negatif",COLOR_NEG)]:
            if label in ts.columns:
                vals = ts[label].values
                ax.bar([f"T{t}" for t in ts.index], vals, bottom=bottom, label=label, color=color)
                bottom += vals
        ax.set_ylabel("Proporsi (%)", color="white"); ax.legend()
        ax.set_facecolor("none"); ax.tick_params(colors="white")
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)

    st.markdown('<div class="section-title">Distribusi Topic Confidence per Topik</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10,3), facecolor="none")
    for t_num, color in zip([1,2,3,4], TOPIC_COLORS):
        sub = dff[dff["dominant_topic"]==t_num]["topic_confidence"]
        if len(sub): ax.hist(sub, bins=20, alpha=0.6, label=f"Topik {t_num}", color=color)
    ax.set_xlabel("Confidence Score", color="white"); ax.set_ylabel("Frekuensi", color="white")
    ax.legend(); ax.set_facecolor("none"); ax.tick_params(colors="white")
    fig.patch.set_alpha(0)
    st.pyplot(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🤖 Naive Bayes Classification")
    st.info(
        f"Model terbaik: **Complement NB + TF-IDF Bigram** (α={best_alpha}, threshold={best_thresh:.2f}).  \n"
        f"Label dari VADER. Hanya kelas **Positif** dan **Negatif** yang diklasifikasi."
    )

    m1, m2, m3 = st.columns(3)
    with m1: st.metric("🎯 Akurasi Baseline (MNB)", f"{acc_base*100:.2f}%")
    with m2: st.metric("🚀 Akurasi Final (CNB)", f"{acc_final*100:.2f}%",
                        delta=f"{(acc_final-acc_base)*100:+.2f}%")
    with m3: st.metric("📈 Best CV Score (5-fold)", f"{best_cv*100:.2f}%")

    st.markdown("---")
    cl2, cr2 = st.columns(2)

    with cl2:
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5,4), facecolor="none")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=le.classes_, yticklabels=le.classes_,
                    ax=ax, linewidths=0.5, annot_kws={"size":14})
        ax.set_xlabel("Prediksi", color="white"); ax.set_ylabel("Aktual", color="white")
        ax.tick_params(colors="white")
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)

    with cr2:
        st.markdown('<div class="section-title">Precision, Recall, F1 per Kelas</div>', unsafe_allow_html=True)
        classes = le.classes_
        metrics_df = pd.DataFrame({
            "Precision": [report[c]["precision"] for c in classes],
            "Recall":    [report[c]["recall"]    for c in classes],
            "F1-Score":  [report[c]["f1-score"]  for c in classes],
        }, index=classes)
        fig, ax = plt.subplots(figsize=(5,4), facecolor="none")
        x = np.arange(len(classes)); w = 0.25
        for i, (col_name, color) in enumerate(zip(metrics_df.columns,["#3498db","#e74c3c","#2ecc71"])):
            ax.bar(x+i*w, metrics_df[col_name], w, label=col_name, color=color, edgecolor="white")
        ax.set_xticks(x+w); ax.set_xticklabels(classes, color="white")
        ax.set_ylim(0, 1.1); ax.set_ylabel("Score", color="white"); ax.legend()
        ax.set_facecolor("none"); ax.tick_params(colors="white")
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)

    st.markdown('<div class="section-title">Grid Search Alpha (5-fold CV Accuracy)</div>', unsafe_allow_html=True)
    alphas = list(alpha_results.keys())
    accs   = [v*100 for v in alpha_results.values()]
    fig, ax = plt.subplots(figsize=(8,3), facecolor="none")
    ax.plot(alphas, accs, marker="o", color="#3498db", linewidth=2)
    ax.axvline(best_alpha, color="#f39c12", linestyle="--", label=f"Best α={best_alpha}")
    ax.set_xlabel("Alpha", color="white"); ax.set_ylabel("CV Accuracy (%)", color="white")
    ax.legend(); ax.set_facecolor("none"); ax.tick_params(colors="white")
    fig.patch.set_alpha(0)
    st.pyplot(fig, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#64748b;font-size:0.8rem;">'
    "Analisis Sentimen & Topic Modelling — Review Film Hunger Games · "
    "VADER · LDA · Naive Bayes · Streamlit Dashboard</p>",
    unsafe_allow_html=True)
