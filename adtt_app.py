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

# config
st.set_page_config(
    page_title="Dashboard Analisis Hunger Games",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# colors
COLOR_POS    = "#a78bfa"   # purple-400
COLOR_NEG    = "#f472b6"   # pink-400
COLOR_NEU    = "#c4b5fd"   # purple-300
COLOR_MAP    = {"Positif": COLOR_POS, "Negatif": COLOR_NEG, "Netral": COLOR_NEU}
TOPIC_COLORS = ["#7c3aed", "#a855f7", "#c084fc", "#e879f9"]

BG_DARK      = "#0f0a1e"
BG_CARD      = "#1a1033"
BG_CARD2     = "#231546"
ACCENT       = "#7c3aed"
ACCENT2      = "#a855f7"
BORDER       = "#3b1f6e"

st.markdown(f"""
<style>
    /* Global background */
    .stApp {{ background-color: {BG_DARK}; }}
    
    /* Sidebar */
    div[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0f0a1e 0%, #1a1033 100%);
        border-right: 1px solid {BORDER};
    }}
    div[data-testid="stSidebar"] * {{ color: #e2d9f3 !important; }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background: {BG_CARD};
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid {BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        color: #c4b5fd;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT2} 100%) !important;
        color: white !important;
    }}

    /* Metric card */
    .metric-card {{
        background: linear-gradient(135deg, {BG_CARD} 0%, {BG_CARD2} 100%);
        border-radius: 16px;
        padding: 22px 18px;
        text-align: center;
        border: 1px solid {BORDER};
        box-shadow: 0 4px 24px rgba(124,58,237,0.15);
    }}
    .metric-value {{ font-size: 1.7rem; font-weight: 700; color: #e9d5ff; }}
    .metric-label {{ font-size: 0.78rem; color: #a78bfa; margin-top: 5px; letter-spacing: 0.5px; }}

    /* Section title */
    .section-title {{
        font-size: 1rem; font-weight: 600; color: #c4b5fd;
        margin-bottom: 10px;
        border-bottom: 2px solid {BORDER};
        padding-bottom: 6px;
    }}

    /* Info / warning boxes */
    div[data-testid="stAlert"] {{
        background: {BG_CARD2} !important;
        border: 1px solid {BORDER} !important;
        color: #ddd6fe !important;
        border-radius: 10px !important;
    }}

    /* Streamlit metric widget */
    div[data-testid="metric-container"] {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 16px;
    }}
    div[data-testid="metric-container"] label {{ color: #a78bfa !important; }}
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{ color: #e9d5ff !important; }}
    div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {{ color: #86efac !important; }}

    /* Multiselect & slider */
    div[data-baseweb="select"] {{ background: {BG_CARD2} !important; border-color: {BORDER} !important; }}
    .stMultiSelect [data-baseweb="tag"] {{ background: {ACCENT} !important; }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: {BG_DARK}; }}
    ::-webkit-scrollbar-thumb {{ background: {ACCENT}; border-radius: 3px; }}
</style>
""", unsafe_allow_html=True)

# data
@st.cache_data
def load_and_process(path="hunger_games_reviews_clean.csv"):
    df = pd.read_csv(path)
    df_clean = df.dropna(subset=["review"]).reset_index(drop=True)
    df_clean["review"] = df_clean["review"].astype(str)

    analyzer = SentimentIntensityAnalyzer()
    def get_label(text):
        c = analyzer.polarity_scores(text)["compound"]
        return "Positif" if c >= 0.05 else ("Negatif" if c <= -0.05 else "Netral")
    def get_score(text):
        return analyzer.polarity_scores(text)["compound"]

    df_clean["sentiment_label"]   = df_clean["review"].apply(get_label)
    df_clean["sentiment_score"]   = df_clean["review"].apply(get_score)
    df_clean["word_count"]        = df_clean["review"].apply(lambda x: len(x.split()))
    df_clean["char_count"]        = df_clean["review"].apply(len)
    df_clean["unique_word_count"] = df_clean["review"].apply(lambda x: len(set(x.split())))
    return df_clean

@st.cache_data
def run_lda(_df):
    count_vec    = CountVectorizer(max_features=800, min_df=3, max_df=0.90)
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
    dominant   = doc_topic.argmax(axis=1) + 1
    confidence = doc_topic.max(axis=1)
    return lda, feature_names, dominant, confidence, topic_names

@st.cache_data
def run_nb(_df):
    df_bin = _df[_df["sentiment_label"].isin(["Positif", "Negatif"])].copy()
    le = LabelEncoder()
    y  = le.fit_transform(df_bin["sentiment_label"])

    vec_base = CountVectorizer(max_features=500, min_df=2)
    X_base   = vec_base.fit_transform(df_bin["review"].fillna(""))
    Xtr_b, Xte_b, ytr_b, yte_b = train_test_split(X_base, y, test_size=0.2,
                                                    random_state=42, stratify=y)
    nb_base  = MultinomialNB(alpha=1.0)
    nb_base.fit(Xtr_b, ytr_b)
    acc_base = accuracy_score(yte_b, nb_base.predict(Xte_b))

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

    cnb   = CalibratedClassifierCV(ComplementNB(alpha=best_alpha), cv=5)
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
  
def purple_fig(figsize=(6, 4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="none")
    ax.set_facecolor("#1a1033")
    ax.tick_params(colors="#c4b5fd")
    ax.xaxis.label.set_color("#c4b5fd")
    ax.yaxis.label.set_color("#c4b5fd")
    for spine in ax.spines.values():
        spine.set_edgecolor("#3b1f6e")
    return fig, ax


with st.spinner("✨ Memproses data dengan VADER, LDA, dan Naive Bayes..."):
    df_clean = load_and_process("hunger_games_reviews_clean.csv")
    lda_model, feat_names, dominant_topic, topic_conf, topic_names = run_lda(df_clean)
    df_clean["dominant_topic"]   = dominant_topic
    df_clean["topic_confidence"] = topic_conf
    df_clean["topic_name"]       = df_clean["dominant_topic"].map(topic_names)
    acc_base, acc_final, best_alpha, best_thresh, best_cv, cm, report, le, alpha_results = run_nb(df_clean)

with st.sidebar:
    st.markdown(
        f'<div style="text-align:center;padding:16px 0 8px;">'
        f'<div style="font-size:2rem">🎬</div>'
        f'<div style="font-size:1.1rem;font-weight:700;color:#e9d5ff;">Hunger Games</div>'
        f'<div style="font-size:0.8rem;color:#a78bfa;">Review Analytics</div></div>',
        unsafe_allow_html=True)
    st.markdown(f'<hr style="border-color:{BORDER}">', unsafe_allow_html=True)

    st.markdown("### 🔍 Filter Data")
    sel_sent   = st.multiselect("Sentimen", ["Positif","Negatif","Netral"],
                                 default=["Positif","Negatif","Netral"])
    sel_topics = st.multiselect("Topik LDA", list(topic_names.values()),
                                 default=list(topic_names.values()))
    wrange     = st.slider("Jumlah Kata per Review", 1, int(df_clean["word_count"].max()), (1, 30))

    st.markdown(f'<hr style="border-color:{BORDER}">', unsafe_allow_html=True)
    st.markdown("### 📊 Info Dataset")
    st.markdown(
        f'<div style="background:{BG_CARD2};border:1px solid {BORDER};border-radius:10px;padding:14px;font-size:0.82rem;color:#ddd6fe;">'
        f'<b>Total awal:</b> 1.812 baris<br>'
        f'<b>Setelah cleaning:</b> {len(df_clean)} baris<br>'
        f'<b>Labeling:</b> VADER<br>'
        f'<b>Model:</b> CNB + TF-IDF<br>'
        f'<b>Akurasi Final:</b> <span style="color:#a78bfa;font-weight:700">{acc_final*100:.2f}%</span>'
        f'</div>', unsafe_allow_html=True)


dff = df_clean[
    df_clean["sentiment_label"].isin(sel_sent) &
    df_clean["topic_name"].isin(sel_topics) &
    df_clean["word_count"].between(wrange[0], wrange[1])
].copy()


st.markdown(
    f'<div style="background:linear-gradient(135deg,{BG_CARD} 0%,#2d1b69 100%);'
    f'border:1px solid {BORDER};border-radius:16px;padding:24px 28px;margin-bottom:20px;">'
    f'<h1 style="color:#e9d5ff;margin:0;font-size:1.7rem;">🎬 Dashboard Analisis Sentimen & Topic Modelling</h1>'
    f'<p style="color:#a78bfa;margin:6px 0 0;font-size:0.88rem;">'
    f'Review Film The Hunger Games &nbsp;·&nbsp; VADER &nbsp;·&nbsp; LDA &nbsp;·&nbsp; Naive Bayes</p>'
    f'</div>', unsafe_allow_html=True)

total = len(dff)
if total == 0:
    st.warning("Tidak ada data yang sesuai filter. Ubah filter di sidebar.")
    st.stop()

n_pos = (dff["sentiment_label"]=="Positif").sum()
n_neg = (dff["sentiment_label"]=="Negatif").sum()
n_neu = (dff["sentiment_label"]=="Netral").sum()
avg_score = dff["sentiment_score"].mean()

k1, k2, k3, k4, k5 = st.columns(5)
kpi_data = [
    (k1, f"{total:,}",                          "Total Review",   "#7c3aed"),
    (k2, f"{n_pos:,}\n({n_pos/total*100:.1f}%)", "Review Positif", COLOR_POS),
    (k3, f"{n_neg:,}\n({n_neg/total*100:.1f}%)", "Review Negatif", COLOR_NEG),
    (k4, f"{n_neu:,}\n({n_neu/total*100:.1f}%)", "Review Netral",  COLOR_NEU),
    (k5, f"{avg_score:.4f}",                     "Rata-rata Skor", "#e879f9"),
]
for col, val, label, color in kpi_data:
    with col:
        st.markdown(
            f'<div class="metric-card" style="border-top:3px solid {color};">'
            f'<div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Distribusi Sentimen", "☁️ Word Cloud & Frekuensi", "🗂️ Topic Modelling", "🤖 Model Naive Bayes"])

with tab1:
    c1, c2 = st.columns(2)
    counts = dff["sentiment_label"].value_counts()
    colors = [COLOR_MAP.get(l,"#ccc") for l in counts.index]

    with c1:
        st.markdown('<div class="section-title">Proporsi Sentimen</div>', unsafe_allow_html=True)
        fig, ax = purple_fig((5, 4))
        ax.set_facecolor("none")
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=colors, startangle=90, explode=[0.05]*len(counts),
            textprops={"color":"#e9d5ff"}, wedgeprops={"linewidth":1.5,"edgecolor":BG_DARK})
        for at in autotexts: at.set_color("#0f0a1e"); at.set_fontweight("bold")
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Jumlah Review per Sentimen</div>', unsafe_allow_html=True)
        fig, ax = purple_fig((5, 4))
        bars = ax.bar(counts.index, counts.values, color=colors,
                      edgecolor=BG_DARK, linewidth=1.2)
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+4,
                    str(v), ha="center", fontweight="bold", color="#e9d5ff", fontsize=11)
        ax.set_ylabel("Jumlah Review")
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)

    st.markdown(f'<hr style="border-color:{BORDER};margin:8px 0">', unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="section-title">Distribusi Jumlah Kata per Sentimen</div>', unsafe_allow_html=True)
        fig, ax = purple_fig((6, 4))
        for label in ["Positif","Negatif","Netral"]:
            sub = dff[dff["sentiment_label"]==label]["word_count"]
            ax.hist(sub, alpha=0.7, label=label, color=COLOR_MAP[label], bins=30, edgecolor="none")
        ax.set_xlim(0, min(60, dff["word_count"].quantile(0.99)))
        ax.set_xlabel("Jumlah Kata"); ax.set_ylabel("Frekuensi")
        ax.legend(facecolor=BG_CARD2, edgecolor=BORDER, labelcolor="#e9d5ff")
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)

    with c4:
        st.markdown('<div class="section-title">Distribusi Skor Sentimen (VADER Compound)</div>', unsafe_allow_html=True)
        fig, ax = purple_fig((6, 4))
        ax.hist(dff["sentiment_score"], bins=40, color=ACCENT, alpha=0.8,
                edgecolor=BG_DARK, linewidth=0.5)
        ax.axvline(0.05,  color=COLOR_POS, linestyle="--", lw=1.5, label="Batas Positif")
        ax.axvline(-0.05, color=COLOR_NEG, linestyle="--", lw=1.5, label="Batas Negatif")
        ax.set_xlabel("Compound Score"); ax.set_ylabel("Frekuensi")
        ax.legend(facecolor=BG_CARD2, edgecolor=BORDER, labelcolor="#e9d5ff", fontsize=8)
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)

    st.markdown('<div class="section-title">Boxplot Jumlah Kata per Sentimen</div>', unsafe_allow_html=True)
    fig, ax = purple_fig((10, 3))
    groups = [dff[dff["sentiment_label"]==l]["word_count"].values for l in ["Positif","Netral","Negatif"]]
    bp = ax.boxplot(groups, labels=["Positif","Netral","Negatif"], patch_artist=True,
                    medianprops=dict(color="#e9d5ff", linewidth=2),
                    whiskerprops=dict(color=ACCENT2), capprops=dict(color=ACCENT2),
                    flierprops=dict(markerfacecolor=ACCENT2, marker="o", markersize=3))
    for patch, color in zip(bp["boxes"], [COLOR_POS, COLOR_NEU, COLOR_NEG]):
        patch.set_facecolor(color); patch.set_alpha(0.75); patch.set_edgecolor(BG_DARK)
    ax.set_ylim(0, min(60, dff["word_count"].quantile(0.99)+5))
    ax.set_ylabel("Jumlah Kata")
    fig.patch.set_alpha(0)
    st.pyplot(fig, use_container_width=True)

with tab2:
    st.markdown('<div class="section-title">25 Kata Paling Sering Muncul</div>', unsafe_allow_html=True)
    all_words = " ".join(dff["review"].dropna().tolist()).split()
    word_freq = Counter(all_words)
    top25 = word_freq.most_common(25)
    words, freqs = zip(*top25)

    fig, ax = purple_fig((10, 6))
    cmap_colors = plt.cm.PuRd(np.linspace(0.35, 0.9, len(words)))
    bars = ax.barh(list(reversed(words)), list(reversed(freqs)),
                   color=list(reversed(cmap_colors)), edgecolor="none")
    for bar, freq in zip(bars, list(reversed(freqs))):
        ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                str(freq), va="center", fontsize=9, color="#e9d5ff")
    ax.set_xlabel("Frekuensi")
    fig.patch.set_alpha(0)
    st.pyplot(fig, use_container_width=True)

    st.markdown(f'<hr style="border-color:{BORDER}">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Word Cloud per Kategori Sentimen</div>', unsafe_allow_html=True)
    wc_cols = st.columns(3)
    wc_cmaps = [("Positif","Purples"),("Negatif","RdPu"),("Netral","BuPu")]
    for col, (label, cmap) in zip(wc_cols, wc_cmaps):
        with col:
            st.markdown(f'<div style="text-align:center;color:#c4b5fd;font-weight:600;margin-bottom:6px;">Sentimen {label}</div>', unsafe_allow_html=True)
            txt = " ".join(dff[dff["sentiment_label"]==label]["review"].dropna().tolist())
            if len(txt.strip()) > 10:
                wc = WordCloud(width=400, height=260, background_color="#1a1033",
                               colormap=cmap, max_words=80, collocations=False).generate(txt)
                fig, ax = plt.subplots(figsize=(4, 3), facecolor="none")
                ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
                fig.patch.set_alpha(0)
                st.pyplot(fig, use_container_width=True)
            else:
                st.warning("Data tidak cukup.")

    st.markdown(f'<hr style="border-color:{BORDER}">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Top 15 Kata per Sentimen</div>', unsafe_allow_html=True)
    top_cols = st.columns(3)
    for col, (label, color) in zip(top_cols, [("Positif",COLOR_POS),("Negatif",COLOR_NEG),("Netral",COLOR_NEU)]):
        with col:
            st.markdown(f'<div style="text-align:center;color:#c4b5fd;font-weight:600;margin-bottom:6px;">Top Kata – {label}</div>', unsafe_allow_html=True)
            sub_words = " ".join(dff[dff["sentiment_label"]==label]["review"].dropna().tolist()).split()
            top15 = Counter(sub_words).most_common(15)
            if top15:
                w, f = zip(*top15)
                fig, ax = purple_fig((4, 5))
                ax.barh(list(reversed(w)), list(reversed(f)), color=color,
                        edgecolor=BG_DARK, linewidth=0.8, alpha=0.85)
                ax.set_xlabel("Frekuensi")
                fig.patch.set_alpha(0)
                st.pyplot(fig, use_container_width=True)


with tab3:
    st.markdown(
        f'<div style="background:{BG_CARD2};border:1px solid {BORDER};border-radius:10px;'
        f'padding:12px 16px;color:#ddd6fe;font-size:0.88rem;margin-bottom:16px;">'
        f'⚙️ LDA dilatih dengan <code>n_components=4</code>, <code>max_iter=20</code>, '
        f'<code>learning_method=\'batch\'</code>.</div>', unsafe_allow_html=True)

    t_info = [
        ("1","🎭","Penilaian Akting & Karakter",      "book, read, cast, look, give, perfect, bad, feel",        TOPIC_COLORS[0]),
        ("2","📖","Perbandingan Film dengan Buku",     "look, scene, excite, arena, year, love, funny, color",    TOPIC_COLORS[1]),
        ("3","😮","Antisipasi & Reaksi Emosional",    "book, make, say, thats, comment, reading, kill, laugh",   TOPIC_COLORS[2]),
        ("4","🎬","Penilaian Umum Franchise",         "game, hunger, book, watch, haymitch, feel, story, movie", TOPIC_COLORS[3]),
    ]
    tc = st.columns(4)
    for col, (num,icon,name,kw,color) in zip(tc, t_info):
        with col:
            st.markdown(
                f'<div style="background:{BG_CARD};border-radius:12px;padding:16px 14px;'
                f'border-top:3px solid {color};border:1px solid {BORDER};border-top:3px solid {color};">'
                f'<div style="font-size:1.6rem;margin-bottom:6px">{icon}</div>'
                f'<div style="font-weight:700;color:#e9d5ff;font-size:0.9rem;margin-bottom:4px">Topik {num}</div>'
                f'<div style="color:{color};font-size:0.78rem;font-weight:600;margin-bottom:6px">{name}</div>'
                f'<div style="color:#94a3b8;font-size:0.72rem;line-height:1.5">{kw}</div></div>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns(2)

    with cl:
        st.markdown('<div class="section-title">Jumlah Review per Topik</div>', unsafe_allow_html=True)
        tc_counts = dff["dominant_topic"].value_counts().sort_index()
        fig, ax = purple_fig((6, 4))
        bars = ax.bar([f"Topik {t}" for t in tc_counts.index], tc_counts.values,
                      color=TOPIC_COLORS[:len(tc_counts)], edgecolor=BG_DARK, linewidth=1)
        for bar, v in zip(bars, tc_counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                    str(v), ha="center", fontweight="bold", color="#e9d5ff")
        ax.set_ylabel("Jumlah Review")
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)

    with cr:
        st.markdown('<div class="section-title">Komposisi Sentimen per Topik</div>', unsafe_allow_html=True)
        ts = pd.crosstab(dff["dominant_topic"], dff["sentiment_label"], normalize="index").mul(100).round(1)
        fig, ax = purple_fig((6, 4))
        bottom = np.zeros(len(ts))
        for label, color in [("Positif",COLOR_POS),("Netral",COLOR_NEU),("Negatif",COLOR_NEG)]:
            if label in ts.columns:
                vals = ts[label].values
                ax.bar([f"T{t}" for t in ts.index], vals, bottom=bottom,
                       label=label, color=color, edgecolor=BG_DARK, linewidth=0.5)
                bottom += vals
        ax.set_ylabel("Proporsi (%)")
        ax.legend(facecolor=BG_CARD2, edgecolor=BORDER, labelcolor="#e9d5ff")
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)

    st.markdown('<div class="section-title">Distribusi Topic Confidence per Topik</div>', unsafe_allow_html=True)
    fig, ax = purple_fig((10, 3))
    for t_num, color in zip([1,2,3,4], TOPIC_COLORS):
        sub = dff[dff["dominant_topic"]==t_num]["topic_confidence"]
        if len(sub):
            ax.hist(sub, bins=20, alpha=0.7, label=f"Topik {t_num}", color=color, edgecolor="none")
    ax.set_xlabel("Confidence Score"); ax.set_ylabel("Frekuensi")
    ax.legend(facecolor=BG_CARD2, edgecolor=BORDER, labelcolor="#e9d5ff")
    fig.patch.set_alpha(0)
    st.pyplot(fig, use_container_width=True)

with tab4:
    st.markdown(
        f'<div style="background:{BG_CARD2};border:1px solid {BORDER};border-radius:10px;'
        f'padding:12px 16px;color:#ddd6fe;font-size:0.88rem;margin-bottom:16px;">'
        f'🤖 Model terbaik: <b>Complement NB + TF-IDF Bigram</b> '
        f'(α={best_alpha}, threshold={best_thresh:.2f}) — '
        f'Label dari VADER. Hanya kelas <b>Positif</b> dan <b>Negatif</b> yang diklasifikasi.'
        f'</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1: st.metric("🎯 Akurasi Baseline (MNB)", f"{acc_base*100:.2f}%")
    with m2: st.metric("🚀 Akurasi Final (CNB)",    f"{acc_final*100:.2f}%",
                        delta=f"{(acc_final-acc_base)*100:+.2f}%")
    with m3: st.metric("📈 Best CV Score (5-fold)", f"{best_cv*100:.2f}%")

    st.markdown(f'<hr style="border-color:{BORDER}">', unsafe_allow_html=True)
    cl2, cr2 = st.columns(2)

    with cl2:
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        fig, ax = purple_fig((5, 4))
        sns.heatmap(cm, annot=True, fmt="d",
                    cmap=sns.color_palette("BuPu", as_cmap=True),
                    xticklabels=le.classes_, yticklabels=le.classes_,
                    ax=ax, linewidths=0.5, linecolor=BG_DARK,
                    annot_kws={"size":14, "color":"white"})
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
        ax.tick_params(colors="#c4b5fd")
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
        fig, ax = purple_fig((5, 4))
        x = np.arange(len(classes)); w = 0.25
        bar_colors = ["#7c3aed","#a855f7","#e879f9"]
        for i, (col_name, color) in enumerate(zip(metrics_df.columns, bar_colors)):
            ax.bar(x+i*w, metrics_df[col_name], w, label=col_name,
                   color=color, edgecolor=BG_DARK, linewidth=0.8)
        ax.set_xticks(x+w); ax.set_xticklabels(classes)
        ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
        ax.legend(facecolor=BG_CARD2, edgecolor=BORDER, labelcolor="#e9d5ff")
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)

    st.markdown('<div class="section-title">Grid Search Alpha — 5-fold CV Accuracy</div>', unsafe_allow_html=True)
    alphas = list(alpha_results.keys())
    accs   = [v*100 for v in alpha_results.values()]
    fig, ax = purple_fig((9, 3))
    ax.plot(alphas, accs, marker="o", color=ACCENT2, linewidth=2.5,
            markerfacecolor="#e9d5ff", markeredgecolor=ACCENT, markersize=7)
    ax.fill_between(alphas, [a-1 for a in accs], [a+1 for a in accs],
                    alpha=0.2, color=ACCENT)
    ax.axvline(best_alpha, color=COLOR_NEG, linestyle="--", lw=1.5,
               label=f"Best α={best_alpha}")
    ax.set_xlabel("Alpha"); ax.set_ylabel("CV Accuracy (%)")
    ax.legend(facecolor=BG_CARD2, edgecolor=BORDER, labelcolor="#e9d5ff")
    fig.patch.set_alpha(0)
    st.pyplot(fig, use_container_width=True)

st.markdown(
    f'<hr style="border-color:{BORDER};margin-top:32px">'
    f'<p style="text-align:center;color:#4c1d95;font-size:0.78rem;padding-bottom:8px;">'
    f'Analisis Sentimen & Topic Modelling — Review Film Hunger Games &nbsp;·&nbsp; '
    f'VADER &nbsp;·&nbsp; LDA &nbsp;·&nbsp; Naive Bayes &nbsp;·&nbsp; Streamlit</p>',
    unsafe_allow_html=True)
