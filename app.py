import streamlit as st
import pickle
import re
import numpy as np

# -----------------------
# Page Config (MUST be first Streamlit command)
# -----------------------
st.set_page_config(
    page_title="Fake News Detector | AI-Powered Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# Load Model
# -----------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# -----------------------
# Text Cleaning
# -----------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------
# Get top contributing words
# -----------------------
def get_top_words(text, vectorizer, model, top_n=10):
    """Get the words that most influenced the prediction."""
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    feature_names = vectorizer.get_feature_names_out()

    # Get coefficients for the positive class (Real=1)
    coefs = model.coef_[0]

    # Get non-zero feature indices from the input vector
    nonzero_indices = vector.nonzero()[1]

    if len(nonzero_indices) == 0:
        return [], []

    # Calculate word contributions: tfidf_value * coefficient
    word_scores = []
    for idx in nonzero_indices:
        word = feature_names[idx]
        tfidf_val = vector[0, idx]
        contribution = tfidf_val * coefs[idx]
        word_scores.append((word, contribution))

    # Sort by absolute contribution
    word_scores.sort(key=lambda x: abs(x[1]), reverse=True)

    # Split into fake-leaning and real-leaning
    fake_words = [(w, s) for w, s in word_scores if s < 0][:top_n]
    real_words = [(w, s) for w, s in word_scores if s > 0][:top_n]

    return fake_words, real_words

# -----------------------
# Premium CSS Styling
# -----------------------
st.markdown("""
<style>
    /* ===== Google Font Import ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ===== Root Variables ===== */
    :root {
        --bg-primary: #0a0e1a;
        --bg-secondary: #111827;
        --bg-card: rgba(17, 24, 39, 0.7);
        --bg-glass: rgba(255, 255, 255, 0.03);
        --border-glass: rgba(255, 255, 255, 0.08);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-blue: #3b82f6;
        --accent-cyan: #06b6d4;
        --accent-purple: #8b5cf6;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-amber: #f59e0b;
        --gradient-primary: linear-gradient(135deg, #3b82f6, #8b5cf6);
        --gradient-success: linear-gradient(135deg, #10b981, #06b6d4);
        --gradient-danger: linear-gradient(135deg, #ef4444, #f59e0b);
        --shadow-glow-blue: 0 0 40px rgba(59, 130, 246, 0.15);
        --shadow-glow-green: 0 0 40px rgba(16, 185, 129, 0.15);
        --shadow-glow-red: 0 0 40px rgba(239, 68, 68, 0.15);
    }

    /* ===== Global Styles ===== */
    .stApp {
        font-family: 'Inter', sans-serif !important;
    }
    
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }

    /* ===== Animations ===== */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }

    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(30px); }
        to { opacity: 1; transform: translateX(0); }
    }

    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.2); }
        50% { box-shadow: 0 0 40px rgba(59, 130, 246, 0.4); }
    }

    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes bounceIn {
        0% { transform: scale(0.3); opacity: 0; }
        50% { transform: scale(1.05); }
        70% { transform: scale(0.9); }
        100% { transform: scale(1); opacity: 1; }
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    @keyframes borderGlow {
        0%, 100% { border-color: rgba(59, 130, 246, 0.3); }
        50% { border-color: rgba(139, 92, 246, 0.6); }
    }

    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    /* ===== Hero Header ===== */
    .hero-header {
        text-align: center;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(139, 92, 246, 0.08), rgba(6, 182, 212, 0.08));
        border: 1px solid var(--border-glass);
        border-radius: 20px;
        backdrop-filter: blur(20px);
        animation: fadeInUp 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }

    .hero-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -50%;
        width: 200%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #3b82f6, #8b5cf6, #06b6d4, transparent);
        animation: shimmer 3s linear infinite;
    }

    .hero-icon {
        font-size: 3.5rem;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
        margin-bottom: 0.5rem;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 4s ease infinite;
        margin: 0.5rem 0;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        font-weight: 400;
        margin-top: 0.5rem;
        line-height: 1.6;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 6px 16px;
        border-radius: 50px;
        font-size: 0.8rem;
        color: #10b981;
        font-weight: 600;
        margin-top: 1rem;
        letter-spacing: 0.5px;
    }

    /* ===== Glass Card ===== */
    .glass-card {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        animation: fadeInUp 0.6s ease-out;
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.15);
        transform: translateY(-2px);
    }

    .card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ===== Stat Cards ===== */
    .stat-card {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.6s ease-out;
    }

    .stat-card:hover {
        transform: translateY(-5px);
        border-color: rgba(59, 130, 246, 0.3);
        box-shadow: var(--shadow-glow-blue);
    }

    .stat-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }

    .stat-value {
        font-size: 1.8rem;
        font-weight: 800;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .stat-label {
        font-size: 0.8rem;
        color: var(--text-muted);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.25rem;
    }

    /* ===== Result Cards ===== */
    .result-real {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(6, 182, 212, 0.05));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        animation: bounceIn 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        box-shadow: var(--shadow-glow-green);
    }

    .result-fake {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(245, 158, 11, 0.05));
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        animation: bounceIn 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        box-shadow: var(--shadow-glow-red);
    }

    .result-icon {
        font-size: 4rem;
        display: block;
        margin-bottom: 0.5rem;
        animation: bounceIn 0.8s ease-out;
    }

    .result-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    .result-title-real {
        color: #10b981;
    }

    .result-title-fake {
        color: #ef4444;
    }

    .result-desc {
        color: var(--text-secondary);
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* ===== Confidence Gauge ===== */
    .confidence-container {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        animation: fadeInUp 0.7s ease-out;
    }

    .confidence-value {
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: -2px;
    }

    .confidence-high { color: #10b981; }
    .confidence-medium { color: #f59e0b; }
    .confidence-low { color: #ef4444; }

    .confidence-bar-outer {
        width: 100%;
        height: 8px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        overflow: hidden;
        margin-top: 1rem;
    }

    .confidence-bar-inner {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-out;
    }

    .confidence-label {
        font-size: 0.8rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    /* ===== Word Tags ===== */
    .word-tag {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 3px;
        transition: all 0.2s ease;
        cursor: default;
    }

    .word-tag:hover {
        transform: scale(1.08);
    }

    .word-tag-fake {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.25);
        color: #fca5a5;
    }

    .word-tag-real {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.25);
        color: #6ee7b7;
    }

    /* ===== Instructions ===== */
    .instruction-step {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        padding: 1rem;
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 12px;
        margin-bottom: 0.75rem;
        transition: all 0.3s ease;
        animation: fadeInLeft 0.5s ease-out;
    }

    .instruction-step:hover {
        border-color: rgba(59, 130, 246, 0.3);
        transform: translateX(5px);
    }

    .step-number {
        width: 36px;
        height: 36px;
        min-width: 36px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 0.9rem;
        color: white;
        background: var(--gradient-primary);
    }

    .step-content h4 {
        margin: 0 0 4px 0;
        font-size: 0.95rem;
        color: var(--text-primary);
        font-weight: 600;
    }

    .step-content p {
        margin: 0;
        font-size: 0.85rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }

    /* ===== Sidebar Styling ===== */
    .sidebar-card {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }

    .sidebar-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.75rem;
    }

    .tech-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 3px;
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.2);
        color: #93c5fd;
    }

    /* ===== Section Divider ===== */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-glass), transparent);
        margin: 2rem 0;
    }

    /* ===== Textarea Styling ===== */
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 1px solid var(--border-glass) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
    }

    .stTextArea textarea:focus {
        border-color: rgba(59, 130, 246, 0.5) !important;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.1) !important;
    }

    /* ===== Button Styling ===== */
    .stButton > button {
        width: 100%;
        background: var(--gradient-primary) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 40px rgba(59, 130, 246, 0.3) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ===== Warning/Tips ===== */
    .tip-box {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-left: 3px solid #f59e0b;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.25rem;
        font-size: 0.85rem;
        color: var(--text-secondary);
        line-height: 1.6;
    }

    .tip-box strong {
        color: #f59e0b;
    }

    /* ===== Footer ===== */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        color: var(--text-muted);
        font-size: 0.8rem;
        border-top: 1px solid var(--border-glass);
        margin-top: 3rem;
    }

    .footer a {
        color: var(--accent-blue);
        text-decoration: none;
    }

    /* ===== Scrollbar ===== */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

    /* ===== Hide Streamlit defaults ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Hero Header
# -----------------------
st.markdown("""
<div class="hero-header">
    <div class="hero-icon">🛡️</div>
    <div class="hero-title">Fake News Detector</div>
    <div class="hero-subtitle">
        AI-Powered News Authenticity Analysis — Instantly verify if a news article is <strong>Real</strong> or <strong>Fabricated</strong>
    </div>
    <div class="hero-badge">
        <span>●</span> MODEL ACCURACY: 98.93%
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-title">🧠 Model Information</div>
        <table style="width:100%; font-size:0.85rem; color: #94a3b8;">
            <tr><td style="padding:6px 0; color:#64748b;">Algorithm</td><td style="text-align:right; font-weight:600;">Logistic Regression</td></tr>
            <tr><td style="padding:6px 0; color:#64748b;">Vectorizer</td><td style="text-align:right; font-weight:600;">TF-IDF (5000 features)</td></tr>
            <tr><td style="padding:6px 0; color:#64748b;">Accuracy</td><td style="text-align:right; font-weight:600; color:#10b981;">98.93%</td></tr>
            <tr><td style="padding:6px 0; color:#64748b;">Dataset Size</td><td style="text-align:right; font-weight:600;">44,898 articles</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-title">🏗️ Tech Stack</div>
        <div>
            <span class="tech-tag">Python</span>
            <span class="tech-tag">Scikit-Learn</span>
            <span class="tech-tag">Streamlit</span>
            <span class="tech-tag">NLP</span>
            <span class="tech-tag">TF-IDF</span>
            <span class="tech-tag">Regex</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-title">📊 Dataset Source</div>
        <p style="font-size:0.85rem; color:#94a3b8; line-height:1.5; margin:0;">
            Trained on the <strong style="color:#f1f5f9;">Fake & Real News Dataset</strong> containing
            23,481 fake articles and 21,417 real articles from various news sources.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-title">⚠️ Disclaimer</div>
        <p style="font-size:0.8rem; color:#94a3b8; line-height:1.5; margin:0;">
            This tool provides an AI-based prediction and should be used as a 
            <strong style="color:#f59e0b;">supplementary tool</strong>, not the sole basis 
            for determining news authenticity. Always cross-check with trusted sources.
        </p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------
# How to Use Section
# -----------------------
with st.expander("📖 How to Use This Tool", expanded=False):
    st.markdown("""
    <div class="instruction-step" style="animation-delay: 0.1s;">
        <div class="step-number">1</div>
        <div class="step-content">
            <h4>Paste the News Article</h4>
            <p>Copy the full text of the news article you want to analyze and paste it in the text area below. The more text you provide, the more accurate the result.</p>
        </div>
    </div>
    <div class="instruction-step" style="animation-delay: 0.2s;">
        <div class="step-number">2</div>
        <div class="step-content">
            <h4>Click "Analyze Article"</h4>
            <p>Hit the analyze button. The AI model will clean the text, convert it to numerical features using TF-IDF, and run it through the trained classifier.</p>
        </div>
    </div>
    <div class="instruction-step" style="animation-delay: 0.3s;">
        <div class="step-number">3</div>
        <div class="step-content">
            <h4>Review the Results</h4>
            <p>You'll see: <strong>Verdict</strong> (Real/Fake), <strong>Confidence Score</strong> (how certain the model is), and <strong>Key Words</strong> that influenced the decision.</p>
        </div>
    </div>
    <div class="tip-box" style="margin-top: 0.75rem;">
        <strong>💡 Pro Tips:</strong><br>
        • Paste the <strong>full article body</strong> — headlines alone are too short for reliable results.<br>
        • The model works best with <strong>English political news</strong> articles.<br>
        • A confidence below 70% means the model is uncertain — verify with other sources.<br>
        • URLs, special characters, and numbers are automatically removed during analysis.
    </div>
    """, unsafe_allow_html=True)

# -----------------------
# Input Section
# -----------------------
st.markdown("""
<div class="card-title" style="margin-top: 1rem;">
    <span>📝</span> PASTE NEWS ARTICLE FOR ANALYSIS
</div>
""", unsafe_allow_html=True)

news_text = st.text_area(
    "Paste the news article text here:",
    height=200,
    placeholder="Paste the full news article here...\n\nExample: 'WASHINGTON (Reuters) - The United States announced today that...'",
    label_visibility="collapsed"
)

# Character / word count
if news_text:
    word_count = len(news_text.split())
    char_count = len(news_text)
    quality = "🟢 Excellent" if word_count >= 50 else ("🟡 Moderate" if word_count >= 20 else "🔴 Too Short")
    st.markdown(f"""
    <div style="display:flex; gap:2rem; font-size:0.8rem; color:#64748b; padding: 0.25rem 0;">
        <span>📊 {word_count} words • {char_count} characters</span>
        <span>Quality: {quality}</span>
    </div>
    """, unsafe_allow_html=True)

# Analyze button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analyze = st.button("🔍  Analyze Article", use_container_width=True)

# -----------------------
# Prediction & Results
# -----------------------
if analyze:
    if news_text.strip() == "":
        st.warning("⚠️ Please paste a news article to analyze.")
    elif len(news_text.split()) < 5:
        st.warning("⚠️ The text is too short. Please paste at least a few sentences for reliable analysis.")
    else:
        with st.spinner("🔄 Analyzing article..."):
            import time
            time.sleep(0.8)  # Brief pause for visual feedback

            cleaned = clean_text(news_text)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            probability = model.predict_proba(vector)
            confidence = max(probability[0]) * 100
            fake_prob = probability[0][0] * 100
            real_prob = probability[0][1] * 100

            # Get top contributing words
            fake_words, real_words = get_top_words(news_text, vectorizer, model)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Result and confidence columns
        res_col1, res_col2 = st.columns([3, 2])

        with res_col1:
            if prediction == 1:
                st.markdown("""
                <div class="result-real">
                    <span class="result-icon">✅</span>
                    <div class="result-title result-title-real">REAL NEWS</div>
                    <div class="result-desc">
                        This article appears to be <strong>authentic and credible</strong>. 
                        The language patterns are consistent with legitimate news reporting.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-fake">
                    <span class="result-icon">🚨</span>
                    <div class="result-title result-title-fake">FAKE NEWS DETECTED</div>
                    <div class="result-desc">
                        This article shows <strong>characteristics of fabricated content</strong>. 
                        Please verify with trusted news sources before sharing.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            # Confidence level text and color
            if confidence >= 85:
                conf_class = "confidence-high"
                conf_label = "High Confidence"
                bar_color = "#10b981"
            elif confidence >= 70:
                conf_class = "confidence-medium"
                conf_label = "Moderate Confidence"
                bar_color = "#f59e0b"
            else:
                conf_class = "confidence-low"
                conf_label = "Low Confidence"
                bar_color = "#ef4444"

            st.markdown(f"""
            <div class="confidence-container">
                <div class="confidence-label">Model Confidence</div>
                <div class="confidence-value {conf_class}">{confidence:.1f}%</div>
                <div style="font-size:0.85rem; color:#94a3b8; margin-top:0.25rem;">{conf_label}</div>
                <div class="confidence-bar-outer">
                    <div class="confidence-bar-inner" style="width:{confidence}%; background:linear-gradient(90deg, {bar_color}, {bar_color}88);"></div>
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:1rem; font-size:0.8rem;">
                    <div>
                        <div style="color:#ef4444; font-weight:700;">{fake_prob:.1f}%</div>
                        <div style="color:#64748b;">Fake</div>
                    </div>
                    <div>
                        <div style="color:#10b981; font-weight:700;">{real_prob:.1f}%</div>
                        <div style="color:#64748b;">Real</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Word analysis section
        if fake_words or real_words:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="card-title">
                <span>🔬</span> KEY WORDS ANALYSIS — Words that influenced the prediction
            </div>
            """, unsafe_allow_html=True)

            wcol1, wcol2 = st.columns(2)

            with wcol1:
                if real_words:
                    st.markdown('<p style="font-size:0.85rem; color:#10b981; font-weight:600; margin-bottom:0.5rem;">✅ Words Suggesting Real News</p>', unsafe_allow_html=True)
                    tags = "".join([f'<span class="word-tag word-tag-real">{w}</span>' for w, s in real_words[:8]])
                    st.markdown(f'<div>{tags}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<p style="font-size:0.85rem; color:#64748b;">No strong real-news indicators found.</p>', unsafe_allow_html=True)

            with wcol2:
                if fake_words:
                    st.markdown('<p style="font-size:0.85rem; color:#ef4444; font-weight:600; margin-bottom:0.5rem;">🚩 Words Suggesting Fake News</p>', unsafe_allow_html=True)
                    tags = "".join([f'<span class="word-tag word-tag-fake">{w}</span>' for w, s in fake_words[:8]])
                    st.markdown(f'<div>{tags}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<p style="font-size:0.85rem; color:#64748b;">No strong fake-news indicators found.</p>', unsafe_allow_html=True)

# -----------------------
# Dataset Statistics
# -----------------------
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="card-title">
    <span>📊</span> TRAINING DATASET OVERVIEW
</div>
""", unsafe_allow_html=True)

scol1, scol2, scol3, scol4 = st.columns(4)

with scol1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-icon">📰</div>
        <div class="stat-value">44,898</div>
        <div class="stat-label">Total Articles</div>
    </div>
    """, unsafe_allow_html=True)

with scol2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-icon">🚫</div>
        <div class="stat-value">23,481</div>
        <div class="stat-label">Fake Articles</div>
    </div>
    """, unsafe_allow_html=True)

with scol3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-icon">✅</div>
        <div class="stat-value">21,417</div>
        <div class="stat-label">Real Articles</div>
    </div>
    """, unsafe_allow_html=True)

with scol4:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-icon">🎯</div>
        <div class="stat-value">98.93%</div>
        <div class="stat-label">Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------
# Footer
# -----------------------
st.markdown("""
<div class="footer">
    <p>🛡️ <strong>Fake News Detector</strong> — AI-Powered News Authenticity Analysis</p>
    <p>Built with Scikit-Learn, TF-IDF Vectorization & Streamlit</p>
    <p style="margin-top:0.5rem; font-size:0.75rem;">
        © 2026 — For educational and research purposes only
    </p>
</div>
""", unsafe_allow_html=True)