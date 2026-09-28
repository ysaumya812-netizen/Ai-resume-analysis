import streamlit as st
import matplotlib.pyplot as plt
import PyPDF2
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ResumeAI",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# NLTK
# ============================================================

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)


# ============================================================
# CUSTOM CSS  (UI only)
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #0F172A;
        --ink-soft: #475569;
        --muted: #64748B;
        --line: #E2E8F0;
        --bg: #F6F8FB;
        --card: #FFFFFF;
        --accent: #0D9488;
        --accent-dark: #0F766E;
        --accent-soft: #CCFBF1;
    }

    /* ---------- Base ---------- */
    html, body, .stApp, .stApp * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* keep Streamlit icon fonts working */
    .stApp [data-testid="stIconMaterial"],
    .stApp .material-icons,
    .stApp span[class*="material"] {
        font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    }

    .stApp {
        background-color: var(--bg);
        color: var(--ink);
    }

    .block-container {
        max-width: 1120px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    header, #MainMenu, footer {
        visibility: hidden;
    }

    h1, h2, h3, h4 {
        color: var(--ink);
        letter-spacing: -0.01em;
    }

    p, li, label, span {
        color: var(--ink);
    }

    /* ---------- Hero ---------- */
    .hero {
        background: var(--ink);
        border-radius: 20px;
        padding: 44px 44px 40px 44px;
        margin-bottom: 26px;
        position: relative;
        overflow: hidden;
    }

    .hero::after {
        content: "";
        position: absolute;
        right: -80px;
        top: -80px;
        width: 320px;
        height: 320px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(13,148,136,0.55) 0%, rgba(13,148,136,0) 70%);
    }

    .hero-title {
        font-size: 40px;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #FFFFFF;
        line-height: 1.1;
        margin-bottom: 10px;
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #CBD5E1;
        max-width: 560px;
        line-height: 1.55;
        position: relative;
        z-index: 1;
    }

    .hero-steps {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 24px;
        position: relative;
        z-index: 1;
    }

    .hero-step {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.14);
        color: #E2E8F0;
        padding: 7px 14px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
    }

    .intro-text {
        color: var(--ink-soft);
        font-size: 15px;
        margin-bottom: 22px;
    }

    /* ---------- Cards (bordered containers) ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--card);
        border: 1px solid var(--line) !important;
        border-radius: 16px !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        padding: 8px 10px;
    }

    .card-title {
        font-size: 18px;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 2px;
    }

    .card-description {
        color: var(--muted);
        font-size: 14px;
        margin-bottom: 8px;
    }

    /* ---------- File uploader ---------- */
    div[data-testid="stFileUploader"] section,
    div[data-testid="stFileUploaderDropzone"] {
        background: #F8FAFC;
        border: 1.5px dashed #94A3B8;
        border-radius: 12px;
    }

    div[data-testid="stFileUploader"] section:hover,
    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent);
        background: #F0FDFA;
    }

    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] span {
        color: var(--ink-soft);
    }

    div[data-testid="stFileUploader"] button {
        background: #FFFFFF;
        color: var(--ink);
        border: 1px solid var(--line);
        border-radius: 8px;
        font-weight: 600;
    }

    /* ---------- Text area ---------- */
    div[data-testid="stTextArea"] textarea,
    textarea {
        color: var(--ink) !important;
        background-color: #F8FAFC !important;
        border: 1px solid var(--line) !important;
        border-radius: 12px !important;
        font-size: 14.5px !important;
        line-height: 1.55 !important;
    }

    div[data-testid="stTextArea"] textarea:focus,
    textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(13,148,136,0.15) !important;
        background-color: #FFFFFF !important;
    }

    /* ---------- Analyze button ---------- */
    div.stButton > button {
        width: 100%;
        height: 54px;
        border-radius: 12px;
        border: none;
        background: var(--accent);
        color: #FFFFFF;
        font-size: 17px;
        font-weight: 700;
        letter-spacing: 0.01em;
        transition: background 0.15s ease, box-shadow 0.15s ease;
    }

    div.stButton > button:hover {
        background: var(--accent-dark);
        color: #FFFFFF;
        box-shadow: 0 8px 20px rgba(13,148,136,0.28);
    }

    div.stButton > button:focus-visible {
        outline: 3px solid rgba(13,148,136,0.4);
        outline-offset: 2px;
    }

    div.stButton > button p {
        color: #FFFFFF;
    }

    /* ---------- Section title ---------- */
    .section-title {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: var(--ink);
        margin-top: 34px;
        margin-bottom: 16px;
        padding-left: 14px;
        border-left: 4px solid var(--accent);
    }

    /* ---------- Metrics ---------- */
    div[data-testid="stMetric"] {
        background: var(--card);
        padding: 20px 22px;
        border-radius: 16px;
        border: 1px solid var(--line);
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    div[data-testid="stMetricLabel"] p {
        color: var(--muted);
        font-size: 14px;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: var(--ink);
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    /* ---------- Progress bar ---------- */
    div[data-testid="stProgress"] > div > div {
        background-color: #E2E8F0;
        border-radius: 999px;
    }

    div[data-testid="stProgress"] > div > div > div {
        background: var(--accent) !important;
        border-radius: 999px;
    }

    /* ---------- Alerts ---------- */
    div[data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid var(--line);
    }

    /* ---------- AI result ---------- */
    .ai-note {
        color: var(--muted);
        font-size: 13px;
        margin-bottom: 4px;
    }

    .ai-box h3 {
        font-size: 18px;
        font-weight: 700;
        margin-top: 22px;
        padding-bottom: 6px;
        border-bottom: 1px solid var(--line);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] h3 {
        font-size: 18px;
        font-weight: 700;
        margin-top: 18px;
        color: var(--ink);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] li {
        line-height: 1.6;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid var(--line);
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--ink);
    }

    section[data-testid="stSidebar"] hr {
        border-color: var(--line);
    }

    section[data-testid="stSidebar"] div[data-testid="stAlert"] {
        background: #F0FDFA;
        border: 1px solid #99F6E4;
    }

    /* ---------- Footer ---------- */
    .custom-footer {
        text-align: center;
        color: var(--muted);
        font-size: 14px;
        padding-top: 26px;
        margin-top: 44px;
        border-top: 1px solid var(--line);
        line-height: 1.7;
    }

    .custom-footer b {
        color: var(--ink);
    }

    /* ---------- Mobile ---------- */
    @media (max-width: 720px) {
        .hero { padding: 28px 22px; }
        .hero-title { font-size: 30px; }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 📄 ResumeAI")

    st.markdown("---")

    st.markdown("### ✨ About")

    st.info(
        """
        This tool helps you:

        • Measure resume-job similarity  
        • Identify matching skills  
        • Find missing skills  
        • Get AI suggestions  
        • Improve your resume
        """
    )

    st.markdown("### ⚙️ How it works")

    st.write(
        """
        1. Upload your resume PDF

        2. Paste the job description

        3. Click **Analyze Resume**

        4. Get your match score

        5. Get AI-powered suggestions
        """
    )

    st.markdown("---")

    st.markdown("### 🛠 Technologies")

    st.write(
        """
        🐍 Python

        🎨 Streamlit

        🧠 NLP

        📊 TF-IDF

        🔍 Cosine Similarity

        🤖 Ollama / Gemma
        """
    )


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    '<div class="hero">'
    '<div class="hero-title">📄 ResumeAI</div>'
    '<div class="hero-subtitle">AI-Powered Resume &amp; Job Match Analyzer</div>'
    '<div class="hero-steps">'
    '<span class="hero-step">Upload resume</span>'
    '<span class="hero-step">Paste job description</span>'
    '<span class="hero-step">Get match score</span>'
    '<span class="hero-step">Get AI suggestions</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="intro-text">'
    "Analyze how closely your resume matches a job description "
    "and discover ways to improve your application."
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# INPUT SECTION
# ============================================================

col1, col2 = st.columns(2, gap="large")


with col1:

    with st.container(border=True):

        st.markdown(
            """
            <div class="card-title">
                📄 Upload Your Resume
            </div>

            <div class="card-description">
                Upload your resume in PDF format.
            </div>
            """,
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Upload Resume",
            type=["pdf"],
            label_visibility="collapsed"
        )


with col2:

    with st.container(border=True):

        st.markdown(
            """
            <div class="card-title">
                💼 Job Description
            </div>

            <div class="card-description">
                Paste the job description below.
            </div>
            """,
            unsafe_allow_html=True
        )

        job_description = st.text_area(
            "Job Description",
            height=210,
            placeholder=(
                "Example:\n\n"
                "We are looking for a Python developer "
                "with experience in Machine Learning, "
                "SQL, Git and Streamlit..."
            ),
            label_visibility="collapsed"
        )


st.write("")


# ============================================================
# PDF FUNCTION
# ============================================================

def extract_text_from_pdf(uploaded_file):

    try:

        pdf_reader = PyPDF2.PdfReader(uploaded_file)

        text = ""

        for page in pdf_reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

        return text

    except Exception as e:

        st.error(
            f"Error reading PDF: {e}"
        )

        return ""


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# REMOVE STOPWORDS
# ============================================================

def remove_stopwords(text):

    stop_words = set(
        stopwords.words("english")
    )

    words = word_tokenize(text)

    return " ".join(
        [
            word
            for word in words
            if word not in stop_words
        ]
    )


# ============================================================
# SIMILARITY
# ============================================================

def calculate_similarity(
    resume_text,
    job_description
):

    resume_processed = remove_stopwords(
        clean_text(resume_text)
    )

    job_processed = remove_stopwords(
        clean_text(job_description)
    )

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(
        [
            resume_processed,
            job_processed
        ]
    )

    score = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0] * 100

    return (
        round(score, 2),
        resume_processed,
        job_processed
    )


# ============================================================
# OLLAMA AI FUNCTION
# ============================================================

def ai_resume_analysis(
    resume_text,
    job_description
):

    try:

        from ollama import chat

        prompt = f"""
You are an AI Resume Analyzer.

Analyze the candidate's resume against the given job description.

RESUME:

{resume_text}

JOB DESCRIPTION:

{job_description}

Give the result in exactly these sections:

### 1. Matching Skills

List the skills from the job description that are already present in the resume.

### 2. Missing Skills

List important skills from the job description that are missing from the resume.

### 3. Resume Strengths

Give 3-5 important strengths of this resume for this particular job.

### 4. Improvement Suggestions

Give specific suggestions to improve the resume for this job.

### 5. Suggested Professional Summary

Write a short 3-4 line professional summary tailored to this job.

### 6. Keywords to Add

List important job-related keywords that could naturally be added to the resume.

IMPORTANT:

Do not invent experience, education, projects,
certifications, or skills that are not supported
by the resume.
"""

        response = chat(
            model="gemma3:1b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.message.content

    except Exception as e:

        return f"""
### ⚠️ AI Analysis Error

Could not connect to the local Ollama model.

Please make sure Ollama is running and the model is installed.

Error:

`{e}`
"""


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "✨ Analyze Resume",
    use_container_width=True
):

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not uploaded_file:

        st.warning(
            "⚠️ Please upload your resume first."
        )

        st.stop()


    if not job_description.strip():

        st.warning(
            "⚠️ Please paste the job description."
        )

        st.stop()


    # --------------------------------------------------------
    # EXTRACT RESUME
    # --------------------------------------------------------

    with st.spinner(
        "📖 Reading your resume..."
    ):

        resume_text = extract_text_from_pdf(
            uploaded_file
        )


    if not resume_text:

        st.error(
            "❌ Could not extract text from the PDF."
        )

        st.stop()


    # --------------------------------------------------------
    # CALCULATE SCORE
    # --------------------------------------------------------

    with st.spinner(
        "📊 Calculating resume match..."
    ):

        (
            similarity_score,
            resume_processed,
            job_processed
        ) = calculate_similarity(
            resume_text,
            job_description
        )


    # ========================================================
    # RESULTS
    # ========================================================

    st.markdown(
        '<div class="section-title">📊 Resume Match Results</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    if similarity_score < 40:

        status = "Needs Improvement"
        bar_color = "#DC2626"

    elif similarity_score < 70:

        status = "Good Match"
        bar_color = "#D97706"

    else:

        status = "Excellent Match"
        bar_color = "#0D9488"


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "🎯 Match Score",
            f"{similarity_score:.2f}%"
        )


    with col2:

        st.metric(
            "📄 Resume",
            "Analyzed"
        )


    with col3:

        st.metric(
            "📌 Status",
            status
        )


    # --------------------------------------------------------
    # PROGRESS BAR + GRAPH
    # --------------------------------------------------------

    st.write("")

    with st.container(border=True):

        st.markdown(
            '<div class="card-title">Resume Compatibility</div>',
            unsafe_allow_html=True
        )

        st.progress(
            int(similarity_score)
        )

        # ====================================================
        # GRAPH
        # ====================================================

        fig, ax = plt.subplots(
            figsize=(10, 1.1)
        )

        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        # background track
        ax.barh(
            [0],
            [100],
            height=0.5,
            color="#E2E8F0"
        )

        # score bar
        ax.barh(
            [0],
            [similarity_score],
            height=0.5,
            color=bar_color
        )

        ax.set_xlim(
            0,
            100
        )

        ax.set_yticks([])

        ax.set_xlabel(
            "Match Percentage",
            color="#64748B",
            fontsize=9
        )

        ax.set_title(
            "Resume vs Job Description",
            color="#0F172A",
            fontsize=11,
            fontweight="bold",
            loc="left"
        )

        ax.tick_params(
            axis="x",
            colors="#64748B",
            labelsize=8,
            length=0
        )

        for spine in ax.spines.values():
            spine.set_visible(False)

        st.pyplot(
            fig,
            use_container_width=True
        )


    # ========================================================
    # SCORE MESSAGE
    # ========================================================

    if similarity_score < 40:

        st.warning(
            "⚠️ Low match. Consider tailoring your resume "
            "more closely to the job requirements."
        )

    elif similarity_score < 70:

        st.info(
            "ℹ️ Good match. Adding relevant keywords "
            "could improve your alignment."
        )

    else:

        st.success(
            "🎉 Excellent match! Your resume strongly "
            "aligns with this job description."
        )


    # ========================================================
    # AI ANALYSIS
    # ========================================================

    st.markdown(
        '<div class="section-title">🤖 AI Resume Analysis</div>',
        unsafe_allow_html=True
    )


    with st.spinner(
        "🤖 AI is analyzing your resume..."
    ):

        ai_result = ai_resume_analysis(
            resume_text,
            job_description
        )


    # ========================================================
    # AI RESULT CARD
    # ========================================================

    with st.container(border=True):

        st.markdown(ai_result)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="custom-footer">'
    '<b>ResumeAI</b> · AI-powered Resume Analysis<br>'
    'Built with Python • Streamlit • NLP • TF-IDF • Ollama 🤖'
    '</div>',
    unsafe_allow_html=True
)