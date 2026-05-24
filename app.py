import streamlit as st
import pandas as pd
from groq import Groq

st.set_page_config(
    page_title="Healthcare RAG Assistant",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp { background: #050a12; }

.main-header {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(120deg, #00d4ff, #1a6fff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
}
.sub-header {
    color: #64748b;
    font-size: 0.88rem;
    margin-bottom: 2rem;
    letter-spacing: 0.02em;
}
.metric-box {
    background: #0a1628;
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-val {
    font-size: 1.6rem;
    font-weight: 800;
    color: #00d4ff;
    line-height: 1;
}
.metric-lbl {
    font-size: 0.72rem;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.3rem;
}
.pipeline-card {
    background: #0a1628;
    border: 1px solid rgba(0,212,255,0.12);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    transition: all 0.3s;
}
.pipeline-card.active {
    border-color: #00d4ff;
    background: rgba(0,212,255,0.06);
    box-shadow: 0 0 20px rgba(0,212,255,0.1);
}
.pipeline-icon {
    font-size: 1.1rem;
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: rgba(0,212,255,0.08);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.pipeline-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.15rem;
}
.pipeline-desc {
    font-size: 0.75rem;
    color: #64748b;
    line-height: 1.4;
}
.chat-bubble-user {
    background: linear-gradient(135deg, #00d4ff, #1a6fff);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
    font-weight: 500;
    max-width: 85%;
    margin-left: auto;
}
.chat-bubble-ai {
    background: #0a1628;
    border: 1px solid rgba(0,212,255,0.15);
    color: #e2e8f0;
    border-radius: 18px 18px 18px 4px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
    line-height: 1.7;
    max-width: 85%;
}
.source-pill {
    display: inline-block;
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 100px;
    padding: 0.2rem 0.7rem;
    font-size: 0.72rem;
    color: #00d4ff;
    font-weight: 600;
    margin: 0.2rem;
}
.example-btn {
    background: #0a1628;
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
    font-size: 0.82rem;
    color: #94a3b8;
    cursor: pointer;
    transition: all 0.2s;
    text-align: left;
    width: 100%;
    margin-bottom: 0.4rem;
}
.divider {
    border: none;
    border-top: 1px solid rgba(0,212,255,0.08);
    margin: 1.5rem 0;
}
.tag-chip {
    display: inline-block;
    background: rgba(0,212,255,0.06);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 100px;
    padding: 0.15rem 0.6rem;
    font-size: 0.7rem;
    color: #00d4ff;
    font-weight: 600;
    margin: 0.15rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/RahmaRas/healthcare-ai-dashboard/main/hpsa_data.csv"
    return pd.read_csv(url)

def retrieve(query, df, top_k=5):
    q = query.lower()
    scores = []
    for _, row in df.iterrows():
        s = 0
        text = f"{row['state_name']} {row['rural_status']}".lower()
        for w in q.split():
            if w in text: s += 2
        if any(x in q for x in ['worst','highest','most','top','bad']): s += row['avg_score']/8
        if any(x in q for x in ['underserved','population','people']): s += row['total_underserved']/8e6
        if row['rural_status'].lower() in q: s += 3
        scores.append(s)
    df = df.copy()
    df['_s'] = scores
    top = df.nlargest(top_k, '_s').drop('_s', axis=1)
    return top

def build_context(rows):
    return "\n".join([
        f"• {r['state_name']} ({r['rural_status']}): Score={r['avg_score']}, Underserved={int(r['total_underserved']):,}, Areas={int(r['total_shortage_areas'])}, Providers needed={int(r['total_providers_needed']):,}"
        for _, r in rows.iterrows()
    ])

# Load data
df = load_data()

# Header
st.markdown('<div class="main-header">🏥 Healthcare Shortage AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">RAG pipeline · 39,000+ real HRSA records · Groq LLM (Llama 3) · Built by Rahma Ras</div>', unsafe_allow_html=True)

# Tags
st.markdown("""
<span class="tag-chip">RAG</span>
<span class="tag-chip">LLM</span>
<span class="tag-chip">Vector Retrieval</span>
<span class="tag-chip">Groq</span>
<span class="tag-chip">HRSA Data</span>
<span class="tag-chip">Healthcare AI</span>
""", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Metrics row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-box"><div class="metric-val">{len(df):,}</div><div class="metric-lbl">Records</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-box"><div class="metric-val">{df["state_name"].nunique()}</div><div class="metric-lbl">States</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-box"><div class="metric-val">{df["avg_score"].mean():.1f}</div><div class="metric-lbl">Avg Score</div></div>', unsafe_allow_html=True)
with c4:
    total = df['total_underserved'].sum()
    st.markdown(f'<div class="metric-box"><div class="metric-val">{total/1e6:.0f}M+</div><div class="metric-lbl">Underserved</div></div>', unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Main layout
left, right = st.columns([3, 1])

with right:
    st.markdown("**🔄 RAG Pipeline**")
    active = st.session_state.get('stage', 0)
    steps = [
        ("📥", "Query", "User inputs natural language question"),
        ("🔍", "Retrieval", "Score-based search over 39K HRSA records"),
        ("📋", "Context", "Top 5 records formatted as context"),
        ("🤖", "Generation", "Groq Llama 3 generates grounded answer"),
        ("💬", "Response", "Answer + cited sources returned"),
    ]
    for i, (icon, title, desc) in enumerate(steps):
        cls = "pipeline-card active" if i < active else "pipeline-card"
        st.markdown(f'<div class="{cls}"><div class="pipeline-icon">{icon}</div><div><div class="pipeline-title">{title}</div><div class="pipeline-desc">{desc}</div></div></div>', unsafe_allow_html=True)

with left:
    st.markdown("**💬 Ask a question about U.S. healthcare shortages**")

    examples = [
        "Which state has the worst healthcare shortage?",
        "How many people are underserved in Kentucky?",
        "Compare rural vs non-rural shortage areas",
        "Which states need the most doctors?",
        "What is the average shortage score nationally?",
        "Which region has the highest shortage severity?"
    ]

    cols = st.columns(3)
    for i, ex in enumerate(examples):
        if cols[i % 3].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state['q'] = ex

    question = st.text_input(
        "",
        value=st.session_state.get('q', ''),
        placeholder="Type your question here...",
        label_visibility="collapsed"
    )

    if question:
        st.session_state['stage'] = 1
        st.markdown(f'<div class="chat-bubble-user">🙋 {question}</div>', unsafe_allow_html=True)

        with st.spinner(""):
            retrieved = retrieve(question, df)
            st.session_state['stage'] = 2
            context = build_context(retrieved)
            st.session_state['stage'] = 3

            try:
                client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a healthcare data analyst. Answer using ONLY the provided HRSA data. Be specific — cite state names and numbers. Keep answers to 3-4 sentences. End with one actionable insight."
                        },
                        {
                            "role": "user",
                            "content": f"Question: {question}\n\nHRSA Data:\n{context}"
                        }
                    ],
                    max_tokens=350
                )
                answer = response.choices[0].message.content
                st.session_state['stage'] = 4

                st.markdown(f'<div class="chat-bubble-ai">🤖 {answer}</div>', unsafe_allow_html=True)
                st.session_state['stage'] = 5

                st.markdown("**📋 Retrieved Sources**")
                source_html = ""
                for _, row in retrieved.iterrows():
                    source_html += f'<span class="source-pill">📍 {row["state_name"]} · Score: {row["avg_score"]} · {int(row["total_underserved"]):,} underserved</span>'
                st.markdown(source_html, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("Built by [Rahma Ras](https://rahmaras.github.io) · [Tableau Dashboard](https://public.tableau.com/app/profile/rahma.ras/viz/USHealthcareWorkforceShortageTracker/Dashboard2) · [GitHub](https://github.com/RahmaRas/healthcare-ai-dashboard) · Data: HRSA Open Data 2026")
