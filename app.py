import streamlit as st
import pandas as pd
import numpy as np
from groq import Groq

st.set_page_config(
    page_title="Healthcare AI Assistant",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
body { background: #050a12; }
.main-header {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(120deg, #00d4ff, #1a6fff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
}
.sub-header { color: #64748b; font-size: 0.9rem; margin-bottom: 1.5rem; }
.rag-step {
    background: #0a1628;
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
}
.rag-step.active { border-color: #00d4ff; background: rgba(0,212,255,0.05); }
.source-card {
    background: #0a1628;
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 8px;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/RahmaRas/healthcare-ai-dashboard/main/hpsa_data.csv"
    df = pd.read_csv(url)
    return df

def retrieve_context(query, df, top_k=5):
    query_lower = query.lower()
    keywords = query_lower.split()
    scores = []
    for _, row in df.iterrows():
        score = 0
        row_text = f"{row['state_name']} {row['rural_status']}".lower()
        for kw in keywords:
            if kw in row_text:
                score += 2
        if any(w in query_lower for w in ['worst', 'highest', 'most', 'top']):
            score += row['avg_score'] / 10
        if any(w in query_lower for w in ['underserved', 'population', 'people']):
            score += row['total_underserved'] / 1e7
        if any(w in query_lower for w in ['rural', 'urban', 'suburban']):
            if row['rural_status'].lower() in query_lower:
                score += 3
        scores.append(score)
    df['_score'] = scores
    top = df.nlargest(top_k, '_score')
    df.drop('_score', axis=1, inplace=True)
    return top

def format_context(retrieved):
    lines = []
    for _, row in retrieved.iterrows():
        lines.append(
            f"State: {row['state_name']} | Area: {row['rural_status']} | "
            f"Shortage Score: {row['avg_score']} | "
            f"Underserved Population: {int(row['total_underserved']):,} | "
            f"Shortage Areas: {int(row['total_shortage_areas'])} | "
            f"Providers Needed: {int(row['total_providers_needed']):,}"
        )
    return "\n".join(lines)

# Header
st.markdown('<div class="main-header">🏥 Healthcare Shortage AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">RAG-powered chatbot · Real HRSA data · Built with Groq LLM · Demonstrates Genaiva-style AI pipeline</div>', unsafe_allow_html=True)

# Layout
col_main, col_pipeline = st.columns([2, 1])

with col_pipeline:
    st.markdown("### 🔄 RAG Pipeline")
    st.markdown('<div class="rag-step">📥 <b>Step 1: User Query</b><br>Natural language question received</div>', unsafe_allow_html=True)
    st.markdown('<div class="rag-step">🔍 <b>Step 2: Retrieval</b><br>Keyword + score-based search over 39K HRSA records</div>', unsafe_allow_html=True)
    st.markdown('<div class="rag-step">📋 <b>Step 3: Context Building</b><br>Top 5 relevant records extracted and formatted</div>', unsafe_allow_html=True)
    st.markdown('<div class="rag-step">🤖 <b>Step 4: Generation</b><br>Groq LLM (Llama 3) generates answer from context</div>', unsafe_allow_html=True)
    st.markdown('<div class="rag-step">💬 <b>Step 5: Response</b><br>Grounded answer with cited sources returned</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Data Stats")
    try:
        df = load_data()
        st.metric("Records", f"{len(df):,}")
        st.metric("States", df['state_name'].nunique())
        st.metric("Avg Score", f"{df['avg_score'].mean():.1f}")
    except:
        st.info("Loading data...")

with col_main:
    st.markdown("### 💬 Ask the AI")

    examples = [
        "Which state has the worst healthcare shortage?",
        "How many people are underserved in Kentucky?",
        "Compare rural vs non-rural shortage areas",
        "Which states need the most doctors?",
        "What is the average shortage score nationally?"
    ]

    st.markdown("**Try these questions:**")
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state['question'] = ex

    question = st.text_input(
        "Or type your own question:",
        value=st.session_state.get('question', ''),
        placeholder="e.g. Which states have the most underserved populations?"
    )

    if question:
        try:
            df = load_data()
            with st.spinner("🔍 Retrieving relevant data..."):
                retrieved = retrieve_context(question, df.copy())
                context = format_context(retrieved)

            with st.spinner("🤖 Generating answer..."):
                client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a healthcare data analyst specializing in U.S. workforce shortages.
Answer questions using ONLY the provided HRSA data context.
Be specific — cite state names, numbers, and scores.
Keep answers clear and 3-5 sentences. End with one actionable insight."""
                        },
                        {
                            "role": "user",
                            "content": f"Question: {question}\n\nRelevant HRSA Data:\n{context}"
                        }
                    ],
                    max_tokens=400
                )
                answer = response.choices[0].message.content

            st.success("✅ Answer generated")
            st.markdown("### 🤖 AI Response")
            st.markdown(answer)

            st.markdown("### 📋 Retrieved Sources (Top 5)")
            for _, row in retrieved.iterrows():
                st.markdown(f"""<div class="source-card">
                    📍 <b>{row['state_name']}</b> — {row['rural_status']} |
                    Score: <b>{row['avg_score']}</b> |
                    Underserved: <b>{int(row['total_underserved']):,}</b> |
                    Providers needed: <b>{int(row['total_providers_needed']):,}</b>
                </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown("Built by [Rahma Ras](https://rahmaras.github.io) · Demonstrates RAG pipeline used at Genaiva Voice AI · [Portfolio](https://rahmaras.github.io) · [GitHub](https://github.com/RahmaRas)")
