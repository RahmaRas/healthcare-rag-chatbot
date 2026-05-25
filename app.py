import streamlit as st
import pandas as pd
from groq import Groq
import base64

st.set_page_config(
    page_title="Healthcare AI Assistant — Rahma Ras",
    page_icon="🏥",
    layout="wide"
)

PHOTO_B64 = "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"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: #050a12; }
.stTextInput input {
    background: #0a1628 !important;
    border: 1px solid rgba(0,212,255,0.25) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    padding: 0.75rem 1rem !important;
}
.stTextInput input:focus { border-color: #00d4ff !important; box-shadow: 0 0 15px rgba(0,212,255,0.2) !important; }
.stButton button {
    background: transparent !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    border-color: #00d4ff !important;
    color: #00d4ff !important;
    background: rgba(0,212,255,0.06) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<style>
.header-bar {{
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding: 1.5rem 0 1rem;
    border-bottom: 1px solid rgba(0,212,255,0.1);
    margin-bottom: 1.5rem;
}}
.header-photo {{
    width: 72px;
    height: 72px;
    border-radius: 50%;
    object-fit: cover;
    object-position: center top;
    border: 2.5px solid #00d4ff;
    box-shadow: 0 0 20px rgba(0,212,255,0.3);
    flex-shrink: 0;
}}
.header-info h1 {{
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(120deg, #00d4ff, #1a6fff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.15rem;
    line-height: 1;
}}
.header-info p {{
    font-size: 0.82rem;
    color: #64748b;
    margin: 0;
}}
.header-tags {{ display: flex; gap: 0.4rem; flex-wrap: wrap; margin-top: 0.5rem; }}
.tag {{
    background: rgba(0,212,255,0.06);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 100px;
    padding: 0.15rem 0.6rem;
    font-size: 0.68rem;
    color: #00d4ff;
    font-weight: 600;
}}
.chat-container {{
    max-height: 480px;
    overflow-y: auto;
    padding: 1rem 0;
}}
.msg-row {{
    display: flex;
    align-items: flex-end;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
}}
.msg-row.user {{ flex-direction: row-reverse; }}
.avatar {{
    width: 38px;
    height: 38px;
    border-radius: 50%;
    object-fit: cover;
    object-position: center top;
    flex-shrink: 0;
    border: 1.5px solid rgba(0,212,255,0.3);
}}
.bubble-user {{
    background: linear-gradient(135deg, #00d4ff, #1a6fff);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 0.85rem 1.1rem;
    font-size: 0.9rem;
    font-weight: 500;
    max-width: 70%;
    line-height: 1.55;
}}
.bubble-bot {{
    background: #0a1628;
    border: 1px solid rgba(0,212,255,0.15);
    color: #e2e8f0;
    border-radius: 18px 18px 18px 4px;
    padding: 0.85rem 1.1rem;
    font-size: 0.9rem;
    max-width: 70%;
    line-height: 1.65;
}}
.bot-name {{
    font-size: 0.72rem;
    font-weight: 700;
    color: #00d4ff;
    margin-bottom: 0.3rem;
    letter-spacing: 0.03em;
}}
.source-row {{ margin-top: 0.5rem; display: flex; flex-wrap: wrap; gap: 0.3rem; }}
.source-pill {{
    background: rgba(0,212,255,0.06);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 100px;
    padding: 0.15rem 0.6rem;
    font-size: 0.7rem;
    color: #00d4ff;
    font-weight: 600;
}}
.pipeline-step {{
    background: #0a1628;
    border: 1px solid rgba(0,212,255,0.1);
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.4rem;
    font-size: 0.78rem;
    color: #64748b;
    transition: all 0.3s;
}}
.pipeline-step.active {{
    border-color: #00d4ff;
    color: #e2e8f0;
    background: rgba(0,212,255,0.05);
    box-shadow: 0 0 12px rgba(0,212,255,0.1);
}}
.stat-box {{
    background: #0a1628;
    border: 1px solid rgba(0,212,255,0.12);
    border-radius: 10px;
    padding: 0.75rem;
    text-align: center;
    margin-bottom: 0.5rem;
}}
.stat-val {{ font-size: 1.4rem; font-weight: 800; color: #00d4ff; line-height: 1; }}
.stat-lbl {{ font-size: 0.68rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin-top: 0.2rem; }}
</style>
<div class="header-bar">
    <img src="data:image/jpeg;base64,{PHOTO_B64}" class="header-photo"/>
    <div class="header-info">
        <h1>Healthcare Shortage AI Assistant</h1>
        <p>Multi-turn RAG chatbot · Built by Rahma Ras · Real HRSA data · Groq LLM</p>
        <div class="header-tags">
            <span class="tag">RAG Pipeline</span>
            <span class="tag">Multi-turn Chat</span>
            <span class="tag">Groq Llama 3</span>
            <span class="tag">39K+ Records</span>
            <span class="tag">Healthcare AI</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/RahmaRas/healthcare-ai-dashboard/main/hpsa_data.csv"
    return pd.read_csv(url)

def retrieve(query, df, top_k=4):
    q = query.lower()
    scores = []
    for _, row in df.iterrows():
        s = 0
        text = f"{row['state_name']} {row['rural_status']}".lower()
        for w in q.split():
            if len(w) > 3 and w in text: s += 2
        if any(x in q for x in ['worst','highest','most','top','bad','severe']): s += row['avg_score']/8
        if any(x in q for x in ['underserved','population','people','million']): s += row['total_underserved']/8e6
        if row['rural_status'].lower() in q: s += 3
        scores.append(s)
    df = df.copy()
    df['_s'] = scores
    return df.nlargest(top_k, '_s').drop('_s', axis=1)

def build_context(rows):
    return "\n".join([
        f"• {r['state_name']} ({r['rural_status']}): Score={r['avg_score']}, Underserved={int(r['total_underserved']):,}, Areas={int(r['total_shortage_areas'])}, Providers needed={int(r['total_providers_needed']):,}"
        for _, r in rows.iterrows()
    ])

df = load_data()

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'stage' not in st.session_state:
    st.session_state.stage = 0

col_chat, col_side = st.columns([3, 1])

with col_side:
    st.markdown("**🔄 RAG Pipeline**")
    steps = [
        ("📥", "Query received"),
        ("🔍", "Retrieving data"),
        ("📋", "Building context"),
        ("🤖", "Generating answer"),
        ("💬", "Response ready"),
    ]
    for i, (icon, label) in enumerate(steps):
        cls = "pipeline-step active" if i < st.session_state.stage else "pipeline-step"
        st.markdown(f'<div class="{cls}">{icon} {label}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📊 Data**")
    st.markdown(f'<div class="stat-box"><div class="stat-val">{len(df):,}</div><div class="stat-lbl">Records</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stat-box"><div class="stat-val">{df["state_name"].nunique()}</div><div class="stat-lbl">States</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stat-box"><div class="stat-val">{df["total_underserved"].sum()/1e6:.0f}M</div><div class="stat-lbl">Underserved</div></div>', unsafe_allow_html=True)

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.stage = 0
        st.rerun()

with col_chat:
    if not st.session_state.messages:
        st.markdown("**Try asking:**")
        examples = [
            "Which state has the worst shortage?",
            "How many people are underserved in Kentucky?",
            "Compare rural vs urban shortages",
            "Which states need the most doctors?",
        ]
        c1, c2 = st.columns(2)
        for i, ex in enumerate(examples):
            if (c1 if i%2==0 else c2).button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state['pending'] = ex
                st.rerun()

    # ── CHAT HISTORY ──────────────────────────────────────────────────────────
    # USER bubble: their photo on the RIGHT
    # BOT  bubble: Rahma's photo on the LEFT  (looks like Rahma is answering)
    chat_html = '<div class="chat-container">'
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            # User message — photo on right, bubble on left of photo
            chat_html += f'''
            <div class="msg-row user">
                <img src="data:image/jpeg;base64,{PHOTO_B64}" class="avatar"/>
                <div class="bubble-user">{msg["content"]}</div>
            </div>'''
        else:
            # Bot/Rahma answer — Rahma's photo on left, bubble on right of photo
            sources_html = ""
            if msg.get('sources'):
                sources_html = '<div class="source-row">' + ''.join(
                    [f'<span class="source-pill">📍 {s}</span>' for s in msg['sources']]
                ) + '</div>'
            chat_html += f'''
            <div class="msg-row">
                <img src="data:image/jpeg;base64,{PHOTO_B64}" class="avatar"/>
                <div class="bubble-bot">
                    <div class="bot-name">Rahma Ras · Healthcare AI</div>
                    {msg["content"]}{sources_html}
                </div>
            </div>'''
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

    # ── INPUT ─────────────────────────────────────────────────────────────────
    question = st.text_input(
        "", placeholder="Ask about U.S. healthcare shortages...",
        key="input", label_visibility="collapsed"
    )

    if 'pending' in st.session_state and st.session_state.pending:
        question = st.session_state.pending
        st.session_state.pending = None

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.stage = 1

        retrieved = retrieve(question, df)
        st.session_state.stage = 2
        context = build_context(retrieved)
        st.session_state.stage = 3

        history = [
            {"role": "system", "content": (
                "You are Rahma Ras, a healthcare data analyst and Master's student at UMBC "
                "specializing in U.S. workforce shortages. Answer using ONLY the provided HRSA data context. "
                "Be specific — cite state names and numbers. "
                "Remember the conversation history — if the user references something from earlier, use that context. "
                "Keep answers to 3-4 sentences. End with one follow-up question to keep the conversation going."
            )}
        ]
        for m in st.session_state.messages[:-1]:
            history.append({"role": m["role"], "content": m["content"]})
        history.append({
            "role": "user",
            "content": f"Question: {question}\n\nHRSA Data:\n{context}"
        })

        try:
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",   # ← FIXED MODEL
                messages=history,
                max_tokens=350
            )
            answer = response.choices[0].message.content
            st.session_state.stage = 4

            sources = [
                f"{row['state_name']} (Score: {row['avg_score']})"
                for _, row in retrieved.iterrows()
            ]
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
            st.session_state.stage = 5

        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error: {str(e)}",
                "sources": []
            })

        st.rerun()

st.markdown("---")
st.markdown(
    "Built by [Rahma Ras](https://rahmaras.github.io) · "
    "[Dashboard](https://healthcare-shortage-dashboard.streamlit.app) · "
    "[GitHub](https://github.com/RahmaRas/healthcare-ai-dashboard) · Data: HRSA 2026"
)
