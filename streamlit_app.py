"""
Interface do Payroll Chatbot.
Execute com: streamlit run streamlit_app.py
"""
from __future__ import annotations

import json

import streamlit as st

st.set_page_config(
    page_title="Payroll Chatbot",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.chat_engine import ChatEngine
from app.utils.formatting import fmt_competency

st.markdown(
    """
    <style>
    .source-badge {
        background: #1e3a5f;
        color: #a8d8ea;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.78rem;
        margin-right: 4px;
    }
    .intent-badge {
        background: #2d4a22;
        color: #b5e48c;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        margin-left: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "engine" not in st.session_state:
    st.session_state.engine = ChatEngine()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

engine: ChatEngine = st.session_state.engine

with st.sidebar:
    st.title("💼 Payroll Chatbot")
    st.caption("Powered by Gemini + RAG")
    st.divider()

    st.subheader("📊 Dataset")
    rag = engine.get_rag()
    employees = rag.get_all_employees()
    competencies = rag.get_competencies()

    if employees:
        st.markdown("**Funcionários:**")
        for e in employees:
            st.markdown(f"- {e}")

    if competencies:
        st.markdown("**Competências disponíveis:**")
        comp_labels = [fmt_competency(c) for c in competencies]
        st.markdown(", ".join(comp_labels))

    st.divider()

    st.subheader("⚡ Perguntas rápidas")
    quick_questions = [
        "Qual o salário líquido da Ana Souza em maio de 2025?",
        "Quais os bônus do Bruno Lima no primeiro semestre?",
        "Qual o total de INSS descontado da Ana em 2025?",
        "Mostre todos os pagamentos de Bruno Lima.",
        "Qual foi o maior salário líquido do semestre?"
    ]

    for q in quick_questions:
        if st.button(q, use_container_width=True, key=f"quick_{q[:20]}"):
            st.session_state._quick_input = q

    st.divider()

    if st.button("Limpar histórico", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        engine.clear_history()
        st.rerun()

    st.subheader("📥 Evidências")
    if st.session_state.last_sources:
        json_bytes = json.dumps(
            st.session_state.last_sources,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")
        st.download_button(
            label="Baixar evidências (JSON)",
            data=json_bytes,
            file_name="evidences.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.caption("Nenhuma evidência disponível.")

st.title("💬 Chat com o Payroll Bot")
st.caption(
    "Faça perguntas sobre folha de pagamento, "
    "solicite buscas na web, ou converse livremente."    
)

for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

        if msg.get("sources"):
            badges = " ".join(
                f'<span class="source-badge">📋 {s["source_label"]}</span>'
                for s in msg["sources"]
            )
            st.markdown(f"**Fontes:** {badges}", unsafe_allow_html=True)

        if msg.get("intents"):
            badges = " ".join(
                f'<span class="intent-badge">{i}</span>'
                for i in msg["intents"]
            )
            st.markdown(
                f"**Intenções:** {badges}",
                unsafe_allow_html=True,
            )

        if msg.get("web_sources"):
            with st.expander("🌐 Fontes web"):
                for ws in msg["web_sources"]:
                    st.markdown(f"- [{ws['title']}]({ws['url']})")


quick_input = st.session_state.pop("_quick_input", None)

user_input = st.chat_input("Digite sua pergunta...") or quick_input

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Processando..."):
            response = engine.chat(user_input)

        st.markdown(response.text)

        if response.sources:
            st.session_state.last_sources = response.sources
            badges = " ".join(
                f'<span class="source-badge">📋 {s["source_label"]}</span>'
                for s in response.sources
            )
            st.markdown(f"**Fontes:** {badges}", unsafe_allow_html=True)

        if response.intents:
            badges = " ".join(
                f'<span class="intent-badge">{i}</span>'
                for i in response.intents
            ) 
            st.markdown(
                f"**Intenções detectadas:** {badges}",
                unsafe_allow_html=True
            )

        if response.web_results:
            with st.expander("🌐 Fontes web consultadas"):
                for wr in response.web_results:
                    st.markdown(f"- [{wr.title}]({wr.url})\n  > {wr.snippet}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response.text,
        "sources": response.sources,
        "intents": response.intents,
        "web_sources": [
            {"title": wr.title, "url": wr.url}
            for wr in response.web_results
        ],
    })