"""
Orquestrador central do chatbot.

Responsabilidades:
1. Sanitizar entrada do usuário
2. Detectar intenção
3. Acionar RAG e/ou busca web conforme necessidade
4. Montar prompt com contexto
5. Chamar o Gemini
6. Manter histórico da conversa
7. Retornar resposta estruturada com evidências
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.rag.payroll_rag import PayrollChunk, PayrollRAG
from app.tools.intent_router import Intent, detect_intent
from app.tools.web_search import SearchResult, format_search_results, web_search
from app.utils.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    MAX_CONVERSATION_TURNS,
    get_logger,
)

from app.utils.formatting import sanitize_input

logger = get_logger(__name__)

_SYSTEM_PROMPT = """Você é um assistente especializado em Recursos Humanos, \
com acesso à folha de pagamento da empresa.

Regras obrigatórias:
1. Responda SEMPRE em português do Brasil.
2. Ao usar dados de folha de pagamento, cite obrigatoriamente a fonte \
no formato (Fonte: <employee_id>, <competência>).
3. Formate valores monetários em BRL — ex.: R$ 8.418,75.
4. Ao usar resultados de busca na web, cite a URL no final da resposta.
5. Seja preciso, conciso e profissional.
6. Se não souber a resposta, diga claramente que não possui a informação.
7. Nunca invente dados de folha de pagamento.
"""


@dataclass
class ChatMessage:
    """
    Representa uma mensagem na conversa.
    role: quem falou — "user" (usuário) ou "model" (Gemini)
    content: o texto da mensagem
    """
    role: str
    content: str


@dataclass
class ChatResponse:
    """
    Resposta completa do sistema para uma mensagem do usuário.

    Além do texto, carrega as evidências usadas para gerar a resposta.
    Isso permite que a interface mostre as fontes consultadas.
    """
    text: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    web_results: list[SearchResult] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)

    def sources_json(self) -> str:
        """Serializa as fontes como JSON para download pela interface."""
        return json.dumps(self.sources, indent=2, ensure_ascii=False)
    

class ChatEngine:
    """
    Motor principal do chatbot.
    Instanciado uma vez e reutilizado durante toda a sessão.
    """

    def __init__(self) -> None:
        self._rag = PayrollRAG()

        self._history: list[ChatMessage] = []

        self._llm_available = bool(GEMINI_API_KEY)

        if self._llm_available:
            self._init_gemini()
            try:
                self._rag.build_embeddings()
            except Exception as exc:
                logger.warning("embedding_init_failed %s", str(exc))

    def _init_gemini(self) -> None:
        """Configura o cliente Gemini."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            self._genai = genai

            self._model = genai.GenerativeModel(
                model_name=GEMINI_MODEL, 
                system_instruction=_SYSTEM_PROMPT,
            )
            logger.info('gemini_ready model=%s', GEMINI_MODEL)
        except Exception as exc:
            logger.error('gemini_init_error %s', str(exc))
            self._llm_available = False

    def chat(self, user_message: str) -> ChatResponse:
        """
        Processa uma mensagem do usuário e retorna a resposta do sistema.

        Passos:
        1. Sanitiza a entrada
        2. Detecta intenção
        3. Aciona RAG e 
        4. Aciona busca web conforme necessidade
        5. Monta prompt com contexto
        6. Chama Gemini
        7. Atualiza histórico
        8. Retorna resposta estruturada
        """
        # PASSO 1 — Sanitização
        try:
            user_message = sanitize_input(user_message)
        except ValueError as exc:
            return ChatResponse(
                text=f"Entrada rejeitada pelo sistema de segurança: {exc}",
                intents=["BLOCKED"],
            )
        
        # PASSO 2 — Detecção de intenção
        intents = detect_intent(user_message)
        intent_names = [i.name for i in intents]
        logger.info("intent_detected intents=%s", intent_names)

        # PASSO 3 — RAG (se pergunta sobre folha)
        payroll_chunks: list[PayrollChunk] = []
        rag_context = ""
        if Intent.PAYROLL in intents:
            payroll_chunks = self._rag.retrieve(user_message, top_k=6)
            if payroll_chunks:
                rag_context = self._rag.format_context(payroll_chunks)
                logger.info("rag_retrieved count=%d", len(payroll_chunks))

        # PASSO 4 — Busca web (se solicitado)
        web_results: list[SearchResult] = []
        web_context = ""
        if Intent.WEB_SEARCH in intents:
            search_query = (
                user_message
                .replace("busque", "")
                .replace("pesquise", "")
                .strip()
            )
            web_results = web_search(search_query, num_results=3)
            if web_results:
                web_context = format_search_results(web_results)

        # PASSO 5 — Montagem do prompt
        augmented_prompt = self._build_prompt(
            user_message, rag_context, web_context
        )

        # PASSO 6 — Chamada ao Gemini
        response_text = self._call_llm(augmented_prompt)

        # PASSO 7 — Atualização do histórico
        self._history.append(ChatMessage(role="user", content=user_message))
        self._history.append(ChatMessage(role="model", content=response_text))
        self._trim_history()

        # PASSO 8 - Monta objeto de resposta com evidências
        sources = [
            {
                "employee_id": c.employee_id,
                "name": c.name,
                "competency": c.competency,
                "net_pay": c.data.get("net_pay"),
                "payment_date": c.data.get("payment_date"),
                "source_label": c.source_label,
            }
            for c in payroll_chunks
        ]

        return ChatResponse(
            text=response_text,
            sources=sources,
            web_results=web_results,
            intents=intent_names,
        )
    
    def _build_prompt(
        self,
        user_message: str,
        rag_context: str,
        web_context: str,
    ) -> str:

        parts = []

        if web_context:
            parts.append(web_context)

        if rag_context:
            parts.append(rag_context)
            parts.append("Com base nesses dados, responda:")

        parts.append(user_message)

        prompt = "\n\n".join(parts)

        return prompt if prompt.strip() else user_message
        
    def _build_history_for_gemini(self) -> list[dict]:
        """
        Converte o histórico interno para o formato que o Gemini espera.

        Por que converter: o Gemini usa {"role": ..., "parts": [...]}
        enquanto internamente usamos ChatMessage com role e content.
        Separar os formatos protege o resto do código de mudanças
        na API do Gemini.
        """
        messages = []
        for msg in self._history[-(MAX_CONVERSATION_TURNS * 2):]:
            messages.append({"role": msg.role, "parts": [msg.content]})
        return messages
    
    def _call_llm(self, prompt: str) -> str:
        """
        Envia o prompt para o Gemini e retorna o texto da resposta.
        Retorna mensagem de erro legível se o LLM não estiver disponível.
        """
        if not self._llm_available:
            return (
                f"⚠️ LLM não configurado. "
                f"Defina GEMINI_API_KEY no arquivo .env\n\n"
                f"Informação recuperada do dataset:\n{prompt}"                
            )
        try:
            history = self._build_history_for_gemini()
            chat = self._model.start_chat(history=history)
            response = chat.send_message(prompt)
            return response.text
        except Exception as exc:
            logger.error("llm_call_error %s", str(exc))
            return f"❌ Erro ao chamar o modelo: {exc}"
        
    def _trim_history(self) -> None:
        """
        Mantém o histórico dentro do limite configurado.

        Por que limitar: o Gemini tem um limite de tokens por requisição.
        Conversas muito longas ultrapassam esse limite e causam erro.
        Truncar o histórico antigo preserva o contexto recente —
        que é o mais relevante para a conversa atual.
        """
        max_msgs = MAX_CONVERSATION_TURNS * 2
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]

    def clear_history(self) -> None:
        """Limpa o histórico da conversa."""
        self._history.clear()

    def get_rag(self) -> PayrollRAG:
        """Expõe o RAG para a interface acessar metadados do dataset."""
        return self._rag