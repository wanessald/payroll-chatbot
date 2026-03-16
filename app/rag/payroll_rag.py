"""
Pipeline RAG sobre o dataset de folha de pagamento.

Estratégia:
1. Carrega o CSV uma vez e constrói um chunk de texto por linha
2. Embeds cada chunk com a API de embeddings do Gemini (cache em memória)
3. Na consulta, embeds a pergunta e ranqueia chunks por similaridade cosseno
4. Retorna os top-k chunks como contexto + objetos de evidência estruturados

Tolerâncias implementadas:
- Nomes: busca parcial, case-insensitive, sem acentos
- Competências: aceita 10+ formatos diferentes
- Valores monetários: normalizados antes da comparação
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd 

from app.utils.config import EMBEDDING_MODEL, GEMINI_API_KEY, PAYROLL_CSV_PATH, get_logger
from app.utils.formatting import fmt_brl, fmt_competency


logger = get_logger(__name__)

_MONTH_MAP: dict[str, str] = {
    "janeiro": "01", "jan": "01",
    "fevereiro": "02", "fev": "02", "feb": "02",
    "marco": "03", "mar": "03",
    "abril": "04", "abr": "04", "apr": "04",
    "maio": "05", "mai": "05", "may": "05",
    "junho": "06", "jun": "06",
    "julho": "07", "jul": "07",
    "agosto": "08", "ago": "08", "aug": "08",
    "setembro": "09", "set": "09", "sep": "09",
    "outubro": "10", "out": "10", "oct": "10",
    "novembro": "11", "nov": "11",
    "dezembro": "12", "dez": "12", "dec": "12",
}

def _normalise_text(text: str) -> str:
    nfkd = unicodedata.normalize('NFKD', text.lower())
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def _parse_competency(comp: str) -> str | None:
    raw = _normalise_text(comp.strip())

    m = re.match(r"(\d{4})[-/](\d{1,2})$", raw)
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}"
    
    m = re.match(r"(\d{1,2})[-/](\d{4})$", raw)
    if m:
        return f"{m.group(2)}-{m.group(1).zfill(2)}"
    
    m = re.match(r"([a-z]+)[\s/\-](\d{2,4})$", raw)
    if m:
        month_name, year = m.group(1), m.group(2)
        month_num = _MONTH_MAP.get(month_name)
        if month_num:
            if len(year) == 2:
                year = "20" + year
            return f"{year}-{month_num}"

    return None

@dataclass
class PayrollChunk:
    employee_id: str
    name: str
    competency: str
    data: dict[str, Any]       # linha original do CSV como dicionário
    text: str                  # texto rico gerado para embedding
    embedding: list[float] = field(default_factory=list)  # vetor numérico

    @property
    def source_label(self) -> str:
        return f"{self.employee_id}, {self.competency}"
    
def _build_chunk_text(row: pd.Series) -> str:
        return (
        f"Funcionário: {row['name']} (ID {row['employee_id']}). "
        f"Competência: {fmt_competency(row['competency'])} ({row['competency']}). "
        f"Salário base: {fmt_brl(row['base_salary'])}. "
        f"Bônus: {fmt_brl(row['bonus'])}. "
        f"Benefícios (VT/VR): {fmt_brl(row['benefits_vt_vr'])}. "
        f"Outros rendimentos: {fmt_brl(row['other_earnings'])}. "
        f"INSS: {fmt_brl(row['deductions_inss'])}. "
        f"IRRF: {fmt_brl(row['deductions_irrf'])}. "
        f"Outras deduções: {fmt_brl(row['other_deductions'])}. "
        f"Pagamento líquido: {fmt_brl(row['net_pay'])}. "
        f"Data de pagamento: {row['payment_date']}."
    )


class PayrollRAG:
    def __init__(self) -> None:
        self._chunks: list[PayrollChunk] = []
        self._embedding_ready = False
        self._df: pd.DataFrame = pd.DataFrame()
        self._load_data()
    
    def _load_data(self) -> None:
        path: Path = PAYROLL_CSV_PATH
        if not path.exists():
            logger.error("payroll_csv_not_found path=%s", str(path))
            return
        
        self._df = pd.read_csv(path)

        for _, row in self._df.iterrows():
            self._chunks.append(
                PayrollChunk(
                    employee_id=row["employee_id"],
                    name=row["name"],
                    competency=row["competency"],
                    data=row.to_dict(),
                    text=_build_chunk_text(row),
                )
            )
        logger.info("payroll_loaded rows=%d", len(self._chunks))

    def build_embeddings(self) -> None:
        if self._embedding_ready:
            return
        
        if not GEMINI_API_KEY:
            logger.warning("no_gemini_key_skipping_embeddings")
            return
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)

            texts = [c.text for c in self._chunks]
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=texts,
                task_type="retrieval_document",
            )
            for chunk, emb in zip(self._chunks, result["embedding"]):
                chunk.embedding = emb

            self._embedding_ready = True
            logger.info("embeddings_built count=%d", len(self._chunks))
        except Exception as exc:
            logger.error("embedding_error=%s", str(exc))

    def _embed_query(self, query: str) -> list[float] | None:
        if not GEMINI_API_KEY:
            return None
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=[query],
                task_type="retrieval_query",
            )
            return result["embedding"]
        except Exception as exc:
            logger.error("query_embedding_error=%s", str(exc))
            return None
        
    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        va, vb = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return np.dot(va, vb).item() / denom

    def _semantic_search(self, query: str, top_k: int = 5) -> list[PayrollChunk]:
        """Busca semântica via embeddings + similaridade cosseno."""
        q_emb = self._embed_query(query)
        if q_emb is None or not self._embedding_ready:
            return []
        
        scored = [
            (self._cosine(q_emb, c.embedding), c) 
            for c in self._chunks 
            if c.embedding
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]
    
    def _keyword_search(self, query:str, top_k: int = 5) -> list[PayrollChunk]:
        """
        Busca por palavras-chave quando embeddings não estão disponíveis.
        Extrai nome, employee_id e competência da pergunta e filtra o DataFrame.
        """
        q_norm = _normalise_text(query)
        comp_found: str | None = None

        # Tenta extrair competência da pergunta
        for pat in [r"\b(\d{4})[-/](\d{1,2})\b", r"\b(\d{1,2})[-/](\d{4})\b"]:
            m = re.search(pat, query)
            if m:
                raw = m.group(0)
                comp_found = _parse_competency(raw)
                break

        if not comp_found:
            for word in q_norm.split():
                c = _parse_competency(word)
                if c:
                    comp_found = c
                    break

        name_tokens = [
            chunk.name for chunk in self._chunks
            if any(tok in q_norm for tok in _normalise_text(chunk.name).split())
        ]

        emp_ids: list[str] = re.findall(r"\bE\d{3}\b", query.upper())

        mask = pd.Series([True] * len(self._df))
        if emp_ids:
            mask &= self._df["employee_id"].isin(emp_ids)
        if name_tokens:
            mask &= self._df["name"].isin(name_tokens)
        if comp_found:
            mask &= self._df["competency"] == comp_found

        filtered = self._df[mask]
        if filtered.empty and not (emp_ids or name_tokens or comp_found):
            filtered = self._df  # fallback para evitar resposta vazia

        result_chunks = []
        for _, row in filtered.head(top_k).iterrows():
            for chunk in self._chunks:
                if (chunk.employee_id == row["employee_id"]
                        and chunk.competency == row["competency"]):
                    result_chunks.append(chunk)
        return result_chunks
    
    def retrieve(self, query: str, top_k: int = 5) -> list[PayrollChunk]:
        """
        Ponto de entrada principal da busca.
        Usa semântica se disponível, keyword como fallback.
        """
        if self._embedding_ready:
            results = self._semantic_search(query, top_k)
            if results:
                return results
        return self._keyword_search(query, top_k)

    def format_context(self, chunks: list[PayrollChunk]) -> str:
        """Formata chunks recuperados como bloco de contexto para o LLM."""
        lines = ["=== DADOS DE FOLHA DE PAGAMENTO (contexto recuperado) ==="]
        for i, chunk in enumerate(chunks, 1):
            lines.append(f"\n[{i} Fonte: {chunk.source_label}]")
            lines.append(chunk.text)
        return "\n".join(lines)
    
    def get_all_employees(self) -> list[str]:
        return sorted(self._df["name"].unique().tolist()) if not self._df.empty else []
    
    def get_competencies(self) -> list[str]:
        return sorted(self._df["competency"].unique().tolist()) if not self._df.empty else []