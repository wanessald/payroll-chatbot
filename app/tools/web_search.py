"""
app/tools/web_search.py

Busca na web via Google Custom Search JSON API.
Retorna lista vazia silenciosamente quando credenciais
não estão configuradas — o sistema continua funcionando.
"""
from __future__ import annotations
from dataclasses import dataclass
import requests
from app.utils.config import GOOGLE_CSE_API_KEY, GOOGLE_CSE_ID, get_logger

logger = get_logger(__name__)

_CSE_URL = 'https://www.googleapis.com/customsearch/v1'


@dataclass
class SearchResult:
    """
    Representa um resultado de busca na web.

    Três campos porque são os únicos que o usuário precisa ver:
    - title: o título da página
    - url: o endereço para citar como fonte
    - snippet: o trecho relevante do conteúdo
    """
    title: str
    url: str
    snippet: str

    def as_text(self) -> str:
        """Formata o resultado para inserção no contexto do LLM."""
        return f"**{self.title}**\n{self.snippet}\nFonte: {self.url}"
    

def web_search(query: str, num_results: int = 4) -> list[SearchResult]:
    """
    Executa busca no Google e retorna resultados formatados.

    Retorna lista vazia (não levanta erro) quando:
    - Credenciais não configuradas
    - API retorna erro
    - Timeout de rede

    Por que não levanta erro: busca web é funcionalidade
    opcional. O sistema não deve quebrar se ela falhar.
    """
    if not GOOGLE_CSE_API_KEY or not GOOGLE_CSE_ID:
        logger.warning("web_search_disabled: missing credentials")
        return []

    try:
        params = {
            'key': GOOGLE_CSE_API_KEY,
            'cx': GOOGLE_CSE_ID,
            'q': query,
            'num': min(num_results, 10),
            'hl': 'pt-BR',
        }

        resp = requests.get(_CSE_URL, params=params, timeout=10)

        resp.raise_for_status()

        items = resp.json().get('items', [])
        results = [
            SearchResult(
                title=item.get('title', ''),
                url=item.get('link', ''),
                snippet=item.get('snippet', ''),
            )
            for item in items
        ]
        logger.info("web_search_ok query=%s count=%d", query, len(results))
        return results
    except requests.RequestException as exc:
        logger.error("web_search_error query=%s error=%s", query, str(exc))
        return []
    

def format_search_results(results: list[SearchResult]) -> str:
    """Formata lista de resultados como bloco de contexto para o LLM."""
    if not results:
        return ''
    header = '=== RESULTADOS DA WEB ==='
    body = '\n\n'.join(r.as_text() for r in results)
    return f'{header}\n{body}'