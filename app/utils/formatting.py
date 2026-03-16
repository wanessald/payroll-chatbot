"""
Funções utilitárias de formatação e sanitização de entrada.
"""
from __future__ import annotations
import unicodedata


def _normalize_text(text: str) -> str:
    """
    Converte texto para minúsculas e remove acentos.
    
    Exemplo: 'Março' → 'marco', 'Ação' → 'acao'
    
    Por que: buscas no dataset precisam ser tolerantes a acentuação. 
    O usuário pode digitar 'março' ou 'marco', ambos devem encontrar o mesmo resultado.
    """

    nfkd = unicodedata.normalize('NFKD', text.lower())
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

def fmt_brl(value: float | int | str) -> str:
    """
    Formata um número como moeda brasileira.
    
    Exemplos:
        8418.75  → 'R$ 8.418,75'
        8000     → 'R$ 8.000,00'
        '7447.5' → 'R$ 7.447,50'
    
    Por que: o Brasil usa ponto como separador de milhar e vírgula como separador decimal, 
    o oposto do padrão americano usado pelo Python nativamente.
    """
    try:
        v = float(value)
    except (ValueError, TypeError):
        return str(value)
    
    formatted = f"{v:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    return f"R$ {formatted}"

def fmt_competency(comp: str) -> str:
    """
    Converte o formato interno de competência para português.
    
    Exemplo: '2025-01' → 'Janeiro/2025'
    
    Por que: o dataset armazena competências como
    'YYYY-MM' por ser ordenável e parseável. Mas o usuário
    precisa ver 'Janeiro/2025' — legível e familiar.
    """
    months = {
        "01": "Janeiro",  "02": "Fevereiro", "03": "Março",
        "04": "Abril",    "05": "Maio",       "06": "Junho",
        "07": "Julho",    "08": "Agosto",     "09": "Setembro",
        "10": "Outubro",  "11": "Novembro",   "12": "Dezembro",
    }

    try:
        year, month = comp.split('-')
        return f"{months.get(month, month)}/{year}"
    except Exception:
        return comp
    
def sanitize_input(text: str, max_length: int = 2000) -> str:
    """
    Sanitiza a entrada do usuário antes de enviar ao LLM.
    
    Faz três coisas:
    1. Remove espaços extras e caracteres nulos
    2. Trunca textos muito longos
    3. Detecta tentativas de prompt injection
    
    Por que: sem sanitização, um usuário mal-intencionado pode tentar sobrescrever as instruções do
    sistema com comandos como 'ignore previous instructions'.
    Isso é chamado de prompt injection.
    """

    text = text.replace("\x00", "").strip()[:max_length]

    INJECTION_PATTERNS = [
        "ignore previous instructions",
        "ignore all previous",
        "disregard your instructions",
        "you are now",
        "forget your instructions",
        "new instructions:",
        "system prompt:",
        "###system",
        "<|system|>",
        "[system]",
    ]

    lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if pattern in lower:
            raise ValueError(
                f"Entrada suspeita detectada (possível prompt injection): '{pattern}'"
            )
        
    return text