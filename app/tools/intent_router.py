"""
Detecta a intenção da mensagem do usuário e decide
quais ferramentas o ChatEngine deve acionar.
"""
from __future__ import annotations

import re
import unicodedata
from enum import Enum, auto


def _normalise_text(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

class Intent(Enum):
    PAYROLL = auto()
    WEB_SEARCH = auto()
    GENERAL = auto()


_PAYROLL_KEYWORDS = [
    "salario", "folha", "pagamento", "liquido",
    "inss", "irrf", "bonus", "beneficio",
    "competencia", "holerite", "contracheque",
    "ana souza", "bruno lima", "e001", "e002",
    "rendimento", "deducao", "vt", "vr",
]

_WEB_KEYWORDS = [
    "busque", "pesquise", "procure na internet",
    "busca na web", "acesse", "noticia", "pesquisa",
    "google", "atual", "atualidade", "hoje",
]

_DATE_PATTERN = re.compile(
    r"\b(20\d{2}[-/]\d{1,2}|\d{1,2}[-/]20\d{2}|janeiro|fevereiro|marco|"
    r"abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\b"
)


def detect_intent(message: str) -> list[Intent]:
    """
    Analisa a mensagem e retorna lista de intenções detectadas.

    Retorna lista porque uma mensagem pode ter múltiplas intenções:
    "Busque o salário da Ana em maio" → [PAYROLL, WEB_SEARCH]

    Ordem da lista indica prioridade — primeiro item é a
    intenção principal.
    """
    norm = _normalise_text(message)
    intents: list[Intent] = []

    payroll_hit = any(kw in norm for kw in _PAYROLL_KEYWORDS)

    date_hit = bool(_DATE_PATTERN.search(norm))

    if payroll_hit or date_hit:
        intents.append(Intent.PAYROLL)

    if any(kw in norm for kw in _WEB_KEYWORDS):
        intents.append(Intent.WEB_SEARCH)

    if not intents:
        intents.append(Intent.GENERAL)

    return intents