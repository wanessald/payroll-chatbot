"""
Configuração centralizada carregada a partir do arquivo .env
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

try:
    import structlog
    _HAS_STRUCTLOG = True
except ImportError:
    structlog = None
    _HAS_STRUCTLOG = False

try:
    from dotenv import load_dotenv
    _ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'

GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', '')
GEMINI_MODEL: str = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')
EMBEDDING_MODEL: str = os.getenv('EMBEDDING_MODEL', 'models/text-embedding-004')

PAYROLL_CSV_PATH: Path = BASE_DIR / os.getenv('PAYROLL_CSV_PATH', 'data/payroll.csv')

GOOGLE_CSE_API_KEY: str = os.getenv('GOOGLE_CSE_API_KEY', '')
GOOGLE_CSE_ID: str = os.getenv('GOOGLE_CSE_ID', '')

LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
MAX_CONVERSATION_TURNS: int = int(os.getenv('MAX_CONVERSATION_TURNS', '20'))

def setup_logging() -> None:
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
        level=level,
        stream=sys.stdout
    )
    if _HAS_STRUCTLOG:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt='iso'),
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(level),
            logger_factory=structlog.PrintLoggerFactory(),
        )

def get_logger(name: str):
    if _HAS_STRUCTLOG:
        return structlog.get_logger(name)
    return logging.getLogger(name)


setup_logging()
logger = get_logger(__name__)