# Payroll Chatbot

Assistente conversacional com RAG sobre folha de pagamento, powered by Gemini + Streamlit.

## Setup

### 1. Pré-requisitos
- Python 3.11+
- Git

### 2. Instalar dependências
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
### 3. Configurar variáveis de ambiente
```bash
# Copie o template e preencha com suas chaves
cp .env.example .env
```
Variáveis obrigatórias:
- GEMINI_API_KEY: chave da API do Google AI Studio

Variáveis opcionais (busca web):
- GOOGLE_CSE_API_KEY: chave da API do Google Cloud
- GOOGLE_CSE_ID: ID do mecanismo de busca

### 4. Rodar os testes
```bash
python -m pytest tests/ -v
```
### 5. Subir a aplicação
```bash
streamlit run streamlit_app.py
```
Acesse: http://localhost:8501

## Arquitetura
```bash
\streamlit_app.py        -> Interface do usuário
app/
  chat_engine.py        -> Orquestrador central
  rag/
    payroll_rag.py      -> RAG sobre folha de pagamento
  tools/
    intent_router.py    -> Deteccao de intencao
    web_search.py       -> Busca na web (opcional)
  utils/
    config.py           -> Configuracao centralizada
    formatting.py       -> Formatacao e sanitizacao
data/
  payroll.csv           -> Dataset de folha de pagamento
tests/
  test_payroll.py       -> 33 testes automatizados
```
## Decisoes Tecnicas

### Por que RAG em vez de fine-tuning?
Fine-tuning treina o modelo com novos dados -- caro e lento.
RAG recupera dados relevantes em tempo real -- barato e flexivel.
Para um dataset de folha de pagamento que muda mensalmente,
RAG e a escolha correta.

### Por que RAG hibrido (semantico + keyword)?
- Com chave Gemini: embeddings semanticos toleram parafrase
- Sem chave / em testes: filtros pandas cobrem casos principais
O sistema funciona em ambos os cenarios.

### Por que nao LangChain?
Dataset pequeno (12 linhas) nao justifica o overhead.
Implementacao manual garante entendimento completo do pipeline.

### Por que Streamlit?
Prototipagem rapida de interfaces de dados em Python puro.
Sem necessidade de HTML/CSS/JavaScript.

## Limitacoes
- Dataset sintetico com 12 registros
- Busca web requer credenciais opcionais do Google Cloud
- Embeddings recalculados a cada reinicio (sem cache persistente)
- Sem autenticacao de usuario

## Testes
33 testes automatizados cobrindo:
- Formatacao de moeda BRL
- Parser de competencias (10 formatos)
- RAG por nome, ID, competencia
- Valores monetarios especificos do dataset
- Deteccao de intencao
- Guardrail de prompt injection
