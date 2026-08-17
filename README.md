# Tech Challenge Fase 3 — Assistente Médico com LLM + LangChain

> Grupo 61 | Pós-graduação em IA — POSTECH

## Sobre o projeto

Assistente virtual médico treinado com dados hospitalares próprios, capaz de auxiliar condutas clínicas, responder dúvidas de médicos e sugerir procedimentos com base em protocolos internos. Fluxos de decisão automatizados coordenados com LangChain e LangGraph.

## Stack

| Componente | Tecnologia |
|---|---|
| Modelo base | `meta-llama/Llama-3.2-3B-Instruct` |
| Fine-tuning | MLX-LM + LoRA (Apple Silicon) |
| Dataset | PubMedQA + dados sintéticos hospitalares |
| Pipeline | LangChain |
| Fluxo automatizado | LangGraph |
| Banco de dados | SQLite + SQLAlchemy |
| Segurança | Guardrails custom + audit log JSON |

## Pré-requisitos

- Python 3.11+
- Apple Silicon (M1/M2/M3) recomendado para fine-tuning via MLX
- Conta no [HuggingFace](https://huggingface.co) com token de acesso
- Acesso ao modelo `meta-llama/Llama-3.2-3B-Instruct` (solicitar em huggingface.co/meta-llama)

## Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/oryange/tech-challenge-fase3.git
cd tech-challenge-fase3

# 2. Crie e ative o ambiente virtual
python -m venv venv
source venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure as variáveis de ambiente
cp .env.example .env
# Edite .env com seu token HuggingFace e caminhos desejados
```

## Como executar o pipeline completo

```bash
# 1. Baixar e processar PubMedQA
python src/data/loader.py

# 2. Gerar dados sintéticos hospitalares
python src/data/synthetic_generator.py

# 3. Curar e unificar o dataset
python src/data/curator.py

# 4. Fine-tuning do modelo (requer Apple Silicon)
python src/fine_tuning/trainer.py

# 5. Popular banco de dados com pacientes sintéticos
python src/database/seed.py

# 6. Iniciar assistente médico interativo
python src/assistant/chain.py

# 7. Executar fluxo LangGraph
python src/graph/clinical_flow.py
```

## Testes

```bash
# Rodar todos os testes (exceto integração)
pytest tests/ -m "not integration"

# Rodar testes de integração (requer modelo carregado)
pytest tests/ -m integration
```

## Estrutura do projeto

```
tech-challenge-fase3/
├── data/
│   ├── processed/          # Dataset curado (JSONL)
│   ├── synthetic/          # Dados sintéticos hospitalares
│   └── fine_tuned/         # Adapters LoRA (gerado localmente)
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_fine_tuning.ipynb
│   └── 03_langchain_demo.ipynb
├── src/
│   ├── data/               # Pipeline de dados
│   ├── fine_tuning/        # Fine-tuning MLX-LM
│   ├── llm/                # Wrapper LLM + guardrails
│   ├── assistant/          # Pipeline LangChain
│   ├── graph/              # Fluxo LangGraph
│   ├── database/           # SQLAlchemy models + seed
│   └── logging/            # Audit logger
├── tests/
├── docs/
│   ├── relatorio-tecnico.md
│   └── diagrama-langchain.md
├── .env.example
├── requirements.txt
└── pytest.ini
```

## Equipe

Grupo 61 — POSTECH IA
