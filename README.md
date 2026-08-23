# Tech Challenge Fase 3 — Assistente Médico com LLM + LangChain

> Grupo 24 | Pós-graduação em IA — POSTECH

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

- Python 3.11+ (o projeto é desenvolvido e validado em 3.13)
- Apple Silicon (M1/M2/M3) para fine-tuning via MLX — em Linux/x86 o `mlx-lm` não é instalado (o restante do projeto funciona normalmente)
- Conta no [HuggingFace](https://huggingface.co) com token de acesso
- Acesso aprovado ao modelo `meta-llama/Llama-3.2-3B-Instruct` — **peça primeiro**, ver abaixo

### Acesso ao modelo (faça isso antes de tudo)

O `meta-llama/Llama-3.2-3B-Instruct` é *gated*: exige aceitar a licença da Meta e aguardar
aprovação, que pode levar de minutos a dias. É a única dependência do projeto com espera
humana, então dispare no início e siga com o resto do setup em paralelo.

1. Logado no HuggingFace, abra
   [huggingface.co/meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
   e preencha o formulário de acesso
2. Crie um token em [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens),
   tipo *Read*. Se optar por *fine-grained*, marque **"Read access to contents of all public
   gated repos you can access"** — sem isso o token não abre o modelo mesmo com o acesso aprovado

Nada disso bloqueia o pipeline de dados: o PubMedQA é público.

## Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/oryange/tech-challenge-group-24.git
cd tech-challenge-group-24

# 2. Crie e ative o ambiente virtual — fixe o interpretador em 3.13.
#    Não use só "python3": em máquinas onde ele aponta para 3.14, parte das wheels
#    (o mlx-lm entre elas) ainda não existe e a instalação falha.
python3.13 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure as variáveis de ambiente
cp .env.example .env
# Edite .env e cole o MESMO token em HUGGINGFACE_TOKEN e HF_TOKEN.
# São dois nomes de propósito: HF_TOKEN é o que a biblioteca huggingface_hub lê.

# 5. Autentique a CLI do HuggingFace
#    O .env cobre só o código Python; comandos de CLI (mlx_lm, downloads) usam o token
#    gravado em ~/.cache/huggingface/token por este login.
hf auth login

# 6. Registre o kernel do Jupyter, para os notebooks rodarem com o venv
#    Sem isso os notebooks usam o Python do sistema e os imports falham.
python -m ipykernel install --user --name tc-fase3 --display-name "TC Fase 3"
```

O `.env` é ignorado pelo git (regra `*.env`) — nunca comite um token.

### Versões das dependências

O `requirements.txt` declara piso e teto para cada pacote (ex.: `langchain>=1.0.0,<2.0.0`),
então um major novo não entra sozinho no ambiente. Os tetos foram conferidos contra uma
instalação limpa em venv com Python 3.13 — não os afrouxe sem reinstalar do zero.

O projeto usa **LangChain 1.x**: o pipeline é LCEL (`prompt | llm`) com
`RunnableWithMessageHistory`. As APIs `LLMChain` e `ConversationBufferMemory` são da linha
0.x e saíram para o `langchain-classic`; o `scripts/check_env.py` falha de propósito se elas
reaparecerem, porque isso indica que o ecossistema foi rebaixado.

Para reproduzir exatamente o mesmo conjunto de versões em outra máquina, gere um lock a
partir do ambiente já validado:

```bash
pip freeze > requirements-lock.txt   # gerar (commitar quando a stack estiver estável)
pip install -r requirements-lock.txt # reproduzir
```

## Verificação do setup

Depois da instalação, rode a verificação de ambiente a partir da **raiz do repositório**,
com o venv ativo:

```bash
# 1. Verificação completa do ambiente
python -m scripts.check_env

# 2. Rodar a suíte de testes — a saída esperada é exit code 0, sem falhas.
pytest tests/ -v; echo "exit code: $?"
```

O `scripts/check_env.py` checa, em sete seções: se o venv está ativo; as dependências
instaladas com suas versões (tratando o `mlx-lm` como aviso fora de Apple Silicon); os
pacotes de `src/`; as APIs que o projeto usa (`LLM`, `ChatPromptTemplate`,
`RunnableWithMessageHistory`, `StateGraph`, `mlx_lm.generate`); que as APIs legadas do
LangChain 0.x (`LLMChain`, `ConversationBufferMemory`) **não** estão presentes; se o MLX
executa na GPU; e as variáveis do `.env`.

Ele sai com **exit 0** quando o ambiente está pronto e **exit 1** se algo essencial falta,
então serve direto em CI. As variáveis do `.env` são apenas avisos — só importam quando o
fine-tuning e o assistente são executados.

## Como executar o pipeline completo

Rode sempre a partir da raiz do repositório e na forma `python -m pacote.modulo`, nunca
`python src/pacote/modulo.py`: a segunda forma coloca o diretório do arquivo no `sys.path`
em vez da raiz, e qualquer `import src...` interno quebra com `No module named 'src'`.

```bash
# 1. Baixar e processar PubMedQA
python -m src.data.loader

# 2. Gerar dados sintéticos hospitalares
python -m src.data.synthetic_generator

# 3. Curar e unificar o dataset
python -m src.data.curator

# 4. Fine-tuning do modelo (requer Apple Silicon)
python -m src.fine_tuning.trainer

# 5. Avaliar o modelo — ROUGE-L e BLEU-4, baseline vs fine-tuned
#    Gera docs/evaluation_results.json, que alimenta as métricas do relatório técnico
python -m src.fine_tuning.evaluator

# 6. Popular banco de dados com pacientes sintéticos
python -m src.database.seed

# 7. Iniciar assistente médico interativo
python -m src.assistant.chain

# 8. Executar fluxo LangGraph
python -m src.graph.clinical_flow
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
tech-challenge-group-24/
├── data/
│   ├── raw/                # PubMedQA baixado (gerado localmente, não versionado)
│   ├── processed/          # Dataset curado (JSONL)
│   ├── synthetic/          # Dados sintéticos hospitalares
│   ├── database/           # SQLite de pacientes — DB_PATH (gerado localmente)
│   └── fine_tuned/adapters/ # Pesos LoRA — ADAPTER_PATH (gerado localmente)
├── logs/                   # audit.jsonl — AUDIT_LOG_PATH (gerado localmente)
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
│   └── audit/              # Audit logger (nome evita sombrear o `logging` da stdlib)
├── scripts/
│   └── check_env.py        # Verificação de ambiente (python -m scripts.check_env)
├── tests/
├── docs/
│   ├── relatorio-tecnico.md
│   ├── diagramas.md
│   └── evaluation_results.json  # Métricas ROUGE-L/BLEU-4 (gerado pelo evaluator)
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── CHECKLIST_FASE3.md
├── requirements.txt
├── pytest.ini
└── README.md
```

## Equipe

Grupo 24 — Pós-graduação em IA (POSTECH)

- Oryange Strifezze
- Larissa Nunes
