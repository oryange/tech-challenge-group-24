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

- Python 3.11+
- Apple Silicon (M1/M2/M3) para fine-tuning via MLX — em Linux/x86 o `mlx-lm` não é instalado (o restante do projeto funciona normalmente)
- Conta no [HuggingFace](https://huggingface.co) com token de acesso
- Acesso ao modelo `meta-llama/Llama-3.2-3B-Instruct` (solicitar em huggingface.co/meta-llama)

## Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/oryange/tech-challenge-group-24.git
cd tech-challenge-group-24

# 2. Crie e ative o ambiente virtual
python -m venv venv
source venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure as variáveis de ambiente
cp .env.example .env
# Edite .env com seu token HuggingFace e caminhos desejados
```

### Versões das dependências

O `requirements.txt` declara piso e teto para cada pacote (ex.: `langchain>=0.3.0,<1.0.0`),
então um major novo não entra sozinho no ambiente. Para reproduzir exatamente o mesmo
conjunto de versões em outra máquina, gere um lock a partir do ambiente já validado:

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

# 2. Rodar o pytest. Neste estágio ainda não existem testes, então a saída esperada é
#    'no tests ran' com exit code 5 (o código do pytest para "nenhum teste coletado").
#    Não é falha de setup: significa que o pytest achou o pytest.ini e a pasta tests/.
pytest tests/ -v; echo "exit code: $?"
```

O `scripts/check_env.py` checa, em sete seções: se o venv está ativo; as dependências
instaladas com suas versões (tratando o `mlx-lm` como aviso fora de Apple Silicon); os
pacotes de `src/`; as APIs que cada PR vai usar (`LLM`, `ChatPromptTemplate`,
`RunnableWithMessageHistory`, `StateGraph`, `mlx_lm.generate`); que as APIs legadas do
LangChain 0.x (`LLMChain`, `ConversationBufferMemory`) **não** estão presentes; se o MLX
executa na GPU; e as variáveis do `.env`.

Ele sai com **exit 0** quando o ambiente está pronto e **exit 1** se algo essencial falta,
então serve direto em CI. As variáveis do `.env` são apenas avisos — só importam a partir
do PR 04.

> **Nota:** Os dois comandos acima devem passar já no PR 01 — eles só checam dependências,
> pacotes e a coleta do pytest. O que ainda **não** funciona são os comandos da seção
> seguinte: os módulos dentro de `src/` são implementados incrementalmente por PR, então
> até o PR 02 ser mergeado `python -m src.data.loader` falha com
> `No module named src.data.loader` (o módulo ainda não existe) e um
> `from src.data import loader` falha com `cannot import name 'loader' from 'src.data'`.

<!-- separador: mantém as duas notas como blocos distintos (markdownlint MD028) -->

> **Nota para CI:** exit code 5 não é sucesso para a maioria dos runners. Enquanto a
> ausência de testes for esperada, use `pytest tests/ -v || [ $? -eq 5 ]` para tolerar
> só esse código. A partir do PR 02 (primeiros testes) o esperado passa a ser exit 0,
> e essa tolerância deve ser removida — senão ela mascara uma suíte que parou de coletar.

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
