# Tech Challenge Fase 3 — Checklist de PRs

> Assistente Médico com LLM Fine-tuned + LangChain + LangGraph  
> Grupo 61 | MacBook Pro M3 24GB | Python 3.13

---

## Stack decidida

| Componente | Tecnologia |
|---|---|
| Modelo base | `meta-llama/Llama-3.2-3B-Instruct` |
| Fine-tuning | MLX-LM + LoRA (nativo Apple Silicon M3) |
| Dataset | PubMedQA + dados sintéticos hospitalares |
| Pipeline | LangChain |
| Fluxo automatizado | LangGraph |
| Banco de dados | SQLite + SQLAlchemy |
| Segurança | Guardrails custom + audit log JSON |

---

## Estrutura de pastas do projeto

```
tech-challenge-fase3/
├── data/
│   ├── raw/                        # PubMedQA baixado
│   ├── processed/                  # Dataset curado para fine-tuning (JSONL)
│   ├── synthetic/                  # Protocolos e laudos sintéticos
│   ├── database/                   # SQLite com pacientes sintéticos
│   └── fine_tuned/adapters/        # Pesos LoRA após fine-tuning
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_fine_tuning.ipynb
│   └── 03_langchain_demo.ipynb
├── src/
│   ├── data/                       # Pipeline de dados
│   ├── fine_tuning/                # Fine-tuning MLX-LM
│   ├── llm/                        # Wrapper LLM + guardrails
│   ├── assistant/                  # LangChain pipeline
│   ├── graph/                      # LangGraph flow
│   ├── database/                   # SQLAlchemy models + seed
│   └── logging/                    # Audit logger
├── tests/
├── docs/
│   ├── relatorio-tecnico.md
│   └── diagrama-langchain.md
├── logs/
├── .env.example
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## PRs — Divisão de trabalho

### PR 01 — Setup do projeto
**Responsável:** qualquer uma  
**Entrega:** repositório inicial pronto para desenvolvimento

- [ ] Criar repositório `tech-challenge-fase3` no GitHub
- [ ] Criar estrutura de pastas conforme acima
- [ ] Criar `requirements.txt` com todas as dependências
- [ ] Criar `.env.example` com as variáveis de ambiente
- [ ] Criar `pytest.ini`
- [ ] Criar todos os `__init__.py` dos pacotes em `src/`
- [ ] Criar `README.md` inicial (pode ser esqueleto, será completado depois)
- [ ] Criar `tests/__init__.py`
- [ ] Adicionar `.gitignore` (ignorar `data/raw/`, `data/fine_tuned/`, `logs/`, `.env`, `__pycache__`)

**Dependências de outras PRs:** nenhuma — deve ser a primeira

---

### PR 02 — Pipeline de dados
**Responsável:** Pessoa A  
**Entrega:** dados prontos para fine-tuning em `data/processed/dataset.jsonl`

- [ ] `src/data/loader.py`
  - Baixa PubMedQA via HuggingFace `datasets`
  - Filtra subset `pqa_labeled`
  - Converte para formato instruction-tuning: `{"instruction", "input", "output", "source"}`
  - Salva em `data/processed/pubmedqa.jsonl`
  - Executável diretamente: `python src/data/loader.py`

- [ ] `src/data/anonymizer.py`
  - Remove PII com regex: nomes, datas, CPF, telefones, emails
  - Substitui por tokens: `[PACIENTE]`, `[DATA]`, `[MÉDICO]`, `[PACIENTE_ID]`
  - Funções: `anonymize(text)` e `anonymize_record(dict)`

- [ ] `src/data/synthetic_generator.py`
  - Gera ~100 registros sintéticos: protocolos CID-10, laudos e FAQs médicas
  - Usa templates com variação aleatória (sem dados reais)
  - Salva em `data/synthetic/synthetic_hospital.jsonl`
  - Executável diretamente: `python src/data/synthetic_generator.py`

- [ ] `src/data/curator.py`
  - Merge de `pubmedqa.jsonl` + `synthetic_hospital.jsonl`
  - Aplica anonimização em todos os registros
  - Remove duplicatas e filtra respostas muito curtas (<20 palavras)
  - Salva dataset final em `data/processed/dataset.jsonl`
  - Executável diretamente: `python src/data/curator.py`

- [ ] `notebooks/01_data_preparation.ipynb`
  - Célula 1: carrega e exibe estatísticas do PubMedQA (total, distribuição de labels)
  - Célula 2: demonstra anonimização com exemplos antes/depois
  - Célula 3: exibe exemplos dos dados sintéticos gerados
  - Célula 4: estatísticas do dataset final (contagem, distribuição de fontes, tamanho médio)

- [ ] `tests/test_data.py`
  - `test_anonymize_removes_name()` — regex de nome substitui corretamente
  - `test_anonymize_removes_date()` — datas são substituídas
  - `test_anonymize_record_keys_preserved()` — instruction/input/output preservados
  - `test_to_instruction_format()` — saída tem os campos esperados
  - `test_synthetic_generator_output_count()` — gera a quantidade certa de registros
  - `test_curator_deduplication()` — duplicatas são removidas
  - `test_curator_quality_filter()` — registros curtos são filtrados

**Dependências de outras PRs:** PR 01

---

### PR 03 — Banco de dados e dados de pacientes
**Responsável:** Pessoa B  
**Entrega:** SQLite com pacientes sintéticos pronto para consulta

- [ ] `src/database/models.py`
  - `Base` declarativa SQLAlchemy
  - Model `Patient`: `id`, `name_anon`, `age`, `blood_type`, `allergies`, `conditions`
  - Model `Exam`: `id`, `patient_id` (FK), `type`, `status` (pending/done), `result`, `date`
  - Model `Protocol`: `id`, `condition`, `cid_code`, `procedure`, `notes`
  - Função `get_engine(db_path)` e `create_tables(engine)`

- [ ] `src/database/seed.py`
  - Cria e popula o banco com dados sintéticos (sem PII real):
    - 20 pacientes com `name_anon` = `[PACIENTE_001]` ... `[PACIENTE_020]`
    - 2–4 exames por paciente (alguns com `status=pending`)
    - 8–10 protocolos hospitalares (um por condição CID-10)
  - Executável diretamente: `python src/database/seed.py`

- [ ] `tests/test_data.py` (adicionar ou criar `test_database.py`)
  - `test_patient_creation()` — cria paciente e recupera do banco
  - `test_exam_pending_query()` — filtra exames pendentes por paciente
  - `test_protocol_by_condition()` — busca protocolo por condição
  - `test_seed_populates_records()` — seed cria o número esperado de registros

**Dependências de outras PRs:** PR 01

---

### PR 04 — Fine-tuning com MLX-LM
**Responsável:** Pessoa A  
**Entrega:** adapters LoRA treinados + notebook com métricas

> **Pré-requisito:** PR 02 mergeado e `data/processed/dataset.jsonl` gerado localmente

- [ ] `src/fine_tuning/config.py`
  - Dataclass `LoRAConfig` com todos os hiperparâmetros:
    - `model = "meta-llama/Llama-3.2-3B-Instruct"`
    - `lora_layers = 8`, `lora_rank = 8`, `lora_alpha = 16.0`
    - `learning_rate = 1e-4`, `num_iters = 500`, `batch_size = 4`
    - `max_seq_length = 512`, `val_batches = 25`
  - Método `to_mlx_args()` que retorna lista de args para `mlx_lm.lora`

- [ ] `src/fine_tuning/trainer.py`
  - Função `_prepare_mlx_data(dataset_path, output_dir)`:
    - Split 90/10 treino/validação
    - Converte para formato de prompt MLX-LM: `<s>[INST] ... [/INST] ... </s>`
    - Salva `data/processed/mlx/train.jsonl` e `valid.jsonl`
  - Função `train(config)`:
    - Chama `_prepare_mlx_data`
    - Executa `python -m mlx_lm.lora` via `subprocess` com os args do config
    - Salva adapters em `data/fine_tuned/adapters/`
  - Executável diretamente: `python src/fine_tuning/trainer.py`

- [ ] `src/fine_tuning/evaluator.py`
  - Função `evaluate(model_path, adapter_path, test_samples)`:
    - Gera respostas com `mlx_lm.generate`
    - Calcula ROUGE-L e BLEU-4 contra respostas de referência
    - Retorna dict com métricas
  - Função `save_results(metrics, path)`:
    - Salva em `docs/evaluation_results.json`
  - Executável diretamente: `python src/fine_tuning/evaluator.py`

- [ ] `notebooks/02_fine_tuning.ipynb`
  - Célula 1: instala dependências, configura `LoRAConfig`
  - Célula 2: prepara dados MLX, exibe split treino/validação
  - Célula 3: executa fine-tuning (ou carrega resultados pré-computados)
  - Célula 4: curvas de loss treino vs validação (gráfico)
  - Célula 5: avaliação ROUGE-L e BLEU-4 — baseline vs fine-tuned
  - Célula 6: exemplos qualitativos de respostas antes/depois do fine-tuning

**Dependências de outras PRs:** PR 01, PR 02

---

### PR 05 — LLM wrapper e guardrails
**Responsável:** Pessoa B  
**Entrega:** classe LLM compatível com LangChain + sistema de guardrails

- [ ] `src/llm/model.py`
  - Classe `MedicalMLXLLM(LLM)` herdando de `langchain_core.language_models.llm.LLM`
  - `_llm_type = "medical-mlx"`
  - `_call(prompt, stop, run_manager)`: chama `mlx_lm.generate` com adapter carregado
  - `_identifying_params`: expõe `model_path` e `adapter_path`
  - Carregamento lazy do modelo (singleton com cache)
  - Parâmetros: `model_path`, `adapter_path`, `max_tokens=512`, `temperature=0.2`

- [ ] `src/llm/guardrails.py`
  - `sanitize_input(text)`:
    - Remove tentativas de prompt injection (padrões `ignore previous`, `you are now`, `jailbreak`)
    - Limita tamanho máximo de input (trunca em 2000 chars)
  - `check_prescription_attempt(text)`:
    - Detecta intent de prescrição direta (palavras-chave: "prescrever", "receitar", "dose de", "administrar")
    - Retorna `(bool, warning_message)`
  - `validate_response(response)`:
    - Verifica se a resposta contém `[Fonte:` e `[Requer validação médica]`
    - Se não, adiciona footer padrão: `\n[Requer validação médica por profissional habilitado]`
  - `apply_guardrails(query, response)`:
    - Wrapper que aplica todas as checagens e retorna resposta segura

- [ ] `tests/test_guardrails.py`
  - `test_sanitize_removes_injection()` — prompt injection detectado e removido
  - `test_check_prescription_triggers()` — palavras de prescrição ativam o guardrail
  - `test_check_prescription_passes_question()` — pergunta normal não ativa guardrail
  - `test_validate_response_adds_disclaimer()` — disclaimer adicionado quando ausente
  - `test_validate_response_keeps_existing()` — não duplica disclaimer se já existe
  - `test_apply_guardrails_full_flow()` — fluxo completo com mock

**Dependências de outras PRs:** PR 01

---

### PR 06 — Audit logger
**Responsável:** Pessoa A ou B  
**Entrega:** sistema de logging estruturado para auditoria

- [ ] `src/logging/audit_logger.py`
  - Classe `AuditLogger`:
    - `__init__(log_path)`: inicializa com path do arquivo JSONL
    - `log(query, response, patient_id, source, guardrail_triggered, session_id)`:
      - Escreve linha JSON: `{"timestamp", "session_id", "patient_id", "query", "response_preview" (200 chars), "source", "guardrail_triggered"}`
    - `get_session_logs(session_id)`: retorna todos os logs de uma sessão
    - `get_patient_logs(patient_id)`: retorna histórico de consultas de um paciente
  - Instância global `audit_logger` configurada via `.env`

- [ ] `tests/test_audit_logger.py`
  - `test_log_creates_file()` — cria arquivo JSONL se não existir
  - `test_log_entry_has_required_fields()` — todos os campos obrigatórios presentes
  - `test_log_truncates_response()` — response_preview limitado a 200 chars
  - `test_get_session_logs_filters_correctly()` — filtra por session_id
  - `test_get_patient_logs_filters_correctly()` — filtra por patient_id

**Dependências de outras PRs:** PR 01

---

### PR 07 — Assistente LangChain
**Responsável:** Pessoa B  
**Entrega:** pipeline LangChain completo e funcional

> **Pré-requisito:** PRs 03, 05, 06 mergeados

- [ ] `src/assistant/prompts.py`
  - `SYSTEM_PROMPT`: define o papel do assistente, limites éticos, obrigatoriedade de citar fonte
  - `MEDICAL_TEMPLATE`: template com `{system}`, `{patient_context}`, `{history}`, `{question}`
  - Função `build_prompt(question, patient_context, history)` → string formatada

- [ ] `src/assistant/retriever.py`
  - Classe `PatientRetriever`:
    - `__init__(db_path)`: conecta ao SQLite via SQLAlchemy
    - `get_patient_context(patient_id)`:
      - Retorna dict com dados do paciente, exames recentes e protocolos relevantes
      - Formata como string para injetar no prompt
    - `get_pending_exams(patient_id)`: retorna lista de exames com `status=pending`
    - `get_protocols(condition)`: busca protocolos por condição clínica

- [ ] `src/assistant/chain.py`
  - Classe `MedicalAssistant`:
    - `__init__(llm, retriever, audit_logger)`: composição dos três componentes
    - `ask(question, patient_id, session_id)`:
      1. Sanitiza input via guardrails
      2. Recupera contexto do paciente via `retriever`
      3. Checa prescrição via guardrails
      4. Monta prompt com `build_prompt`
      5. Chama LLM via LangChain chain
      6. Valida resposta via guardrails
      7. Loga via audit_logger
      8. Retorna `{"response", "source", "guardrail_triggered", "patient_context_used"}`
    - `create_chain(llm)`: retorna `LLMChain` com `ConversationBufferMemory`
  - Executável interativo: `python src/assistant/chain.py`

- [ ] `tests/test_chain.py`
  - `test_ask_returns_required_fields()` — resposta tem todos os campos
  - `test_ask_triggers_guardrail_on_prescription()` — prescrição direta é bloqueada
  - `test_ask_includes_patient_context()` — contexto do paciente aparece na resposta
  - `test_ask_logs_to_audit()` — audit logger é chamado com parâmetros corretos
  - Todos os testes com mock do LLM (sem chamar o modelo real)

**Dependências de outras PRs:** PR 03, PR 05, PR 06

---

### PR 08 — Fluxo LangGraph
**Responsável:** Pessoa A  
**Entrega:** fluxo de decisão clínica automatizado

> **Pré-requisito:** PRs 03, 05, 06 mergeados

- [ ] `src/graph/clinical_flow.py`
  - TypedDict `ClinicalState`:
    - `patient_id: str`
    - `exams: list[dict]`
    - `pending_exams: list[dict]`
    - `alerts: list[str]`
    - `suggestions: str`
    - `requires_validation: bool`
    - `session_id: str`

  - Nós do grafo:
    - `intake_node(state)`: recebe dados iniciais, busca paciente no banco
    - `check_exams_node(state)`: consulta exames pendentes via `PatientRetriever`
    - `alert_team_node(state)`: formata e registra alertas para exames pendentes
    - `suggest_treatment_node(state)`: chama LLM para sugerir conduta
    - `human_validation_node(state)`: marca resposta como `requires_validation=True`

  - `build_graph()`: monta `StateGraph` com:
    - Borda condicional após `check_exams`: se há pendentes → `alert_team`, senão → `suggest_treatment`
    - Todas as arestas de transição explícitas
    - Retorna grafo compilado

  - `run_clinical_flow(patient_id, session_id)`: executa o grafo e retorna estado final
  - Executável: `python src/graph/clinical_flow.py`

- [ ] `tests/test_graph.py`
  - `test_intake_node_loads_patient()` — paciente carregado corretamente
  - `test_check_exams_finds_pending()` — detecta exames pendentes
  - `test_alert_generated_when_exams_pending()` — alerta emitido
  - `test_suggestion_generated_when_no_pending()` — sugestão gerada quando sem pendências
  - `test_requires_validation_always_true()` — validação humana sempre marcada
  - `test_full_flow_execution()` — fluxo completo retorna estado válido
  - Todos com mock do banco e do LLM

**Dependências de outras PRs:** PR 03, PR 05, PR 06

---

### PR 09 — Notebooks de demo e documentação
**Responsável:** Pessoa A ou B (podem dividir)  
**Entrega:** notebooks executados + documentação completa

- [ ] `notebooks/03_langchain_demo.ipynb`
  - Célula 1: setup e imports
  - Célula 2: inicializa `MedicalAssistant` com modelo fine-tuned
  - Célula 3: pergunta clínica simples → resposta com fonte citada
  - Célula 4: tentativa de prescrição direta → guardrail ativa
  - Célula 5: consulta com `patient_id` → contexto do paciente injetado
  - Célula 6: executa fluxo LangGraph completo para um paciente
  - Célula 7: exibe `logs/audit.jsonl` com as interações registradas

- [ ] `docs/relatorio-tecnico.md` — relatório obrigatório da Fase 3:
  - Introdução e objetivo
  - Arquitetura geral do sistema
  - Processo de fine-tuning (modelo, dados, técnica LoRA, hiperparâmetros)
  - Descrição do assistente médico e do pipeline LangChain
  - Diagrama do fluxo LangGraph (Mermaid)
  - Métricas de avaliação do modelo (ROUGE-L, BLEU-4 — baseline vs fine-tuned)
  - Segurança: guardrails, logging, explainability
  - Conclusão e trabalhos futuros

- [ ] `docs/diagrama-langchain.md`
  - Diagrama Mermaid do pipeline LangChain
  - Diagrama Mermaid do fluxo LangGraph

- [ ] `README.md` completo:
  - Descrição do projeto
  - Pré-requisitos (Python 3.11+, Apple Silicon recomendado, HuggingFace token)
  - Instalação passo a passo
  - Como rodar o pipeline completo (5 comandos em sequência)
  - Como rodar os testes (`pytest tests/`)
  - Estrutura do projeto
  - Equipe

**Dependências de outras PRs:** PRs 02, 04, 07, 08

---

### PR 10 — Integração final e testes de ponta a ponta
**Responsável:** Pessoa A ou B juntas  
**Entrega:** projeto completo, todos os testes passando

- [ ] Rodar `pytest tests/` — todos os testes passam (exceto `@integration`)
- [ ] Executar o pipeline completo de ponta a ponta:
  1. `python src/data/loader.py`
  2. `python src/data/synthetic_generator.py`
  3. `python src/data/curator.py`
  4. `python src/database/seed.py`
  5. `python src/assistant/chain.py` (interativo)
- [ ] Executar fluxo LangGraph: `python src/graph/clinical_flow.py`
- [ ] Verificar `logs/audit.jsonl` com registros reais
- [ ] Executar fine-tuning (pode ser rodado uma vez localmente e commitado o notebook executado)
- [ ] Revisar todos os notebooks — garantir que estão executados com output visível
- [ ] Revisão final do `docs/relatorio-tecnico.md`

---

## Ordem sugerida de PRs

```
PR 01 (setup)
  ├── PR 02 (dados)        → PR 04 (fine-tuning)  →  PR 09 (docs/demo)
  ├── PR 03 (banco)        ↗
  ├── PR 05 (LLM/guardrails)  →  PR 07 (LangChain)  →  PR 09
  ├── PR 06 (audit logger)    →  PR 08 (LangGraph)   →  PR 09
  └──────────────────────────────────────────────── PR 10 (integração final)
```

**Paralelismo possível após PR 01:**
- Pessoa A: PR 02 → PR 04 → PR 08
- Pessoa B: PR 03 → PR 05 → PR 06 → PR 07

---

## Entregáveis obrigatórios cobertos

| Entregável (Fase 3) | PR |
|---|---|
| Pipeline de fine-tuning | PR 04 |
| Integração com LangChain | PR 07 |
| Fluxos do LangGraph | PR 08 |
| Dataset anonimizado/sintético | PR 02 |
| Relatório técnico detalhado | PR 09 |
| Diagrama do fluxo LangChain | PR 09 |
| Avaliação do modelo | PR 04 |
| Logging e auditoria | PR 06 |
| Guardrails e limites | PR 05 |
| Explainability (citação de fonte) | PR 05 + PR 07 |
