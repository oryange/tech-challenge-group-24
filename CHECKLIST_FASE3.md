# Tech Challenge Fase 3 — Checklist de PRs

> Assistente Médico com LLM Fine-tuned + LangChain + LangGraph  
> Grupo 24 | Requer Python 3.11+ (máquina de desenvolvimento: MacBook Pro M3 24GB, Python 3.13)

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
tech-challenge-group-24/
├── data/
│   ├── raw/                        # PubMedQA baixado
│   ├── processed/                  # Dataset curado para fine-tuning (JSONL)
│   ├── synthetic/                  # Protocolos e laudos sintéticos
│   ├── database/                   # SQLite com pacientes sintéticos
│   └── fine_tuned/adapters/        # Pesos LoRA após fine-tuning (ADAPTER_PATH)
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
│   └── audit/                      # Audit logger (evita sombrear o `logging` da stdlib)
├── scripts/
│   └── check_env.py                # Verificação de ambiente (exit 0/1, usável em CI)
├── tests/
├── docs/
│   ├── relatorio-tecnico.md
│   ├── diagramas.md
│   └── evaluation_results.json     # Métricas do evaluator (gerado no PR 04)
├── logs/
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── requirements.txt
├── pytest.ini
├── CHECKLIST_FASE3.md
└── README.md
```

---

## PRs — Divisão de trabalho

### PR 01 — Setup do projeto ✅
**Responsável:** qualquer uma  
**Entrega:** repositório inicial pronto para desenvolvimento  
**Branch:** `feat/pr01-setup-projeto`

- [x] Criar repositório `tech-challenge-group-24` no GitHub
- [x] Criar estrutura de pastas conforme acima
- [x] Criar `requirements.txt` com todas as dependências
- [x] Criar `.env.example` com as variáveis de ambiente
- [x] Criar `pytest.ini`
- [x] Criar todos os `__init__.py` dos pacotes em `src/`
- [x] Criar `README.md` inicial (pode ser esqueleto, será completado depois)
- [x] Criar `tests/__init__.py`
- [x] Adicionar `.gitignore` (ignorar `data/raw/`, `data/fine_tuned/`, `data/database/`,
      `logs/`, `.env`, `__pycache__` — com `.gitkeep` versionado em cada um deles;
      `data/processed/` e `data/synthetic/` ficam versionados de propósito, são entregáveis)
- [x] Adicionar `.pre-commit-config.yaml` (hoje `repos: []` — sem hooks; existe para não
      bloquear commits em quem tem o pre-commit instalado globalmente)
- [x] Criar `scripts/check_env.py` — verificação de ambiente com exit 0/1
      (`python -m scripts.check_env`): venv ativo, versões das dependências, pacotes de
      `src/`, APIs que cada PR usa, ausência das APIs legadas do LangChain 0.x, MLX na GPU
      e variáveis do `.env`. Rodar isto antes de começar qualquer PR

**Dependências de outras PRs:** nenhuma — deve ser a primeira

---

### PR 02 — Pipeline de dados ✅
**Responsável:** Pessoa A  
**Entrega:** dados prontos para fine-tuning em `data/processed/dataset.jsonl`

- [x] `src/data/loader.py`
  - Baixa PubMedQA via HuggingFace `datasets`
  - Filtra subset `pqa_labeled`
  - Converte para formato instruction-tuning: `{"instruction", "input", "output", "source"}`
  - Salva em `data/processed/pubmedqa.jsonl`
  - Executável diretamente: `python -m src.data.loader`

- [x] `src/data/anonymizer.py`
  - Remove PII com regex: nomes, datas, CPF, telefones, emails
  - Substitui por tokens: `[PACIENTE]`, `[DATA]`, `[MÉDICO]`, `[PACIENTE_ID]`
  - Funções: `anonymize(text)` e `anonymize_record(dict)`

- [x] `src/data/synthetic_generator.py`
  - Gera ~100 registros sintéticos cobrindo os quatro tipos exigidos pelo enunciado
    ("protocolos médicos do hospital; exemplos de perguntas frequentes feitas por médicos;
    modelos de laudos, receitas e procedimentos internos"):
    - protocolos CID-10
    - FAQs médicas
    - modelos de laudo
    - modelos de receita e de procedimento interno (posologia sempre fictícia, com o
      disclaimer de validação humana — o assistente nunca prescreve por conta própria)
  - Usa templates com variação aleatória (sem dados reais)
  - Campo `source` identifica o tipo do registro, para o assistente citar a fonte depois
  - Salva em `data/synthetic/synthetic_hospital.jsonl`
  - Executável diretamente: `python -m src.data.synthetic_generator`

- [x] `src/data/curator.py`
  - Merge de `pubmedqa.jsonl` + `synthetic_hospital.jsonl`
  - Aplica anonimização em todos os registros
  - Remove duplicatas e filtra respostas muito curtas (<20 palavras)
  - Salva dataset final em `data/processed/dataset.jsonl`
  - Executável diretamente: `python -m src.data.curator`

- [x] `notebooks/01_data_preparation.ipynb`
  - Célula 1: carrega e exibe estatísticas do PubMedQA (total, distribuição de labels)
  - Célula 2: demonstra anonimização com exemplos antes/depois
  - Célula 3: exibe exemplos dos dados sintéticos gerados
  - Célula 4: estatísticas do dataset final (contagem, distribuição de fontes, tamanho médio)

- [x] `tests/test_data.py`
  - `test_anonymize_removes_name()` — regex de nome substitui corretamente
  - `test_anonymize_removes_date()` — datas são substituídas
  - `test_anonymize_record_keys_preserved()` — instruction/input/output preservados
  - `test_to_instruction_format()` — saída tem os campos esperados
  - `test_synthetic_generator_output_count()` — gera a quantidade certa de registros
  - `test_curator_deduplication()` — duplicatas são removidas
  - `test_curator_quality_filter()` — registros curtos são filtrados

**Dependências de outras PRs:** PR 01

---

### PR 03 — Banco de dados e dados de pacientes ✅
**Responsável:** Pessoa B  
**Entrega:** SQLite com pacientes, exames, protocolos e histórico de consultas sintéticos

- [x] `src/database/models.py`
  - `Base` declarativa SQLAlchemy
  - Model `Patient`: `id`, `name_anon`, `age`, `blood_type`, `allergies`, `conditions`
  - Model `Exam`: `id`, `patient_id` (FK), `type`, `status` (pending/done), `result`, `date`
  - Model `Protocol`: `id`, `condition`, `cid_code`, `procedure`, `notes`
  - Model `Consultation` (o "prontuário" que o enunciado cita — histórico clínico datado):
    `id`, `patient_id` (FK), `date`, `chief_complaint` (queixa), `assessment` (avaliação),
    `plan` (conduta), `physician_anon`
    - `physician_anon` usa o token `[MÉDICO]` gerado pelo `anonymizer.py`, mantendo a
      coerência de anonimização com o resto do projeto
    - é o que dá dimensão temporal ao contexto: sem histórico não há como demonstrar as
      "informações atualizadas do paciente" que o enunciado exige, nem dar ao assistente
      uma fonte datada para citar (`[Fonte: consulta de DD/MM/AAAA]`)
  - Função `get_engine(db_path)` e `create_tables(engine)`
  - `get_engine` deve criar o diretório pai de `db_path` (`os.makedirs(..., exist_ok=True)`)
    antes de abrir a conexão — SQLite não cria o diretório e falha com
    `unable to open database file` se `DB_PATH` apontar para um caminho novo

- [x] `src/database/seed.py`
  - Cria e popula o banco com dados sintéticos (sem PII real):
    - 20 pacientes com `name_anon` = `[PACIENTE_001]` ... `[PACIENTE_020]`
    - 2–4 exames por paciente (alguns com `status=pending`)
    - 8–10 protocolos hospitalares (um por condição CID-10)
    - 2–3 consultas por paciente, em datas decrescentes (a mais recente primeiro), com
      queixa/avaliação/conduta coerentes com as `conditions` daquele paciente
  - Executável diretamente: `python -m src.database.seed`

- [x] `tests/test_data.py` (adicionar ou criar `test_database.py`)
  - `test_patient_creation()` — cria paciente e recupera do banco
  - `test_exam_pending_query()` — filtra exames pendentes por paciente
  - `test_protocol_by_condition()` — busca protocolo por condição
  - `test_seed_populates_records()` — seed cria o número esperado de registros
  - `test_consultations_ordered_by_date_desc()` — histórico do paciente vem da mais
    recente para a mais antiga (é o que garante o "atualizadas" do contexto)

**Dependências de outras PRs:** PR 01

---

### PR 04 — Fine-tuning com MLX-LM ✅
**Responsável:** Pessoa A  
**Entrega:** adapters LoRA treinados + notebook com métricas  
**Branch:** `feat/pr04-fine-tuning` (mergeado no `main` pelo PR #4)

> **Pré-requisito:** PR 02 mergeado e `data/processed/dataset.jsonl` gerado localmente

- [x] `src/fine_tuning/config.py`
  - Dataclass `LoRAConfig` com todos os hiperparâmetros:
    - `model = "meta-llama/Llama-3.2-3B-Instruct"`
    - `lora_layers = 8`, `lora_rank = 8`, `lora_alpha = 16.0`
    - `learning_rate = 1e-4`, `num_iters = 500`, `batch_size = 4`
    - `max_seq_length = 512`, `val_batches = 25`
  - Método `to_mlx_args()` que retorna lista de args para `mlx_lm.lora`

- [x] `src/fine_tuning/trainer.py`
  - Função `_prepare_mlx_data(dataset_path, output_dir)`:
    - Split 90/10 treino/validação
    - ~~Converte para formato de prompt MLX-LM: `<s>[INST] ... [/INST] ... </s>`~~
      Grava `{messages}` com os papéis user/assistant e deixa o MLX-LM aplicar o
      `chat_template` do próprio modelo. Escrever `[INST]` à mão produziria o template do
      Mistral, e o modelo aqui é o Llama-3.2 — o treino veria uma estrutura e a inferência
      outra. O porquê está em `trainer.py`, e o `evaluator._build_prompt` e o
      `model._aplicar_chat_template` do PR 05 seguem o mesmo formato de propósito.
    - Salva `data/processed/mlx/train.jsonl` e `valid.jsonl`
  - Função `train(config)`:
    - Chama `_prepare_mlx_data`
    - Executa `python -m mlx_lm.lora` via `subprocess` com os args do config
    - Salva adapters em `data/fine_tuned/adapters/`
  - Executável diretamente: `python -m src.fine_tuning.trainer`

- [x] `src/fine_tuning/evaluator.py`
  - Função `evaluate(model_path, adapter_path, test_samples)`:
    - Gera respostas com `mlx_lm.generate`
    - Calcula ROUGE-L e BLEU-4 contra respostas de referência
    - Retorna dict com métricas
  - Função `save_results(metrics, path)`:
    - Salva em `docs/evaluation_results.json`
  - Executável diretamente: `python -m src.fine_tuning.evaluator`
  - Além do plano: `available_checkpoints`, `best_checkpoint` e `materialize_checkpoint`
    escolhem o checkpoint pela validação em vez de assumir que o último é o melhor, e
    `compare()` roda baseline vs fine-tuned no mesmo processo — que é o que o
    `_MODELOS_CARREGADOS` do PR 05 tem de saber distinguir

- [x] `notebooks/02_fine_tuning.ipynb`
  - Célula 1: instala dependências, configura `LoRAConfig`
  - Célula 2: prepara dados MLX, exibe split treino/validação
  - Célula 3: executa fine-tuning (ou carrega resultados pré-computados)
  - Célula 4: curvas de loss treino vs validação (gráfico)
  - Célula 5: avaliação ROUGE-L e BLEU-4 — baseline vs fine-tuned
  - Célula 6: exemplos qualitativos de respostas antes/depois do fine-tuning

**Dependências de outras PRs:** PR 01, PR 02

---

### PR 05 — LLM wrapper e guardrails ✅
**Responsável:** Pessoa B  
**Entrega:** classe LLM compatível com LangChain + sistema de guardrails  
**Branch:** `feat/pr05-llm-guardrails`

- [x] `src/llm/model.py`
  - Classe `MedicalMLXLLM(LLM)` herdando de `langchain_core.language_models.llms.LLM`
    (LangChain 1.x — confirmar o caminho do import com
    `python -c "from langchain_core.language_models.llms import LLM; print('ok')"`
    antes de escrever o resto da classe)
  - `_llm_type = "medical-mlx"`
  - `_call(prompt, stop, run_manager)`: chama `mlx_lm.generate` com adapter carregado
    - o `stop` é aplicado depois da geração (o `mlx_lm.generate` não recebe a lista), na
      ocorrência mais à esquerda, para o contrato do LangChain valer
    - aplica o `chat_template` do tokenizer, tudo no papel `user` — igual ao treino, pelo
      mesmo motivo do `evaluator._build_prompt`
  - `_identifying_params`: expõe `model_path` e `adapter_path`
  - Carregamento lazy do modelo (singleton com cache)
    - chave `(modelo, adapter, revisão)`: com o adapter fora da chave, a comparação
      baseline vs fine-tuned do notebook recebe os mesmos pesos nas duas pontas
  - Parâmetros: `model_path`, `adapter_path`, `max_tokens=512`, `temperature=0.2`
  - `from_env()`: reusa o `LoRAConfig` para resolver `BASE_MODEL`/`ADAPTER_PATH` e liga
    `MAX_TOKENS` e `TEMPERATURE`, que estavam no `.env.example` sem nada consumindo
  - `model_config = ConfigDict(protected_namespaces=())`: o Pydantic v2 reserva o prefixo
    `model_` e avisa a cada instanciação de um campo `model_path`

- [x] `src/llm/guardrails.py`
  - `sanitize_input(text)`:
    - Remove tentativas de prompt injection (padrões `ignore previous`, `you are now`, `jailbreak`)
    - Limita tamanho máximo de input (trunca em 2000 chars)
    - Trunca **antes** de rodar as regex, e substitui por marcador em vez de apagar:
      apagar cola as pontas e reconstrói a instrução que se queria eliminar
    - Regex sem quantificador aninhado — um `(\s+\w+)+` transformaria a sanitização num
      vetor de negação de serviço por backtracking
  - `check_prescription_attempt(text)`:
    - Detecta intent de prescrição direta (palavras-chave: "prescrever", "receitar", "dose de", "administrar")
    - Casa por radical (`prescr(ev|iç|ic|it)`), não por flexão listada uma a uma
    - Retorna `(bool, warning_message)`
  - `validate_response(response)`:
    - Verifica se a resposta contém `[Fonte:` e `[Requer validação médica]`
    - Se não, adiciona footer padrão: `\n[Requer validação médica por profissional habilitado]`
    - A marca é comparada por **prefixo**, senão uma variante da frase leva rodapé dobrado
    - Fonte ausente é reportada, nunca preenchida: um `[Fonte:]` fabricado aqui destrói a
      explainability que o campo existe para dar
  - `apply_guardrails(query, response)`:
    - Wrapper que aplica todas as checagens e retorna resposta segura
    - Checa prescrição nos **dois** lados: a pergunta diz se pediram, a resposta diz se o
      modelo entregou posologia sem que ninguém pedisse — o caso mais perigoso
    - Devolve `ResultadoGuardrails(resposta, guardrail_triggered, tem_fonte, motivos)`,
      que é o que o PR 07 retorna e o PR 06 registra

- [x] `tests/test_guardrails.py`
  - `test_sanitize_removes_injection()` — prompt injection detectado e removido
  - `test_check_prescription_triggers()` — palavras de prescrição ativam o guardrail
  - `test_check_prescription_passes_question()` — pergunta normal não ativa guardrail
  - `test_validate_response_adds_disclaimer()` — disclaimer adicionado quando ausente
  - `test_validate_response_keeps_existing()` — não duplica disclaimer se já existe
  - `test_apply_guardrails_full_flow()` — fluxo completo com mock
  - `test_sanitize_neutraliza_sem_reconstruir_o_ataque()` — saída sem nenhum padrão residual
  - `test_apply_guardrails_detecta_prescricao_so_na_resposta()` — posologia não solicitada
  - `test_apply_guardrails_nao_inventa_fonte()` — ausência reportada, não remendada

- [x] `tests/test_llm_model.py` (não estava no plano; o wrapper também precisa de teste)
  - `mlx_lm` é substituído por módulo falso em `sys.modules`, não por `monkeypatch` sobre o
    pacote real: a suíte precisa rodar fora do Apple Silicon, onde o `mlx-lm` nem é instalado
  - cobre chat template, `stop`, repasse de `temperature`/`max_tokens`, cache do modelo,
    separação baseline vs fine-tuned e `from_env()`

**Dependências de outras PRs:** PR 01

---

### PR 06 — Audit logger ✅
**Responsável:** Pessoa A ou B  
**Entrega:** sistema de logging estruturado para auditoria  
**Branch:** `feat/pr06-audit-logger`

- [x] `src/audit/audit_logger.py`
  - Classe `AuditLogger`:
    - `__init__(log_path)`: inicializa com path do arquivo JSONL e cria o diretório pai
      (`os.makedirs(..., exist_ok=True)`) — sem isso, um `AUDIT_LOG_PATH` em diretório
      inexistente estoura `FileNotFoundError` na primeira escrita
    - `log(query, response, patient_id, source, guardrail_triggered, session_id)`:
      - Escreve linha JSON: `{"timestamp", "session_id", "patient_id", "query", "response_preview" (200 chars), "source", "guardrail_triggered"}`
      - `query` e `response_preview` passam pelo `anonymize` do PR 02 antes de gravar: a
        pergunta é texto livre digitado na hora e pode trazer nome, telefone ou prontuário
        reais — dado que nenhum pipeline anterior viu, porque eles anonimizam dataset e
        banco, não a conversa. Este arquivo é o único que persiste esse texto em disco, e
        ainda é exibido no notebook de demonstração e no vídeo de entrega
      - Anonimiza **antes** de recortar em 200 caracteres: recortando primeiro, o corte cai
        no meio de um dado e a regra deixa de casar (um CPF partido perde os dois dígitos
        finais que a regra exige), gravando o pedaço em claro
      - `patient_id` fica fora da anonimização — já é token do seed do PR 03
        (`[PACIENTE_007]`), e anonimizá-lo destruiria a chave de filtro sem proteger nada
      - `ensure_ascii=False`, senão "asmática" vira "asmática" e a trilha fica
        ilegível justamente na hora de exibi-la
    - `get_session_logs(session_id)`: retorna todos os logs de uma sessão
    - `get_patient_logs(patient_id)`: retorna histórico de consultas de um paciente
      (atravessa sessões de propósito — é o histórico do paciente, não o da conversa)
    - leitura pula linha corrompida em vez de estourar: o caso real é o processo morrer no
      meio de uma escrita, e uma trilha que se recusa a abrir por causa disso perde as
      entradas íntegras junto com a quebrada
  - ~~Instância global `audit_logger`~~ → acessor `get_audit_logger()`, configurado via
    `.env` (fecha o `AUDIT_LOG_PATH`, que era checado pelo `check_env` sem ninguém lê-lo —
    mesmo defeito que o PR 04 corrigiu no `BASE_MODEL`)
    - a instância construída no import lia o ambiente **antes** do `load_dotenv`, porque num
      programa de linha de comando os imports acontecem antes de qualquer execução: o
      `AUDIT_LOG_PATH` do `.env` era ignorado sem nada denunciar, e o PR 08 acabaria
      gravando num arquivo diferente do que o assistente do PR 07 usa
    - de quebra, o `mkdir` do construtor deixa de ser efeito colateral de import

- [x] `tests/test_audit_logger.py`
  - `test_log_creates_file()` — cria arquivo JSONL se não existir
  - `test_log_entry_has_required_fields()` — todos os campos obrigatórios presentes
  - `test_log_truncates_response()` — response_preview limitado a 200 chars
  - `test_get_session_logs_filters_correctly()` — filtra por session_id
  - `test_get_patient_logs_filters_correctly()` — filtra por patient_id
  - `test_log_anonimiza_antes_de_recortar()` — CPF a cavaleiro do corte não vaza
  - `test_log_anonimiza_pii_da_pergunta()` — nome na pergunta vira `[PACIENTE]`
  - `test_linha_corrompida_nao_derruba_a_leitura()` — entradas íntegras sobrevivem

**Dependências de outras PRs:** PR 01

---

### PR 07 — Assistente LangChain ✅
**Responsável:** Pessoa B  
**Entrega:** pipeline LangChain completo e funcional  
**Branch:** `feat/pr07-assistente`

> **Pré-requisito:** PRs 03, 05, 06 mergeados

- [x] `src/assistant/prompts.py`
  - `SYSTEM_PROMPT`: define o papel do assistente, limites éticos, obrigatoriedade de citar fonte
  - `MEDICAL_TEMPLATE`: template com `{system}`, `{patient_context}`, `{history}`, `{question}`
  - Função `build_prompt(question, patient_context, history)` → string formatada
  - Contexto e pergunta entram em blocos delimitados (`<contexto_do_paciente>`,
    `<pergunta_do_medico>`) e o `SYSTEM_PROMPT` declara que o conteúdo deles é **dado, nunca
    instrução** — é a proteção estrutural contra prompt injection que o `guardrails.py` do
    PR 05 aponta como sendo a de verdade, em oposição à denylist de padrões conhecidos
  - Tags em vez de cerca de crase: o fechamento precisa ser difícil de falsificar de dentro
  - Sem paciente selecionado o bloco leva um aviso explícito em vez de ficar vazio — bloco
    vazio faz o modelo preencher a lacuna sozinho, o oposto de citar fonte

- [x] `src/assistant/retriever.py`
  - Classe `PatientRetriever`:
    - `__init__(db_path)`: conecta ao SQLite via SQLAlchemy
    - `get_patient_context(patient_id)`:
      - Retorna dict com dados do paciente, exames recentes, protocolos relevantes e as
        **2 consultas mais recentes** (queixa, avaliação, conduta e data)
      - Formata como string para injetar no prompt, com a data de cada consulta visível —
        é o que permite ao assistente citar `[Fonte: consulta de DD/MM/AAAA]`
      - Exames pendentes saem em bloco próprio e **afirmados**, inclusive o caso "nenhum". A
        lista única com `status: done|pending` obrigava o modelo a deduzir a ausência de
        pendência a partir de dois `done`, e observando as respostas ele não deduzia:
        ignorava o bloco de exames e recitava o de protocolos. Um fato que precisa ser
        inferido não é fato no contexto — tem de estar escrito. Medido: o acerto factual
        passou de 0/3 para 3/3 nos pacientes conferidos na hora da mudança
      - Sem método novo de propósito: o histórico entra no contexto que o PR 07 já monta,
        evitando abrir superfície nova na interface do retriever
    - `get_pending_exams(patient_id)`: retorna lista de exames com `status=pending`
    - `get_protocols(condition)`: busca protocolos por condição clínica
      - Comparação exata e insensível a caixa, não `LIKE '%...%'`: `%` e `_` são curingas, e
        um termo vindo da interface casaria o catálogo inteiro dentro do prompt
    - `listar_pacientes()` (não estava no plano): o `patient_id` é token gerado pelo seed, e
      sem forma de descobrir quais existem a única saída era abrir o SQLite na mão — pedir um
      identificador que o usuário não tem como conhecer é o mesmo que não pedir nada
  - Todas as consultas passam pelo ORM, que vincula os valores como parâmetro — nenhuma
    string de SQL montada por concatenação ou f-string
  - `patient_id` validado por allowlist (`[PACIENTE_NNN]`) antes de chegar ao banco. Não é o
    que impede SQL injection (o ORM já impede): serve para transformar identificador
    malformado em erro claro, em vez de resultado vazio que na tela vira "paciente sem dados"
  - Paciente inexistente levanta `PacienteNaoEncontrado` em vez de devolver contexto vazio —
    responder sem contexto, mas parecendo que teve, é pior que falhar

- [x] `src/assistant/chain.py`
  - Classe `MedicalAssistant`:
    - `__init__(llm, retriever, audit_logger)`: composição dos três componentes
    - `ask(question, patient_id, session_id)`:
      1. Sanitiza input via guardrails
      2. Recupera contexto do paciente via `retriever`
      3. Checa prescrição via guardrails
      4. Monta prompt com `build_prompt`
      5. Invoca a chain LCEL (`chain.invoke({...})`)
      6. Valida resposta via guardrails
      7. Loga via audit_logger
      8. Retorna `{"response", "source", "guardrail_triggered", "patient_context_used"}`
    - `create_chain(llm)`: monta a chain no estilo **LCEL** do LangChain 1.x —
      `prompt | llm`, envolvida em `RunnableWithMessageHistory` para o histórico da conversa
      (`InMemoryChatMessageHistory` por `session_id`)
      - `LLMChain` e `ConversationBufferMemory` **não** existem mais no pacote principal do
        LangChain 1.x: foram para o `langchain-classic`. Não usar — o projeto fica preso à
        linha 0.3 e o pip rebaixa todo o ecossistema em volta
      - ⚠️ **Desvio do plano, com medição:** a `RunnableWithMessageHistory` foi removida. O
        `InMemoryChatMessageHistory` por `session_id` continua sendo o armazenamento, mas o
        histórico é injetado como **texto** no slot `{history}` do `MEDICAL_TEMPLATE` (que o
        próprio plano define como texto), não como turnos de mensagem.
        Motivo medido, `[PACIENTE_005]`, similaridade entre a resposta da 1ª e da 2ª
        pergunta, sendo elas completamente diferentes:

        | Forma do histórico | temp=0.2 | temp=0.7 |
        |---|---|---|
        | turnos `AI:` (`MessagesPlaceholder`) | 100% | 100% |
        | sem histórico (sessão nova) | 14% | 26% |
        | texto no bloco de dado | 24% | 10% |

        Com turno de assistente o modelo copiava literalmente a própria resposta anterior,
        nas duas temperaturas. É o mesmo motivo pelo qual o PR 05 não manda papel `system`:
        este modelo foi fine-tuned em pares soltos e nunca viu conversa multi-turno, então
        um bloco `AI: <resposta>` é estrutura fora da distribuição dele e a continuação mais
        provável é repeti-la. Bônus: a `RunnableWithMessageHistory` já estava deprecada, e
        os 17 `LangChainDeprecationWarning` da suíte sumiram junto.
      - Só os 3 últimos turnos entram, e a resposta anterior é cortada em 200 caracteres:
        resposta longa realimentada volta a ancorar a repetição, em versão atenuada
      - O contexto do paciente **não** entra no histórico: entraria de novo a cada turno e o
        prompt cresceria de forma quadrática
      - O rodapé do guardrail também não é realimentado: ensinar o modelo a escrevê-lo
        sozinho faria a marca deixar de distinguir o que o guardrail garantiu do que o
        modelo inventou
      - O histórico é atributo de instância, não global de módulo — duas instâncias não
        enxergam a conversa uma da outra e um teste não herda o histórico do anterior
    - O passo 3 usa o aviso de prescrição como **reforço dentro do contexto**, antes da
      inferência: quem pede posologia faz o modelo receber o limite junto do dado, em vez de
      só levar o carimbo depois
    - O passo 4 monta o prompt pelo `ChatPromptTemplate`, que consome as mesmas constantes de
      bloco do `prompts.py`. O `build_prompt` continua sendo a forma string (notebook e
      relatório técnico) — as duas partem dos mesmos marcadores para não divergirem
    - Só a pergunta saneada e o recorte da resposta vão para a trilha do PR 06. O contexto
      clínico fica de fora: já está no banco, e copiá-lo para um arquivo aberto no notebook e
      gravado no vídeo espalharia dado de paciente sem responder nada a mais na auditoria
  - Executável interativo: `python -m src.assistant.chain`
    - O paciente é validado na entrada, não na primeira pergunta: validar só dentro do `ask`
      fazia a interface reclamar depois de a pergunta já ter sido escrita, e descartá-la
    - `?` lista os pacientes do banco; Ctrl-C encerra em qualquer um dos dois campos
    - Passos numerados ("passo 1 de 2 — paciente") e erro específico para quem digita a
      pergunta no campo do paciente: dizer só "não está no banco" faz a pessoa tentar outro
      nome, quando o problema é ela estar no campo errado
    - `--paciente` / `--pergunta` fazem uma pergunta só e encerram, e `--listar` lista os
      pacientes sem carregar o modelo. O interativo é ruim para testar: obriga a esperar o
      carregamento e redigitar tudo a cada rodada, e o resultado não dá para repetir nem
      colar num relatório
    - O `--paciente` passa pela mesma normalização do passo 1. Encontrado rodando: `--paciente 5`
      — a forma que o próprio interativo ensina — chegava cru ao retriever e era recusado com
      uma mensagem pedindo o token completo. Duas formas de dizer o mesmo paciente na mesma
      interface, uma delas rejeitada, é defeito da interface
    - `MedicalAssistant.preload()` carrega o modelo logo depois do banner. O carregamento é
      preguiçoso (e deve ser, para quem importa a classe), mas na interface isso tornava a
      mensagem "carregando modelo e adapters" falsa: o prompt voltava na hora e a espera de
      ~6s caía dentro da primeira pergunta, junto com o que o MLX imprime no stderr ao
      inicializar. Não economiza tempo — põe a espera onde ela foi anunciada
    - Silencia o ruído do `transformers` e do `huggingface_hub` só no modo interativo (a
      variável tem de entrar no ambiente antes do import), e o aviso de depreciação só no
      `main()` — na suíte ele continua visível, que é onde serve de lembrete

- [x] `tests/test_chain.py`
  - `test_ask_returns_required_fields()` — resposta tem todos os campos
  - `test_ask_triggers_guardrail_on_prescription()` — prescrição direta é bloqueada
  - `test_ask_includes_patient_context()` — contexto do paciente aparece na resposta
  - `test_ask_logs_to_audit()` — audit logger é chamado com parâmetros corretos
  - Todos os testes com mock do LLM (sem chamar o modelo real)
  - O falso LLM **herda de `LLM`** em vez de ser um `Mock()`: um mock solto aceitaria
    qualquer coisa no operador `|` e o teste passaria mesmo com a chain montada errada
  - Cobre ainda: pergunta saneada antes de entrar no prompt, posologia oferecida sem ninguém
    pedir, histórico isolado entre sessões e entre instâncias, contexto não repetido a cada
    turno, paciente inexistente não chega a chamar o modelo, e o contexto clínico não
    aparecendo no `audit.jsonl`

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
  - Executável: `python -m src.graph.clinical_flow`

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
  - Avaliação do modelo: métricas ROUGE-L e BLEU-4, baseline vs fine-tuned
    (números vindos de `docs/evaluation_results.json`)
  - Análise dos resultados — o enunciado pede avaliação **e** análise, então não basta
    tabelar: interpretar onde o fine-tuning melhorou e onde não, com exemplos de respostas
    antes/depois, hipóteses para os casos ruins e limitações (tamanho do dataset, 3B
    parâmetros, LoRA de 8 camadas)
  - Segurança: guardrails, logging, explainability
  - Conclusão e trabalhos futuros

- [ ] `docs/diagramas.md`
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
  1. `python -m src.data.loader`
  2. `python -m src.data.synthetic_generator`
  3. `python -m src.data.curator`
  4. `python -m src.database.seed`
  5. `python -m src.assistant.chain` (interativo)
- [ ] Executar fluxo LangGraph: `python -m src.graph.clinical_flow`
- [ ] Verificar `logs/audit.jsonl` com registros reais
- [ ] Executar fine-tuning: `python -m src.fine_tuning.trainer` (pode ser rodado uma vez
      localmente e commitado o notebook executado)
- [ ] Executar a avaliação: `python -m src.fine_tuning.evaluator`
  - confirmar que `docs/evaluation_results.json` foi gerado com as métricas baseline vs
    fine-tuned — é a fonte dos números do relatório técnico e da demonstração do vídeo
- [ ] Revisar todos os notebooks — garantir que estão executados com output visível
- [ ] Revisão final do `docs/relatorio-tecnico.md`

---

### PR 11 — Vídeo de demonstração
**Responsável:** Pessoa A e B juntas
**Entrega:** vídeo de até 15 minutos (entregável obrigatório da Fase 3)

> **Pré-requisito:** PR 10 concluído — o vídeo grava o sistema já funcionando de ponta a ponta

Os quatro itens abaixo são exigidos explicitamente pelo enunciado e devem aparecer no vídeo:

- [ ] Treinamento e funcionamento da LLM personalizada
  - mostrar o fine-tuning (pode ser execução gravada ou o notebook `02_fine_tuning.ipynb`
    com as curvas de loss e as métricas ROUGE-L/BLEU-4)
- [ ] Execução de um fluxo automatizado
  - rodar `python -m src.graph.clinical_flow` e mostrar o caminho condicional do LangGraph
    (exames pendentes → alerta, sem pendências → sugestão de conduta)
- [ ] Resposta a perguntas clínicas contextualizadas
  - consulta com `patient_id` mostrando o contexto do paciente injetado no prompt
    e a fonte citada na resposta
  - fazer ao menos uma pergunta que só se responde com o histórico (ex.: "o que mudou
    desde a última consulta?"), para evidenciar as "informações atualizadas do paciente"
- [ ] Logs e validação das respostas
  - exibir `logs/audit.jsonl` com os registros da demo e o guardrail de prescrição sendo
    acionado (resposta marcada como `[Requer validação médica]`)

- [ ] Duração final ≤ 15 minutos
- [ ] Link do vídeo adicionado ao `README.md` e ao `docs/relatorio-tecnico.md`

---

## Ordem sugerida de PRs

```
PR 01 (setup)
  ├── PR 02 (dados)        → PR 04 (fine-tuning)  →  PR 09 (docs/demo)
  ├── PR 03 (banco)        ↗
  ├── PR 05 (LLM/guardrails)  →  PR 07 (LangChain)  →  PR 09
  ├── PR 06 (audit logger)    →  PR 08 (LangGraph)   →  PR 09
  └──────────────────────────────────────────────── PR 10 (integração final) → PR 11 (vídeo)
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
| Consulta a base estruturada (prontuários e registros) | PR 03 + PR 07 |
| Contexto atualizado do paciente | PR 03 + PR 07 |
| Fluxos do LangGraph | PR 08 |
| Dataset anonimizado/sintético | PR 02 |
| Relatório técnico detalhado | PR 09 |
| Diagrama do fluxo LangGraph | PR 09 |
| Avaliação do modelo e análise dos resultados | PR 04 + PR 09 |
| Logging e auditoria | PR 06 |
| Guardrails e limites (nunca prescrever sem validação humana) | PR 05 + PR 08 |
| Explainability (citação de fonte) | PR 05 + PR 07 |
| Vídeo de até 15 minutos | PR 11 |

---

## Pendências abertas na integração do PR 07

Medidas rodando o assistente completo contra o modelo fine-tuned real, não em teste com
mock. Nenhuma delas é defeito do PR 07 — o prompt chega ao modelo correto e completo, o que
dá para conferir trocando o LLM pelo `FakeLLM` de `tests/test_chain.py`.

### P1 — `TEMPERATURE=0.2` derruba o acerto factual (config, PR 07) ✅

Perguntando "quais exames estão pendentes?" aos 8 primeiros pacientes e conferindo a resposta
contra o `get_pending_exams` do banco, já com o bloco explícito de pendentes no contexto:

| Cenário | Acerto factual | Repetição média |
|---|---|---|
| `temp=0.2` (anterior) | 2/8 | 37% |
| `temp=0.7` (atual) | 6/8 | 45% |

- [x] `TEMPERATURE_PADRAO` em `src/llm/model.py` passou de `0.2` para `0.7`. A leitura
      original — "é configuração, nenhuma linha de código" — estava incompleta: o padrão
      embutido é o que vale para quem clona o repositório sem definir a variável, e deixá-lo
      em `0.2` faria o valor recomendado e o comportamento de fábrica discordarem.
- [x] `TEMPERATURE=0.7` no `.env` e no `.env.example`. O `from_env()` do PR 05 lê a variável,
      e ela **vence** o padrão embutido — trocar só o código teria deixado o valor efetivo em
      `0.2` sem nada denunciar. Conferido pelo caminho da aplicação (`load_dotenv` seguido de
      `MedicalMLXLLM.from_env().temperature`), que é o que responde "qual valor o assistente
      usa de fato", e não o que está escrito em cada arquivo separadamente.
- [ ] **Não** adicionar `repetition_penalty`: mexeria no `src/llm/model.py`, que é do PR 05,
      e a medição não sustentou o ganho.

A troca é acerto por fluidez, e vale: uma resposta correta e repetitiva é revisável, uma
resposta fluente e errada não. Uma medição anterior, feita antes do bloco explícito de
pendentes e olhando **só** repetição, dava `temp=0.2` em 71-77% e `temp=0.7` em 19-26% — a
conclusão é a mesma, o fundamento é este aqui.

Ressalva de método: 8 pacientes, uma amostra cada, sem seed fixa. A variância por paciente é
alta; o agregado entre cenários é o que se sustenta, o resultado de um paciente isolado não.
O `[PACIENTE_002]` (o único sem nenhum exame pendente) erra nas duas temperaturas — afirmar
uma ausência é o caso mais difícil e o que menos existe no treino.

### P2 — O modelo recita protocolo em vez de responder a pergunta (dataset, PR 04)

A temperatura não resolve isto. Perguntando "quais exames estão pendentes deste paciente?":

- `[PACIENTE_002]`, que **não tem nenhum exame pendente**, recebe o protocolo de
  gastroenterite recitado. O modelo nunca diz "nenhum pendente".
- `[PACIENTE_005]`, que tem exatamente um (`hemoglobina glicada`), tem o exame citado como
  conduta de protocolo, não como resposta. E junto vem alucinação: "meta de glicemia below
  6.5 mmol/l" — palavra em inglês e unidade errada (hemoglobina glicada é em %).

Composição do `data/processed/mlx/train.jsonl` (903 exemplos), que explica o padrão:

| Medida | Valor |
|---|---|
| Exemplos em inglês (PubMedQA) | 808 (89%) |
| Exemplos em português (sintéticos) | 95 (11%) |
| Exemplos que respondem sobre **dados estruturados de um paciente** | 0 |
| Exemplos cuja resposta cita `[Fonte:` | 91 (10%) |

- [ ] Avaliar o desequilíbrio 89/11 entre PubMedQA e sintéticos em português
- [ ] Gerar exemplos do formato que o assistente realmente usa: contexto estruturado de
      paciente na entrada, resposta que lê **aquele** contexto — inclusive o caso "não há
      nada pendente", que hoje não existe no treino
- [ ] Só 10% dos exemplos citam fonte, mas o `SYSTEM_PROMPT` do PR 07 exige citação em toda
      resposta. É o que produz citação improvisada e malformada (`[Fonte:CID A09]`,
      `[Fonte: avaliação:E11]`) — a explainability é cobrada na inferência e quase não é
      treinada

### P3 — Revisão de segurança: o que ficou fora do PR 07

Revisão de segurança da branch (Python, 6 arquivos): 0 críticos, 0 altos. Compliant no que
mais importava — SQL injection pelo `patient_id` (ORM com parâmetros vinculados mais a
allowlist ancorada), log injection no JSONL (`json.dumps` escapa `\n` e aspas), a ordem
anonimiza-antes-de-recortar, as regex do `guardrails.py` (lineares, sem quantificador
aninhado) e a camada estrutural de prompt injection.

Corrigido dentro deste PR:

- `neutralizar_delimitadores` passou a casar por regex tolerante a caixa e a espaço
  (`</PERGUNTA_DO_MEDICO>` e `</ pergunta_do_medico >` fechavam o bloco e passavam intactos)
- `AuditLogger.log` ganhou teto de 2000 caracteres antes do `anonymize`, porque `log()` é API
  pública e nem todo chamador passa pelo `sanitize_input` do PR 05
- a trilha e o diretório dela passaram a ser criados em `0600`/`0700`

O que **não** foi corrigido, e por quê:

- [ ] `DB_PATH` sem `expanduser()`/`resolve()` no `retriever.from_env`. O arquivo do PR 07
      espelha o `src/database/seed.py` de propósito — consertar só um lado faz o assistente
      ler o home de verdade enquanto o seed popula um diretório chamado `~`, e a falha aparece
      como "paciente sem dados". A correção sai nos dois, no arquivo do PR 03.
- [ ] O `anonymize` do PR 02 é denylist **ancorada em contexto**: redige nome precedido de
      "paciente"/"Dr.", mas `"João Silva ainda está com febre?"` — como um médico digita — vai
      em claro para o `audit.jsonl`. A função foi escrita para curar dataset, onde o texto é
      estruturado, e está sendo reusada sobre digitação livre. Uma passada não ancorada, ou
      falhar fechado quando não dá para redigir com confiança, é mudança no `anonymizer.py`
      (PR 02) e afeta o dataset inteiro.
- [ ] O `response_preview` de 200 caracteres é derivado do contexto clínico. O PR 07 acerta ao
      não gravar o contexto cru, mas a resposta o reafirma — e o arquivo é aberto no notebook e
      gravado no vídeo de entrega. Decidir se `source` + `guardrail_triggered` já respondem as
      perguntas de auditoria (e o preview sai) é decisão de produto do PR 06, não ajuste local.
- **Containment de caminho não é bug e não será adicionado.** `AUDIT_LOG_PATH` e `DB_PATH`
  saem do `.env` de quem roda o comando — mesma decisão e mesmo motivo já documentados em
  `config._do_ambiente`: não há fronteira de privilégio para defender, e apontar log ou banco
  para fora do repositório é o motivo de as variáveis existirem.
- `_historicos` e `_MODELOS_CARREGADOS` sem eviction só importam se a classe virar serviço; a
  CLI usa uma sessão fixa. Fica para o PR que expuser o assistente por HTTP, se houver.
