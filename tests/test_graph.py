"""Testes do fluxo LangGraph de decisão clínica.

Nenhum peso é carregado e nenhum modelo é chamado: o LLM é um `FakeLLM` que herda de
`langchain_core.language_models.llms.LLM`, como no `test_chain.py`, e o banco é um SQLite
temporário populado pelo seed de `src/database/seed.py`. Herdar do `LLM` real em vez de usar um
mock solto é o que garante que a chain exercitada aqui é a mesma que roda em produção — e, neste
arquivo, que o `suggest_treatment_node` está mesmo passando pelo `MedicalAssistant` inteiro em vez
de falar com o modelo por fora dos guardrails.

Sobre os pacientes usados: o seed é determinístico (`SEED_PADRAO`), então com
`total_pacientes=3` o `[PACIENTE_001]` sai sem exame pendente e com duas alergias registradas,
e o `[PACIENTE_002]` sai com dois pendentes. São os dois ramos da borda condicional, e as
fixtures conferem a premissa antes de o teste valer alguma coisa — um seed que mude de
comportamento tem de quebrar como premissa falsa, não como asserção enigmática lá na frente.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.language_models.llms import LLM

from src.assistant.chain import MedicalAssistant
from src.assistant.retriever import PacienteNaoEncontrado, PatientRetriever
from src.audit.audit_logger import AuditLogger
from src.database.seed import seed
from src.graph.clinical_flow import (
    ClinicalState,
    Dependencias,
    PERGUNTA_DE_CONDUTA,
    SessaoInvalida,
    _tem_exames_pendentes,
    _validar_session_id,
    alert_team_node,
    build_graph,
    check_exams_node,
    human_validation_node,
    intake_node,
    main,
    nova_sessao,
    run_clinical_flow,
    suggest_treatment_node,
)
from src.llm.guardrails import RODAPE_VALIDACAO

SEM_PENDENTES = "[PACIENTE_001]"
COM_PENDENTES = "[PACIENTE_002]"
RESPOSTA_PADRAO = "Manter conduta atual e reavaliar em 30 dias. [Fonte: exames do paciente]"


class FakeLLM(LLM):
    """LLM programável que registra os prompts recebidos."""

    resposta: str = RESPOSTA_PADRAO
    prompts: list = []

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _call(self, prompt, stop=None, run_manager=None, **kwargs) -> str:
        self.prompts.append(prompt)
        return self.resposta


@pytest.fixture
def banco(tmp_path):
    caminho = tmp_path / "hospital.db"
    seed(caminho, total_pacientes=3)
    return caminho


@pytest.fixture
def trilha(tmp_path):
    return tmp_path / "audit.jsonl"


@pytest.fixture
def llm():
    return FakeLLM(prompts=[])


@pytest.fixture
def deps(banco, trilha, llm):
    retriever = PatientRetriever(banco)
    dependencias = Dependencias(
        retriever=retriever,
        assistant=MedicalAssistant(
            llm=llm,
            retriever=retriever,
            audit_logger=AuditLogger(trilha),
        ),
        audit_logger=AuditLogger(trilha),
    )

    # Premissa das fixtures, conferida uma vez: se o seed mudar, o teste tem de falhar aqui,
    # dizendo o que deixou de valer, e não três asserções adiante.
    assert retriever.get_pending_exams(SEM_PENDENTES) == []
    assert retriever.get_pending_exams(COM_PENDENTES) != []
    return dependencias


def estado_inicial(patient_id: str, session_id: str = "teste-01") -> ClinicalState:
    return {
        "patient_id": patient_id,
        "exams": [],
        "pending_exams": [],
        "alerts": [],
        "suggestions": "",
        "requires_validation": False,
        "session_id": session_id,
    }


def ler_trilha(caminho) -> list[dict]:
    if not caminho.exists():
        return []
    return [json.loads(linha) for linha in caminho.read_text(encoding="utf-8").splitlines()]


# ------------------------------------------------------------------------------- nós


def test_intake_node_loads_patient(deps):
    saida = intake_node(estado_inicial(COM_PENDENTES), deps)

    assert saida["patient_id"] == COM_PENDENTES
    assert len(saida["exams"]) == 3
    assert {"tipo", "status", "resultado", "data"} <= set(saida["exams"][0])
    # O nó zera os campos derivados: um estado herdado de execução anterior não pode
    # sobreviver ao intake e ser lido adiante como se fosse deste paciente.
    assert saida["pending_exams"] == []
    assert saida["alerts"] == []
    assert saida["suggestions"] == ""
    assert saida["requires_validation"] is False


def test_intake_node_falha_alto_com_paciente_inexistente(deps):
    # Seguir com o estado vazio faria o fluxo rodar inteiro, não achar pendência nenhuma
    # (porque não achou paciente nenhum) e terminar afirmando que está tudo em ordem.
    with pytest.raises(PacienteNaoEncontrado):
        intake_node(estado_inicial("[PACIENTE_999]"), deps)


def test_intake_node_recusa_identificador_fora_da_allowlist(deps):
    with pytest.raises(ValueError):
        intake_node(estado_inicial("1 OR 1=1"), deps)


def test_check_exams_finds_pending(deps):
    saida = check_exams_node(estado_inicial(COM_PENDENTES), deps)

    assert len(saida["pending_exams"]) == 2
    assert {"tipo", "data"} <= set(saida["pending_exams"][0])


def test_check_exams_sem_pendencia_devolve_lista_vazia(deps):
    assert check_exams_node(estado_inicial(SEM_PENDENTES), deps)["pending_exams"] == []


def test_alert_generated_when_exams_pending(deps):
    base = estado_inicial(COM_PENDENTES)
    estado = {**base, **check_exams_node(base, deps)}

    alertas = alert_team_node(estado, deps)["alerts"]

    # Cabeçalho com a contagem, mais uma linha por exame: quem lê precisa saber quantos são
    # antes de ler a lista, e quais são para agir sobre eles.
    assert len(alertas) == 3
    assert COM_PENDENTES in alertas[0]
    for exame in estado["pending_exams"]:
        assert any(exame["tipo"] in alerta for alerta in alertas)


def test_alerta_de_exame_pendente_e_registrado_na_trilha(deps, trilha):
    base = estado_inicial(COM_PENDENTES)
    estado = {**base, **check_exams_node(base, deps)}

    alert_team_node(estado, deps)

    entradas = ler_trilha(trilha)
    assert len(entradas) == 1
    assert entradas[0]["patient_id"] == COM_PENDENTES
    assert entradas[0]["session_id"] == "teste-01"
    assert entradas[0]["motivos"] == ["exames_pendentes"]


def test_suggestion_generated_when_no_pending(deps, llm):
    saida = suggest_treatment_node(estado_inicial(SEM_PENDENTES), deps)

    assert RESPOSTA_PADRAO.split(".")[0] in saida["suggestions"]
    # A pergunta do fluxo é a constante do módulo, não texto montado com dado do prontuário.
    assert PERGUNTA_DE_CONDUTA in llm.prompts[0]


def test_suggestion_passa_pelo_assistente_e_nao_pelo_llm_cru(deps, trilha):
    # É a decisão central do módulo: pelo assistente, a sugestão sai com contexto do paciente,
    # rodapé de validação garantido e trilha escrita. Falando direto com o LLM, o ramo
    # automatizado — o que roda sem ninguém olhando — teria menos garantias que a CLI.
    saida = suggest_treatment_node(estado_inicial(SEM_PENDENTES), deps)

    assert RODAPE_VALIDACAO in saida["suggestions"]
    assert ler_trilha(trilha)[0]["patient_id"] == SEM_PENDENTES


def test_alergia_do_prontuario_vira_alerta_do_fluxo(deps, llm):
    # O [PACIENTE_001] tem dipirona registrada como alergia, e o modelo a oferece.
    llm.resposta = "Sugiro dipirona para a dor. [Fonte: exames do paciente]"

    alertas = suggest_treatment_node(estado_inicial(SEM_PENDENTES), deps)["alerts"]

    # Em campo próprio, e não só embutida no texto: um consumidor do estado que não leia a
    # sugestão inteira continua enxergando a contraindicação.
    assert any("dipirona" in alerta for alerta in alertas)


def test_human_validation_node_impoe_o_rodape_quando_ele_falta(deps):
    estado = {**estado_inicial(SEM_PENDENTES), "suggestions": "Reavaliar em 30 dias."}

    saida = human_validation_node(estado, deps)

    assert saida["requires_validation"] is True
    assert saida["suggestions"].endswith(RODAPE_VALIDACAO)


def test_human_validation_node_nao_duplica_o_rodape(deps):
    ja_marcada = f"Reavaliar em 30 dias.\n{RODAPE_VALIDACAO}"
    estado = {**estado_inicial(SEM_PENDENTES), "suggestions": ja_marcada}

    assert human_validation_node(estado, deps)["suggestions"] == ja_marcada


def test_human_validation_node_nao_marca_sugestao_vazia(deps):
    # No ramo do alerta não há conduta: um rodapé sozinho afirmaria precisar de validação sem
    # ter nada dentro para validar.
    saida = human_validation_node(estado_inicial(COM_PENDENTES), deps)

    assert saida["suggestions"] == ""
    assert saida["requires_validation"] is True


# ------------------------------------------------------------------- borda condicional


@pytest.mark.parametrize(
    ("pendentes", "esperado"),
    [([], "suggest_treatment"), ([{"tipo": "hemograma", "data": "01/08/2026"}], "alert_team")],
)
def test_borda_condicional_escolhe_pelo_estado_do_prontuario(pendentes, esperado):
    estado = {**estado_inicial(COM_PENDENTES), "pending_exams": pendentes}

    assert _tem_exames_pendentes(estado) == esperado


# ------------------------------------------------------------------------ fluxo inteiro


def test_full_flow_execution(deps):
    estado = run_clinical_flow(SEM_PENDENTES, session_id="teste-fluxo", deps=deps)

    assert set(ClinicalState.__annotations__) <= set(estado)
    assert estado["patient_id"] == SEM_PENDENTES
    assert estado["session_id"] == "teste-fluxo"
    assert estado["exams"]
    assert estado["requires_validation"] is True


@pytest.mark.parametrize("patient_id", [SEM_PENDENTES, COM_PENDENTES])
def test_requires_validation_always_true(deps, patient_id):
    # Os dois ramos passam pelo `human_validation`, e é a aresta explícita de cada um que
    # garante isso — nenhum caminho termina antes do nó que impõe a validação humana.
    assert run_clinical_flow(patient_id, deps=deps)["requires_validation"] is True


def test_fluxo_com_pendencia_alerta_e_nao_sugere_conduta(deps, llm):
    estado = run_clinical_flow(COM_PENDENTES, deps=deps)

    assert estado["alerts"]
    # Escolha clínica do fluxo: conduta sobre exame que ainda não voltou parece completa e não
    # é. O modelo nem chega a ser chamado neste ramo.
    assert estado["suggestions"] == ""
    assert llm.prompts == []


def test_fluxo_sem_pendencia_sugere_conduta(deps, llm):
    estado = run_clinical_flow(SEM_PENDENTES, deps=deps)

    assert estado["pending_exams"] == []
    assert RODAPE_VALIDACAO in estado["suggestions"]
    assert len(llm.prompts) == 1


def test_execucao_do_fluxo_deixa_a_trilha_filtravel_por_sessao(deps, trilha):
    run_clinical_flow(COM_PENDENTES, session_id="auditoria-01", deps=deps)
    run_clinical_flow(SEM_PENDENTES, session_id="auditoria-01", deps=deps)

    assert len(AuditLogger(trilha).get_session_logs("auditoria-01")) == 2


# -------------------------------------------------------------------------- sessão


def test_sessao_gerada_e_imprevisivel_e_identificavel():
    primeira, segunda = nova_sessao(), nova_sessao()

    assert primeira != segunda
    assert primeira.startswith("fluxo-")
    assert _validar_session_id(primeira) == primeira


@pytest.mark.parametrize(
    "invalida",
    [
        "",
        "sessão com espaço",
        "a" * 65,
        # O campo não passa pela anonimização do `AuditLogger`, que cobre pergunta, resposta
        # e fonte.
        # Sem allowlist ele vira depósito de texto livre no arquivo que o vídeo grava.
        "paciente Maria Silva, 11987654321",
        "../../etc/passwd",
        '{"json": "forjado"}',
    ],
)
def test_session_id_fora_da_allowlist_e_recusado(invalida):
    with pytest.raises(SessaoInvalida):
        _validar_session_id(invalida)


def test_run_clinical_flow_recusa_sessao_invalida_antes_de_tocar_no_banco(deps):
    with pytest.raises(SessaoInvalida):
        run_clinical_flow(SEM_PENDENTES, session_id="sessão inválida", deps=deps)


# ---------------------------------------------------------------------------- grafo


def test_build_graph_liga_os_cinco_nos(deps):
    grafo = build_graph(deps).get_graph()

    assert {"intake", "check_exams", "alert_team", "suggest_treatment", "human_validation"} <= set(
        grafo.nodes
    )


# ------------------------------------------------------------------------------- CLI


def test_main_roda_todos_os_pacientes_e_mostra_os_dois_caminhos(deps, monkeypatch, capsys):
    monkeypatch.setattr(Dependencias, "from_env", classmethod(lambda cls: deps))

    main([])

    saida = capsys.readouterr().out
    assert "Caminho tomado após check_exams: alert_team" in saida
    assert "Caminho tomado após check_exams: suggest_treatment" in saida


def test_main_com_paciente_aceita_a_forma_abreviada(deps, monkeypatch, capsys):
    monkeypatch.setattr(Dependencias, "from_env", classmethod(lambda cls: deps))

    main(["--paciente", "2"])

    saida = capsys.readouterr().out
    assert COM_PENDENTES in saida
    assert SEM_PENDENTES not in saida


def test_main_com_paciente_inexistente_encerra_com_mensagem(deps, monkeypatch):
    monkeypatch.setattr(Dependencias, "from_env", classmethod(lambda cls: deps))

    with pytest.raises(SystemExit, match="não está no banco"):
        main(["--paciente", "999"])
