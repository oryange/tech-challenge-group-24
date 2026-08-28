"""Testes da trilha de auditoria."""

from __future__ import annotations

import json

import pytest

from src.audit.audit_logger import PREVIEW_CARACTERES, AuditLogger

CAMPOS_OBRIGATORIOS = {
    "timestamp",
    "session_id",
    "patient_id",
    "query",
    "response_preview",
    "source",
    "guardrail_triggered",
}


@pytest.fixture
def logger(tmp_path) -> AuditLogger:
    return AuditLogger(tmp_path / "audit.jsonl")


def _log(logger: AuditLogger, **kwargs) -> dict:
    padrao = {
        "query": "Quais exames estão pendentes?",
        "response": "Hemograma e glicemia. [Fonte: exames do paciente]",
        "patient_id": "[PACIENTE_007]",
        "source": "exames do paciente",
        "guardrail_triggered": False,
        "session_id": "sessao-01",
    }
    return logger.log(**{**padrao, **kwargs})


def test_log_creates_file(logger):
    assert not logger.log_path.exists()

    _log(logger)

    assert logger.log_path.exists()


def test_cria_o_diretorio_pai(tmp_path):
    # Falha na construção, não na primeira consulta clínica.
    destino = tmp_path / "sem" / "nada" / "aqui" / "audit.jsonl"

    AuditLogger(destino)

    assert destino.parent.is_dir()


def test_log_entry_has_required_fields(logger):
    entrada = _log(logger)

    assert set(entrada) == CAMPOS_OBRIGATORIOS


def test_log_grava_uma_linha_json_por_interacao(logger):
    _log(logger, session_id="a")
    _log(logger, session_id="b")

    linhas = logger.log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(linhas) == 2
    assert [json.loads(linha)["session_id"] for linha in linhas] == ["a", "b"]


def test_log_truncates_response(logger):
    entrada = _log(logger, response="R" * (PREVIEW_CARACTERES * 3))

    assert len(entrada["response_preview"]) == PREVIEW_CARACTERES


def test_log_anonimiza_pii_da_pergunta(logger):
    entrada = _log(logger, query="O paciente João Silva relata febre há 3 dias.")

    assert "João Silva" not in entrada["query"]
    assert "[PACIENTE]" in entrada["query"]


def test_log_anonimiza_antes_de_recortar(logger):
    # O CPF fica a cavaleiro do corte de 200 caracteres. Recortando primeiro, a regra do CPF
    # deixa de casar (ela exige os dois dígitos finais) e o pedaço da esquerda seria gravado
    # em claro. Anonimizando primeiro, o CPF inteiro vira token antes de qualquer corte.
    enchimento = "Evolução do quadro sem intercorrências relevantes. "
    prefixo = (enchimento * 5)[: PREVIEW_CARACTERES - 14]

    entrada = _log(logger, response=f"{prefixo}CPF 123.456.789-01 conforme cadastro.")

    assert "123.456" not in entrada["response_preview"]
    assert "123.456" not in logger.log_path.read_text(encoding="utf-8")


def test_patient_id_nao_e_anonimizado(logger):
    # Já é token do seed do PR 03; anonimizá-lo destruiria a chave de filtro.
    entrada = _log(logger, patient_id="[PACIENTE_007]")

    assert entrada["patient_id"] == "[PACIENTE_007]"
    assert logger.get_patient_logs("[PACIENTE_007]") == [entrada]


def test_preserva_acentuacao_no_arquivo(logger):
    _log(logger, query="Qual a conduta na crise asmática?")

    assert "asmática" in logger.log_path.read_text(encoding="utf-8")


def test_get_session_logs_filters_correctly(logger):
    _log(logger, session_id="sessao-A", query="Primeira da A.")
    _log(logger, session_id="sessao-B", query="Única da B.")
    _log(logger, session_id="sessao-A", query="Segunda da A.")

    da_sessao = logger.get_session_logs("sessao-A")

    assert len(da_sessao) == 2
    # Ordem de acontecimento preservada — é o que faz a trilha reconstruir a conversa.
    assert [e["query"] for e in da_sessao] == ["Primeira da A.", "Segunda da A."]


def test_get_patient_logs_filters_correctly(logger):
    _log(logger, patient_id="[PACIENTE_001]", session_id="s1")
    _log(logger, patient_id="[PACIENTE_002]", session_id="s1")
    _log(logger, patient_id="[PACIENTE_001]", session_id="s2")

    do_paciente = logger.get_patient_logs("[PACIENTE_001]")

    assert len(do_paciente) == 2
    # Atravessa sessões de propósito: é o histórico do paciente, não o da conversa.
    assert {e["session_id"] for e in do_paciente} == {"s1", "s2"}


def test_consulta_em_trilha_inexistente(logger):
    assert logger.get_session_logs("qualquer") == []
    assert logger.get_patient_logs("qualquer") == []


def test_linha_corrompida_nao_derruba_a_leitura(logger):
    _log(logger, session_id="s1", query="Íntegra antes.")
    with logger.log_path.open("a", encoding="utf-8") as arquivo:
        arquivo.write('{"session_id": "s1", "query": pela met\n')
    _log(logger, session_id="s1", query="Íntegra depois.")

    entradas = logger.get_session_logs("s1")

    assert [e["query"] for e in entradas] == ["Íntegra antes.", "Íntegra depois."]


def test_from_env_le_o_caminho(monkeypatch, tmp_path):
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "outro" / "audit.jsonl"))

    logger = AuditLogger.from_env()

    assert logger.log_path == tmp_path / "outro" / "audit.jsonl"


def test_from_env_ancora_relativo_na_raiz(monkeypatch):
    monkeypatch.setenv("AUDIT_LOG_PATH", "logs/audit.jsonl")

    logger = AuditLogger.from_env()

    assert logger.log_path.is_absolute()
    assert logger.log_path.parts[-2:] == ("logs", "audit.jsonl")
