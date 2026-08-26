"""Testes do banco de dados e do seed."""

from __future__ import annotations

import re
from datetime import date

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.data.anonymizer import anonymize
from src.data.synthetic_generator import CONDICOES
from src.database.models import (
    Consultation,
    Exam,
    Patient,
    Protocol,
    create_tables,
    get_engine,
)
from src.database.seed import TOTAL_PACIENTES, seed

REFERENCIA = date(2026, 6, 1)


@pytest.fixture
def engine(tmp_path):
    # Subdiretório inexistente de propósito: get_engine tem de criá-lo.
    motor = get_engine(tmp_path / "novo" / "hospital.db")
    create_tables(motor)
    return motor


@pytest.fixture
def engine_populado(tmp_path):
    caminho = tmp_path / "seed" / "hospital.db"
    seed(caminho, referencia=REFERENCIA)
    return get_engine(caminho)


# --------------------------------------------------------------------------------------
# models.py
# --------------------------------------------------------------------------------------


def test_get_engine_cria_diretorio_pai(tmp_path):
    # SQLite não cria o diretório: sem isso, um DB_PATH novo estoura
    # "unable to open database file".
    destino = tmp_path / "a" / "b" / "hospital.db"

    create_tables(get_engine(destino))

    assert destino.is_file()


def test_patient_creation(engine):
    with Session(engine) as sessao:
        sessao.add(
            Patient(
                name_anon="[PACIENTE_001]",
                age=54,
                blood_type="O+",
                allergies="penicilina",
                conditions="asma",
            )
        )
        sessao.commit()

    with Session(engine) as sessao:
        paciente = sessao.scalars(select(Patient)).one()

        assert paciente.name_anon == "[PACIENTE_001]"
        assert paciente.age == 54
        assert paciente.blood_type == "O+"
        assert paciente.conditions == "asma"


def test_exam_pending_query(engine):
    with Session(engine) as sessao:
        paciente = Patient(name_anon="[PACIENTE_001]", age=40, blood_type="A+")
        paciente.exams = [
            Exam(type="espirometria", status="pending", date=REFERENCIA),
            Exam(type="radiografia de tórax", status="done", result="sem alterações", date=REFERENCIA),
        ]
        outro = Patient(name_anon="[PACIENTE_002]", age=61, blood_type="B-")
        outro.exams = [Exam(type="urocultura", status="pending", date=REFERENCIA)]
        sessao.add_all([paciente, outro])
        sessao.commit()
        paciente_id = paciente.id

    with Session(engine) as sessao:
        pendentes = sessao.scalars(
            select(Exam).where(Exam.patient_id == paciente_id, Exam.status == "pending")
        ).all()

        assert len(pendentes) == 1
        assert pendentes[0].type == "espirometria"
        assert pendentes[0].result is None


def test_protocol_by_condition(engine):
    with Session(engine) as sessao:
        sessao.add_all(
            [
                Protocol(condition="asma", cid_code="J45", procedure="nebulização assistida"),
                Protocol(condition="migrânea", cid_code="G43", procedure="escala de dor"),
            ]
        )
        sessao.commit()

    with Session(engine) as sessao:
        protocolo = sessao.scalars(select(Protocol).where(Protocol.condition == "asma")).one()

        assert protocolo.cid_code == "J45"
        assert protocolo.procedure == "nebulização assistida"


def test_exames_seguem_o_paciente_ao_ser_removido(engine):
    # cascade delete-orphan: apagar o paciente não pode deixar exame órfão apontando
    # para um patient_id inexistente.
    with Session(engine) as sessao:
        paciente = Patient(name_anon="[PACIENTE_001]", age=33, blood_type="AB+")
        paciente.exams = [Exam(type="hemoglobina glicada", status="pending", date=REFERENCIA)]
        sessao.add(paciente)
        sessao.commit()

        sessao.delete(paciente)
        sessao.commit()

        assert sessao.scalars(select(Exam)).all() == []


# --------------------------------------------------------------------------------------
# seed.py
# --------------------------------------------------------------------------------------


def test_seed_populates_records(engine_populado):
    with Session(engine_populado) as sessao:
        pacientes = sessao.scalars(select(Patient)).all()
        protocolos = sessao.scalars(select(Protocol)).all()
        exames = sessao.scalars(select(Exam)).all()
        consultas = sessao.scalars(select(Consultation)).all()

    assert len(pacientes) == TOTAL_PACIENTES
    assert len(protocolos) == len(CONDICOES)
    assert 2 * TOTAL_PACIENTES <= len(exames) <= 4 * TOTAL_PACIENTES
    assert 2 * TOTAL_PACIENTES <= len(consultas) <= 3 * TOTAL_PACIENTES


def test_seed_quantidades_por_paciente(engine_populado):
    # As faixas valem por paciente, não no agregado: um paciente sem nenhum exame passaria
    # pela soma total e deixaria o retriever sem contexto.
    with Session(engine_populado) as sessao:
        for paciente in sessao.scalars(select(Patient)).all():
            assert 2 <= len(paciente.exams) <= 4, paciente.name_anon
            assert 2 <= len(paciente.consultations) <= 3, paciente.name_anon


def test_seed_name_anon_segue_a_convencao_de_tokens(engine_populado):
    with Session(engine_populado) as sessao:
        nomes = [p.name_anon for p in sessao.scalars(select(Patient).order_by(Patient.id)).all()]

    assert nomes[0] == "[PACIENTE_001]"
    assert nomes[-1] == f"[PACIENTE_{TOTAL_PACIENTES:03d}]"
    assert all(re.fullmatch(r"\[PACIENTE_\d{3}\]", nome) for nome in nomes)


def test_consultations_ordered_by_date_desc(engine_populado):
    # É o que garante o "atualizadas" do contexto: o retriever injeta as 2 primeiras
    # consultas no prompt, então a mais recente tem de vir primeiro.
    with Session(engine_populado) as sessao:
        for paciente in sessao.scalars(select(Patient)).all():
            datas = [c.date for c in paciente.consultations]

            assert len(datas) >= 2
            assert datas == sorted(datas, reverse=True), paciente.name_anon


def test_seed_historico_conta_uma_trajetoria(engine_populado):
    # É o que dá resposta a "o que mudou desde a última consulta?": a consulta mais antiga é
    # sempre o primeiro atendimento e cada consulta traz uma conduta diferente da anterior.
    with Session(engine_populado) as sessao:
        for paciente in sessao.scalars(select(Patient)).all():
            consultas = paciente.consultations  # mais recente primeiro

            assert "primeiro atendimento" in consultas[-1].assessment, paciente.name_anon
            assert "primeiro atendimento" not in consultas[0].assessment, paciente.name_anon
            planos = [c.plan for c in consultas]
            assert len(set(planos)) == len(planos), paciente.name_anon


def test_seed_exames_do_paciente_nao_repetem_tipo(engine_populado):
    # Regressão: sortear o tipo com reposição dava fichas com o mesmo exame três vezes,
    # com resultado idêntico.
    with Session(engine_populado) as sessao:
        for paciente in sessao.scalars(select(Patient)).all():
            tipos = [exame.type for exame in paciente.exams]

            assert len(set(tipos)) == len(tipos), paciente.name_anon


def test_seed_gera_exames_pendentes(engine_populado):
    # Sem pendências, a borda condicional do fluxo LangGraph nunca seria exercitada.
    with Session(engine_populado) as sessao:
        pendentes = sessao.scalars(select(Exam).where(Exam.status == "pending")).all()

    assert pendentes
    assert all(exame.result is None for exame in pendentes)


def test_seed_pendente_e_mais_recente_que_a_primeira_consulta(engine_populado):
    # Exame aguardando resultado com data anterior à consulta que o solicitou seria
    # incoerente no contexto que o retriever monta com as duas coisas juntas.
    with Session(engine_populado) as sessao:
        for paciente in sessao.scalars(select(Patient)).all():
            primeiro_atendimento = paciente.consultations[-1].date
            for exame in paciente.exams:
                if exame.status == "pending":
                    assert exame.date >= primeiro_atendimento, paciente.name_anon


def test_seed_exames_concluidos_tem_resultado(engine_populado):
    with Session(engine_populado) as sessao:
        concluidos = sessao.scalars(select(Exam).where(Exam.status == "done")).all()

    assert concluidos
    assert all(exame.result for exame in concluidos)


def test_seed_nao_contem_pii(engine_populado):
    # Anonimizar o conteúdo do banco tem de ser operação nula: ele já nasce com tokens.
    with Session(engine_populado) as sessao:
        for paciente in sessao.scalars(select(Patient)).all():
            assert anonymize(paciente.name_anon) == paciente.name_anon
        for consulta in sessao.scalars(select(Consultation)).all():
            assert consulta.physician_anon == "[MÉDICO]"
            for campo in (consulta.chief_complaint, consulta.assessment, consulta.plan):
                assert anonymize(campo) == campo


def test_seed_e_idempotente(tmp_path):
    # Rodar o seed duas vezes não pode duplicar pacientes.
    caminho = tmp_path / "hospital.db"

    primeira = seed(caminho, referencia=REFERENCIA)
    segunda = seed(caminho, referencia=REFERENCIA)

    assert primeira == segunda
    with Session(get_engine(caminho)) as sessao:
        assert len(sessao.scalars(select(Patient)).all()) == TOTAL_PACIENTES


def test_seed_e_deterministico_por_seed(tmp_path):
    def fichas(caminho, seed_aleatoria):
        seed(caminho, seed_aleatoria=seed_aleatoria, referencia=REFERENCIA)
        with Session(get_engine(caminho)) as sessao:
            return [
                (p.name_anon, p.age, p.blood_type, p.conditions)
                for p in sessao.scalars(select(Patient).order_by(Patient.id)).all()
            ]

    assert fichas(tmp_path / "a.db", 42) == fichas(tmp_path / "b.db", 42)
    assert fichas(tmp_path / "c.db", 42) != fichas(tmp_path / "d.db", 7)


def test_seed_conditions_casa_direto_com_protocol_condition(engine_populado):
    # É o contrato do `get_protocols(condition)` do retriever: o valor gravado em
    # `Patient.conditions` tem de casar com `Protocol.condition` sem nenhum parse.
    with Session(engine_populado) as sessao:
        for paciente in sessao.scalars(select(Patient)).all():
            for condicao in paciente.conditions.split(", "):
                protocolo = sessao.scalars(
                    select(Protocol).where(Protocol.condition == condicao)
                ).one()

                assert protocolo.cid_code, f"{paciente.name_anon}: {condicao}"


def test_seed_conditions_nao_embute_cid(engine_populado):
    # Regressão: guardar "asma (J45)" obrigaria o retriever a parsear a string
    # antes de consultar os protocolos.
    with Session(engine_populado) as sessao:
        for paciente in sessao.scalars(select(Patient)).all():
            assert "(" not in paciente.conditions, paciente.name_anon


def test_seed_consultas_coerentes_com_as_condicoes(engine_populado):
    with Session(engine_populado) as sessao:
        for paciente in sessao.scalars(select(Patient)).all():
            condicoes = set(paciente.conditions.split(", "))
            for consulta in paciente.consultations:
                avaliada = consulta.assessment.split(" (CID-10 ")[0]

                assert avaliada in condicoes, paciente.name_anon
