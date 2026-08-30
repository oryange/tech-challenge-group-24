"""Busca no banco o contexto clínico que vai para o prompt.

Uso como biblioteca:

    from src.assistant.retriever import PatientRetriever

    retriever = PatientRetriever.from_env()
    contexto = retriever.get_patient_context("[PACIENTE_007]")

É a peça que transforma as quatro tabelas do PR 03 no bloco de texto que o PR 07 injeta como
dado delimitado. A formatação mora aqui, e não no `prompts.py`, porque ela depende do formato
das tabelas: quem mexer no schema tem de mexer na formatação junto, e as duas coisas ficando
no mesmo arquivo isso é difícil de esquecer.

Sobre as consultas ao banco: todas passam pelo ORM do SQLAlchemy, que vincula os valores como
parâmetros. Nenhuma string de SQL é montada por concatenação ou f-string neste módulo — é o
que fecha a porta de SQL injection pelo `patient_id`, que chega da interface do assistente.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import Consultation, Exam, Patient, Protocol, get_engine

RAIZ = Path(__file__).resolve().parents[2]
DB_PATH_PADRAO = RAIZ / "data" / "database" / "hospital.db"

# Quantas consultas entram no contexto. Duas é o que o plano do projeto pede, e o número tem
# razão de ser: o histórico existe para responder "o que mudou desde a última vez?", e isso
# se responde comparando a atual com a anterior. Mais consultas empurrariam a pergunta para
# longe do fim do prompt sem acrescentar comparação nenhuma.
CONSULTAS_NO_CONTEXTO = 2

# Allowlist do identificador de paciente. O ORM já vincula o valor como parâmetro, então isto
# não é o que impede SQL injection — é validação de formato pelo caminho menos permissivo, e
# serve principalmente para transformar um identificador malformado num erro claro em vez de
# num resultado vazio silencioso, que na tela vira "paciente sem dados".
#
# O formato é o token que o seed do PR 03 grava em `Patient.name_anon` e que o PR 06 usa como
# chave de filtro do audit log — os três precisam concordar.
_PATIENT_ID_VALIDO = re.compile(r"^\[PACIENTE_\d{1,6}\]$")


class PacienteNaoEncontrado(LookupError):
    """Nenhum paciente com esse identificador no banco."""


def _formatar_data(valor: Any) -> str:
    """Data em DD/MM/AAAA, que é o formato que a citação de fonte usa."""
    return valor.strftime("%d/%m/%Y")


def _lista(campo: str) -> list[str]:
    """Quebra os campos separados por vírgula do PR 03 (`allergies`, `conditions`)."""
    return [item.strip() for item in (campo or "").split(",") if item.strip()]


class PatientRetriever:
    """Lê o banco de pacientes sintéticos e formata o contexto para o prompt."""

    def __init__(self, db_path: str | os.PathLike[str] = DB_PATH_PADRAO) -> None:
        self.db_path = Path(db_path)
        self._sessao = sessionmaker(bind=get_engine(self.db_path))

    @classmethod
    def from_env(cls) -> "PatientRetriever":
        """Constrói a partir de `DB_PATH` no `.env`.

        Mesma resolução do `src.database.seed`: caminho relativo é ancorado na raiz do
        repositório, senão o banco encontrado mudaria conforme o diretório de onde o comando
        foi disparado — e o assistente leria um banco diferente do que o seed populou.
        """
        caminho = Path(os.getenv("DB_PATH") or DB_PATH_PADRAO)
        if not caminho.is_absolute():
            caminho = RAIZ / caminho
        return cls(caminho)

    @staticmethod
    def _validar_patient_id(patient_id: str) -> str:
        if not isinstance(patient_id, str) or not _PATIENT_ID_VALIDO.match(patient_id):
            raise ValueError(
                "patient_id fora do formato esperado. Use o token do banco, "
                "por exemplo '[PACIENTE_007]'."
            )
        return patient_id

    def _buscar_paciente(self, sessao: Session, patient_id: str) -> Patient:
        paciente = (
            sessao.query(Patient).filter(Patient.name_anon == self._validar_patient_id(patient_id)).one_or_none()
        )
        if paciente is None:
            raise PacienteNaoEncontrado(f"Nenhum paciente com o identificador {patient_id}.")
        return paciente

    def listar_pacientes(self) -> list[str]:
        """Identificadores de todos os pacientes do banco, em ordem.

        Existe para a interface: o `patient_id` é um token gerado pelo seed, e sem uma forma
        de descobrir quais existem a única saída é abrir o SQLite na mão. Pedir um
        identificador que o usuário não tem como conhecer é o mesmo que não pedir nada.
        """
        with self._sessao() as sessao:
            return [
                nome
                for (nome,) in sessao.query(Patient.name_anon).order_by(Patient.name_anon).all()
            ]

    def get_pending_exams(self, patient_id: str) -> list[dict[str, str]]:
        """Exames com `status='pending'`, do mais recente para o mais antigo."""
        with self._sessao() as sessao:
            paciente = self._buscar_paciente(sessao, patient_id)
            exames = (
                sessao.query(Exam)
                .filter(Exam.patient_id == paciente.id, Exam.status == "pending")
                .order_by(Exam.date.desc())
                .all()
            )
            return [{"tipo": e.type, "data": _formatar_data(e.date)} for e in exames]

    def get_protocols(self, condition: str) -> list[dict[str, str]]:
        """Protocolos hospitalares de uma condição clínica.

        A comparação é exata e insensível a caixa, não `LIKE '%...%'`. O valor entra vinculado
        como parâmetro nos dois casos, então não é questão de injeção: é que `%` e `_` dentro
        de um `LIKE` são curingas, e um termo vindo da interface poderia casar o catálogo
        inteiro e despejar todos os protocolos dentro do prompt.
        """
        if not condition:
            return []
        with self._sessao() as sessao:
            protocolos = (
                sessao.query(Protocol)
                .filter(func.lower(Protocol.condition) == condition.strip().lower())
                .order_by(Protocol.cid_code)
                .all()
            )
            return [
                {
                    "condicao": p.condition,
                    "cid": p.cid_code,
                    "conduta": p.procedure,
                    "observacoes": p.notes,
                }
                for p in protocolos
            ]

    def get_patient_context(self, patient_id: str) -> dict[str, Any]:
        """Contexto completo do paciente, já formatado para entrar no prompt.

        Devolve o dicionário com os dados estruturados **e** o texto em `contexto`. Os dois
        juntos de propósito: o texto é o que vai para o modelo, e os campos estruturados são o
        que os testes e o notebook checam sem precisar fazer parsing do texto de volta.

        Sem método novo para o histórico, como o plano pede: as duas consultas mais recentes
        entram no mesmo contexto, com a data visível em cada uma — é ela que permite ao
        assistente citar `[Fonte: consulta de DD/MM/AAAA]` em vez de um "segundo o histórico"
        que ninguém consegue conferir depois.
        """
        with self._sessao() as sessao:
            paciente = self._buscar_paciente(sessao, patient_id)

            condicoes = _lista(paciente.conditions)
            alergias = _lista(paciente.allergies)

            exames = (
                sessao.query(Exam)
                .filter(Exam.patient_id == paciente.id)
                .order_by(Exam.date.desc())
                .all()
            )
            consultas = (
                sessao.query(Consultation)
                .filter(Consultation.patient_id == paciente.id)
                .order_by(Consultation.date.desc())
                .limit(CONSULTAS_NO_CONTEXTO)
                .all()
            )

            dados = {
                "patient_id": paciente.name_anon,
                "idade": paciente.age,
                "tipo_sanguineo": paciente.blood_type,
                "alergias": alergias,
                "condicoes": condicoes,
                "exames": [
                    {
                        "tipo": e.type,
                        "status": e.status,
                        "resultado": e.result,
                        "data": _formatar_data(e.date),
                    }
                    for e in exames
                ],
                "exames_pendentes": [
                    {"tipo": e.type, "data": _formatar_data(e.date)}
                    for e in exames
                    if e.status == "pending"
                ],
                "consultas": [
                    {
                        "data": _formatar_data(c.date),
                        "queixa": c.chief_complaint,
                        "avaliacao": c.assessment,
                        "conduta": c.plan,
                        "medico": c.physician_anon,
                    }
                    for c in consultas
                ],
            }

        # Fora da sessão de propósito: a formatação só usa o dicionário já materializado, e
        # deixá-la aqui garante que nenhum acesso preguiçoso a relacionamento dispare uma
        # query depois que a sessão fechou.
        dados["protocolos"] = [
            protocolo for condicao in condicoes for protocolo in self.get_protocols(condicao)
        ]
        dados["contexto"] = self._formatar(dados)
        return dados

    @staticmethod
    def _formatar(dados: dict[str, Any]) -> str:
        """Monta o texto do contexto. Rótulos explícitos, uma informação por linha.

        O formato é declarativo e sem prosa porque o bloco é lido pelo modelo como dado, não
        como texto corrido: rótulo e valor na mesma linha é o que permite ao modelo apontar de
        onde tirou cada afirmação quando for citar a fonte.
        """
        linhas = [
            f"Identificação: {dados['patient_id']}",
            f"Idade: {dados['idade']} anos",
            f"Tipo sanguíneo: {dados['tipo_sanguineo']}",
            f"Alergias: {', '.join(dados['alergias']) or 'nenhuma registrada'}",
            f"Condições: {', '.join(dados['condicoes']) or 'nenhuma registrada'}",
        ]

        # Os pendentes vêm em bloco próprio e afirmados, inclusive quando não há nenhum. A
        # lista única com `status` obrigava o modelo a deduzir a ausência de pendência a
        # partir de dois `done`, e observando as respostas ele não deduzia: ignorava o bloco
        # de exames e recitava o de protocolos. Um fato que precisa ser inferido não é fato
        # no contexto — tem de estar escrito.
        # Cada linha carrega o próprio estado no começo, e não só o cabeçalho do bloco.
        # Observado com o modelo real: perguntado sobre exames pendentes, ele pegou uma linha
        # do bloco de realizados — que tinha resultado normal registrado — e afirmou que
        # estava pendente. Cabeçalho de bloco é contexto que se perde quando o modelo copia
        # uma linha isolada; rótulo na linha viaja junto com ela.
        linhas.append("")
        if dados["exames_pendentes"]:
            linhas.append("Exames PENDENTES (aguardando realização ou resultado):")
            for exame in dados["exames_pendentes"]:
                linhas.append(f"- PENDENTE: {exame['tipo']} (solicitado em {exame['data']})")
        else:
            linhas.append("Exames PENDENTES: nenhum. Todos os exames deste paciente já "
                          "foram realizados e têm resultado.")

        linhas.append("")
        linhas.append("Exames JÁ REALIZADOS (estes NÃO estão pendentes):")
        realizados = [e for e in dados["exames"] if e["status"] != "pending"]
        if realizados:
            for exame in realizados:
                resultado = exame["resultado"] or "sem resultado registrado"
                linhas.append(
                    f"- JÁ REALIZADO em {exame['data']}: {exame['tipo']} — resultado: {resultado}"
                )
        else:
            linhas.append("- nenhum")

        linhas.append("")
        linhas.append(f"Consultas mais recentes (até {CONSULTAS_NO_CONTEXTO}):")
        if dados["consultas"]:
            for consulta in dados["consultas"]:
                linhas.append(f"- Consulta de {consulta['data']}, {consulta['medico']}:")
                linhas.append(f"    Queixa: {consulta['queixa']}")
                linhas.append(f"    Avaliação: {consulta['avaliacao']}")
                linhas.append(f"    Conduta: {consulta['conduta']}")
        else:
            linhas.append("- nenhuma consulta registrada")

        linhas.append("")
        linhas.append("Protocolos hospitalares aplicáveis:")
        if dados["protocolos"]:
            for protocolo in dados["protocolos"]:
                observacoes = f" ({protocolo['observacoes']})" if protocolo["observacoes"] else ""
                linhas.append(
                    f"- {protocolo['condicao']} [CID {protocolo['cid']}]: "
                    f"{protocolo['conduta']}{observacoes}"
                )
        else:
            linhas.append("- nenhum protocolo cadastrado para as condições deste paciente")

        return "\n".join(linhas)
