"""Popula o SQLite com pacientes, exames, protocolos e consultas sintéticos.

Uso, a partir da raiz do repositório:

    python -m src.database.seed

Grava em `DB_PATH` (padrão `data/database/hospital.db`, definido no `.env`). O banco não é
versionado — é gerado localmente por este módulo, o que mantém o repositório sem qualquer
arquivo de dados de paciente, mesmo sintético.

O catálogo clínico (condições, CID-10, exames, condutas) é **reaproveitado** de
`src.data.synthetic_generator`: os protocolos que o assistente consulta no banco precisam
falar dos mesmos quadros que o modelo viu no fine-tuning. Duas listas paralelas divergiriam
na primeira alteração, e o assistente citaria protocolo de uma condição que o modelo nunca
treinou.
"""

from __future__ import annotations

import os
import random
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.orm import Session

from src.data.anonymizer import TOKEN_MEDICO
from src.data.synthetic_generator import CONDICOES
from src.database.models import (
    Base,
    Consultation,
    Exam,
    Patient,
    Protocol,
    create_tables,
    get_engine,
)

RAIZ = Path(__file__).resolve().parents[2]
DB_PATH_PADRAO = RAIZ / "data" / "database" / "hospital.db"

TOTAL_PACIENTES = 20
SEED_PADRAO = 42

TIPOS_SANGUINEOS = ("A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-")
ALERGIAS = ("penicilina", "dipirona", "sulfa", "iodo", "látex", "AINE")

QUEIXAS = {
    "J45": "dispneia e sibilância a esforços habituais",
    "I10": "cefaleia occipital matinal e tontura",
    "E11": "poliúria e cansaço nas últimas semanas",
    "J18": "tosse produtiva com febre há três dias",
    "N39.0": "disúria e aumento da frequência urinária",
    "K29.7": "epigastralgia em queimação após as refeições",
    "J44": "dispneia progressiva e tosse matinal crônica",
    "G43": "cefaleia pulsátil unilateral com fotofobia",
    "A09": "diarreia aquosa e náuseas há dois dias",
    "I21": "dor torácica opressiva com irradiação para o braço esquerdo",
}

# Exames complementares que não são específicos de uma condição. Existem para que um
# paciente com uma única condição não termine com 3 exames idênticos: a rotina laboratorial
# aparece junto do exame do quadro, como aconteceria numa ficha real.
EXAMES_GERAIS: tuple[tuple[str, str], ...] = (
    ("hemograma completo", "série vermelha e branca sem alterações significativas"),
    ("função renal (ureia e creatinina)", "taxa de filtração glomerular preservada"),
    ("eletrocardiograma de repouso", "ritmo sinusal, sem alterações agudas"),
    ("perfil lipídico", "LDL acima da meta para o risco cardiovascular estimado"),
)

# Estágios da evolução clínica, do mais RECENTE para o mais ANTIGO. Ordenar em vez de
# sortear é o que dá ao histórico uma trajetória de verdade (início -> ajuste -> melhora):
# com frases aleatórias, "o que mudou desde a última consulta?" — a pergunta que demonstra o
# contexto atualizado do paciente — não tem resposta nos dados.
ESTAGIOS = (
    ("melhora parcial dos sintomas, com boa adesão ao tratamento", "manter {conduta}"),
    ("sintomas persistentes, sem sinais de gravidade", "ajuste da conduta: {conduta}"),
    ("primeiro atendimento do quadro", "início de {conduta}; solicitado {exame}"),
)


def _condicao_por_nome(nome: str) -> dict[str, str]:
    return next(c for c in CONDICOES if c["nome"] == nome)


def build_protocols() -> list[Protocol]:
    """Um protocolo por condição do catálogo CID-10."""
    return [
        Protocol(
            condition=condicao["nome"],
            cid_code=condicao["cid"],
            procedure=condicao["procedimento"],
            notes=(
                f"Avaliação: solicitar {condicao['exame']} para confirmação diagnóstica. "
                f"Conduta: {condicao['conduta']}. "
                f"Achado esperado no exame: {condicao['achado']}."
            ),
        )
        for condicao in CONDICOES
    ]


def build_patients(rng: random.Random, total: int = TOTAL_PACIENTES) -> list[Patient]:
    """Pacientes identificados só pelo token de anonimização (`[PACIENTE_001]`)."""
    pacientes = []
    for numero in range(1, total + 1):
        condicoes = rng.sample(CONDICOES, rng.randint(1, 2))
        alergias = rng.sample(ALERGIAS, rng.randint(0, 2))
        pacientes.append(
            Patient(
                name_anon=f"[PACIENTE_{numero:03d}]",
                age=rng.randint(18, 89),
                blood_type=rng.choice(TIPOS_SANGUINEOS),
                allergies=", ".join(alergias),
                # Só o nome da condição, sem o CID: é a chave que o retriever do assistente
                # usa em `get_protocols(condition)` para casar com `Protocol.condition`. Guardar
                # `"asma (J45)"` obrigaria o retriever a parsear a string antes de consultar.
                conditions=", ".join(c["nome"] for c in condicoes),
            )
        )
    return pacientes


def _condicoes_do_paciente(paciente: Patient) -> list[dict[str, str]]:
    """Devolve as condições do catálogo listadas em `conditions` (`"asma, migrânea"`)."""
    return [_condicao_por_nome(nome) for nome in paciente.conditions.split(", ")]


def build_exams(paciente: Patient, rng: random.Random, referencia: date) -> list[Exam]:
    """2 a 4 exames por paciente, parte deles ainda pendente de resultado.

    Os exames pendentes são o gatilho da borda condicional do fluxo LangGraph, então o seed
    garante que existam pacientes com pendência — sem eles, aquele caminho do grafo nunca
    seria exercitado numa demonstração.
    """
    # Um exame específico por condição do paciente, e o restante preenchido com rotina
    # laboratorial. Sortear o tipo com reposição dava fichas com o mesmo exame repetido três
    # vezes, com resultado idêntico — o que não ajuda ninguém a ler o caso.
    especificos = [(c["exame"], c["achado"]) for c in _condicoes_do_paciente(paciente)]
    gerais = list(EXAMES_GERAIS)
    rng.shuffle(gerais)
    catalogo = especificos + gerais

    total = rng.randint(2, 4)
    exames = []
    for indice, (tipo, achado) in enumerate(catalogo[:total]):
        pendente = rng.random() < 0.35
        # Pendente é exame recém-solicitado, então data recente; concluído pode ser antigo.
        # Sem essa distinção, aparecia exame aguardando resultado com data anterior à
        # consulta que o solicitou — o retriever injeta os dois no mesmo contexto.
        dias = rng.randint(0, 15) if pendente else rng.randint(20, 120) + indice
        exames.append(
            Exam(
                type=tipo,
                status="pending" if pendente else "done",
                result=None if pendente else achado,
                date=referencia - timedelta(days=dias),
            )
        )
    return exames


def build_consultations(
    paciente: Patient, rng: random.Random, referencia: date
) -> list[Consultation]:
    """2 a 3 consultas por paciente, da mais recente para a mais antiga.

    Todas acompanham a **mesma** condição — a primeira da ficha. É o que faz o histórico
    contar uma trajetória (primeiro atendimento -> ajuste -> melhora) em vez de três fotos
    soltas: sem isso, "o que mudou desde a última consulta?" não tem resposta nos dados, e é
    justamente a pergunta que demonstra o contexto atualizado exigido pelo enunciado.
    """
    condicao = _condicoes_do_paciente(paciente)[0]
    total = rng.randint(2, 3)
    # A consulta mais antiga é sempre o primeiro atendimento (último item de ESTAGIOS),
    # mesmo quando o histórico tem só duas entradas — um histórico que começa por "ajuste da
    # conduta" descreveria um tratamento que nunca foi iniciado.
    estagios = list(ESTAGIOS[: total - 1]) + [ESTAGIOS[-1]]

    consultas = []
    dias_atras = rng.randint(3, 30)
    for indice in range(total):
        evolucao, plano = estagios[indice]
        consultas.append(
            Consultation(
                date=referencia - timedelta(days=dias_atras),
                chief_complaint=QUEIXAS[condicao["cid"]],
                assessment=f"{condicao['nome']} (CID-10 {condicao['cid']}): {evolucao}.",
                plan=f"{plano.format(**condicao)}. Reavaliação clínica agendada.",
                physician_anon=TOKEN_MEDICO,
            )
        )
        dias_atras += rng.randint(30, 120)
    return consultas


def seed(
    db_path: str | os.PathLike[str] = DB_PATH_PADRAO,
    total_pacientes: int = TOTAL_PACIENTES,
    seed_aleatoria: int = SEED_PADRAO,
    referencia: date | None = None,
) -> dict[str, int]:
    """Recria o banco do zero e devolve a contagem de registros por tabela.

    As tabelas são derrubadas antes de recriar: rodar o seed duas vezes tem de deixar o
    banco no mesmo estado, e não com pacientes duplicados. São dados sintéticos gerados por
    este próprio módulo, então não há o que preservar.
    """
    referencia = referencia or date.today()
    rng = random.Random(seed_aleatoria)

    engine = get_engine(db_path)
    Base.metadata.drop_all(engine)
    create_tables(engine)

    protocolos = build_protocols()
    pacientes = build_patients(rng, total_pacientes)

    total_exames = 0
    total_consultas = 0
    for paciente in pacientes:
        paciente.exams = build_exams(paciente, rng, referencia)
        paciente.consultations = build_consultations(paciente, rng, referencia)
        total_exames += len(paciente.exams)
        total_consultas += len(paciente.consultations)

    with Session(engine) as sessao:
        sessao.add_all(protocolos)
        sessao.add_all(pacientes)
        sessao.commit()

    return {
        "pacientes": len(pacientes),
        "exames": total_exames,
        "protocolos": len(protocolos),
        "consultas": total_consultas,
    }


def main() -> dict[str, int]:
    load_dotenv(RAIZ / ".env")
    db_path = Path(os.getenv("DB_PATH") or DB_PATH_PADRAO)
    if not db_path.is_absolute():
        db_path = RAIZ / db_path

    contagens = seed(db_path)

    print(f"Banco: {db_path}")
    for tabela, total in contagens.items():
        print(f"{tabela.capitalize():12} {total}")
    return contagens


if __name__ == "__main__":
    main()
