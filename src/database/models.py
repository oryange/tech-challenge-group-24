"""Modelos SQLAlchemy do banco de pacientes sintéticos.

Uso, a partir da raiz do repositório:

    python -m src.database.seed   # cria as tabelas e popula com dados sintéticos

Quatro tabelas cobrem o que o enunciado da Fase 3 pede como "base estruturada": paciente,
exame, protocolo hospitalar e o prontuário (aqui `Consultation`) que dá dimensão temporal ao
contexto — sem histórico datado não há como o assistente citar "informações atualizadas do
paciente" nem uma fonte no formato `[Fonte: consulta de DD/MM/AAAA]`.

Nenhum dado real: `name_anon` e `physician_anon` só recebem os tokens que `src.data.anonymizer`
já produziria (`[PACIENTE_001]`, `[MÉDICO]`), mantendo a mesma convenção de anonimização usada
no dataset de fine-tuning.
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import Column, Date, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True)
    name_anon = Column(String, nullable=False, unique=True)
    age = Column(Integer, nullable=False)
    blood_type = Column(String, nullable=False)
    # allergies/conditions ficam como string separada por vírgula: SQLite não tem tipo
    # array nativo, e um paciente sintético raramente precisa de mais que 2-3 itens em
    # cada campo — uma tabela associativa seria complexidade sem benefício aqui.
    allergies = Column(String, nullable=False, default="")
    conditions = Column(String, nullable=False, default="")

    exams = relationship("Exam", back_populates="patient", cascade="all, delete-orphan")
    consultations = relationship(
        "Consultation",
        back_populates="patient",
        cascade="all, delete-orphan",
        order_by="desc(Consultation.date)",
    )


class Exam(Base):
    __tablename__ = "exams"

    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    type = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")  # "pending" | "done"
    result = Column(String, nullable=True)
    date = Column(Date, nullable=False)

    patient = relationship("Patient", back_populates="exams")


class Protocol(Base):
    __tablename__ = "protocols"

    id = Column(Integer, primary_key=True)
    condition = Column(String, nullable=False)
    cid_code = Column(String, nullable=False, unique=True)
    procedure = Column(String, nullable=False)
    notes = Column(Text, nullable=False, default="")


class Consultation(Base):
    __tablename__ = "consultations"

    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    date = Column(Date, nullable=False)
    chief_complaint = Column(Text, nullable=False)
    assessment = Column(Text, nullable=False)
    plan = Column(Text, nullable=False)
    physician_anon = Column(String, nullable=False)

    patient = relationship("Patient", back_populates="consultations")


def get_engine(db_path: str | os.PathLike[str]) -> Engine:
    """Cria a engine SQLite, garantindo que o diretório do arquivo exista.

    SQLite não cria o diretório pai por conta própria: um `DB_PATH` apontando para uma
    pasta nova falha com "unable to open database file" antes mesmo de chegar a criar
    tabelas. Isso é o que faz um clone limpo do repositório (sem `data/database/` ainda
    populado) funcionar de primeira.
    """
    caminho = Path(db_path)
    os.makedirs(caminho.parent, exist_ok=True)
    return create_engine(f"sqlite:///{caminho}")


def create_tables(engine: Engine) -> None:
    Base.metadata.create_all(engine)
