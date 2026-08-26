"""Anonimização de PII antes do fine-tuning.

Uso como biblioteca:

    from src.data.anonymizer import anonymize, anonymize_record

O `curator.py` aplica isto em todos os registros do dataset. A premissa é que os dados
sintéticos hospitalares podem conter identificadores em português (nomes com título, CPF,
telefone, prontuário), enquanto o PubMedQA é texto científico em inglês.

Por isso as regras são **conservadoras e ancoradas em contexto**: nome só é substituído
quando vem precedido de um marcador ("Dr.", "paciente", "Sra."). Uma regra genérica de
"duas palavras capitalizadas = nome" destruiria termos científicos do PubMedQA
("Programmed Cell Death", nomes de genes, espécies) e degradaria o dataset de treino.
"""

from __future__ import annotations

import re
from typing import Iterable

TOKEN_PACIENTE = "[PACIENTE]"
TOKEN_MEDICO = "[MÉDICO]"
TOKEN_DATA = "[DATA]"
TOKEN_PACIENTE_ID = "[PACIENTE_ID]"
TOKEN_TELEFONE = "[TELEFONE]"
TOKEN_EMAIL = "[EMAIL]"

CAMPOS_ANONIMIZAVEIS = ("instruction", "input", "output")

_MESES = (
    "janeiro|fevereiro|março|marco|abril|maio|junho|julho|agosto|setembro|outubro|"
    "novembro|dezembro"
)
_PALAVRA_NOME = r"[A-ZÀ-Ý][\wÀ-ÿ'’-]+"
# Partículas em minúscula no meio do nome. Sem elas o match para no conectivo e o sobrenome
# sobra no texto ("paciente Maria da Silva Santos" -> "paciente [PACIENTE] da Silva Santos"),
# que é o formato de nome mais comum no Brasil.
_PARTICULA = r"d[aeo]s?"
# Nome com até 4 palavras, sem consumir o espaço final: se consumisse, a substituição
# precisaria recolocá-lo e isso exigia uma limpeza de espaços que mutilava texto legítimo
# (`P = .04` virava `P =.04` nos abstracts do PubMedQA).
_NOME_COMPLETO = (
    rf"{_PALAVRA_NOME}(?:\s+(?:(?:{_PARTICULA})\s+)?{_PALAVRA_NOME}){{0,3}}"
)

# A ORDEM importa: e-mail antes de telefone e CPF (contém dígitos e pontos que as outras
# regras casariam parcialmente), e datas antes de telefone (dd/mm/aaaa vs sequências de
# dígitos). Cada item é (regex compilada, substituição).
_REGRAS: tuple[tuple[re.Pattern[str], str], ...] = (
    # E-mail
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), TOKEN_EMAIL),
    # Identificadores de paciente rotulados: prontuário 123456, CNS, matrícula
    (
        re.compile(
            r"\b(?P<cue>prontu[áa]rio|matr[íi]cula|CNS|cart[ãa]o nacional de sa[úu]de)"
            r"(?P<sep>\s*(?:n[º°]?\.?)?\s*:?\s*)\d(?:[\d.\-/]*\d)?",
            re.IGNORECASE,
        ),
        lambda m: f"{m.group('cue')}{m.group('sep')}{TOKEN_PACIENTE_ID}",
    ),
    # CPF formatado ou rotulado
    (re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"), TOKEN_PACIENTE_ID),
    (re.compile(r"\bCPF\s*:?\s*\d{11}\b", re.IGNORECASE), f"CPF: {TOKEN_PACIENTE_ID}"),
    # Datas: dd/mm/aaaa, dd-mm-aaaa, aaaa-mm-dd e "12 de março de 2024"
    (re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"), TOKEN_DATA),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), TOKEN_DATA),
    (re.compile(rf"\b\d{{1,2}}\s+de\s+(?:{_MESES})\s+de\s+\d{{4}}\b", re.IGNORECASE), TOKEN_DATA),
    # Telefone brasileiro. Exige parênteses de DDD, +55 ou uma palavra-marcador: o padrão
    # solto `\d{4,5}-\d{4}` casava intervalos de anos ("2000-2012") em 40 campos do
    # PubMedQA e destruía o período dos estudos.
    (re.compile(r"\+55\s?\(?\d{2}\)?\s?9?\d{4,5}[-\s]?\d{4}\b"), TOKEN_TELEFONE),
    (re.compile(r"\(\d{2}\)\s?9?\d{4,5}[-\s]?\d{4}\b"), TOKEN_TELEFONE),
    (
        re.compile(
            r"\b(?P<cue>telefone|tel|celular|fone|contato|whatsapp)(?P<sep>\.?\s*:?\s*)"
            r"\+?\d[\d\s()-]{7,}\d",
            re.IGNORECASE,
        ),
        lambda m: f"{m.group('cue')}{m.group('sep')}{TOKEN_TELEFONE}",
    ),
    # Profissional de saúde: "Dr. João Carlos Silva", "DRA. MARIA" -> [MÉDICO]
    (re.compile(rf"\b(?:(?i:Dr|Dra|Doutor|Doutora))\.?\s+{_NOME_COMPLETO}"), TOKEN_MEDICO),
    # Nome de paciente, sempre ancorado num marcador que é preservado. O (?i:...) fica
    # restrito ao cue: aplicar re.IGNORECASE no padrão inteiro faria [A-ZÀ-Ý] casar
    # minúscula e o nome passaria a engolir palavra comum ("paciente com asma").
    # O separador aceita ":" ("Paciente: João Silva" é a forma padrão em laudo) mas NÃO
    # aceita ponto: "paciente. Solicitar exame" casaria o início da frase seguinte.
    (
        re.compile(
            rf"\b(?P<cue>(?i:paciente))(?P<sep>\s*:?\s+)(?P<nome>{_NOME_COMPLETO})"
        ),
        lambda m: f"{m.group('cue')}{m.group('sep')}{TOKEN_PACIENTE}",
    ),
    # Abreviações, onde o ponto pertence ao próprio marcador e não ao separador
    (
        re.compile(
            rf"\b(?P<cue>(?i:Sr|Sra|Srta))(?P<sep>\.?\s*:?\s+)(?P<nome>{_NOME_COMPLETO})"
        ),
        lambda m: f"{m.group('cue')}{m.group('sep')}{TOKEN_PACIENTE}",
    ),
)


def anonymize(texto: str | None) -> str:
    """Substitui PII por tokens. Idempotente: rodar duas vezes não altera o resultado."""
    if not texto:
        return ""
    resultado = texto
    for regex, substituicao in _REGRAS:
        resultado = regex.sub(substituicao, resultado)
    # Nenhuma normalização de espaço aqui de propósito: as regras não consomem o espaço
    # que segue o trecho substituído, e um "limpa-espaços" genérico alteraria texto sem
    # PII nenhuma — o que é pior que o problema que resolveria.
    return resultado


def anonymize_record(
    registro: dict[str, str], campos: Iterable[str] = CAMPOS_ANONIMIZAVEIS
) -> dict[str, str]:
    """Anonimiza os campos de texto preservando todas as chaves do registro.

    O campo `source` fica **fora** da anonimização de propósito: ele carrega a referência
    da fonte (`PubMedQA:21645374`) usada na citação exigida pelo requisito de
    explainability. Anonimizá-lo destruiria a rastreabilidade sem proteger ninguém — não é
    dado de paciente.
    """
    anonimizado = dict(registro)
    for campo in campos:
        if campo in anonimizado:
            anonimizado[campo] = anonymize(anonimizado[campo])
    return anonimizado
