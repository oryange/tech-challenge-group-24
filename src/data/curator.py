"""Curadoria final: une, anonimiza, deduplica e filtra o dataset de fine-tuning.

Uso, a partir da raiz do repositório:

    python -m src.data.loader              # gera data/processed/pubmedqa.jsonl
    python -m src.data.synthetic_generator # gera data/synthetic/synthetic_hospital.jsonl
    python -m src.data.curator             # gera data/processed/dataset.jsonl

O `dataset.jsonl` produzido aqui é a entrega do PR 02 e a entrada do fine-tuning (PR 04).
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from src.data.anonymizer import anonymize_record
from src.data.loader import load_jsonl, save_jsonl

RAIZ = Path(__file__).resolve().parents[2]

ENTRADA_PUBMEDQA = RAIZ / "data" / "processed" / "pubmedqa.jsonl"
ENTRADA_SINTETICA = RAIZ / "data" / "synthetic" / "synthetic_hospital.jsonl"
SAIDA_PADRAO = RAIZ / "data" / "processed" / "dataset.jsonl"

MIN_PALAVRAS_RESPOSTA = 20
SEED_EMBARALHAMENTO = 42


def _chave_dedup(registro: dict[str, str]) -> tuple[str, str]:
    """Identidade do exemplo para deduplicação: o par pergunta + contexto.

    Não inclui o `output`: dois exemplos com a mesma pergunta e respostas diferentes ainda
    são o mesmo prompt para o modelo, e manter os dois só ensina resposta inconsistente.
    """
    return (
        " ".join(registro.get("instruction", "").lower().split()),
        " ".join(registro.get("input", "").lower().split()),
    )


def deduplicate(registros: list[dict[str, str]]) -> list[dict[str, str]]:
    """Remove exemplos repetidos preservando a ordem e a primeira ocorrência."""
    vistos: set[tuple[str, str]] = set()
    unicos: list[dict[str, str]] = []
    for registro in registros:
        chave = _chave_dedup(registro)
        if chave in vistos:
            continue
        vistos.add(chave)
        unicos.append(registro)
    return unicos


def filter_by_quality(
    registros: list[dict[str, str]], min_palavras: int = MIN_PALAVRAS_RESPOSTA
) -> list[dict[str, str]]:
    """Descarta exemplos sem pergunta/resposta ou com resposta curta demais.

    Resposta muito curta no PubMedQA costuma ser uma conclusão truncada ("Yes.", "No
    difference was found."), que como alvo de treino ensina o modelo a responder sem
    fundamentar — o oposto do que um assistente clínico precisa fazer.
    """
    filtrados = []
    for registro in registros:
        instrucao = registro.get("instruction", "").strip()
        resposta = registro.get("output", "").strip()
        if not instrucao or not resposta:
            continue
        if len(resposta.split()) < min_palavras:
            continue
        filtrados.append(registro)
    return filtrados


def curate(
    caminhos: tuple[Path, ...] = (ENTRADA_PUBMEDQA, ENTRADA_SINTETICA),
    min_palavras: int = MIN_PALAVRAS_RESPOSTA,
    seed: int = SEED_EMBARALHAMENTO,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Executa o pipeline de curadoria e devolve (registros, estatísticas)."""
    brutos: list[dict[str, str]] = []
    for caminho in caminhos:
        brutos.extend(load_jsonl(caminho))

    anonimizados = [anonymize_record(r) for r in brutos]
    sem_duplicatas = deduplicate(anonimizados)
    filtrados = filter_by_quality(sem_duplicatas, min_palavras)

    # Embaralhamento determinístico. O PR 04 faz split sequencial 90/10 treino/validação:
    # sem embaralhar, os registros hospitalares (que vêm todos no fim do arquivo) cairiam
    # inteiros na validação e o modelo nunca treinaria com os dados próprios do hospital.
    embaralhados = list(filtrados)
    random.Random(seed).shuffle(embaralhados)

    estatisticas = {
        "lidos": len(brutos),
        "duplicatas_removidas": len(anonimizados) - len(sem_duplicatas),
        "curtos_removidos": len(sem_duplicatas) - len(filtrados),
        "final": len(embaralhados),
    }
    return embaralhados, estatisticas


def main(caminho: Path = SAIDA_PADRAO) -> int:
    registros, estatisticas = curate()
    gravados = save_jsonl(registros, caminho)

    tipos = Counter(r["source"].split(":")[0] for r in registros)
    palavras = [len(r["output"].split()) for r in registros]

    print(f"Lidos:                {estatisticas['lidos']}")
    print(f"Duplicatas removidas: {estatisticas['duplicatas_removidas']}")
    print(f"Respostas curtas (<{MIN_PALAVRAS_RESPOSTA} palavras) removidas: {estatisticas['curtos_removidos']}")
    print(f"Gravados:             {gravados}")
    print(f"Arquivo: {caminho.relative_to(RAIZ)}")
    print(f"Fontes: {dict(tipos)}")
    print(f"Palavras na resposta: min={min(palavras)} media={sum(palavras)/len(palavras):.0f} max={max(palavras)}")
    return gravados


if __name__ == "__main__":
    main()
