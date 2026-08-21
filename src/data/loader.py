"""Carrega o PubMedQA e converte para o formato de instruction-tuning do projeto.

Uso, a partir da raiz do repositório:

    python -m src.data.loader

Baixa o subset `pqa_labeled` (1.000 pares de pergunta/resposta clínica revisados por
especialistas), converte cada registro para `{"instruction", "input", "output", "source"}`
e grava em `data/processed/pubmedqa.jsonl`.

O campo `source` carrega o `pubid` de origem (`PubMedQA:21645374`), que é o que permite ao
assistente citar a fonte da informação — requisito de explainability da Fase 3.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from datasets import load_dataset
from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[2]

DATASET_ID = "qiaojin/PubMedQA"
DATASET_CONFIG = "pqa_labeled"
SAIDA_PADRAO = RAIZ / "data" / "processed" / "pubmedqa.jsonl"

# Caracteres que o Unicode trata como quebra de linha mas o json.dumps(ensure_ascii=False)
# NÃO escapa. Um deles cru no texto parte a linha do JSONL em duas e quebra qualquer leitor
# que use splitlines(). Há de fato um U+2029 no pqa_labeled (registro PubMedQA:28177278).
QUEBRAS_NAO_ESCAPADAS = str.maketrans(
    {
        " ": " ",  # LINE SEPARATOR
        " ": " ",  # PARAGRAPH SEPARATOR
        "\x0b": " ",  # VERTICAL TAB
        "\x0c": " ",  # FORM FEED
        "\x85": " ",  # NEXT LINE
    }
)


def normalize_text(texto: str | None) -> str:
    """Remove quebras exóticas e espaços das pontas, mantendo acentuação intacta."""
    return (texto or "").translate(QUEBRAS_NAO_ESCAPADAS).strip()


def format_context(context: dict[str, Any]) -> str:
    """Junta as seções do abstract preservando os rótulos (BACKGROUND, RESULTS, ...).

    O PubMedQA entrega `contexts` e `labels` como listas paralelas — verificado: nos 1.000
    registros do `pqa_labeled` os dois têm sempre o mesmo tamanho. Ainda assim o zip é
    defensivo, porque um desalinhamento silenciaria seções inteiras do contexto clínico.
    """
    trechos = [normalize_text(t) for t in (context.get("contexts") or [])]
    rotulos = [normalize_text(r) for r in (context.get("labels") or [])]
    if len(rotulos) == len(trechos):
        return "\n\n".join(
            f"{rotulo}: {trecho}".strip() for rotulo, trecho in zip(rotulos, trechos)
        )
    return "\n\n".join(trechos)


def to_instruction_format(registro: dict[str, Any]) -> dict[str, str]:
    """Converte um registro cru do PubMedQA para o formato de instruction-tuning."""
    return {
        "instruction": normalize_text(registro.get("question")),
        "input": format_context(registro.get("context") or {}),
        "output": normalize_text(registro.get("long_answer")),
        "source": f"PubMedQA:{registro.get('pubid')}",
    }


def load_pubmedqa(dataset_id: str = DATASET_ID, config: str = DATASET_CONFIG) -> Iterator[dict]:
    """Baixa o dataset e devolve os registros já convertidos.

    Descarta registros sem pergunta ou sem resposta: eles não servem para fine-tuning e
    passariam adiante como exemplos vazios. Nos dados atuais isso não remove nada — os
    1.000 registros têm `long_answer` preenchido — mas protege contra mudança na origem.
    """
    dataset = load_dataset(dataset_id, config)["train"]
    for registro in dataset:
        convertido = to_instruction_format(registro)
        if convertido["instruction"] and convertido["output"]:
            yield convertido


def save_jsonl(registros: Iterable[dict], caminho: Path = SAIDA_PADRAO) -> int:
    """Grava um JSONL (um objeto por linha) e devolve quantos registros foram escritos.

    Cria o diretório pai se necessário: `data/processed/` existe no repositório, mas um
    caminho customizado não teria o diretório e o `open` estouraria FileNotFoundError.
    """
    caminho = Path(caminho)
    os.makedirs(caminho.parent, exist_ok=True)
    total = 0
    with caminho.open("w", encoding="utf-8") as arquivo:
        for registro in registros:
            arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
            total += 1
    return total


def load_jsonl(caminho: Path) -> list[dict]:
    """Lê um JSONL gravado por `save_jsonl`.

    Fica ao lado do `save_jsonl` de propósito: a garantia de "um registro por linha física"
    (que exigiu normalizar U+2028/U+2029) vale para os dois lados, e separá-los abriria
    espaço para um leitor divergir do escritor.
    """
    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(
            f"{caminho} não existe. Rode as etapas anteriores do pipeline antes desta."
        )
    with caminho.open(encoding="utf-8") as arquivo:
        return [json.loads(linha) for linha in arquivo if linha.strip()]


def main(caminho: Path = SAIDA_PADRAO) -> int:
    load_dotenv(RAIZ / ".env")

    print(f"Baixando {DATASET_ID} ({DATASET_CONFIG})...")
    registros = list(load_pubmedqa())

    total = save_jsonl(registros, caminho)

    palavras = [len(r["output"].split()) for r in registros]
    fontes = Counter(r["source"].split(":")[0] for r in registros)
    print(f"\nRegistros gravados: {total}")
    print(f"Arquivo: {caminho.relative_to(RAIZ)}")
    print(f"Fontes: {dict(fontes)}")
    if palavras:
        media = sum(palavras) / len(palavras)
        curtos = sum(1 for p in palavras if p < 20)
        print(f"Palavras na resposta: min={min(palavras)} media={media:.0f} max={max(palavras)}")
        print(f"Respostas com menos de 20 palavras: {curtos} (o curator vai filtrar)")
    return total


if __name__ == "__main__":
    main()
