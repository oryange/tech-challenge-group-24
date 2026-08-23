"""Avaliação do modelo: ROUGE-L e BLEU-4, baseline contra fine-tuned.

Uso, a partir da raiz do repositório:

    python -m src.fine_tuning.trainer     # treina e grava os adapters
    python -m src.fine_tuning.evaluator   # gera docs/evaluation_results.json

Duas escolhas metodológicas que sustentam os números:

* **A avaliação usa `valid.jsonl`, o conjunto que o treino não viu.** Medir no `train.jsonl`
  mediria memorização e daria um ganho artificialmente alto — inútil como evidência de que
  o fine-tuning funcionou.
* **A geração é gulosa (sem amostragem).** Duas execuções sobre o mesmo modelo têm de dar o
  mesmo número, ou a comparação baseline/fine-tuned mistura o efeito do treino com a
  variação da amostragem.

Sobre as métricas: ROUGE-L mede a maior subsequência comum (recall de conteúdo) e BLEU-4 a
precisão de n-gramas até 4. Nenhuma das duas avalia correção clínica — são medidas de
sobreposição textual com a resposta de referência, e é assim que devem ser lidas no
relatório.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

from src.data.loader import load_jsonl
from src.fine_tuning.config import LoRAConfig, relativo_a_raiz

RAIZ = Path(__file__).resolve().parents[2]

RESULTADOS_PADRAO = RAIZ / "docs" / "evaluation_results.json"
TOTAL_AMOSTRAS_PADRAO = 50
MAX_TOKENS_PADRAO = 256


_CHECKPOINT = re.compile(r"^(?P<iter>\d+)_adapters\.safetensors$")


def available_checkpoints(adapter_dir: Path) -> dict[int, Path]:
    """Mapeia iteração -> arquivo, para os checkpoints intermediários salvos pelo MLX-LM.

    O MLX grava `0000200_adapters.safetensors` a cada `save_every` iterações, além do
    `adapters.safetensors` final. Só os intermediários entram aqui: o final não carrega o
    número da iteração no nome e é tratado separadamente por quem chama.
    """
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.is_dir():
        return {}
    encontrados = {}
    for arquivo in adapter_dir.iterdir():
        casamento = _CHECKPOINT.match(arquivo.name)
        if casamento:
            encontrados[int(casamento["iter"])] = arquivo
    return encontrados


def best_checkpoint(
    historico_validacao: list[dict[str, float]], adapter_dir: Path
) -> tuple[int, Path] | None:
    """Escolhe o checkpoint com menor loss de validação entre os que existem em disco.

    Considera apenas iterações que têm **as duas coisas**: ponto de validação medido e
    arquivo salvo. Com `steps_per_eval=50` e `save_every=100` os dois conjuntos não
    coincidem — validação em 50, 100, 150..., checkpoint em 100, 200, 300... — então
    escolher pelo menor valor da curva inteira apontaria para uma iteração sem arquivo.

    Descarta `nan`: uma rodada divergente não tem "melhor" checkpoint, e `min()` sobre NaN
    devolve resultado arbitrário dependendo da ordem da lista.
    """
    disponiveis = available_checkpoints(adapter_dir)
    if not disponiveis:
        return None

    candidatos = [
        (ponto["loss"], ponto["iter"])
        for ponto in historico_validacao
        if ponto["iter"] in disponiveis and not math.isnan(ponto["loss"])
    ]
    if not candidatos:
        return None

    _, melhor_iter = min(candidatos)
    return melhor_iter, disponiveis[melhor_iter]


def materialize_checkpoint(checkpoint: Path, adapter_dir: Path, destino: Path) -> Path:
    """Copia um checkpoint para um diretório carregável pelo `mlx_lm.load`.

    O `load` espera um diretório com `adapters.safetensors` e `adapter_config.json`; o
    checkpoint intermediário tem outro nome e fica junto dos demais. Copiar (em vez de
    renomear) preserva os checkpoints originais para uma segunda comparação.
    """
    destino = Path(destino)
    os.makedirs(destino, exist_ok=True)
    shutil.copyfile(checkpoint, destino / "adapters.safetensors")
    shutil.copyfile(Path(adapter_dir) / "adapter_config.json", destino / "adapter_config.json")
    return destino


def load_test_samples(
    total: int = TOTAL_AMOSTRAS_PADRAO, caminho: Path | None = None
) -> list[dict[str, str]]:
    """Lê as amostras de validação (nunca vistas no treino), na ordem do arquivo.

    Sem embaralhar: o `curator` já embaralhou o dataset com seed fixa, então os primeiros N
    registros da validação são uma amostra estável entre execuções. Sortear aqui faria o
    conjunto de avaliação mudar a cada rodada e as métricas deixariam de ser comparáveis.
    """
    caminho = Path(caminho) if caminho is not None else LoRAConfig().data_dir / "valid.jsonl"
    registros = load_jsonl(caminho)
    return registros[:total]


def compute_metrics(predicoes: list[str], referencias: list[str]) -> dict[str, float]:
    """Calcula ROUGE-L (média de F1) e BLEU-4 (corpus) entre predições e referências."""
    if len(predicoes) != len(referencias):
        raise ValueError(
            f"{len(predicoes)} predições contra {len(referencias)} referências — "
            "cada predição precisa da sua referência."
        )
    if not predicoes:
        raise ValueError("Nenhuma predição para avaliar.")

    from rouge_score import rouge_scorer
    from sacrebleu import corpus_bleu

    marcador = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = [
        marcador.score(referencia, predicao)["rougeL"].fmeasure
        for predicao, referencia in zip(predicoes, referencias)
    ]

    # `corpus_bleu` recebe as referências como lista de streams. Um único stream aqui: cada
    # predição tem exatamente uma resposta de referência.
    bleu = corpus_bleu(predicoes, [referencias])

    return {
        "rouge_l": sum(rouge) / len(rouge),
        "bleu_4": bleu.score,
    }


def _build_prompt(tokenizer: Any, pergunta: str) -> str:
    """Monta o prompt com o chat template do próprio modelo.

    Tem de ser o mesmo template usado no treino. O MLX-LM treinou via `CompletionsDataset`,
    que aplica `apply_chat_template` sobre os papéis user/assistant — se a inferência usasse
    outro formato, a avaliação mediria a diferença de template, não o efeito do treino.
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": pergunta}],
        add_generation_prompt=True,
        tokenize=False,
    )


def generate_responses(
    amostras: Iterable[dict[str, str]],
    model_path: str,
    adapter_path: str | os.PathLike[str] | None = None,
    max_tokens: int = MAX_TOKENS_PADRAO,
    verbose: bool = True,
) -> list[str]:
    """Gera uma resposta por amostra. `adapter_path=None` avalia o modelo base."""
    from mlx_lm import generate, load

    if adapter_path is not None and not Path(adapter_path).is_dir():
        raise FileNotFoundError(
            f"{adapter_path} não existe. Rode 'python -m src.fine_tuning.trainer' antes."
        )

    model, tokenizer = load(
        model_path, adapter_path=str(adapter_path) if adapter_path else None
    )

    respostas = []
    amostras = list(amostras)
    for indice, amostra in enumerate(amostras, start=1):
        prompt = _build_prompt(tokenizer, amostra["prompt"])
        resposta = generate(model, tokenizer, prompt, max_tokens=max_tokens)
        respostas.append(resposta.strip())
        if verbose:
            print(f"  {indice}/{len(amostras)}", end="\r", flush=True)
    if verbose:
        print()
    return respostas


def evaluate(
    model_path: str,
    adapter_path: str | os.PathLike[str] | None,
    test_samples: list[dict[str, str]],
    max_tokens: int = MAX_TOKENS_PADRAO,
) -> dict[str, Any]:
    """Avalia um modelo sobre as amostras e devolve métricas mais alguns exemplos.

    Os exemplos vão no retorno de propósito: o relatório técnico exige análise dos
    resultados, e ROUGE/BLEU sozinhos não mostram *como* a resposta mudou.
    """
    predicoes = generate_responses(test_samples, model_path, adapter_path, max_tokens)
    referencias = [a["completion"] for a in test_samples]
    metricas = compute_metrics(predicoes, referencias)

    return {
        "metrics": metricas,
        "samples_evaluated": len(test_samples),
        "adapter_path": relativo_a_raiz(adapter_path) if adapter_path else None,
        "examples": [
            {
                "prompt": amostra["prompt"][:300],
                "reference": referencia[:300],
                "prediction": predicao[:300],
            }
            for amostra, referencia, predicao in list(
                zip(test_samples, referencias, predicoes)
            )[:3]
        ],
    }


def save_results(metrics: dict[str, Any], path: Path = RESULTADOS_PADRAO) -> Path:
    """Grava o JSON de resultados que alimenta o relatório técnico e o notebook."""
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def compare(
    config: LoRAConfig | None = None,
    total_amostras: int = TOTAL_AMOSTRAS_PADRAO,
    max_tokens: int = MAX_TOKENS_PADRAO,
    historico_path: Path = RAIZ / "docs" / "training_history.json",
    incluir_melhor_checkpoint: bool = True,
) -> dict[str, Any]:
    """Roda baseline e fine-tuned sobre as mesmas amostras e monta o comparativo.

    Quando há histórico de treino e checkpoints intermediários em disco, avalia também o
    checkpoint de menor loss de validação. O último checkpoint não é necessariamente o
    melhor: se a validação subiu no fim do treino, o modelo salvo ao final generaliza pior
    que um intermediário, e reportar só ele subestimaria o que o fine-tuning consegue.
    """
    config = config or LoRAConfig()
    amostras = load_test_samples(total_amostras)

    print(f"Avaliando {len(amostras)} amostras de validação (nunca vistas no treino).")
    print("Baseline (modelo base, sem adapters)...")
    baseline = evaluate(config.model, None, amostras, max_tokens)
    print("Fine-tuned (último checkpoint)...")
    finetuned = evaluate(config.model, config.adapter_path, amostras, max_tokens)

    resultados = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": config.model,
        "config": config.to_dict(),
        "samples_evaluated": len(amostras),
        "max_tokens": max_tokens,
        "baseline": baseline,
        "fine_tuned": finetuned,
        "delta": {
            chave: finetuned["metrics"][chave] - baseline["metrics"][chave]
            for chave in baseline["metrics"]
        },
    }

    if incluir_melhor_checkpoint and Path(historico_path).is_file():
        historico = json.loads(Path(historico_path).read_text(encoding="utf-8"))
        escolha = best_checkpoint(
            historico["history"]["validation"], config.adapter_path
        )
        if escolha is not None:
            iteracao, arquivo = escolha
            destino = materialize_checkpoint(
                arquivo, config.adapter_path, config.adapter_path.parent / "adapters_best"
            )
            print(f"Melhor checkpoint (iteração {iteracao})...")
            melhor = evaluate(config.model, destino, amostras, max_tokens)
            melhor["iteration"] = iteracao
            melhor["validation_loss"] = next(
                p["loss"] for p in historico["history"]["validation"] if p["iter"] == iteracao
            )
            resultados["best_checkpoint"] = melhor
            resultados["delta_best_checkpoint"] = {
                chave: melhor["metrics"][chave] - baseline["metrics"][chave]
                for chave in baseline["metrics"]
            }

    return resultados


def main() -> dict[str, Any]:
    load_dotenv(RAIZ / ".env")
    resultados = compare()
    caminho = save_results(resultados)

    base = resultados["baseline"]["metrics"]
    fino = resultados["fine_tuned"]["metrics"]
    melhor = resultados.get("best_checkpoint")

    cabecalho = f"\n{'métrica':10} {'baseline':>10} {'fine-tuned':>12} {'delta':>10}"
    if melhor:
        cabecalho += f" {'melhor ckpt':>13} {'delta':>10}"
    print(cabecalho)
    for chave in base:
        linha = (
            f"{chave:10} {base[chave]:>10.4f} {fino[chave]:>12.4f} "
            f"{resultados['delta'][chave]:>+10.4f}"
        )
        if melhor:
            linha += (
                f" {melhor['metrics'][chave]:>13.4f} "
                f"{resultados['delta_best_checkpoint'][chave]:>+10.4f}"
            )
        print(linha)
    if melhor:
        print(
            f"\nMelhor checkpoint: iteração {melhor['iteration']} "
            f"(loss de validação {melhor['validation_loss']:.3f})"
        )
    print(f"\nArquivo: {caminho.relative_to(RAIZ)}")
    return resultados


if __name__ == "__main__":
    main()
