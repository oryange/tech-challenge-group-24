"""Fine-tuning LoRA do modelo base com MLX-LM.

Uso, a partir da raiz do repositório:

    python -m src.data.curator          # gera data/processed/dataset.jsonl
    python -m src.fine_tuning.trainer   # treina e grava os adapters

Duas decisões de formato que valem registro, porque divergem do que um tutorial genérico
sugeriria:

* **O dataset sai no formato `{"prompt", "completion"}`, não em texto com marcadores
  literais.** O MLX-LM aceita três formatos (verificado em `mlx_lm.tuner.datasets`:
  `{prompt, completion}`, `{messages}` e `{text}`). Nos dois primeiros ele aplica o
  `chat_template` do próprio tokenizer; no terceiro, o texto vai cru. Escrever
  `<s>[INST] ... [/INST]` à mão produziria o template do Mistral — e o modelo aqui é o
  Llama 3.2, cujo template é `<|begin_of_text|><|start_header_id|>...`. Seriam tokens que o
  modelo nunca viu no pré-treino, ensinando um formato que a inferência não usa.
* **A loss é calculada só sobre a resposta** (`--mask-prompt`). A pergunta sempre existe na
  inferência; treinar o modelo para reproduzi-la gasta capacidade sem retorno.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.data.loader import load_jsonl, save_jsonl
from src.fine_tuning.config import LoRAConfig, relativo_a_raiz

RAIZ = Path(__file__).resolve().parents[2]

DATASET_PADRAO = RAIZ / "data" / "processed" / "dataset.jsonl"
HISTORICO_PADRAO = RAIZ / "docs" / "training_history.json"
LOG_PADRAO = RAIZ / "logs" / "fine_tuning.log"

FRACAO_TREINO = 0.9

# As duas linhas de progresso que o MLX-LM imprime. Extrair a curva daqui evita ter de
# treinar de novo só para desenhar o gráfico do notebook.
#
# O valor casa `nan`/`inf` além de número: uma rodada divergente tem de aparecer no
# histórico como divergente. A primeira versão aceitava só `[\d.]+`, e o efeito foi uma
# curva de treino vazia numa rodada que produziu `Train loss nan` do início ao fim — o
# gráfico ficava em branco em vez de mostrar que o treino tinha quebrado.
_VALOR = r"-?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf)"
_LINHA_TREINO = re.compile(rf"Iter (?P<iter>\d+): Train loss (?P<loss>{_VALOR})")
_LINHA_VALIDACAO = re.compile(rf"Iter (?P<iter>\d+): Val loss (?P<loss>{_VALOR})")

# Margem mínima de tokens de resposta para um exemplo valer como alvo de treino. Um exemplo
# cujo prompt consome quase todo o limite ensina a completar um texto cortado no meio.
MIN_TOKENS_RESPOSTA = 32


def to_mlx_record(registro: dict[str, str]) -> dict[str, str]:
    """Converte um registro do dataset para o par prompt/completion do MLX-LM.

    O `input` do dataset é o contexto (abstract do PubMedQA ou a condição clínica nos dados
    hospitalares). Ele entra rotulado no prompt em vez de concatenado direto: sem rótulo, o
    modelo não tem como distinguir a pergunta do material de apoio.
    """
    instrucao = (registro.get("instruction") or "").strip()
    contexto = (registro.get("input") or "").strip()
    prompt = f"{instrucao}\n\nContexto:\n{contexto}" if contexto else instrucao
    return {"prompt": prompt, "completion": (registro.get("output") or "").strip()}


def build_token_counter(model: str, revision: str | None = None):
    """Devolve `(prompt, completion) -> (tokens_prompt, tokens_total)` para o modelo dado.

    Usa o `chat_template` do próprio tokenizer, na mesma forma que o
    `mlx_lm.tuner.datasets.CompletionsDataset` usa no treino — é a única forma de a contagem
    aqui corresponder à que o treino vai fazer. Carrega apenas o tokenizer, não os pesos.

    `revision` fixa o commit do modelo no Hub. Sem ela, uma retag upstream pode mudar a
    contagem de tokens e, com ela, quais exemplos o filtro de `max_seq_length` descarta.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)

    def contar(prompt: str, completion: str) -> tuple[int, int]:
        mensagens = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
        total = tokenizer.apply_chat_template(mensagens, return_dict=False)
        so_prompt = tokenizer.apply_chat_template(
            mensagens[:-1], add_generation_prompt=True, return_dict=False
        )
        return len(so_prompt), len(total)

    return contar


def _prepare_mlx_data(
    dataset_path: Path = DATASET_PADRAO,
    output_dir: Path | None = None,
    fracao_treino: float = FRACAO_TREINO,
    max_seq_length: int | None = None,
    contar_tokens=None,
) -> dict[str, int]:
    """Escreve `train.jsonl` e `valid.jsonl` no formato do MLX-LM.

    O split é sequencial de propósito. O `curator` já embaralha o dataset com seed fixa —
    por essa razão exata, documentada lá — então cortar em 90% aqui produz os dois lados com
    a mesma mistura de literatura científica e dados hospitalares, sem sortear de novo (o
    que faria treino e validação mudarem de composição a cada execução).

    Quando `contar_tokens` e `max_seq_length` são informados, descarta os exemplos em que o
    prompt não deixa espaço para a resposta dentro do limite. É a proteção contra a loss NaN:
    com a loss mascarada no prompt, um exemplo truncado antes da resposta não tem token de
    alvo nenhum, e o NaN daí resultante se propaga pelo otimizador.
    """
    output_dir = Path(output_dir) if output_dir is not None else LoRAConfig().data_dir
    registros = load_jsonl(dataset_path)
    if not registros:
        raise ValueError(f"{dataset_path} está vazio — rode o pipeline de dados antes.")

    convertidos = [to_mlx_record(r) for r in registros]
    convertidos = [r for r in convertidos if r["prompt"] and r["completion"]]
    apos_conversao = len(convertidos)

    descartados_por_tamanho = 0
    if contar_tokens is not None and max_seq_length is not None:
        couberam = []
        for registro in convertidos:
            tokens_prompt, _ = contar_tokens(registro["prompt"], registro["completion"])
            if tokens_prompt + MIN_TOKENS_RESPOSTA <= max_seq_length:
                couberam.append(registro)
            else:
                descartados_por_tamanho += 1
        convertidos = couberam

    corte = int(len(convertidos) * fracao_treino)
    treino, validacao = convertidos[:corte], convertidos[corte:]
    if not treino or not validacao:
        raise ValueError(
            f"Split inviável: {len(convertidos)} registros geram "
            f"{len(treino)} de treino e {len(validacao)} de validação."
        )

    save_jsonl(treino, output_dir / "train.jsonl")
    save_jsonl(validacao, output_dir / "valid.jsonl")
    return {
        "train": len(treino),
        "validation": len(validacao),
        "descartados_sem_texto": len(registros) - apos_conversao,
        "descartados_por_tamanho": descartados_por_tamanho,
    }


def _write_yaml_config(config: LoRAConfig, destino: Path) -> Path:
    """Grava o YAML com o bloco `lora_parameters`, único caminho para rank/scale/dropout.

    Só hiperparâmetros entram no arquivo. Nada de credenciais: o acesso ao modelo *gated* é
    resolvido pelo `huggingface_hub` a partir do ambiente, então gravar token aqui seria
    espalhar segredo em disco sem necessidade.
    """
    destino = Path(destino)
    os.makedirs(destino.parent, exist_ok=True)
    destino.write_text(
        yaml.safe_dump({"lora_parameters": config.to_lora_parameters()}, sort_keys=True),
        encoding="utf-8",
    )
    return destino


def _loss(bruto: str) -> float | None:
    """Converte o valor de loss da saída do MLX-LM, com `None` para `nan`/`inf`.

    O ponto tem de continuar existindo — é assim que a divergência aparece no histórico e
    no gráfico. Mas guardá-lo como `float('nan')` sabota o artefato: o `json.dumps` escreve
    o literal `NaN`, que não é JSON válido (a RFC 8259 só admite números). O Python relê sem
    reclamar, então dentro do projeto passa liso; fora dele, o `JSON.parse` recusa o arquivo
    e o `jq` faz pior — aceita e converte em silêncio, `NaN` para `null` e `Infinity` para
    `1.7976931348623157e+308`. O registro de que o treino divergiu viraria um número grande
    e finito, que é exatamente a leitura errada.

    `None` vira `null`, que todo parser lê e ninguém confunde com uma medição real.
    """
    valor = float(bruto)
    return valor if math.isfinite(valor) else None


def parse_training_history(linhas: list[str]) -> dict[str, list[dict[str, float | None]]]:
    """Extrai as curvas de loss da saída do MLX-LM."""
    historico: dict[str, list[dict[str, float | None]]] = {"train": [], "validation": []}
    for linha in linhas:
        if (m := _LINHA_TREINO.search(linha)) is not None:
            historico["train"].append({"iter": int(m["iter"]), "loss": _loss(m["loss"])})
        elif (m := _LINHA_VALIDACAO.search(linha)) is not None:
            historico["validation"].append({"iter": int(m["iter"]), "loss": _loss(m["loss"])})
    return historico


def _run_mlx(argumentos: list[str], log_path: Path) -> tuple[list[str], int]:
    """Executa o `mlx_lm lora` num processo separado, ecoando e registrando a saída.

    Devolve `(linhas, código_de_saída)` em vez de levantar exceção: quem chama precisa das
    linhas mesmo quando o treino falha, para gravar o histórico parcial. Uma rodada de uma
    hora que quebra na iteração 490 e não deixa registro nenhum é uma hora perdida às cegas.

    Três pontos sobre a forma da chamada:

    * `sys.executable` em vez de `"python"`: garante o interpretador do venv ativo, mesmo se
      o comando for disparado de um shell onde `python` aponta para outro lugar.
    * `python -m mlx_lm lora` em vez de `python -m mlx_lm.lora`: a segunda forma imprime
      aviso de depreciação nesta versão (0.31.3) e é candidata a sair.
    * Lista de argumentos e `shell=False` (o padrão): nada do que vai em `argumentos` passa
      por interpretação de shell.

    O processo é separado, e não uma chamada a `mlx_lm.lora.run`, porque o treino aloca
    vários GB na GPU: encerrar o processo devolve a memória ao sistema de forma
    determinística, o que importa quando o notebook roda treino e avaliação em seguida.
    """
    os.makedirs(log_path.parent, exist_ok=True)
    comando = [sys.executable, "-m", "mlx_lm", "lora", *argumentos]

    print("$ " + " ".join(comando), flush=True)
    linhas: list[str] = []
    # `buffering=1` (linha a linha) e não o padrão de bloco: com buffer de 8 KB o arquivo
    # fica vazio durante quase todo o treino e só materializa no fim. Um log que aparece
    # depois que a rodada acabou não serve para acompanhar nem para diagnosticar uma queda.
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        processo = subprocess.Popen(
            comando,
            cwd=RAIZ,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert processo.stdout is not None
        for linha in processo.stdout:
            print(linha, end="", flush=True)
            log.write(linha)
            linhas.append(linha.rstrip("\n"))
        codigo = processo.wait()

    return linhas, codigo


def train(
    config: LoRAConfig | None = None,
    dataset_path: Path = DATASET_PADRAO,
    historico_path: Path = HISTORICO_PADRAO,
    log_path: Path = LOG_PADRAO,
) -> dict[str, object]:
    """Prepara os dados, treina e grava adapters e histórico de loss."""
    config = config or LoRAConfig()

    estatisticas = _prepare_mlx_data(
        dataset_path,
        config.data_dir,
        max_seq_length=config.max_seq_length,
        contar_tokens=build_token_counter(config.model, config.model_revision),
    )
    print(
        f"Dados MLX em {config.data_dir}: {estatisticas['train']} de treino, "
        f"{estatisticas['validation']} de validação"
    )
    if estatisticas["descartados_por_tamanho"]:
        print(
            f"  {estatisticas['descartados_por_tamanho']} exemplo(s) descartado(s): o prompt "
            f"não deixa {MIN_TOKENS_RESPOSTA} tokens de resposta dentro do limite de "
            f"{config.max_seq_length}"
        )

    yaml_config = _write_yaml_config(config, config.data_dir / "lora_config.yaml")
    argumentos = [*config.to_mlx_args(), "-c", str(yaml_config)]

    linhas, codigo = _run_mlx(argumentos, log_path)
    historico = parse_training_history(linhas)

    resultado = {
        # Marca o fim da rodada, no mesmo formato que o `evaluator` usa em
        # `evaluation_results.json`. Os dois artefatos são lidos juntos pelo notebook, e sem
        # data de um dos lados não há como saber se a avaliação corresponde a este treino.
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config.to_dict(),
        "dataset": estatisticas,
        "history": historico,
        "adapter_path": relativo_a_raiz(config.adapter_path),
        "exit_code": codigo,
    }
    # Grava ANTES de checar o código de saída: se o treino quebrou na iteração 490, a curva
    # até ali é o único registro do que aconteceu, e é o que permite diagnosticar sem
    # repetir a hora de GPU.
    os.makedirs(historico_path.parent, exist_ok=True)
    # `allow_nan=False`: o padrão do `json` é escrever `NaN`/`Infinity` crus, que não são
    # JSON válido. O `_loss` já converte os não-finitos em `None`, então aqui isso nunca
    # deveria disparar — é justamente por isso que vale a pena. Se um não-finito voltar a
    # entrar por outro caminho, o erro aparece na gravação, e não meses depois em quem
    # tentar ler o arquivo fora do Python.
    historico_path.write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    if codigo != 0:
        raise RuntimeError(
            f"mlx_lm lora terminou com código {codigo}. "
            f"Saída completa em {log_path}; histórico parcial em {historico_path}."
        )
    return resultado


def _fmt(ponto: dict[str, float | None]) -> str:
    """Formata um ponto da curva; `divergiu` quando a loss não é finita.

    Sem isto, `f"{None:.3f}"` levanta `TypeError` e o resumo final quebra justamente na
    rodada em que ele mais importa — a que divergiu.
    """
    loss = ponto["loss"]
    return "divergiu" if loss is None else f"{loss:.3f}"


def main() -> dict[str, object]:
    load_dotenv(RAIZ / ".env")
    resultado = train()

    historico = resultado["history"]  # type: ignore[index]
    treino = historico["train"]
    validacao = historico["validation"]
    print(f"\nAdapters: {resultado['adapter_path']}")
    print(f"Histórico: {HISTORICO_PADRAO.relative_to(RAIZ)}")
    if treino:
        print(f"Loss de treino:     {_fmt(treino[0])} -> {_fmt(treino[-1])}")
    if validacao:
        print(f"Loss de validação:  {_fmt(validacao[0])} -> {_fmt(validacao[-1])}")
    return resultado


if __name__ == "__main__":
    main()
