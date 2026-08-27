"""Hiperparâmetros do fine-tuning LoRA e tradução para a interface do MLX-LM.

O `LoRAConfig` é a única fonte dos hiperparâmetros: o trainer, o evaluator e o notebook
leem daqui, então uma alteração não precisa ser replicada em três lugares.

A tradução para o MLX-LM sai por dois caminhos, e a divisão não é arbitrária:

* `to_mlx_args()` devolve os argumentos de linha de comando.
* `to_lora_parameters()` devolve o bloco que **só** existe no arquivo YAML.

O `mlx_lm lora` não expõe flag de CLI para rank, scale e dropout — eles vivem em
`lora_parameters`, lido de um YAML passado com `-c`. Verificado em `mlx_lm.lora.main`:
argumento de linha de comando tem precedência sobre o YAML, e o que faltar nos dois cai no
`CONFIG_DEFAULTS`. Por isso tudo o que tem flag vai pela linha de comando (fica visível no
log de execução) e só rank/scale/dropout vão pelo arquivo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]

MODELO_BASE_PADRAO = "meta-llama/Llama-3.2-3B-Instruct"
DIR_DADOS_MLX = RAIZ / "data" / "processed" / "mlx"
DIR_ADAPTERS = RAIZ / "data" / "fine_tuned" / "adapters"


def relativo_a_raiz(caminho: Path | str) -> str:
    """Caminho relativo à raiz do repositório, quando estiver dentro dela.

    Usado no que vai para `docs/`: gravar `/Users/<alguem>/.../adapters` num entregável
    versionado cola o resultado à máquina de quem rodou e não reproduz em outra.
    """
    caminho = Path(caminho)
    try:
        return str(caminho.relative_to(RAIZ))
    except ValueError:
        return str(caminho)


def _revisao_em_cache(model: str) -> str | None:
    """SHA do snapshot que o cache local do HuggingFace tem para `main`, se houver.

    Serve ao registro, não ao download: quando ninguém fixou `MODEL_REVISION`, o artefato
    ficaria sem dizer *quais* pesos produziram aquelas métricas. Ler o `refs/main` do cache
    recupera o SHA que de fato foi usado na rodada.

    Note a diferença entre as duas coisas, que o `to_dict` registra separadamente: um SHA
    fixado é uma **garantia** (a próxima rodada baixa o mesmo); um SHA lido do cache é uma
    **observação** (foi o que estava ali naquele dia, e uma retag upstream muda o que a
    próxima rodada vai pegar). Confundir os dois daria uma falsa sensação de reprodutibilidade.
    """
    try:
        from huggingface_hub import constants
    except ImportError:
        return None
    referencia = (
        Path(constants.HF_HUB_CACHE) / f"models--{model.replace('/', '--')}" / "refs" / "main"
    )
    try:
        return referencia.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _do_ambiente(variavel: str, padrao: Path) -> Path:
    """Lê um caminho do `.env`, ancorando o relativo na raiz do repositório.

    `ADAPTER_PATH=data/fine_tuned/adapters` é relativo no `.env`. Sem ancorar na raiz, o
    destino dos adapters mudaria conforme o diretório de onde o comando foi disparado.

    Duas normalizações, cada uma consertando um problema diferente:

    * `expanduser()`, senão `ADAPTER_PATH=~/adapters` — coisa natural de escrever — não dá
      erro nenhum: cria um diretório chamado literalmente `~` dentro do repositório e grava
      os adapters lá. Falha silenciosa, do tipo que só aparece quando falta espaço em disco.
    * `resolve()`, pela integridade do registro. O `relativo_a_raiz` compara caminhos de
      forma lexical, então sem resolver `RAIZ/../../../../tmp/pwn` ele grava
      `../../../../tmp/pwn` no `docs/`, um caminho que *parece* relativo à raiz e não é.
      Com `resolve()`, o registro passa a mostrar o caminho absoluto de verdade.

    O que **não** se faz aqui é exigir que o caminho fique contido na raiz. Mandar os
    adapters para um disco externo é justamente o motivo de a variável existir, e quem a
    define é quem roda o comando — não há fronteira de privilégio a defender.
    """
    valor = os.getenv(variavel)
    if not valor:
        return padrao
    caminho = Path(valor).expanduser()
    if not caminho.is_absolute():
        caminho = RAIZ / caminho
    return caminho.resolve()


@dataclass(frozen=True)
class LoRAConfig:
    """Configuração de uma rodada de fine-tuning.

    Os valores padrão são os do plano do projeto. `frozen=True` porque a configuração é
    gravada junto dos adapters como registro da rodada: se ela pudesse ser mutada depois de
    passar pelo trainer, o registro deixaria de descrever o treino que de fato aconteceu.
    """

    # Lido de `BASE_MODEL` no `.env`, como o `ADAPTER_PATH` logo abaixo. A variável já existia
    # no `.env.example` e já era validada pelo `check_env`, mas nada a consumia: trocá-la não
    # mudava treino nem avaliação, enquanto o check dizia que estava tudo certo. Uma variável
    # que o setup confere e o código ignora custa a quem for depurar exatamente o tempo que o
    # check deveria poupar.
    model: str = field(default_factory=lambda: os.getenv("BASE_MODEL") or MODELO_BASE_PADRAO)

    # Commit do modelo no Hub. `None` significa "o que `main` apontar" — que é o que o resto
    # do projeto evita em toda parte via seed, mas aqui nenhuma seed alcança: uma retag
    # upstream troca os pesos sem mudar uma linha do repositório. Fixar via `MODEL_REVISION`
    # no `.env` fecha essa ponta para o tokenizer e para a avaliação.
    #
    # O treino é o caso que **não** dá para fixar por aqui: o `mlx_lm lora` não expõe flag de
    # revision na linha de comando (só `--model`), então prendê-lo exigiria baixar um snapshot
    # fixado e passar o diretório local. Enquanto isso não existe, o `to_dict` ao menos
    # registra qual revisão a rodada usou.
    model_revision: str | None = field(default_factory=lambda: os.getenv("MODEL_REVISION") or None)

    lora_layers: int = 8
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0

    learning_rate: float = 1e-4
    num_iters: int = 500
    batch_size: int = 4
    val_batches: int = 25

    # O plano previa 512, e 512 quebra o treino neste dataset. Medido sobre os 903 exemplos
    # de treino: com limite de 512 tokens, 118 sequências são truncadas e em 46 delas o
    # prompt sozinho já ocupa os 512 — não sobra nenhum token de resposta. Como a loss é
    # mascarada no prompt (`--mask-prompt`), esses lotes ficam sem alvo algum, a loss vira
    # NaN e o NaN se propaga pelo otimizador, destruindo os pesos. Confirmado numa rodada
    # de 4 iterações: "Iter 2: Train loss nan".
    # Com 1024, nenhuma sequência é truncada (o máximo observado é 875 tokens).
    max_seq_length: int = 1024

    # Não estão no plano: existem para a célula de curvas de loss do notebook. Com o padrão
    # do MLX (`steps_per_eval=200`) uma rodada de 500 iterações renderia 2 pontos de
    # validação, o que não desenha curva nenhuma. Com 50, são 10 pontos.
    steps_per_report: int = 10
    steps_per_eval: int = 50

    seed: int = 42

    data_dir: Path = field(default_factory=lambda: DIR_DADOS_MLX)
    adapter_path: Path = field(default_factory=lambda: _do_ambiente("ADAPTER_PATH", DIR_ADAPTERS))

    @property
    def revisao_efetiva(self) -> str | None:
        """A revisão fixada; na falta dela, a que o cache local registra. `None` se nenhuma."""
        return self.model_revision or _revisao_em_cache(self.model)

    @property
    def lora_scale(self) -> float:
        """Fator de escala do LoRA no formato que o MLX-LM espera.

        Atenção à diferença de convenção. No PEFT/HuggingFace declara-se `alpha` e a escala
        efetiva aplicada é `alpha / rank`. No MLX-LM, `lora_parameters.scale` **é** o
        multiplicador aplicado direto (`y + scale * z`, em `mlx_lm.tuner.lora.LoRALinear`).

        Passar `alpha=16` como `scale` daria uma escala 8x maior que a pretendida. Então a
        conversão é explícita: alpha 16 com rank 8 vira scale 2.0.
        """
        return self.lora_alpha / self.lora_rank

    def to_mlx_args(self) -> list[str]:
        """Argumentos de linha de comando para `mlx_lm lora`.

        Devolve lista (nunca string): o trainer passa isso a `subprocess.run` sem shell, de
        forma que nenhum valor seja reinterpretado pelo shell.
        """
        return [
            "--model", self.model,
            "--train",
            "--data", str(self.data_dir),
            "--fine-tune-type", "lora",
            "--num-layers", str(self.lora_layers),
            "--batch-size", str(self.batch_size),
            "--iters", str(self.num_iters),
            "--val-batches", str(self.val_batches),
            "--learning-rate", str(self.learning_rate),
            "--steps-per-report", str(self.steps_per_report),
            "--steps-per-eval", str(self.steps_per_eval),
            "--max-seq-length", str(self.max_seq_length),
            "--adapter-path", str(self.adapter_path),
            "--seed", str(self.seed),
            # Calcula a loss só sobre a resposta, não sobre a pergunta. Sem isto o modelo
            # gasta capacidade aprendendo a reproduzir o enunciado da pergunta, que na
            # inferência sempre vem dado.
            "--mask-prompt",
        ]

    def to_lora_parameters(self) -> dict[str, float | int]:
        """Bloco `lora_parameters` do YAML — o que não tem equivalente em linha de comando."""
        return {
            "rank": self.lora_rank,
            "scale": self.lora_scale,
            "dropout": self.lora_dropout,
        }

    def to_dict(self) -> dict[str, object]:
        """Registro legível da rodada, para gravar junto das métricas e dos adapters."""
        return {
            "model": self.model,
            "model_revision": self.revisao_efetiva,
            # Distingue garantia de observação: `True` significa que a revisão foi fixada e a
            # próxima rodada baixará a mesma; `False`, que o SHA apenas descreve o que estava
            # no cache naquele dia. Sem esta chave o artefato pareceria reprodutível quando
            # não é.
            "model_revision_fixada": self.model_revision is not None,
            "lora_layers": self.lora_layers,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_scale_mlx": self.lora_scale,
            "lora_dropout": self.lora_dropout,
            "learning_rate": self.learning_rate,
            "num_iters": self.num_iters,
            "batch_size": self.batch_size,
            "max_seq_length": self.max_seq_length,
            "val_batches": self.val_batches,
            "seed": self.seed,
        }
