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


def _do_ambiente(variavel: str, padrao: Path) -> Path:
    """Lê um caminho do `.env`, resolvendo relativo à raiz do repositório.

    `ADAPTER_PATH=data/fine_tuned/adapters` é relativo no `.env`. Sem ancorar na raiz, o
    destino dos adapters mudaria conforme o diretório de onde o comando foi disparado.
    """
    valor = os.getenv(variavel)
    if not valor:
        return padrao
    caminho = Path(valor)
    return caminho if caminho.is_absolute() else RAIZ / caminho


@dataclass(frozen=True)
class LoRAConfig:
    """Configuração de uma rodada de fine-tuning.

    Os valores padrão são os do plano do projeto. `frozen=True` porque a configuração é
    gravada junto dos adapters como registro da rodada: se ela pudesse ser mutada depois de
    passar pelo trainer, o registro deixaria de descrever o treino que de fato aconteceu.
    """

    model: str = MODELO_BASE_PADRAO

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
