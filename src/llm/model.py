"""Wrapper LangChain do modelo fine-tuned, servido pelo MLX.

Uso como biblioteca:

    from src.llm.model import MedicalMLXLLM

    llm = MedicalMLXLLM.from_env()      # lê BASE_MODEL, ADAPTER_PATH, MAX_TOKENS, TEMPERATURE
    resposta = llm.invoke("Qual a conduta inicial na crise asmática?")

É a peça que faz o modelo do PR 04 caber no `prompt | llm` do LCEL: o PR 07 monta a chain
com este objeto e não precisa saber que existe MLX do outro lado.

O `LLM` do LangChain é um modelo Pydantic, então os parâmetros são campos declarados — não
argumentos de `__init__`. É o que permite ao `_identifying_params` descrever a instância no
cache e nos callbacks.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import ConfigDict

MAX_TOKENS_PADRAO = 512
TEMPERATURE_PADRAO = 0.2

# Chave: (modelo, adapter, revisão). Valor: (modelo carregado, tokenizer).
#
# Carregar o Llama-3.2-3B custa dezenas de segundos e alguns GB de RAM, e o `_call` roda uma
# vez por pergunta — sem cache, o assistente interativo recarregaria o modelo inteiro a cada
# mensagem. A chave é a tripla, e não só o caminho do modelo, porque o notebook compara
# baseline e fine-tuned no mesmo processo: com o adapter fora da chave, a segunda instância
# receberia os pesos da primeira e a comparação mediria nada.
_MODELOS_CARREGADOS: dict[tuple[str, str | None, str | None], tuple[Any, Any]] = {}

# Cada entrada do cache são alguns GB de pesos vivos na memória, e a chave é uma combinação
# — não um modelo. Um processo que varre combinações (o notebook comparando revisões, uma
# sessão que troca de adapter) acumularia todas até o processo morrer, e a máquina de
# desenvolvimento tem 24 GB. Dois é o que o caso de uso pede: baseline e fine-tuned lado a
# lado. O descarte é por menos usado recentemente, para que a alternância entre esses dois
# não fique recarregando um a cada troca.
_LIMITE_MODELOS_EM_CACHE = 2


def _carregar(model_path: str, adapter_path: str | None, revision: str | None) -> tuple[Any, Any]:
    """Carrega modelo e tokenizer uma vez por combinação, reaproveitando nas seguintes."""
    chave = (model_path, adapter_path, revision)
    if chave in _MODELOS_CARREGADOS:
        # Reinsere no fim: `dict` preserva a ordem de inserção, então o fim é o uso mais
        # recente e o começo é o candidato natural ao descarte.
        _MODELOS_CARREGADOS[chave] = _MODELOS_CARREGADOS.pop(chave)
        return _MODELOS_CARREGADOS[chave]

    from mlx_lm import load

    carregado = load(model_path, adapter_path=adapter_path, revision=revision)
    while len(_MODELOS_CARREGADOS) >= _LIMITE_MODELOS_EM_CACHE:
        _MODELOS_CARREGADOS.pop(next(iter(_MODELOS_CARREGADOS)))
    _MODELOS_CARREGADOS[chave] = carregado
    return carregado


def _cortar_em_stop(texto: str, stop: list[str] | None) -> str:
    """Trunca na primeira sequência de parada.

    O contrato do `LLM._call` do LangChain é que a saída não contenha nenhuma das sequências
    de `stop`. O `mlx_lm.generate` não recebe essa lista, então o corte é feito aqui — na
    ocorrência mais à esquerda, e não na primeira da lista, senão a ordem em que o chamador
    escreveu as sequências mudaria onde o texto é cortado.
    """
    if not stop:
        return texto
    posicoes = [texto.find(marca) for marca in stop if marca and marca in texto]
    if not posicoes:
        return texto
    return texto[: min(posicoes)].rstrip()


class MedicalMLXLLM(LLM):
    """LLM do assistente médico: Llama-3.2-3B com os adapters LoRA do PR 04.

    `adapter_path=None` serve o modelo base sem fine-tuning — é o que a comparação
    baseline vs fine-tuned do relatório técnico usa, e é o que `from_env(com_adapter=False)`
    monta a partir do `.env`.
    """

    # O Pydantic v2 reserva o prefixo `model_` para uso interno e avisa a cada instanciação
    # de um campo chamado `model_path`. O nome vem do plano do projeto e é o que o
    # `_identifying_params` expõe, então o que cede é a proteção de namespace — que aqui não
    # protege nada: esta classe não tem nenhum atributo `model_` do próprio Pydantic.
    model_config = ConfigDict(protected_namespaces=())

    model_path: str
    adapter_path: str | None = None
    max_tokens: int = MAX_TOKENS_PADRAO
    temperature: float = TEMPERATURE_PADRAO
    revision: str | None = None

    @classmethod
    def from_env(cls, com_adapter: bool = True) -> "MedicalMLXLLM":
        """Constrói a instância a partir do `.env`.

        A resolução dos caminhos vem do `LoRAConfig` em vez de reler `os.getenv` aqui: ele já
        ancora caminho relativo na raiz do repositório e já resolve `~`, e duplicar essa
        lógica é como o treino e a inferência acabam apontando para adapters diferentes.

        `com_adapter=False` serve o modelo base — é o outro lado da comparação baseline vs
        fine-tuned do relatório técnico. O parâmetro existe porque `config.adapter_path` é um
        `Path` com default sempre presente: não há valor de `.env` que produza `None`, então
        sem esta chave o baseline seria inalcançável por aqui e, numa máquina sem adapters
        treinados, `_call` só teria o `FileNotFoundError` a oferecer.

        `MAX_TOKENS` e `TEMPERATURE` estão no `.env.example` desde o PR 01 e até agora nada
        as consumia — mesmo caso do `BASE_MODEL` que o PR 04 ligou ao `LoRAConfig`.
        """
        from src.fine_tuning.config import LoRAConfig

        config = LoRAConfig()
        return cls(
            model_path=config.model,
            adapter_path=str(config.adapter_path) if com_adapter else None,
            revision=config.revisao_efetiva,
            max_tokens=int(os.getenv("MAX_TOKENS") or MAX_TOKENS_PADRAO),
            temperature=float(os.getenv("TEMPERATURE") or TEMPERATURE_PADRAO),
        )

    @property
    def _llm_type(self) -> str:
        return "medical-mlx"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Identidade da instância para cache e callbacks do LangChain.

        Inclui `adapter_path` e `revision` porque são o que distingue baseline de fine-tuned:
        sem eles, um cache de LangChain trataria as duas configurações como o mesmo LLM.
        """
        return {
            "model_path": self.model_path,
            "adapter_path": self.adapter_path,
            "revision": self.revision,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def _aplicar_chat_template(self, tokenizer: Any, prompt: str) -> str:
        """Formata o prompt com o chat template do próprio modelo.

        Tem de ser o mesmo formato do treino, pelo mesmo motivo documentado em
        `evaluator._build_prompt`: o MLX-LM treinou aplicando `apply_chat_template` sobre os
        papéis user/assistant, e servir outro formato mede a diferença de template em vez do
        efeito do fine-tuning.

        O prompt inteiro vai como `user` — inclusive o texto de sistema que o PR 07 monta.
        Não há papel `system` aqui de propósito: o treino não viu nenhum, e introduzir um na
        inferência coloca o modelo diante de uma estrutura que ele nunca aprendeu a seguir.
        """
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """Gera a resposta para um prompt já montado."""
        if self.adapter_path is not None and not Path(self.adapter_path).is_dir():
            raise FileNotFoundError(
                f"{self.adapter_path} não existe. Rode 'python -m src.fine_tuning.trainer' "
                "antes, ou passe adapter_path=None para servir o modelo base."
            )

        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        modelo, tokenizer = _carregar(self.model_path, self.adapter_path, self.revision)
        resposta = generate(
            modelo,
            tokenizer,
            self._aplicar_chat_template(tokenizer, prompt),
            max_tokens=self.max_tokens,
            sampler=make_sampler(temp=self.temperature),
            verbose=False,
        )
        return _cortar_em_stop(resposta.strip(), stop)
