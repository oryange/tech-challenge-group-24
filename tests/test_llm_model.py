"""Testes do wrapper LangChain do modelo MLX.

Nenhum peso é carregado aqui. O `mlx_lm` é substituído por um módulo de mentira em
`sys.modules` — e não por um `monkeypatch.setattr` sobre o pacote real — para que a suíte
rode também fora do Apple Silicon, onde o `mlx-lm` sequer é instalado (o marcador de
plataforma no `requirements.txt` o restringe a `darwin`/`arm64`).
"""

from __future__ import annotations

import sys
import types

import pytest

from src.llm import model as modulo_llm
from src.llm.model import MedicalMLXLLM, _cortar_em_stop


class _TokenizerFalso:
    def apply_chat_template(
        self, mensagens: list[dict[str, str]], add_generation_prompt: bool, tokenize: bool
    ) -> str:
        papeis = ",".join(m["role"] for m in mensagens)
        return f"<{papeis}>{mensagens[0]['content']}"


@pytest.fixture
def mlx_falso(monkeypatch):
    """Registra o que o wrapper pediu ao MLX, sem carregar modelo nenhum."""
    registro: dict = {"loads": [], "generates": []}

    def load(model_path, adapter_path=None, revision=None):
        registro["loads"].append((model_path, adapter_path, revision))
        return ("modelo-falso", _TokenizerFalso())

    def generate(modelo, tokenizer, prompt, max_tokens, sampler, verbose):
        registro["generates"].append(
            {"prompt": prompt, "max_tokens": max_tokens, "sampler": sampler}
        )
        return "  Conduta sugerida.\n\nRodapé que o stop deve cortar.  "

    mlx = types.ModuleType("mlx_lm")
    mlx.load = load
    mlx.generate = generate

    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda temp: f"sampler(temp={temp})"
    mlx.sample_utils = sample_utils

    monkeypatch.setitem(sys.modules, "mlx_lm", mlx)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    # Cache limpo por teste: ele é global no módulo e vazaria de um teste para o outro.
    monkeypatch.setattr(modulo_llm, "_MODELOS_CARREGADOS", {})
    return registro


def _llm(**kwargs) -> MedicalMLXLLM:
    kwargs.setdefault("model_path", "meta-llama/Llama-3.2-3B-Instruct")
    kwargs.setdefault("adapter_path", None)
    return MedicalMLXLLM(**kwargs)


def test_llm_type():
    assert _llm()._llm_type == "medical-mlx"


def test_identifying_params_distingue_baseline_de_finetuned(tmp_path):
    baseline = _llm()._identifying_params
    fine_tuned = _llm(adapter_path=str(tmp_path))._identifying_params

    assert baseline != fine_tuned
    assert baseline["adapter_path"] is None


def test_call_aplica_o_chat_template_do_modelo(mlx_falso):
    _llm().invoke("Qual a conduta na crise asmática?")

    prompt = mlx_falso["generates"][0]["prompt"]
    # Um único papel `user`: o treino não viu `system` nenhum.
    assert prompt.startswith("<user>")
    assert "Qual a conduta na crise asmática?" in prompt


def test_preload_carrega_antes_da_primeira_geracao(mlx_falso):
    llm = _llm()

    llm.preload()

    assert len(mlx_falso["loads"]) == 1
    # Carregou sem gerar nada: é só a espera saindo da primeira pergunta.
    assert mlx_falso["generates"] == []


def test_preload_nao_faz_a_geracao_seguinte_recarregar(mlx_falso):
    llm = _llm()

    llm.preload()
    llm.invoke("Pergunta clínica.")

    # Mesmo cache do `_call`: adiantar o carregamento não o duplica.
    assert len(mlx_falso["loads"]) == 1


def test_preload_falha_com_a_mesma_mensagem_do_call(mlx_falso, tmp_path):
    # O erro precisa ser o que ensina a rodar o trainer, e não um KeyError lá de dentro do
    # MLX — e agora ele aparece no banner de carregamento, não no meio da primeira pergunta.
    llm = _llm(adapter_path=str(tmp_path / "nao-existe"))

    with pytest.raises(FileNotFoundError, match="src.fine_tuning.trainer"):
        llm.preload()

    assert mlx_falso["loads"] == []


def test_call_repassa_temperature_e_max_tokens(mlx_falso):
    _llm(max_tokens=128, temperature=0.7).invoke("Pergunta clínica.")

    chamada = mlx_falso["generates"][0]
    assert chamada["max_tokens"] == 128
    assert chamada["sampler"] == "sampler(temp=0.7)"


def test_call_corta_na_sequencia_de_stop(mlx_falso):
    resposta = _llm().invoke("Pergunta clínica.", stop=["\n\n"])

    assert resposta == "Conduta sugerida."


def test_cortar_em_stop_usa_a_ocorrencia_mais_a_esquerda():
    # A ordem em que o chamador escreveu as sequências não pode mudar onde o texto é cortado.
    texto = "início FIM_A meio FIM_B fim"

    assert _cortar_em_stop(texto, ["FIM_B", "FIM_A"]) == "início"
    assert _cortar_em_stop(texto, ["FIM_A", "FIM_B"]) == "início"


def test_cortar_em_stop_sem_stop_preserva_o_texto():
    assert _cortar_em_stop("resposta inteira", None) == "resposta inteira"
    assert _cortar_em_stop("resposta inteira", ["ausente"]) == "resposta inteira"


def test_modelo_carregado_uma_vez_so(mlx_falso):
    llm = _llm()

    llm.invoke("Primeira pergunta.")
    llm.invoke("Segunda pergunta.")

    assert len(mlx_falso["loads"]) == 1
    assert len(mlx_falso["generates"]) == 2


def test_cache_separa_baseline_de_finetuned(mlx_falso, tmp_path):
    # Sem o adapter na chave, a comparação baseline vs fine-tuned do notebook receberia os
    # mesmos pesos nas duas pontas e mediria zero.
    _llm().invoke("Pergunta.")
    _llm(adapter_path=str(tmp_path)).invoke("Pergunta.")

    assert len(mlx_falso["loads"]) == 2
    assert mlx_falso["loads"][0][1] is None
    assert mlx_falso["loads"][1][1] == str(tmp_path)


def test_cache_descarta_a_combinacao_mais_antiga(mlx_falso, tmp_path):
    # Cada entrada são GB de pesos vivos: sem teto, um processo que varre combinações
    # acumularia todas até estourar a memória da máquina.
    adapters = []
    for indice in range(modulo_llm._LIMITE_MODELOS_EM_CACHE + 1):
        caminho = tmp_path / f"adapter_{indice}"
        caminho.mkdir()
        adapters.append(str(caminho))
        _llm(adapter_path=str(caminho)).invoke("Pergunta.")

    assert len(modulo_llm._MODELOS_CARREGADOS) == modulo_llm._LIMITE_MODELOS_EM_CACHE

    # O primeiro saiu do cache, então volta a ser carregado do zero.
    _llm(adapter_path=adapters[0]).invoke("Pergunta.")

    assert len(mlx_falso["loads"]) == modulo_llm._LIMITE_MODELOS_EM_CACHE + 2


def test_call_falha_cedo_se_o_adapter_nao_existe(mlx_falso, tmp_path):
    inexistente = tmp_path / "adapters_que_nao_foram_treinados"

    with pytest.raises(FileNotFoundError, match="trainer"):
        _llm(adapter_path=str(inexistente)).invoke("Pergunta.")

    assert mlx_falso["loads"] == []


def test_from_env_le_o_ambiente(monkeypatch, tmp_path):
    monkeypatch.setenv("BASE_MODEL", "outra-org/OutroModelo-1B")
    monkeypatch.setenv("MODEL_REVISION", "0cb88a4f764b7a12671c53f0838cd831a0843b95")
    monkeypatch.setenv("ADAPTER_PATH", str(tmp_path))
    monkeypatch.setenv("MAX_TOKENS", "128")
    monkeypatch.setenv("TEMPERATURE", "0.5")

    llm = MedicalMLXLLM.from_env()

    assert llm.model_path == "outra-org/OutroModelo-1B"
    assert llm.adapter_path == str(tmp_path)
    assert llm.revision == "0cb88a4f764b7a12671c53f0838cd831a0843b95"
    assert llm.max_tokens == 128
    assert llm.temperature == 0.5


def test_from_env_serve_o_baseline(monkeypatch, tmp_path):
    # `ADAPTER_PATH` tem default sempre presente no `LoRAConfig`, então não existe valor de
    # ambiente que produza o baseline: sem esta chave, o outro lado da comparação do
    # relatório não teria como ser montado a partir do `.env`.
    monkeypatch.setenv("ADAPTER_PATH", str(tmp_path))

    llm = MedicalMLXLLM.from_env(com_adapter=False)

    assert llm.adapter_path is None
    assert MedicalMLXLLM.from_env().adapter_path == str(tmp_path)


def test_from_env_cai_nos_padroes(monkeypatch):
    for variavel in ("MAX_TOKENS", "TEMPERATURE"):
        monkeypatch.delenv(variavel, raising=False)

    llm = MedicalMLXLLM.from_env()

    assert llm.max_tokens == modulo_llm.MAX_TOKENS_PADRAO
    assert llm.temperature == modulo_llm.TEMPERATURE_PADRAO
