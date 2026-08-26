"""Testes do pipeline de fine-tuning.

Nenhum teste aqui carrega o modelo nem treina: são 6GB de pesos e minutos de GPU. O que se
verifica é a lógica que produz o treino — argumentos, formato dos dados, split, parsing da
saída e cálculo das métricas.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.data.loader import save_jsonl
from src.fine_tuning.config import MODELO_BASE_PADRAO, LoRAConfig
from src.fine_tuning.evaluator import (
    available_checkpoints,
    best_checkpoint,
    compute_metrics,
    load_test_samples,
    materialize_checkpoint,
)
from src.fine_tuning.trainer import (
    MIN_TOKENS_RESPOSTA,
    _prepare_mlx_data,
    _write_yaml_config,
    parse_training_history,
    to_mlx_record,
    train,
)


def _registro(instruction="Qual a conduta?", input="Contexto clínico.", output=None, source="PubMedQA:1"):
    return {
        "instruction": instruction,
        "input": input,
        "output": output if output is not None else " ".join(["palavra"] * 25),
        "source": source,
    }


# --------------------------------------------------------------------------------------
# config.py
# --------------------------------------------------------------------------------------


def test_to_mlx_args_traz_os_hiperparametros_do_plano():
    args = LoRAConfig().to_mlx_args()

    def valor(flag):
        return args[args.index(flag) + 1]

    assert valor("--model") == "meta-llama/Llama-3.2-3B-Instruct"
    assert valor("--num-layers") == "8"
    assert valor("--batch-size") == "4"
    assert valor("--iters") == "500"
    assert valor("--learning-rate") == "0.0001"
    assert valor("--val-batches") == "25"
    assert "--train" in args


def test_max_seq_length_comporta_o_dataset():
    # Desvio deliberado do plano, que previa 512. Medido nos 903 exemplos de treino: com 512,
    # 46 sequências ficam sem nenhum token de resposta após o truncamento e a loss vira NaN
    # (reproduzido numa rodada real). O maior exemplo tem 875 tokens.
    assert LoRAConfig().max_seq_length == 1024


def test_to_mlx_args_e_lista_de_strings():
    # Vai direto para subprocess.run sem shell: qualquer valor não-string estouraria lá,
    # e uma string única seria interpretada como comando.
    args = LoRAConfig().to_mlx_args()

    assert isinstance(args, list)
    assert all(isinstance(a, str) for a in args)


def test_lora_scale_converte_alpha_por_rank():
    # No PEFT a escala efetiva é alpha/rank; no MLX-LM o campo `scale` é o multiplicador
    # aplicado direto. Passar alpha=16 como scale daria escala 8x maior que a pretendida.
    assert LoRAConfig(lora_alpha=16.0, lora_rank=8).lora_scale == 2.0
    assert LoRAConfig(lora_alpha=32.0, lora_rank=8).lora_scale == 4.0


def test_to_lora_parameters_tem_as_chaves_do_mlx():
    # rank, scale e dropout não têm flag de CLI — só existem no YAML, com estes nomes.
    assert set(LoRAConfig().to_lora_parameters()) == {"rank", "scale", "dropout"}


def test_mask_prompt_esta_ativo():
    # A loss tem de ser calculada só sobre a resposta: a pergunta sempre existe na inferência.
    assert "--mask-prompt" in LoRAConfig().to_mlx_args()


def test_adapter_path_do_ambiente_e_ancorado_na_raiz(monkeypatch):
    # ADAPTER_PATH no .env é relativo; sem ancorar, o destino mudaria conforme o cwd.
    monkeypatch.setenv("ADAPTER_PATH", "data/fine_tuned/adapters")

    caminho = LoRAConfig().adapter_path

    assert caminho.is_absolute()
    assert caminho.parts[-3:] == ("data", "fine_tuned", "adapters")


def test_adapter_path_expande_o_til(monkeypatch):
    # `ADAPTER_PATH=~/adapters` é natural de escrever e falhava em silêncio: sem expanduser
    # o `~` vira nome de diretório literal dentro do repositório, e os adapters vão para o
    # lugar errado sem erro nenhum.
    monkeypatch.setenv("ADAPTER_PATH", "~/adapters_de_teste")

    caminho = LoRAConfig().adapter_path

    assert "~" not in caminho.parts
    assert caminho == (Path.home() / "adapters_de_teste").resolve()


def test_adapter_path_e_normalizado_para_o_registro(monkeypatch):
    # Sem resolve(), o `..` sobrevive no caminho e o relativo_a_raiz — que compara de forma
    # lexical — grava algo como `../../tmp/x` no docs/, um caminho que parece interno à raiz
    # sem ser. O registro tem de dizer onde os adapters realmente ficaram.
    monkeypatch.setenv("ADAPTER_PATH", "data/../data/fine_tuned/adapters")

    caminho = LoRAConfig().adapter_path

    assert ".." not in caminho.parts
    assert caminho.parts[-3:] == ("data", "fine_tuned", "adapters")


def test_base_model_do_ambiente_chega_no_treino(monkeypatch):
    # A variável era validada pelo check_env sem ninguém consumi-la: trocá-la no .env não
    # mudava treino nem avaliação. O teste trava as duas pontas — a config e o argumento que
    # de fato vai para o mlx_lm.
    monkeypatch.setenv("BASE_MODEL", "outra-org/OutroModelo-1B")

    config = LoRAConfig()

    assert config.model == "outra-org/OutroModelo-1B"
    assert config.to_dict()["model"] == "outra-org/OutroModelo-1B"
    args = config.to_mlx_args()
    assert args[args.index("--model") + 1] == "outra-org/OutroModelo-1B"


def test_base_model_ausente_cai_no_padrao(monkeypatch):
    monkeypatch.delenv("BASE_MODEL", raising=False)

    assert LoRAConfig().model == MODELO_BASE_PADRAO


def test_model_revision_do_ambiente_entra_no_registro(monkeypatch):
    # Fixar a revisão é a única defesa contra uma retag upstream: nenhuma seed protege os
    # pesos que vêm do Hub.
    monkeypatch.setenv("MODEL_REVISION", "0cb88a4f764b7a12671c53f0838cd831a0843b95")

    registro = LoRAConfig().to_dict()

    assert registro["model_revision"] == "0cb88a4f764b7a12671c53f0838cd831a0843b95"
    assert registro["model_revision_fixada"] is True


def test_model_revision_ausente_nao_se_passa_por_fixada(monkeypatch):
    # Sem MODEL_REVISION o registro ainda pode trazer o SHA lido do cache local, e é aí que
    # mora a armadilha: um SHA observado descreve o que estava na máquina naquele dia, não
    # o que a próxima rodada vai baixar. A flag impede que o artefato pareça reprodutível.
    monkeypatch.delenv("MODEL_REVISION", raising=False)

    registro = LoRAConfig().to_dict()

    assert registro["model_revision_fixada"] is False


def test_config_e_imutavel():
    # A config é gravada como registro da rodada; mutá-la depois do treino faria o registro
    # descrever algo que não aconteceu.
    with pytest.raises(Exception):
        LoRAConfig().num_iters = 1  # type: ignore[misc]


# --------------------------------------------------------------------------------------
# trainer.py
# --------------------------------------------------------------------------------------


def test_to_mlx_record_usa_prompt_e_completion():
    # O MLX-LM aplica o chat_template do próprio modelo neste formato. Escrever
    # "<s>[INST]..." à mão produziria o template do Mistral num modelo Llama 3.2.
    registro = to_mlx_record(_registro(instruction="Pergunta?", input="Ctx", output="Resposta"))

    assert set(registro) == {"prompt", "completion"}
    assert "[INST]" not in registro["prompt"]
    assert registro["completion"] == "Resposta"


def test_to_mlx_record_rotula_o_contexto():
    registro = to_mlx_record(_registro(instruction="Pergunta?", input="Abstract do estudo."))

    assert registro["prompt"].startswith("Pergunta?")
    assert "Contexto:\nAbstract do estudo." in registro["prompt"]


def test_to_mlx_record_sem_contexto_nao_deixa_rotulo_vazio():
    registro = to_mlx_record(_registro(instruction="Pergunta?", input=""))

    assert registro["prompt"] == "Pergunta?"
    assert "Contexto:" not in registro["prompt"]


def test_prepare_mlx_data_divide_90_10(tmp_path):
    entrada = tmp_path / "dataset.jsonl"
    save_jsonl([_registro(instruction=f"p{i}") for i in range(100)], entrada)

    stats = _prepare_mlx_data(entrada, tmp_path / "mlx")

    assert (stats["train"], stats["validation"]) == (90, 10)
    assert (tmp_path / "mlx" / "train.jsonl").is_file()
    assert (tmp_path / "mlx" / "valid.jsonl").is_file()


def test_prepare_mlx_data_nao_vaza_validacao_para_o_treino(tmp_path):
    # É o que sustenta a avaliação: se houver interseção, ROUGE/BLEU medem memorização.
    entrada = tmp_path / "dataset.jsonl"
    save_jsonl([_registro(instruction=f"pergunta {i}") for i in range(100)], entrada)

    _prepare_mlx_data(entrada, tmp_path / "mlx")
    treino = {json.loads(l)["prompt"] for l in (tmp_path / "mlx" / "train.jsonl").read_text(encoding="utf-8").splitlines()}
    validacao = {json.loads(l)["prompt"] for l in (tmp_path / "mlx" / "valid.jsonl").read_text(encoding="utf-8").splitlines()}

    assert treino & validacao == set()


def test_prepare_mlx_data_descarta_prompt_que_nao_deixa_espaco_para_resposta(tmp_path):
    # É a proteção contra a loss NaN. Com --mask-prompt, um exemplo cujo prompt consome todo
    # o limite não tem token de alvo nenhum; a loss fica sem denominador e o NaN resultante
    # se propaga pelo otimizador. Reproduzido numa rodada real com max_seq_length=512.
    entrada = tmp_path / "dataset.jsonl"
    save_jsonl(
        [_registro(instruction=f"curto {i}") for i in range(9)]
        + [_registro(instruction="prompt gigantesco")],
        entrada,
    )

    def contar(prompt, completion):
        # O último registro tem prompt que estoura o limite; os demais cabem.
        return (400, 500) if "gigantesco" in prompt else (50, 150)

    stats = _prepare_mlx_data(
        entrada, tmp_path / "mlx", max_seq_length=256, contar_tokens=contar
    )

    assert stats["descartados_por_tamanho"] == 1
    assert stats["train"] + stats["validation"] == 9


def test_prepare_mlx_data_exige_margem_minima_de_resposta(tmp_path):
    # Prompt que cabe no limite mas deixa menos que MIN_TOKENS_RESPOSTA também é descartado:
    # ensinar o modelo a completar uma resposta cortada no meio é pior que não ensinar.
    # Aqui todos os 20 caem, então o split fica sem dados e a falha tem de ser explícita.
    entrada = tmp_path / "dataset.jsonl"
    save_jsonl([_registro(instruction=f"p{i}") for i in range(20)], entrada)
    prompt_no_limite = 100 - MIN_TOKENS_RESPOSTA + 1

    with pytest.raises(ValueError, match="Split inviável"):
        _prepare_mlx_data(
            entrada,
            tmp_path / "mlx",
            max_seq_length=100,
            contar_tokens=lambda p, c: (prompt_no_limite, 200),
        )


def test_prepare_mlx_data_aceita_prompt_exatamente_no_limite(tmp_path):
    # Limite inclusivo: prompt + MIN_TOKENS_RESPOSTA == max_seq_length ainda cabe.
    entrada = tmp_path / "dataset.jsonl"
    save_jsonl([_registro(instruction=f"p{i}") for i in range(20)], entrada)

    stats = _prepare_mlx_data(
        entrada,
        tmp_path / "mlx",
        max_seq_length=100,
        contar_tokens=lambda p, c: (100 - MIN_TOKENS_RESPOSTA, 200),
    )

    assert stats["descartados_por_tamanho"] == 0


def test_prepare_mlx_data_descarta_registro_sem_resposta(tmp_path):
    entrada = tmp_path / "dataset.jsonl"
    save_jsonl(
        [_registro(instruction=f"p{i}") for i in range(9)] + [_registro(instruction="x", output="")],
        entrada,
    )

    stats = _prepare_mlx_data(entrada, tmp_path / "mlx")

    assert stats["train"] + stats["validation"] == 9
    assert stats["descartados_sem_texto"] == 1


def test_prepare_mlx_data_dataset_vazio_erro_claro(tmp_path):
    entrada = tmp_path / "vazio.jsonl"
    save_jsonl([], entrada)

    with pytest.raises(ValueError, match="vazio"):
        _prepare_mlx_data(entrada, tmp_path / "mlx")


def test_prepare_mlx_data_poucos_registros_erro_claro(tmp_path):
    # Com poucos registros o corte de 90% deixa a validação vazia; melhor falhar dizendo
    # isso do que treinar sem conjunto de validação e reportar loss de validação ausente.
    entrada = tmp_path / "dataset.jsonl"
    save_jsonl([_registro()], entrada)

    with pytest.raises(ValueError, match="Split inviável"):
        _prepare_mlx_data(entrada, tmp_path / "mlx")


def test_write_yaml_config_grava_lora_parameters(tmp_path):
    destino = _write_yaml_config(LoRAConfig(), tmp_path / "sub" / "lora_config.yaml")

    conteudo = yaml.safe_load(destino.read_text(encoding="utf-8"))

    assert conteudo == {"lora_parameters": {"rank": 8, "scale": 2.0, "dropout": 0.0}}


def test_write_yaml_config_nao_grava_credencial(tmp_path, monkeypatch):
    # Regressão: o YAML fica em disco e o token do HuggingFace não tem o que fazer nele.
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_token_de_teste")

    destino = _write_yaml_config(LoRAConfig(), tmp_path / "lora_config.yaml")

    assert "hf_" not in destino.read_text(encoding="utf-8")


def test_train_grava_historico_parcial_quando_o_treino_falha(tmp_path, mocker):
    # Uma rodada de uma hora que quebra na iteração 490 não pode sumir sem registro: a curva
    # até o ponto da falha é o que permite diagnosticar sem repetir o treino.
    entrada = tmp_path / "dataset.jsonl"
    save_jsonl([_registro(instruction=f"p{i}") for i in range(20)], entrada)
    historico_path = tmp_path / "docs" / "training_history.json"

    mocker.patch(
        "src.fine_tuning.trainer._run_mlx",
        return_value=(["Iter 10: Train loss 2.455, Learning Rate 1.000e-04"], 1),
    )
    mocker.patch("src.fine_tuning.trainer.build_token_counter", return_value=None)

    with pytest.raises(RuntimeError, match="histórico parcial"):
        train(
            LoRAConfig(data_dir=tmp_path / "mlx"),
            dataset_path=entrada,
            historico_path=historico_path,
            log_path=tmp_path / "logs" / "fine_tuning.log",
        )

    gravado = json.loads(historico_path.read_text(encoding="utf-8"))
    assert gravado["exit_code"] == 1
    assert gravado["history"]["train"] == [{"iter": 10, "loss": 2.455}]


def test_train_grava_divergencia_em_json_estrito(tmp_path, mocker):
    # Trava o par inteiro: rodada divergente entra, JSON que qualquer parser lê sai, e o
    # best_checkpoint não elege campeão. O NaN cru é traiçoeiro porque o Python relê o
    # arquivo sem reclamar — só quem lê de fora (JSON.parse recusa, jq converte em silêncio)
    # descobre que o artefato está quebrado.
    entrada = tmp_path / "dataset.jsonl"
    save_jsonl([_registro(instruction=f"p{i}") for i in range(20)], entrada)
    historico_path = tmp_path / "docs" / "training_history.json"

    mocker.patch(
        "src.fine_tuning.trainer._run_mlx",
        return_value=(
            [
                "Iter 100: Train loss nan, Learning Rate 1.000e-04",
                "Iter 100: Val loss nan, Val took 1s",
                "Iter 200: Val loss -inf, Val took 1s",
            ],
            0,
        ),
    )
    mocker.patch("src.fine_tuning.trainer.build_token_counter", return_value=None)

    train(
        LoRAConfig(data_dir=tmp_path / "mlx"),
        dataset_path=entrada,
        historico_path=historico_path,
        log_path=tmp_path / "logs" / "fine_tuning.log",
    )

    texto = historico_path.read_text(encoding="utf-8")
    assert "NaN" not in texto and "Infinity" not in texto

    # `parse_constant` é chamado exatamente nos literais que a RFC 8259 não admite; levantar
    # aqui reproduz o rigor de um parser de fora do Python.
    def estrito(literal: str) -> object:
        raise AssertionError(f"literal não-JSON no artefato: {literal}")

    gravado = json.loads(texto, parse_constant=estrito)
    assert gravado["history"]["validation"] == [
        {"iter": 100, "loss": None},
        {"iter": 200, "loss": None},
    ]

    diretorio = _checkpoints(tmp_path, [100, 200])
    assert best_checkpoint(gravado["history"]["validation"], diretorio) is None


def test_parse_training_history_separa_treino_e_validacao():
    linhas = [
        "Iter 10: Train loss 2.345, Learning Rate 1.000e-04, It/sec 1.2",
        "Iter 50: Val loss 2.100, Val took 3.5s",
        "Iter 20: Train loss 1.987, Learning Rate 1.000e-04",
    ]

    historico = parse_training_history(linhas)

    assert historico["train"] == [
        {"iter": 10, "loss": 2.345},
        {"iter": 20, "loss": 1.987},
    ]
    assert historico["validation"] == [{"iter": 50, "loss": 2.1}]


def test_parse_training_history_registra_divergencia():
    # Regressão: a primeira versão da regex aceitava só dígitos, então uma rodada que
    # produziu "Train loss nan" do início ao fim virava histórico vazio — o notebook
    # desenhava um gráfico em branco em vez de mostrar que o treino havia quebrado.
    #
    # O ponto entra, mas com `None` no lugar de `float('nan')`: o valor precisa sobreviver à
    # serialização, e `NaN` não é JSON válido.
    historico = parse_training_history(
        [
            "Iter 2: Train loss nan, Learning Rate 1.000e-04",
            "Iter 2: Val loss nan, Val took 1s",
            "Iter 3: Train loss inf, Learning Rate 1.000e-04",
            "Iter 4: Train loss -inf, Learning Rate 1.000e-04",
        ]
    )

    assert historico["train"] == [
        {"iter": 2, "loss": None},
        {"iter": 3, "loss": None},
        {"iter": 4, "loss": None},
    ]
    assert historico["validation"] == [{"iter": 2, "loss": None}]


def test_parse_training_history_aceita_notacao_cientifica():
    historico = parse_training_history(["Iter 300: Train loss 8.5e-05, Learning Rate 1.000e-04"])

    assert historico["train"] == [{"iter": 300, "loss": 8.5e-05}]


def test_parse_training_history_ignora_ruido():
    historico = parse_training_history(["Loading model...", "", "Trainable parameters: 0.1%"])

    assert historico == {"train": [], "validation": []}


# --------------------------------------------------------------------------------------
# evaluator.py
# --------------------------------------------------------------------------------------


def test_compute_metrics_identico_e_perfeito():
    textos = ["o paciente apresenta melhora do quadro clínico após o tratamento"]

    metricas = compute_metrics(textos, textos)

    assert metricas["rouge_l"] == pytest.approx(1.0)
    assert metricas["bleu_4"] == pytest.approx(100.0)


def test_compute_metrics_texto_diferente_pontua_menos():
    referencia = ["o paciente apresenta melhora do quadro clínico após o tratamento"]
    ruim = ["resultado completamente distinto sem qualquer relação com a referência dada"]

    metricas = compute_metrics(ruim, referencia)

    assert metricas["rouge_l"] < 0.5
    assert metricas["bleu_4"] < 10.0


def test_compute_metrics_exige_pares_alinhados():
    with pytest.raises(ValueError, match="cada predição precisa da sua referência"):
        compute_metrics(["a", "b"], ["a"])


def test_compute_metrics_lista_vazia_erro_claro():
    with pytest.raises(ValueError, match="Nenhuma predição"):
        compute_metrics([], [])


def _checkpoints(tmp_path, iteracoes):
    diretorio = tmp_path / "adapters"
    diretorio.mkdir()
    (diretorio / "adapter_config.json").write_text("{}", encoding="utf-8")
    (diretorio / "adapters.safetensors").write_bytes(b"final")
    for i in iteracoes:
        (diretorio / f"{i:07d}_adapters.safetensors").write_bytes(f"ckpt{i}".encode())
    return diretorio


def test_available_checkpoints_ignora_o_adapter_final(tmp_path):
    # O adapters.safetensors final não carrega número de iteração no nome e é tratado
    # separadamente; misturá-lo aqui daria um "checkpoint" sem iteração conhecida.
    diretorio = _checkpoints(tmp_path, [100, 200, 300])

    assert sorted(available_checkpoints(diretorio)) == [100, 200, 300]


def test_available_checkpoints_diretorio_inexistente(tmp_path):
    assert available_checkpoints(tmp_path / "nao_existe") == {}


@pytest.mark.parametrize(
    "nome",
    [
        "200_adapters.safetensors.bak",  # sufixo extra
        "x200_adapters.safetensors",  # prefixo antes do número
        "200_adapters.safetensors\n",  # `$` casaria antes do \n final; `\Z` não
        "٢٠٠_adapters.safetensors",  # `\d` casa dígito Unicode e int() converte para 200
    ],
)
def test_available_checkpoints_ignora_nome_parecido(tmp_path, nome):
    # Cada intruso vai sozinho no diretório, de propósito. Junto com o checkpoint legítimo o
    # teste não provaria nada: os dois cairiam na mesma chave 200, `sorted(...) == [200]`
    # continuaria verdadeiro, e qual arquivo sobrevive dependeria da ordem do iterdir() — que
    # é justamente o defeito. Sozinho, o resultado é determinístico: ou casa, ou não casa.
    diretorio = tmp_path / "adapters"
    diretorio.mkdir()
    try:
        (diretorio / nome).write_bytes(b"intruso")
    except OSError:  # pragma: no cover - depende do sistema de arquivos
        pytest.skip(f"o sistema de arquivos não aceita o nome {nome!r}")

    assert available_checkpoints(diretorio) == {}


def test_best_checkpoint_escolhe_menor_loss_com_arquivo_em_disco(tmp_path):
    # Validação a cada 50 iterações, checkpoint a cada 100: o menor valor da curva (iter 150)
    # não tem arquivo, então a escolha tem de recair no melhor entre os que existem.
    diretorio = _checkpoints(tmp_path, [100, 200])
    validacao = [
        {"iter": 50, "loss": 1.90},
        {"iter": 100, "loss": 1.75},
        {"iter": 150, "loss": 1.60},  # melhor da curva, mas sem checkpoint salvo
        {"iter": 200, "loss": 1.70},
    ]

    iteracao, arquivo = best_checkpoint(validacao, diretorio)

    assert iteracao == 200
    assert arquivo.name == "0000200_adapters.safetensors"


def test_best_checkpoint_descarta_nan(tmp_path):
    # Uma rodada divergente não tem melhor checkpoint; min() sobre NaN devolveria qualquer um.
    diretorio = _checkpoints(tmp_path, [100, 200])
    validacao = [{"iter": 100, "loss": float("nan")}, {"iter": 200, "loss": 1.70}]

    iteracao, _ = best_checkpoint(validacao, diretorio)

    assert iteracao == 200


def test_best_checkpoint_descarta_none(tmp_path):
    # É assim que a divergência chega hoje: o parse_training_history grava `None`, e
    # `math.isnan(None)` levantaria TypeError.
    diretorio = _checkpoints(tmp_path, [100, 200])
    validacao = [{"iter": 100, "loss": None}, {"iter": 200, "loss": 1.70}]

    iteracao, _ = best_checkpoint(validacao, diretorio)

    assert iteracao == 200


def test_best_checkpoint_descarta_infinito(tmp_path):
    # `-inf` passava pelo filtro antigo (`math.isnan` só pega NaN) e, por ser menor que
    # qualquer real, sempre vencia o min() — apontando para o checkpoint de uma rodada que
    # explodiu, e de forma determinística, o que dá a impressão de escolha deliberada.
    diretorio = _checkpoints(tmp_path, [100, 200])
    validacao = [{"iter": 100, "loss": float("-inf")}, {"iter": 200, "loss": 1.70}]

    iteracao, _ = best_checkpoint(validacao, diretorio)

    assert iteracao == 200


def test_best_checkpoint_rodada_toda_divergente_devolve_none(tmp_path):
    diretorio = _checkpoints(tmp_path, [100, 200])
    validacao = [{"iter": 100, "loss": None}, {"iter": 200, "loss": None}]

    assert best_checkpoint(validacao, diretorio) is None


def test_best_checkpoint_sem_candidato_devolve_none(tmp_path):
    diretorio = _checkpoints(tmp_path, [100])

    assert best_checkpoint([{"iter": 50, "loss": 1.5}], diretorio) is None
    assert best_checkpoint([], diretorio) is None


def test_materialize_checkpoint_monta_diretorio_carregavel(tmp_path):
    # O mlx_lm.load espera adapters.safetensors + adapter_config.json no diretório.
    diretorio = _checkpoints(tmp_path, [200])

    destino = materialize_checkpoint(
        diretorio / "0000200_adapters.safetensors", diretorio, tmp_path / "best"
    )

    assert (destino / "adapters.safetensors").read_bytes() == b"ckpt200"
    assert (destino / "adapter_config.json").is_file()


def test_materialize_checkpoint_preserva_os_originais(tmp_path):
    # Copiar e não mover: os checkpoints têm de sobreviver para uma segunda comparação.
    diretorio = _checkpoints(tmp_path, [200])

    materialize_checkpoint(diretorio / "0000200_adapters.safetensors", diretorio, tmp_path / "best")

    assert (diretorio / "0000200_adapters.safetensors").is_file()
    assert (diretorio / "adapters.safetensors").read_bytes() == b"final"


def test_materialize_checkpoint_nao_deixa_destino_pela_metade(tmp_path):
    # Copiando direto, um adapter_config.json ausente só estoura no segundo copyfile, com o
    # primeiro já gravado — sobra um diretório com metade do que o load espera, mas com a
    # cara de um destino pronto. A falha tem de vir antes de qualquer escrita.
    diretorio = _checkpoints(tmp_path, [200])
    (diretorio / "adapter_config.json").unlink()
    destino = tmp_path / "best"

    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        materialize_checkpoint(diretorio / "0000200_adapters.safetensors", diretorio, destino)

    assert not destino.exists()


def test_materialize_checkpoint_recusa_checkpoint_inexistente(tmp_path):
    diretorio = _checkpoints(tmp_path, [200])
    destino = tmp_path / "best"

    with pytest.raises(FileNotFoundError):
        materialize_checkpoint(diretorio / "0000999_adapters.safetensors", diretorio, destino)

    assert not destino.exists()


def test_load_test_samples_limita_e_preserva_ordem(tmp_path):
    caminho = tmp_path / "valid.jsonl"
    save_jsonl([{"prompt": f"p{i}", "completion": f"c{i}"} for i in range(20)], caminho)

    amostras = load_test_samples(5, caminho)

    assert len(amostras) == 5
    assert [a["prompt"] for a in amostras] == ["p0", "p1", "p2", "p3", "p4"]
