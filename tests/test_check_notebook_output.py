"""Testes do gate de commit que confere o output dos notebooks.

O gate protege o entregável mais frágil da Fase 3: o `03_langchain_demo.ipynb` vai
committado **com output**, e output de notebook não sai mais do histórico do git. Um
controle de segurança sem teste é um controle que ninguém percebe quando para de
funcionar, e este em especial só é exercitado no dia em que alguém erra.

Cada teste abaixo é um caso que o gate precisa acertar, e os que mais importam são os
negativos: um gate que reprova uso legítimo ensina a desligar o hook.
"""

from __future__ import annotations

import json

import pytest

from scripts.check_notebook_output import (
    TAMANHO_MINIMO_DE_SEGREDO,
    _achados_no_notebook,
    _achados_no_texto,
    _segredos_do_env,
    _textos,
    main,
)

TOKEN_REAL = "hf_" + "AbCdEfGhIjKlMnOpQrStUvWxYz012345"
TOKEN_FALSO_DE_TESTE = "hf_token_de_teste"


def _notebook(tmp_path, fonte: str, saidas: list[dict], nome: str = "demo.ipynb"):
    caminho = tmp_path / nome
    caminho.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": fonte.splitlines(keepends=True),
                        "outputs": saidas,
                        "execution_count": 1,
                        "metadata": {},
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )
    return caminho


def _stdout(texto: str) -> dict:
    return {"output_type": "stream", "name": "stdout", "text": [texto]}


def _identificadores(achados: list[str]) -> set[str]:
    """Só as etiquetas `[regra]` dos achados, que é o que os testes afirmam."""
    return {a.split("[", 1)[1].split("]", 1)[0] for a in achados if "[" in a}


# --- PII da trilha de auditoria -------------------------------------------------------


def test_response_preview_no_output_reprova(tmp_path):
    nb = _notebook(tmp_path, "pprint(trilha)\n", [_stdout("{'response_preview': 'conduta...'}")])
    assert "response-preview" in _identificadores(_achados_no_notebook(nb, []))


@pytest.mark.parametrize("saida", ["{'query': 'paciente com febre?'}", '{"query": "x"}'])
def test_query_no_output_reprova_nas_duas_aspas(tmp_path, saida):
    """`get_session_logs` devolve `list[dict]`: um pprint sai com aspas simples, um
    json.dumps com aspas duplas. O gate tem de pegar as duas formas."""
    nb = _notebook(tmp_path, "pprint(trilha)\n", [_stdout(saida)])
    assert "query-na-trilha" in _identificadores(_achados_no_notebook(nb, []))


def test_campo_de_pii_citado_na_fonte_nao_reprova(tmp_path):
    """O caso negativo que mais importa: a célula documenta por que o campo fica de fora.

    Reprovar um comentário que explica a allowlist castiga exatamente a documentação que
    faz a allowlist sobreviver à próxima pessoa que mexer no notebook.
    """
    fonte = "# response_preview e 'query': ficam fora da allowlist de propósito\n"
    nb = _notebook(tmp_path, fonte, [_stdout("2026-09-12 [PACIENTE_002] True")])
    assert _achados_no_notebook(nb, []) == []


def test_allowlist_projetada_passa(tmp_path):
    """A forma que o notebook de demonstração usa de fato."""
    saida = "[{'timestamp': '2026-09-12T14:02:11', 'patient_id': '[PACIENTE_002]', "
    saida += "'guardrail_triggered': False, 'tem_fonte': True, 'alergias_alertadas': ()}]"
    nb = _notebook(tmp_path, "exibir_trilha(sessao)\n", [_stdout(saida)])
    assert _achados_no_notebook(nb, []) == []


# --- Token e segredo ------------------------------------------------------------------


def test_token_no_output_reprova(tmp_path):
    nb = _notebook(tmp_path, "print(os.getenv('HF_TOKEN'))\n", [_stdout(TOKEN_REAL)])
    assert "token-hf" in _identificadores(_achados_no_notebook(nb, []))


def test_token_na_fonte_reprova(tmp_path):
    """Token hardcoded não precisa de output para ser um problema."""
    nb = _notebook(tmp_path, f'os.environ["HF_TOKEN"] = "{TOKEN_REAL}"\n', [])
    assert "token-hf" in _identificadores(_achados_no_notebook(nb, []))


def test_token_falso_de_teste_nao_reprova(tmp_path):
    """`tests/test_fine_tuning.py` usa `hf_token_de_teste` para afirmar que o token real
    não chega ao YAML do LoRA. O gate encontrou esse arquivo na primeira execução sobre o
    repositório, e é por causa dele que o padrão exige o formato real (20+ alfanuméricos
    sem sublinhado) em vez do prefixo sozinho."""
    nb = _notebook(tmp_path, f'monkeypatch.setenv("HF_TOKEN", "{TOKEN_FALSO_DE_TESTE}")\n', [])
    assert _achados_no_notebook(nb, []) == []


def test_segredo_literal_do_env_reprova_sem_vazar_o_valor(tmp_path):
    """A regra que cobre o que o padrão `hf_` não cobre — e que não pode ecoar o segredo.

    Imprimir o trecho casado colocaria o segredo no terminal, no output do pre-commit e no
    log de CI: o gate espalharia o dado que existe para conter.
    """
    segredo = "ZxQv" * 8
    nb = _notebook(tmp_path, "print(chave)\n", [_stdout(f"a chave e {segredo} ok")])
    achados = _achados_no_notebook(nb, [segredo])

    assert "segredo-do-env" in _identificadores(achados)
    assert all(segredo not in achado for achado in achados)


def test_arquivo_de_texto_comum_so_leva_as_regras_de_segredo(tmp_path):
    """`response_preview` é nome de campo: o código e a documentação precisam citá-lo."""
    caminho = tmp_path / "notas.md"
    caminho.write_text("o campo response_preview e o 'query': ficam fora\n", encoding="utf-8")
    assert _achados_no_texto(caminho, []) == []

    caminho.write_text(f"token {TOKEN_REAL}\n", encoding="utf-8")
    assert "token-hf" in _identificadores(_achados_no_texto(caminho, []))


# --- Traceback ------------------------------------------------------------------------


def test_traceback_committado_reprova(tmp_path):
    """Exceção que sobe do carregamento do modelo leva caminho resolvido no output e, em
    erro de cliente HTTP, o token na URL da requisição."""
    erro = {
        "output_type": "error",
        "ename": "PacienteNaoEncontrado",
        "evalue": "[PACIENTE_999]",
        "traceback": ["Traceback (most recent call last):", "  File /Users/..."],
    }
    nb = _notebook(tmp_path, "assistant.ask('x', patient_id='[PACIENTE_999]')\n", [erro])
    assert "traceback" in _identificadores(_achados_no_notebook(nb, []))


# --- Leitura do `.env` ----------------------------------------------------------------


def _env(tmp_path, conteudo: str):
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "src" / "__init__.py").touch()
    (tmp_path / "requirements.txt").touch()
    (tmp_path / ".env").write_text(conteudo, encoding="utf-8")
    return tmp_path


def test_env_le_so_chave_de_segredo_com_valor_longo(tmp_path):
    raiz = _env(
        tmp_path,
        "HF_TOKEN=curto\n"
        f"API_KEY={'x' * TAMANHO_MINIMO_DE_SEGREDO}\n"
        "DB_PATH=data/database/hospital.db\n"
        "# COMENTADO_KEY=valor_comentado_longo\n",
    )
    segredos = _segredos_do_env(raiz)

    assert segredos == ["x" * TAMANHO_MINIMO_DE_SEGREDO]


def test_env_ausente_nao_quebra(tmp_path):
    """Clone limpo e CI não têm `.env`: a regra de literal simplesmente não roda."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").touch()
    (tmp_path / "requirements.txt").touch()
    assert _segredos_do_env(tmp_path) == []


# --- Extração de texto dos outputs ----------------------------------------------------


def test_mime_binario_nao_e_varrido():
    """O base64 de um PNG é string de megabytes onde não se acha vazamento nenhum."""
    saida = {"output_type": "display_data", "data": {"image/png": "iVBORw0KGgo", "text/plain": "figura"}}
    assert _textos([saida]) == ["display_data", "figura"]


def test_textos_desce_em_mime_novo():
    """A varredura é recursiva de propósito: a lista de mime types do Jupyter é aberta, e
    um gate que enumera o que sabe olhar deixa de olhar o que aparecer amanhã."""
    saida = {"output_type": "display_data", "data": {"application/vnd.qualquer+json": {"a": ["segredo"]}}}
    assert "segredo" in _textos([saida])


def test_notebook_ilegivel_vira_achado(tmp_path):
    """Falhar fechado: JSON quebrado não passa como 'nada a reportar'."""
    caminho = tmp_path / "quebrado.ipynb"
    caminho.write_text("{isso nao e json", encoding="utf-8")
    assert len(_achados_no_notebook(caminho, [])) == 1


def test_svg_e_varrido():
    """`image/svg+xml` é texto: rótulo e identificador de um gráfico saem legíveis dele.
    Descartá-lo junto com o PNG seria abrir buraco pela conveniência de um prefixo curto."""
    saida = {"output_type": "display_data", "data": {"image/svg+xml": "<text>[PACIENTE_005]</text>"}}
    assert "<text>[PACIENTE_005]</text>" in _textos([saida])


# --- Os dois modos de aprovar em silêncio ---------------------------------------------


def test_caminho_local_no_output_reprova(tmp_path):
    """Caminho absoluto leva o usuário do SO de quem executou para o histórico. Chega por
    warning de stderr, sem `output_type: error`, então a regra do traceback não o pega."""
    saida = {
        "output_type": "stream",
        "name": "stderr",
        "text": ["/Users/fulano/repo/venv/lib/tqdm/auto.py:21: TqdmWarning: IProgress not found.\n"],
    }
    caminho = _notebook(tmp_path, "import tqdm", [saida])
    achados = _achados_no_notebook(caminho, [])

    assert len(achados) == 1
    assert "caminho-local" in achados[0]
    assert "fulano" not in achados[0]


def test_temporario_do_sistema_no_output_reprova(tmp_path):
    """`/var/folders/...` é por onde o `ipykernel` nomeia a célula em todo `UserWarning`
    emitido de dentro do notebook. Não leva nome de pessoa, leva identificador de sessão do
    SO e PID do kernel — e o output é a parte da entrega que ninguém retira depois."""
    saida = {
        "output_type": "stream",
        "name": "stderr",
        "text": ["/var/folders/vx/h1dj5jxx0_s68/T/ipykernel_58722/2931937533.py:1: UserWarning: x\n"],
    }
    caminho = _notebook(tmp_path, "import warnings", [saida])
    achados = _achados_no_notebook(caminho, [])

    assert len(achados) == 1
    assert "caminho-local" in achados[0]


def test_caminho_relativo_nao_reprova(tmp_path):
    """O negativo que importa: o projeto exibe caminho relativo de propósito (`curto()` na
    célula 2), e um gate que reprovasse isso ensinaria a desligar o hook."""
    saida = {
        "output_type": "stream",
        "name": "stdout",
        "text": ["adapter_path   data/fine_tuned/adapters\n", "var/folders é nome de pasta comum\n"],
    }
    caminho = _notebook(tmp_path, "print(curto(p))", [saida])

    assert _achados_no_notebook(caminho, []) == []


def test_extensao_desconhecida_e_conferida(tmp_path):
    """O filtro de arquivo é denylist de binário, não allowlist de extensão: um token num
    `.sh` ou num `.env.example` importa tanto quanto num `.py`, e aprovar em silêncio o que
    não se sabe ler é indistinguível de estar limpo."""
    alvo = tmp_path / "deploy.sh"
    alvo.write_text(f"export HF_TOKEN={TOKEN_REAL}\n", encoding="utf-8")

    assert main([str(alvo)]) == 1


def test_binario_continua_fora_da_varredura(tmp_path):
    """A contrapartida: o que grep nenhum leria continua pulado, sem custo nem ruído."""
    alvo = tmp_path / "figura.png"
    alvo.write_bytes(b"\x89PNG\r\n\x1a\n" + TOKEN_REAL.encode())

    assert main([str(alvo)]) == 0


def test_arquivo_ilegivel_reprova_em_vez_de_aprovar(tmp_path):
    """Falhar fechado também no texto comum: um `.md` em latin-1 com token atravessaria o
    gate se a exceção virasse lista vazia, e sem uma linha no stderr avisando."""
    caminho = tmp_path / "notas.md"
    caminho.write_bytes(TOKEN_REAL.encode() + b"\n caf\xe9\n")

    achados = _achados_no_texto(caminho, [])

    assert len(achados) == 1
    assert "ilegivel" in achados[0]
