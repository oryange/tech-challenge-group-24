"""Gate de commit: impede que output de notebook leve PII clínica ou token para o histórico.

Roda da raiz do repositório, com o venv ativo:

    python -m scripts.check_notebook_output notebooks/03_langchain_demo.ipynb

Sem argumento, varre os arquivos versionados que sabe checar. Sai com 0 quando nada
casou e 1 no primeiro achado — o `.pre-commit-config.yaml` o invoca como hook.

Por que este arquivo existe
---------------------------
O notebook de demonstração é entregue **com output visível** (exigência da Fase 3), e
output de notebook vai para o histórico do git: não há `.gitignore` que o alcance, não há
permissão de arquivo que o proteja e não há remoção possível depois do push. Isso contorna
de uma vez os três controles que hoje protegem a trilha de auditoria — o `logs/*` ignorado,
o `chmod 0600` do `AuditLogger` e a anonimização do `log()`.

`nbstripout` não serve: ele apaga todo o output, e o output é o entregável. O que serve é
um gate que **reprova o commit** em vez de sanear o arquivo — a mesma escolha que o
`AuditLogger` faz ao recusar texto livre acima do teto em vez de cortá-lo no meio.

A defesa de verdade é a allowlist de campos na célula que exibe a trilha (o notebook
projeta `timestamp`, `patient_id`, `session_id`, `guardrail_triggered`, `motivos`,
`tem_fonte`, `source` e `alergias_alertadas`, e mais nada). Este arquivo é a rede embaixo:
uma denylist curta que pega o caso em que alguém trocou a projeção por um `pprint` da lista
inteira. Denylist não substitui allowlist — ela cobre o dia em que a allowlist sair do lugar.

O que o gate **não** pega, e por quê
-------------------------------------
Nome de pessoa digitado solto na pergunta. O `anonymize` do `src/data/anonymizer.py` é
denylist ancorada em contexto: redige nome precedido de "paciente"/"Dr.", mas
`"João Silva ainda está com febre?"` — como um médico digita — passa em claro. Enumerar
nome arbitrário não é automatizável, e o banco semeado não ajuda: ele guarda só tokens
(`[PACIENTE_001]`, `[MÉDICO]`), então não há lista de nomes reais contra a qual cruzar.
Isso fica como conferência humana, e é item explícito da revisão do PR 10.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

# Os três padrões vêm do plano do PR 09 e cada um cobre um vazamento diferente.
#
# `response_preview` e `query` são os dois campos de texto livre da trilha de auditoria:
# derivam do contexto clínico e são exatamente o que a allowlist da célula deixa de fora.
# O padrão do `query` cobre as duas aspas de propósito — `get_session_logs` devolve
# `list[dict]`, e um `pprint` da lista sai com `'query':`, aspas simples, enquanto um
# `json.dumps` sai com `"query":`.
#
# O de token casa o formato real do token do HuggingFace: `hf_` seguido de 20 ou mais
# alfanuméricos **sem sublinhado** (os emitidos hoje têm 34). O plano do PR 09 pedia
# `hf_[A-Za-z0-9]`, de um caractere só; rodar o gate sobre o repositório mostrou por que
# não dá: `tests/test_fine_tuning.py:328` usa `hf_token_de_teste` num teste de regressão
# que garante que o token real não chega ao YAML do LoRA. Reprovar esse arquivo ensinaria
# a desligar o hook, que é o pior estado possível para um gate de segredo.
#
# Estreitar um gate de segredo é a direção arriscada, então não fica só nisso: a regra
# `segredo-do-env` confere o valor literal do `.env` e pega qualquer token, de qualquer
# formato e qualquer tamanho, na máquina onde o notebook é de fato executado e committado.
# Prefixo do HuggingFace porque é o único provedor que o projeto usa.
REGRAS_DE_OUTPUT: list[tuple[str, re.Pattern[str], str]] = [
    (
        "response-preview",
        re.compile(r"response_preview"),
        "campo de texto livre da trilha (recorte da resposta do modelo sobre contexto clínico)",
    ),
    (
        "query-na-trilha",
        re.compile(r"['\"]query['\"]\s*:"),
        "campo de texto livre da trilha (pergunta do médico)",
    ),
]

REGRA_DE_TOKEN = (
    "token-hf",
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "token do HuggingFace (prefixo hf_ seguido de 20+ alfanuméricos)",
)

# Mime types que não são texto: varrer o base64 de uma imagem não encontra vazamento
# nenhum (ninguém grepa PNG) e só gasta tempo em string de megabytes.
MIME_BINARIO = ("image/", "application/pdf")

EXTENSOES_DE_TEXTO = {".py", ".ipynb", ".md", ".txt", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".json"}

# Chaves do `.env` cujo valor é segredo. O teto de tamanho evita que um valor curto ou
# placeholder (`""`, `"changeme"`) vire um padrão que casa com meio repositório.
CHAVES_DE_SEGREDO = re.compile(r"(TOKEN|SECRET|KEY|PASSWORD|SENHA)", re.IGNORECASE)
TAMANHO_MINIMO_DE_SEGREDO = 12


def _raiz_do_repo() -> pathlib.Path:
    """Raiz do projeto, pelo mesmo critério do `scripts/check_env.py`."""
    for candidato in (pathlib.Path(__file__).resolve().parent.parent, pathlib.Path.cwd()):
        if (candidato / "src" / "__init__.py").is_file() and (candidato / "requirements.txt").is_file():
            return candidato
    return pathlib.Path.cwd()


def _segredos_do_env(raiz: pathlib.Path) -> list[str]:
    """Valores literais de segredo do `.env` local, quando ele existe.

    Oportunista de propósito: o `.env` é ignorado pelo git e não existe em clone limpo nem
    em CI. Quando existe — que é justamente a máquina onde o notebook é executado e
    committado —, dá para conferir a coisa mais direta possível: o token real aparece no
    arquivo? É mais forte que o padrão `hf_`, porque pega qualquer segredo, de qualquer
    formato, inclusive um que não tenha prefixo reconhecível.

    Os valores voltam como lista e nunca são impressos — só usados em `in`.
    """
    caminho = raiz / ".env"
    if not caminho.is_file():
        return []
    segredos = []
    for linha in caminho.read_text(encoding="utf-8", errors="replace").splitlines():
        chave, separador, valor = linha.partition("=")
        if not separador or chave.strip().startswith("#"):
            continue
        valor = valor.strip().strip("'\"")
        if CHAVES_DE_SEGREDO.search(chave) and len(valor) >= TAMANHO_MINIMO_DE_SEGREDO:
            segredos.append(valor)
    return segredos


def _textos(no: object, mime: str | None = None) -> list[str]:
    """Toda string dentro de um nó de output do notebook, menos os mime types binários.

    Desce recursivamente em vez de enumerar `text/plain`, `text/html` e companhia: um gate
    que lista o que sabe olhar deixa de olhar o que aparecer amanhã, e a lista de mime
    types que o Jupyter aceita é aberta.
    """
    if mime and mime.startswith(MIME_BINARIO):
        return []
    if isinstance(no, str):
        return [no]
    if isinstance(no, list):
        return [t for item in no for t in _textos(item, mime)]
    if isinstance(no, dict):
        return [t for chave, valor in no.items() for t in _textos(valor, chave if "/" in chave else mime)]
    return []


def _achados_no_notebook(caminho: pathlib.Path, segredos: list[str]) -> list[str]:
    """Achados de um `.ipynb`, já formatados. Lista vazia quando está limpo."""
    try:
        notebook = json.loads(caminho.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return [f"{caminho}: não foi possível ler o notebook como JSON ({exc.__class__.__name__})"]

    achados = []
    for indice, celula in enumerate(notebook.get("cells", []), start=1):
        fonte = "".join(celula.get("source", []))
        saidas = celula.get("outputs", [])
        texto_de_saida = "\n".join(_textos(saidas))

        # Traceback committado leva caminho resolvido no output e, em erro de cliente HTTP,
        # o token na URL da requisição. Vale a célula inteira, independente do conteúdo.
        if any(saida.get("output_type") == "error" for saida in saidas if isinstance(saida, dict)):
            achados.append(f"{caminho}:célula {indice}: [traceback] célula committada com exceção no output")

        # PII da trilha: só no output. A fonte pode citar `response_preview` num comentário
        # explicando por que o campo fica de fora — reprovar isso é castigar a documentação.
        for identificador, padrao, motivo in REGRAS_DE_OUTPUT:
            if padrao.search(texto_de_saida):
                achados.append(f"{caminho}:célula {indice}: [{identificador}] {motivo}")

        # Token: fonte e output. Não há lugar legítimo para um token neste repositório.
        identificador, padrao, motivo = REGRA_DE_TOKEN
        if padrao.search(fonte) or padrao.search(texto_de_saida):
            achados.append(f"{caminho}:célula {indice}: [{identificador}] {motivo}")

        for segredo in segredos:
            if segredo in fonte or segredo in texto_de_saida:
                achados.append(f"{caminho}:célula {indice}: [segredo-do-env] valor literal de segredo do .env")
                break

    return achados


def _achados_no_texto(caminho: pathlib.Path, segredos: list[str]) -> list[str]:
    """Achados de um arquivo de texto comum: só as regras de segredo.

    As de PII da trilha não se aplicam — `response_preview` e `query` são nomes de campo
    que o código-fonte e a documentação precisam poder mencionar.
    """
    try:
        conteudo = caminho.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    achados = []
    identificador, padrao, motivo = REGRA_DE_TOKEN
    if padrao.search(conteudo):
        achados.append(f"{caminho}: [{identificador}] {motivo}")
    for segredo in segredos:
        if segredo in conteudo:
            achados.append(f"{caminho}: [segredo-do-env] valor literal de segredo do .env")
            break
    return achados


def main(argv: list[str] | None = None) -> int:
    raiz = _raiz_do_repo()
    alvos = [pathlib.Path(a) for a in (argv if argv is not None else sys.argv[1:])]
    if not alvos:
        alvos = sorted(raiz.glob("notebooks/*.ipynb"))

    segredos = _segredos_do_env(raiz)

    achados: list[str] = []
    for alvo in alvos:
        if not alvo.is_file() or alvo.suffix not in EXTENSOES_DE_TEXTO:
            continue
        if alvo.suffix == ".ipynb":
            achados.extend(_achados_no_notebook(alvo, segredos))
        else:
            achados.extend(_achados_no_texto(alvo, segredos))

    if not achados:
        origem = "com o .env local" if segredos else "sem .env local (regra de segredo literal não rodou)"
        print(f"check_notebook_output: {len(alvos)} arquivo(s) conferido(s), nada a reportar — {origem}")
        return 0

    # O conteúdo que casou nunca é impresso: o achado sai como arquivo, célula e regra.
    # Ecoar o trecho colocaria a PII no terminal, no output do pre-commit e no log de CI —
    # espalhando exatamente o dado que o gate existe para conter.
    print("check_notebook_output: commit reprovado\n", file=sys.stderr)
    for achado in achados:
        print(f"  {achado}", file=sys.stderr)
    print(
        "\nO trecho que casou não é exibido de propósito. Abra a célula apontada e:\n"
        "  - trilha de auditoria: exiba pela allowlist de campos, não a entrada inteira\n"
        "  - traceback: rode a célula de novo até passar, e só então committe\n"
        "  - token/segredo: limpe o output, troque o segredo e use load_dotenv()\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
