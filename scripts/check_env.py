"""Verificação de ambiente do Tech Challenge Fase 3.

Roda da raiz do repositório, com o venv ativo:

    python -m scripts.check_env

Sai com código 0 se tudo o que o projeto precisa está disponível, 1 se algo
essencial falhou. Avisos (AVISO) não derrubam o exit code — são coisas que só
importam mais adiante, como o token do HuggingFace.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import pathlib
import sys


def _raiz_do_repo() -> pathlib.Path | None:
    """Acha a raiz do projeto e a coloca no sys.path.

    Sem isso, rodar o script por caminho absoluto (python /tmp/check_env.py) coloca o
    diretório DO ARQUIVO no sys.path em vez da raiz do repo, e os imports de src/
    falham com "No module named 'src'" mesmo com o ambiente perfeito.
    """
    candidatos = [
        pathlib.Path(__file__).resolve().parent.parent,  # scripts/check_env.py -> raiz
        pathlib.Path.cwd(),                              # rodando da raiz
    ]
    for c in candidatos:
        if (c / "src" / "__init__.py").is_file() and (c / "requirements.txt").is_file():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            return c
    return None


RAIZ = _raiz_do_repo()

OK, FALHA, AVISO = "  ok   ", " FALHA ", " aviso "
falhas = 0
avisos = 0


def secao(titulo: str) -> None:
    print(f"\n{titulo}")
    print("-" * len(titulo))


def checa_import(rotulo: str, alvo: str, essencial: bool = True) -> None:
    """alvo: 'pacote' ou 'pacote:nome' para checar um atributo/símbolo."""
    global falhas, avisos
    modulo, _, simbolo = alvo.partition(":")
    try:
        mod = importlib.import_module(modulo)
        if simbolo and not hasattr(mod, simbolo):
            raise ImportError(f"{modulo} não expõe {simbolo}")
    except Exception as exc:  # noqa: BLE001 - queremos reportar qualquer falha
        if essencial:
            falhas += 1
            print(f"{FALHA}| {rotulo:34} | {type(exc).__name__}: {exc}")
        else:
            avisos += 1
            print(f"{AVISO}| {rotulo:34} | {type(exc).__name__}")
        return
    print(f"{OK}| {rotulo:34} | {alvo}")


def checa_ausencia(rotulo: str, alvo: str) -> None:
    """Confirma que uma API legada NÃO está disponível (LangChain 0.x)."""
    global falhas
    modulo, _, simbolo = alvo.partition(":")
    try:
        mod = importlib.import_module(modulo)
        if simbolo and not hasattr(mod, simbolo):
            raise ImportError
    except Exception:  # noqa: BLE001
        print(f"{OK}| {rotulo:34} | ausente, como esperado")
        return
    falhas += 1
    print(f"{FALHA}| {rotulo:34} | {alvo} AINDA existe — não usar no projeto")


def versao(pacote: str) -> str:
    try:
        return importlib.metadata.version(pacote)
    except importlib.metadata.PackageNotFoundError:
        return "não instalado"


def main() -> int:
    global avisos

    secao("1. Interpretador")
    em_venv = sys.prefix != sys.base_prefix
    print(f"{OK if em_venv else FALHA}| python {sys.version.split()[0]:26} | venv: {em_venv}")
    if not em_venv:
        globals()["falhas"] = falhas + 1
        print("         venv inativo — rode 'source venv/bin/activate' antes")

    secao("2. Dependências e versões")
    for pacote in (
        "langchain", "langchain-core", "langchain-community", "langgraph",
        "sqlalchemy", "datasets", "evaluate", "huggingface-hub",
        "numpy", "pandas", "matplotlib", "jupyterlab", "ipykernel",
        "pytest", "nltk", "sacrebleu", "rouge-score", "python-dotenv",
    ):
        v = versao(pacote)
        marca = FALHA if v == "não instalado" else OK
        if v == "não instalado":
            globals()["falhas"] = falhas + 1
        print(f"{marca}| {pacote:34} | {v}")

    v_mlx = versao("mlx-lm")
    if v_mlx == "não instalado":
        avisos += 1
        print(f"{AVISO}| {'mlx-lm':34} | não instalado (esperado fora de macOS/Apple Silicon)")
    else:
        print(f"{OK}| {'mlx-lm':34} | {v_mlx}")

    secao("3. Pacotes do projeto (src/)")
    if RAIZ is None:
        globals()["falhas"] = falhas + 1
        print(f"{FALHA}| {'raiz do repositório':34} | não encontrada")
        print("         rode da raiz do projeto (onde estão src/ e requirements.txt)")
    else:
        print(f"{OK}| {'raiz do repositório':34} | {RAIZ}")
        for p in ("data", "fine_tuning", "llm", "assistant", "graph", "database", "audit"):
            checa_import(f"src.{p}", f"src.{p}")

    secao("4. APIs usadas pelo projeto (LangChain 1.x)")
    checa_import("classe base do LLM", "langchain_core.language_models.llms:LLM")
    checa_import("prompt template", "langchain_core.prompts:ChatPromptTemplate")
    checa_import("LCEL Runnable", "langchain_core.runnables:Runnable")
    checa_import("histórico por sessão", "langchain_core.runnables.history:RunnableWithMessageHistory")
    checa_import("store do histórico", "langchain_core.chat_history:InMemoryChatMessageHistory")
    checa_import("StateGraph (LangGraph)", "langgraph.graph:StateGraph")
    checa_import("SQLAlchemy ORM", "sqlalchemy.orm:declarative_base")

    secao("5. APIs legadas que o projeto NÃO deve usar")
    checa_ausencia("LLMChain (langchain 0.x)", "langchain.chains:LLMChain")
    checa_ausencia("ConversationBufferMemory", "langchain.memory:ConversationBufferMemory")

    secao("6. MLX (fine-tuning)")
    if v_mlx == "não instalado":
        avisos += 1
        print(f"{AVISO}| {'mlx':34} | pulado (sem mlx-lm nesta plataforma)")
    else:
        try:
            import mlx.core as mx

            dev = mx.default_device()
            soma = (mx.array([1.0, 2.0]) * 2).tolist()
            print(f"{OK}| {'mlx device':34} | {dev} (teste: {soma})")
        except Exception as exc:  # noqa: BLE001
            globals()["falhas"] = falhas + 1
            print(f"{FALHA}| {'mlx device':34} | {type(exc).__name__}: {exc}")
        checa_import("mlx_lm.generate", "mlx_lm:generate")
        checa_import("mlx_lm.load", "mlx_lm:load")

    secao("7. Variáveis de ambiente (.env)")
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # noqa: BLE001
        pass
    token = os.getenv("HUGGINGFACE_TOKEN", "")
    if token.startswith("hf_"):
        print(f"{OK}| {'HUGGINGFACE_TOKEN':34} | definido")
    else:
        avisos += 1
        print(f"{AVISO}| {'HUGGINGFACE_TOKEN':34} | ausente — necessário no fine-tuning")
    for var in ("BASE_MODEL", "DB_PATH", "AUDIT_LOG_PATH", "ADAPTER_PATH"):
        valor = os.getenv(var)
        if valor:
            print(f"{OK}| {var:34} | {valor}")
        else:
            avisos += 1
            print(f"{AVISO}| {var:34} | não definido (rode 'cp .env.example .env')")

    print()
    if falhas:
        print(f"RESULTADO: {falhas} falha(s), {avisos} aviso(s) — ambiente NÃO está pronto")
        return 1
    print(f"RESULTADO: ambiente OK ({avisos} aviso(s), nenhum bloqueante)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
