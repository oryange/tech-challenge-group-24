"""Fluxo de decisão clínica automatizado com LangGraph.

Uso como biblioteca:

    from src.graph.clinical_flow import run_clinical_flow

    estado = run_clinical_flow("[PACIENTE_007]")

Uso executável, a partir da raiz do repositório:

    python -m src.graph.clinical_flow

É o "fluxo automatizado" que o enunciado da Fase 3 cobra como entregável, e a diferença dele
para o `src/assistant/chain.py` é o que justifica o módulo existir: o assistente responde uma
pergunta que alguém fez, e este grafo **decide sozinho o que fazer** a partir do estado do
paciente. Ninguém digita a pergunta — o caminho é escolhido pelo dado do prontuário.

A decisão é uma só, e é a borda condicional depois de `check_exams`:

    intake -> check_exams -+-> alert_team ---------+-> human_validation -> END
                           |                       |
                           +-> suggest_treatment --+

**Havendo exame pendente, o grafo alerta e não sugere conduta.** É a escolha clínica do fluxo,
não um atalho de implementação: sugerir conduta sobre um quadro cujo exame ainda não voltou é
justamente o caso em que a sugestão parece completa e não é. O que a equipe precisa naquele
momento é saber o que falta, e é isso que o ramo entrega.

Nada de lógica nova mora aqui. O grafo é a ordem em que as peças já existentes do projeto se
aplicam, e o passo que fala com o modelo passa pelo `MedicalAssistant` inteiro — e não pelo
`MedicalMLXLLM` cru — de propósito: assim o guardrail de validação humana, o alerta de alergia
e a trilha de auditoria valem dentro do grafo exatamente como valem na CLI. Um segundo caminho
até o modelo, com metade das garantias, seria o defeito mais fácil de introduzir neste arquivo.

Sobre o que é registrado em disco: tudo o que este módulo grava passa pelo `AuditLogger`
(`src/audit/`), que anonimiza texto livre antes de escrever. O `patient_id` é o token
pseudonimizado do seed (`[PACIENTE_007]`), não um dado direto de identificação, e é a chave pela
qual a trilha é filtrada depois. O `session_id` vem da linha de comando e por isso é validado por
allowlist antes de entrar no fluxo — ver `_validar_session_id`.
"""

from __future__ import annotations

import os

if __name__ == "__main__":  # pragma: no cover
    # Mesmo motivo do `chain.py`: o `transformers` decide o que imprimir no import, então a
    # variável tem de estar no ambiente antes dele. Só no modo executável, para não mudar o
    # comportamento de quem importa este módulo como biblioteca.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import re
import uuid
from dataclasses import dataclass
from functools import partial
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.assistant.chain import MedicalAssistant
from src.assistant.retriever import PacienteNaoEncontrado, PatientRetriever
from src.audit.audit_logger import AuditLogger
from src.llm.guardrails import validate_response

# A pergunta que o grafo faz ao modelo no lugar do médico. É constante de módulo, e não texto
# montado a partir do estado, por duas razões que apontam para o mesmo lado: o fluxo é
# automatizado e ninguém revisa a pergunta antes de ela ir ao modelo, e uma pergunta interpolada
# com dado do prontuário seria conteúdo do banco entrando no prompt por fora do bloco
# `<contexto_do_paciente>` — que é exatamente a fronteira em que a proteção do `prompts.py` se
# apoia. O contexto do paciente já chega ao modelo pelo caminho certo, pelo `patient_id`.
PERGUNTA_DE_CONDUTA = (
    "Com base no quadro registrado e nos protocolos aplicáveis, qual a conduta sugerida "
    "para este paciente?"
)

# Template do alerta de exame pendente. Fixo no código, com o dado entrando como argumento —
# nunca o contrário. Formatar com uma string vinda de fora como *template* deixa quem a
# controla escrever especificadores de formato, e em `str.format` isso alcança atributos do
# processo (`{0.__class__.__mro__}`). Aqui o template é nosso e só os valores variam.
ALERTA_EXAME_PENDENTE = "Exame pendente: {tipo} (solicitado em {data})"

# O que vai no campo `query` da trilha quando o alerta é registrado. Não houve pergunta de
# médico neste ramo — é o fluxo agindo sozinho —, e deixar o campo vazio produziria uma linha
# sem origem identificável no meio das que vieram da CLI do assistente.
PASSO_DO_ALERTA = "[fluxo automatizado] alert_team: exames pendentes detectados"

CABECALHO_DO_ALERTA = (
    "[ALERTA PARA A EQUIPE] O paciente {patient_id} tem {total} exame(s) pendente(s). "
    "Conduta terapêutica não foi sugerida por este fluxo enquanto houver resultado em aberto."
)

# Allowlist do identificador de sessão. Ele não seleciona nada no banco e não decide caminho no
# grafo, então isto não é controle de acesso: é o que impede que um rótulo vindo da linha de
# comando entre sem limite de tamanho e sem forma na trilha de auditoria, que é o arquivo que o
# notebook abre e o vídeo de entrega grava. Sem teto, o campo vira depósito de texto livre — e
# texto livre no `session_id` não passa pela anonimização, que o `log()` aplica só à pergunta,
# à resposta e à fonte.
#
# O padrão é linear (uma classe de caracteres com repetição limitada, sem quantificador
# aninhado), pela mesma razão que as regex do `guardrails.py` são: um padrão com backtracking
# exponencial transformaria a própria validação em vetor de negação de serviço.
_SESSION_ID_VALIDO = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


class SessaoInvalida(ValueError):
    """`session_id` fora do formato aceito pela trilha de auditoria."""


class ClinicalState(TypedDict):
    """Estado que trafega entre os nós do grafo.

    A divisão entre `exams` e `pending_exams` é deliberada: o primeiro é o que o paciente tem
    registrado, o segundo é o subconjunto que decide o caminho. Guardar só a lista completa
    obrigaria cada nó a refazer o filtro, e "pendente" passaria a ser uma conclusão recalculada em
    vários lugares — que é como duas partes do mesmo fluxo acabam discordando sobre o mesmo
    paciente.

    `requires_validation` começa `False` e é o `human_validation_node` que o levanta. Não é
    redundância com o rodapé que o guardrail já impõe no texto: o rodapé é o que o médico lê, e
    este campo é o que um programa que consuma o estado consegue verificar sem fazer parsing da
    resposta.
    """

    patient_id: str
    exams: list[dict]
    pending_exams: list[dict]
    alerts: list[str]
    suggestions: str
    requires_validation: bool
    session_id: str


@dataclass(frozen=True)
class Dependencias:
    """As peças do projeto que os nós usam.

    Existe para que os nós continuem sendo funções de `(estado, dependências)` — testáveis uma
    a uma, com banco temporário e LLM falso, sem carregar 3B de pesos — e para que o grafo
    inteiro possa ser montado sobre as mesmas peças. A alternativa, cada nó chamando
    `from_env()` por conta própria, faria uma execução do grafo abrir três leituras diferentes
    do `.env` e, pior, três instâncias do modelo.

    O `retriever` aparece aqui **e** dentro do `assistant`, e não é engano: o `intake` e o
    `check_exams` consultam o banco sem falar com o modelo, e passar pelo assistente para isso
    obrigaria a ter um LLM carregado para responder "quantos exames pendentes existem".
    """

    retriever: PatientRetriever
    assistant: MedicalAssistant
    audit_logger: AuditLogger

    @classmethod
    def from_env(cls) -> "Dependencias":
        """Monta tudo a partir do `.env`, reaproveitando o retriever do assistente.

        O assistente é construído primeiro e o retriever dele é reusado, em vez de um segundo
        `PatientRetriever.from_env()`: são duas leituras do mesmo `DB_PATH` que poderiam
        divergir se a variável mudasse no meio, e uma delas seria a que decide o caminho do
        grafo enquanto a outra seria a que monta o contexto do modelo.
        """
        assistente = MedicalAssistant.from_env()
        return cls(
            retriever=assistente.retriever,
            assistant=assistente,
            audit_logger=assistente.audit_logger,
        )


def _validar_session_id(session_id: str) -> str:
    """Valida o `session_id` por allowlist antes de ele chegar à trilha.

    Allowlist e não denylist, e o motivo é o mesmo que o `retriever` registra para o
    `patient_id`: a lista do que é aceito tem borda conhecida, a lista do que é recusado tem a
    borda de quem a escreveu. O alcance aqui é o de um rótulo de sessão — letras, dígitos e os
    quatro separadores que aparecem em identificador de execução — e nada disso precisa de
    espaço, acento ou pontuação livre.
    """
    if not isinstance(session_id, str) or not _SESSION_ID_VALIDO.match(session_id):
        raise SessaoInvalida(
            "session_id fora do formato esperado: até 64 caracteres entre letras, dígitos "
            "e '.', '_', ':' ou '-'."
        )
    return session_id


def nova_sessao() -> str:
    """Identificador de sessão para uma execução do fluxo.

    `uuid4` e não um contador nem um `timestamp`: identificador sequencial ou previsível deixa
    quem lê a trilha enumerar as execuções vizinhas, e o valor não custa nada a mais para ser
    imprevisível. O prefixo é só para quem abre o `audit.jsonl` distinguir de olho o que veio
    do grafo do que veio da CLI do assistente.
    """
    return f"fluxo-{uuid.uuid4().hex[:12]}"


def intake_node(state: ClinicalState, deps: Dependencias) -> dict[str, Any]:
    """Recebe os dados iniciais e carrega o paciente do banco.

    É o único nó que pode falhar por dado de entrada, e ele falha **alto**: `patient_id` que
    não passa na allowlist do retriever ou que não existe no banco derruba a execução com a
    exceção original. Seguir com o estado vazio produziria um fluxo que roda inteiro, não
    encontra pendência nenhuma (porque não encontrou paciente nenhum) e termina afirmando que
    está tudo em ordem — a conclusão exatamente oposta à verdadeira, e indistinguível de uma
    execução legítima quando alguém for ler o resultado depois.

    A consulta é a do `PatientRetriever`, pelo ORM, com o valor vinculado como parâmetro.
    Nenhuma string de SQL é montada aqui.
    """
    dados = deps.retriever.get_patient_context(state["patient_id"])
    return {
        # O identificador que segue no estado é o que o banco tem gravado, e não o que veio na
        # entrada. As duas formas são iguais hoje, porque a allowlist do retriever já exige o
        # token exato; fixar a do banco é o que mantém isso verdadeiro se a normalização da
        # interface um dia aceitar uma forma abreviada.
        "patient_id": dados["patient_id"],
        "exams": dados["exames"],
        "pending_exams": [],
        "alerts": [],
        "suggestions": "",
        "requires_validation": False,
    }


def check_exams_node(state: ClinicalState, deps: Dependencias) -> dict[str, Any]:
    """Consulta os exames pendentes, que são o que decide o caminho do grafo.

    A consulta é refeita no banco em vez de filtrar `state["exams"]` por `status == "pending"`.
    É uma ida a mais ao SQLite e vale a pena: o que decide a borda condicional passa a vir da
    mesma função que o resto do projeto usa para responder "o que está pendente"
    (`get_pending_exams`), e não de um critério reescrito aqui. Duas definições de pendência no
    mesmo sistema é a forma mais silenciosa de o fluxo discordar da resposta que o assistente dá
    sobre o mesmo paciente.
    """
    return {"pending_exams": deps.retriever.get_pending_exams(state["patient_id"])}


def alert_team_node(state: ClinicalState, deps: Dependencias) -> dict[str, Any]:
    """Formata os alertas de exame pendente e registra a passagem na trilha.

    O que vai para o `audit.jsonl` é o alerta já formatado, e ele passa pelo `log()` do
    `AuditLogger` como qualquer outro texto — anonimizado antes de ser escrito. O conteúdo é tipo
    de exame e data, que saem do banco sintético e não de digitação livre, mas a trilha não tem
    como saber disso a partir do argumento: gravar por um caminho que pula a anonimização abriria
    uma exceção na garantia do módulo pelo lado menos vigiado, que é o do texto que "não deveria"
    ter dado pessoal.

    A `query` registrada descreve o passo do fluxo, não uma pergunta de médico, porque não
    houve pergunta nenhuma — este é o ramo automatizado. Deixá-la vazia faria a trilha ficar
    com uma linha sem origem identificável.
    """
    pendentes = state["pending_exams"]
    alertas = [
        CABECALHO_DO_ALERTA.format(patient_id=state["patient_id"], total=len(pendentes)),
        *(
            ALERTA_EXAME_PENDENTE.format(tipo=exame["tipo"], data=exame["data"])
            for exame in pendentes
        ),
    ]
    deps.audit_logger.log(
        query=PASSO_DO_ALERTA,
        response="\n".join(alertas),
        patient_id=state["patient_id"],
        session_id=state["session_id"],
        motivos=("exames_pendentes",),
    )
    return {"alerts": alertas}


def suggest_treatment_node(state: ClinicalState, deps: Dependencias) -> dict[str, Any]:
    """Pede ao modelo uma sugestão de conduta, pelo assistente completo.

    A chamada é `MedicalAssistant.ask` e não `llm.invoke`, e é a decisão mais importante deste
    arquivo. Pelo assistente, a sugestão sai com o contexto do paciente montado do jeito que o
    `chain.py` monta, com o alerta de alergia imposto pelo código, com o rodapé de validação humana
    garantido pelo guardrail e com a interação registrada na trilha. Falando direto com o LLM,
    nada disso valeria dentro do grafo — e o fluxo automatizado, que é justamente o que roda
    sem ninguém olhando, seria o caminho com menos garantias do sistema inteiro.

    O `session_id` do estado é repassado, então a linha que este nó grava no `audit.jsonl` cai
    na mesma sessão das outras da execução: a trilha do fluxo pode ser lida inteira com
    `get_session_logs`.

    As alergias alertadas viram alerta do fluxo, e não só marca no texto. Este ramo é o que não
    tem exame pendente, então o `alert_team_node` não roda e ninguém mais escreve em `alerts` —
    não há sobrescrita a temer. Uma contraindicação registrada no prontuário é o alerta mais
    importante que este grafo produz, e deixá-la só embutida na string da sugestão a esconderia
    de qualquer consumidor do estado que não leia o texto inteiro.
    """
    resultado = deps.assistant.ask(
        question=PERGUNTA_DE_CONDUTA,
        patient_id=state["patient_id"],
        session_id=state["session_id"],
    )
    alertas = [
        f"[ALERTA DE ALERGIA] Prontuário registra alergia a {alergia}."
        for alergia in resultado["alergias_alertadas"]
    ]
    return {"suggestions": resultado["response"], "alerts": alertas}


def human_validation_node(state: ClinicalState, deps: Dependencias) -> dict[str, Any]:
    """Marca a saída como dependente de validação humana. Último nó dos dois ramos.

    O nó é o ponto único por onde tudo sai do grafo, e é por isso que ele faz duas coisas em vez
    de uma. Levanta `requires_validation`, que é a forma verificável por programa; e passa a
    sugestão pelo `validate_response` dos guardrails, que é a forma legível por quem lê o texto.

    A segunda parte é idempotente — `validate_response` não acrescenta nada a um texto que já
    fecha com a marca, e o `ask` já a aplicou —, então na prática ela nunca muda o texto. Ela
    está aqui pelo caso em que muda: o `_MARCA_VALIDACAO_NO_FIM` dos guardrails aceita a marca
    escrita pelo próprio modelo, e há registro entre as limitações conhecidas do projeto de que
    essa marca é forjável. Enquanto isso valer, ter a checagem no nó que se chama "validação
    humana" é o lugar certo para ela estar quando alguém for consertar.

    O ramo do alerta não tem sugestão, e aí não há o que marcar: acrescentar o rodapé a uma
    string vazia produziria um texto que afirma precisar de validação sem ter nada dentro.

    `deps` não é usado, e o parâmetro fica assim mesmo — o editor aponta, e a resposta é esta.
    Os cinco nós têm a mesma assinatura porque o `build_graph` liga todos pelo mesmo
    `partial(..., deps=deps)`; abrir exceção para um deles trocaria um aviso de linter por uma
    ligação diferente das outras quatro, que é o tipo de assimetria que se paga quando alguém
    precisar dar uma dependência a este nó — e vai precisar, no dia em que a validação humana
    deixar de ser uma marca e virar um registro de quem validou.
    """
    return {
        "requires_validation": True,
        "suggestions": validate_response(state["suggestions"]) if state["suggestions"] else "",
    }


def _tem_exames_pendentes(state: ClinicalState) -> str:
    """Aresta condicional: o nome do próximo nó, decidido pelo estado do prontuário.

    A decisão é sobre `pending_exams`, que o `check_exams_node` acabou de trazer do banco — e
    não sobre um sinalizador que tenha vindo de fora junto com a entrada. É o mesmo princípio
    que o `intake_node` aplica ao falhar alto: caminho crítico decidido por dado que o próprio
    fluxo foi buscar na origem, nunca por parâmetro que o chamador conseguiria posicionar.
    """
    return "alert_team" if state["pending_exams"] else "suggest_treatment"


def build_graph(deps: Dependencias | None = None) -> Any:
    """Monta e compila o `StateGraph` do fluxo clínico.

    As dependências entram por `functools.partial` porque o LangGraph chama cada nó com o
    estado e nada mais. Fechando sobre elas aqui, os nós continuam sendo funções de dois
    argumentos — que é o que permite testá-los um a um — sem que o grafo precise carregá-las
    dentro do estado, onde elas trafegariam entre nós e acabariam serializadas junto com ele.

    `deps=None` constrói do `.env`, e isso carrega o modelo: quem só quer inspecionar a forma do
    grafo (o notebook de demonstração desenhando o diagrama, por exemplo) deve passar as
    dependências já montadas, ou montar o grafo com peças falsas.

    Todas as arestas são explícitas, inclusive as duas que chegam ao `human_validation`. É o que
    garante que nenhum ramo termine antes dele — o nó que impõe a validação humana não pode
    depender de o autor de um ramo novo lembrar de ligá-lo.
    """
    deps = deps or Dependencias.from_env()

    grafo = StateGraph(ClinicalState)
    grafo.add_node("intake", partial(intake_node, deps=deps))
    grafo.add_node("check_exams", partial(check_exams_node, deps=deps))
    grafo.add_node("alert_team", partial(alert_team_node, deps=deps))
    grafo.add_node("suggest_treatment", partial(suggest_treatment_node, deps=deps))
    grafo.add_node("human_validation", partial(human_validation_node, deps=deps))

    grafo.add_edge(START, "intake")
    grafo.add_edge("intake", "check_exams")
    grafo.add_conditional_edges(
        "check_exams",
        _tem_exames_pendentes,
        {"alert_team": "alert_team", "suggest_treatment": "suggest_treatment"},
    )
    grafo.add_edge("alert_team", "human_validation")
    grafo.add_edge("suggest_treatment", "human_validation")
    grafo.add_edge("human_validation", END)

    return grafo.compile()


def run_clinical_flow(
    patient_id: str,
    session_id: str | None = None,
    deps: Dependencias | None = None,
) -> ClinicalState:
    """Executa o fluxo para um paciente e devolve o estado final.

    `session_id=None` gera um novo — ver `nova_sessao`. O que vier do chamador é validado por
    allowlist antes de entrar no estado, porque daqui ele vai direto para a trilha de auditoria.

    O estado inicial é montado completo, com todos os campos do `ClinicalState`, e não só com o
    `patient_id`. Um `TypedDict` não valida nada em tempo de execução: campo faltando só aparece
    como `KeyError` dentro do nó que o lê, no meio da execução, com a metade anterior do fluxo
    já tendo escrito na trilha.
    """
    estado_inicial: ClinicalState = {
        "patient_id": patient_id,
        "exams": [],
        "pending_exams": [],
        "alerts": [],
        "suggestions": "",
        "requires_validation": False,
        "session_id": _validar_session_id(session_id or nova_sessao()),
    }
    return build_graph(deps).invoke(estado_inicial)


def _imprimir(estado: ClinicalState) -> None:
    print(f"\nPaciente: {estado['patient_id']}   |   sessão: {estado['session_id']}")
    print(f"Exames registrados: {len(estado['exams'])}   |   "
          f"pendentes: {len(estado['pending_exams'])}")

    caminho = "alert_team" if estado["pending_exams"] else "suggest_treatment"
    print(f"Caminho tomado após check_exams: {caminho}")

    if estado["alerts"]:
        print("\nAlertas:")
        for alerta in estado["alerts"]:
            print(f"  {alerta}")

    if estado["suggestions"]:
        print(f"\nSugestão de conduta:\n{estado['suggestions']}")

    print(f"\n(requer validação humana: {estado['requires_validation']})")


def main(argv: list[str] | None = None) -> None:
    """Interface de linha de comando do fluxo automatizado.

    Sem `--paciente`, roda o fluxo para **todos** os pacientes do banco. É o modo que a
    demonstração usa: o valor do grafo está na borda condicional, e uma execução só mostra um
    dos dois ramos — qual deles depende do paciente que se escolheu, o que é o contrário de
    demonstrar a decisão.
    """
    import argparse

    from dotenv import load_dotenv

    from src.assistant.chain import _normalizar_escolha
    from src.assistant.retriever import RAIZ

    parser = argparse.ArgumentParser(
        prog="python -m src.graph.clinical_flow",
        description="Fluxo de decisão clínica com LangGraph. Sem --paciente, roda para todos.",
    )
    parser.add_argument("--paciente", help="identificador: 7, 007 ou [PACIENTE_007]")
    parser.add_argument("--sessao", help="identificador da sessão na trilha de auditoria")
    args = parser.parse_args(argv)

    load_dotenv(RAIZ / ".env")

    print("Carregando modelo e adapters (pode levar alguns segundos)...", flush=True)
    deps = Dependencias.from_env()
    # Aqui e não na primeira sugestão, pelo mesmo motivo do `chain.main`: a mensagem acima
    # promete que o carregamento está acontecendo agora.
    deps.assistant.preload()

    disponiveis = deps.retriever.listar_pacientes()
    if args.paciente:
        alvo = _normalizar_escolha(args.paciente, disponiveis)
        if alvo is None:
            raise SystemExit(
                f"\n'{args.paciente}' não está no banco. Rode "
                "'python -m src.assistant.chain --listar' para ver os identificadores."
            )
        pacientes = [alvo]
    else:
        pacientes = disponiveis

    if not pacientes:
        raise SystemExit(
            "\nBanco sem pacientes. Rode 'python -m src.database.seed' antes."
        )

    # A sessão é uma só para a execução inteira, e não uma por paciente: assim
    # `get_session_logs` devolve a trilha completa do que este comando fez, que é a pergunta
    # que alguém faz ao auditar uma execução do fluxo.
    sessao = _validar_session_id(args.sessao) if args.sessao else nova_sessao()

    for patient_id in pacientes:
        try:
            _imprimir(run_clinical_flow(patient_id, session_id=sessao, deps=deps))
        except (PacienteNaoEncontrado, ValueError) as erro:
            # A mensagem da exceção é do nosso próprio código (allowlist do retriever ou
            # paciente ausente) e não carrega detalhe interno de banco nem stack trace.
            print(f"\n{patient_id}: {erro}")

    print(f"\nTrilha desta execução: sessão {sessao} em {deps.audit_logger.log_path}")


if __name__ == "__main__":
    main()
