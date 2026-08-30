"""Pipeline LangChain do assistente: recupera contexto, pergunta ao modelo, aplica limites.

Uso como biblioteca:

    from src.assistant.chain import MedicalAssistant

    assistente = MedicalAssistant.from_env()
    resultado = assistente.ask(
        question="Quais exames estão pendentes?",
        patient_id="[PACIENTE_007]",
        session_id="sessao-01",
    )

Uso interativo, a partir da raiz do repositório:

    python -m src.assistant.chain

É a peça que junta as quatro anteriores: o modelo fine-tuned do PR 04 pelo wrapper do PR 05,
os limites de atuação do PR 05, o banco de pacientes do PR 03 e a trilha de auditoria do
PR 06. Nada de lógica clínica mora aqui — este módulo é a ordem em que as peças se aplicam.

A chain é LCEL (`prompt | llm | parser`). O `LLMChain` e o `ConversationBufferMemory` não
existem mais no pacote principal do LangChain 1.x — foram para o `langchain-classic`, e
usá-los prenderia o projeto à linha 0.3.

O histórico da conversa é guardado em `InMemoryChatMessageHistory` por `session_id` e entra
no prompt como **texto**, não como turnos de mensagem. O porquê, com os números, está em
`create_chain`: com turno de assistente este modelo copiava a própria resposta anterior.

Duas coisas neste módulo são remendo, não solução, e estão marcadas como tal no código: o
`cortar_repeticao`, que existe porque o modelo degenera em loop, e a checagem de prescrição
antes da inferência. As correções de raiz são de outros PRs — `repetition_penalty` no PR 05
e o dataset no PR 04 — e estão registradas em "Pendências abertas" no `CHECKLIST_FASE3.md`.
"""

from __future__ import annotations

import os

if __name__ == "__main__":  # pragma: no cover
    # Antes dos imports de propósito, e só no modo interativo. O `transformers` decide o que
    # imprimir no momento em que é importado, então a variável precisa estar no ambiente
    # antes disso — depois já é tarde. Fica sob o guarda de `__main__` para não mudar o
    # comportamento de quem importa este módulo como biblioteca: a demonstração quer a tela
    # limpa, a suíte de testes quer ver os avisos.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import difflib
import re
import warnings
from pathlib import Path
from typing import Any

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from src.assistant.prompts import (
    MEDICAL_TEMPLATE,
    SEM_CONTEXTO,
    SYSTEM_PROMPT,
    neutralizar_delimitadores,
)
from src.assistant.retriever import PacienteNaoEncontrado, PatientRetriever
from src.audit.audit_logger import AuditLogger
from src.llm.guardrails import apply_guardrails, check_prescription_attempt, sanitize_input
from src.llm.model import MedicalMLXLLM

SESSAO_PADRAO = "sessao-local"

# Quantos pares pergunta/resposta anteriores entram no prompt, e quanto de cada resposta.
# Ambos existem pela mesma razão: histórico longo empurra a pergunta atual para longe do fim
# do prompt e ancora o modelo no que já foi dito.
TURNOS_NO_HISTORICO = 3
RESUMO_DA_RESPOSTA = 200

_ABERTURA_FONTE = re.compile(r"\[Fonte:\s*", re.IGNORECASE)

# Corte da resposta quando o modelo entra em loop. Frases curtas ficam de fora porque
# "Conduta:" ou "Achado esperado." repetidos não são degeneração, são estrutura.
#
# O separador é capturado (e não descartado) porque ele carrega a formatação: a resposta vem
# com item de lista em linha própria, e um `split` que engole o `\n` seguido de `" ".join`
# devolve a lista inteira colada num parágrafo só — na tela que vai para a demonstração.
_FIM_DE_FRASE = re.compile(r"((?<=[.!?])\s+)")
FRASE_MINIMA = 25
SIMILARIDADE_DE_REPETICAO = 0.9


def cortar_repeticao(texto: str) -> str:
    """Corta a resposta na primeira frase que repete uma anterior.

    Este modelo degenera: acerta a primeira ou as duas primeiras frases e depois repete uma
    delas até esgotar o `max_tokens`. O conteúdo útil está antes do loop, então o corte não
    perde informação — só para de exibir a mesma frase catorze vezes.

    É paliativo e não conserta a geração: a correção de verdade é `repetition_penalty` no
    `mlx_lm.generate`, que mora no `src/llm/model.py` do PR 05. Fica aqui porque o assistente
    não deve mostrar ao médico uma parede de texto repetido enquanto aquela decisão não sai.

    A comparação é por similaridade e não por igualdade porque o loop degrada junto: as
    repetições vêm com erro de digitação ("hemoglobria" no lugar de "hemoglobina"), e
    igualdade exata deixaria passar exatamente as piores.

    O espaçamento original entre as frases mantidas é preservado: o corte tira o trecho
    repetido e nada mais, então uma resposta em lista continua em lista.
    """
    # Com o separador capturado, o `split` alterna frase, separador, frase — o separador que
    # antecede a frase de índice `i` é o item `i - 1`.
    partes = _FIM_DE_FRASE.split(texto or "")
    mantidas: list[str] = []
    comparaveis: list[str] = []
    for indice in range(0, len(partes), 2):
        frase = partes[indice]
        chave = " ".join(frase.lower().split())
        if len(chave) >= FRASE_MINIMA and any(
            difflib.SequenceMatcher(None, chave, vista).ratio() >= SIMILARIDADE_DE_REPETICAO
            for vista in comparaveis
        ):
            break
        if indice:
            mantidas.append(partes[indice - 1])
        mantidas.append(frase)
        if len(chave) >= FRASE_MINIMA:
            comparaveis.append(chave)
    return "".join(mantidas).strip()


def extrair_fonte(resposta: str) -> str | None:
    """Primeira fonte citada na resposta, ou `None` se o modelo não citou nenhuma.

    A varredura conta a profundidade dos colchetes em vez de casar `\\[Fonte:([^\\]]*)\\]`.
    Regex parando no primeiro `]` está errado aqui porque a fonte aninha colchete de verdade:
    o modelo cita `[Fonte: exames do [PACIENTE_001]]`, com o token do paciente dentro, e o
    corte no primeiro fechamento devolve `exames do [PACIENTE_001` — um identificador
    truncado, que é pior que nenhum, porque vai para a trilha de auditoria parecendo válido.

    Continua linear no tamanho da resposta: uma passada só, sem retrocesso.

    `None` e não string vazia: a ausência de fonte é uma informação que o PR 06 grava e o
    relatório técnico mede como explainability. Um `""` no log seria indistinguível de uma
    fonte que o modelo citou vazia.
    """
    texto = resposta or ""
    inicio = _ABERTURA_FONTE.search(texto)
    if not inicio:
        return None

    profundidade = 1
    for posicao in range(inicio.end(), len(texto)):
        if texto[posicao] == "[":
            profundidade += 1
        elif texto[posicao] == "]":
            profundidade -= 1
            if profundidade == 0:
                return texto[inicio.end() : posicao].strip() or None

    # Colchete nunca fechado: o modelo cortou no meio da citação, provavelmente no limite de
    # `max_tokens`. Uma fonte pela metade não é rastreável, então vale como ausência.
    return None


class MedicalAssistant:
    """Assistente clínico: contexto do paciente, modelo fine-tuned e limites de atuação."""

    def __init__(
        self,
        llm: LLM,
        retriever: PatientRetriever,
        audit_logger: AuditLogger,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.audit_logger = audit_logger
        # O histórico é atributo de instância, e não global de módulo, por duas razões que
        # apontam para o mesmo lado: duas instâncias do assistente não podem enxergar a
        # conversa uma da outra, e um teste não pode herdar o histórico deixado pelo anterior.
        self._historicos: dict[str, InMemoryChatMessageHistory] = {}
        self.chain = self.create_chain(llm)

    @classmethod
    def from_env(cls) -> "MedicalAssistant":
        """Monta o assistente inteiro a partir do `.env`."""
        return cls(
            llm=MedicalMLXLLM.from_env(),
            retriever=PatientRetriever.from_env(),
            audit_logger=AuditLogger.from_env(),
        )

    def _historico_da_sessao(self, session_id: str) -> BaseChatMessageHistory:
        return self._historicos.setdefault(session_id, InMemoryChatMessageHistory())

    @staticmethod
    def _formatar_historico(historico: BaseChatMessageHistory) -> str:
        """Últimos turnos como texto, com a resposta anterior encurtada.

        O corte em `RESUMO_DA_RESPOSTA` não é economia de token: uma resposta anterior longa
        passa a dominar o prompt e volta a ancorar o modelo na repetição — o mesmo efeito,
        atenuado, que o turno `AI:` causava por inteiro. O que o histórico precisa carregar é
        o assunto do turno anterior, para "e o que a última consulta registrou?" ter
        referente, não a resposta completa.
        """
        mensagens = historico.messages[-TURNOS_NO_HISTORICO * 2 :]
        if not mensagens:
            return ""
        linhas = []
        for mensagem in mensagens:
            papel = "Médico perguntou" if mensagem.type == "human" else "Você respondeu"
            texto = mensagem.content[:RESUMO_DA_RESPOSTA].strip()
            reticencias = "..." if len(mensagem.content) > RESUMO_DA_RESPOSTA else ""
            linhas.append(f"{papel}: {texto}{reticencias}")
        return "\n".join(linhas)

    def create_chain(self, llm: LLM) -> Runnable:
        """Monta a chain LCEL: `prompt | llm | parser`, com o `MEDICAL_TEMPLATE` inteiro.

        O histórico entra como **texto**, no slot `{history}` do template, e não como turnos
        `AI:` de um `MessagesPlaceholder`. Não é preferência de estilo — foi medido.

        Com o histórico injetado como turno de assistente, a resposta da segunda pergunta
        saía 100% idêntica à da primeira, nas duas temperaturas, para perguntas completamente
        diferentes: o modelo copiava a própria resposta anterior. Em sessões novas, sem
        histórico, as mesmas perguntas davam respostas 14% e 26% parecidas. Passando o
        histórico como texto dentro do bloco de dado, a similaridade cai para 24% e 10%.

        A explicação é a mesma que o PR 05 usa para não mandar papel `system`: este modelo
        foi fine-tuned em pares pergunta/resposta soltos e nunca viu conversa multi-turno. Um
        bloco `AI: <resposta longa>` é estrutura fora da distribuição dele, e a continuação
        mais provável diante dela é repeti-la.

        O `InMemoryChatMessageHistory` por `session_id` continua sendo o armazenamento, como
        o plano pede. O que saiu foi a `RunnableWithMessageHistory`, que só sabe injetar
        mensagens — e que já estava deprecada de qualquer forma.
        """
        return ChatPromptTemplate.from_messages([("human", MEDICAL_TEMPLATE)]) | llm | StrOutputParser()

    def ask(
        self,
        question: str,
        patient_id: str | None = None,
        session_id: str = SESSAO_PADRAO,
    ) -> dict[str, Any]:
        """Responde uma pergunta clínica e devolve a resposta já dentro dos limites.

        A ordem dos passos é a do plano do projeto, e cada um está onde está por um motivo:

        1. `sanitize_input` **antes** de qualquer outra coisa: o texto que entra no prompt é o
           saneado, e é ele que vai para o log — o original não é persistido em lugar nenhum.
        2. Contexto do paciente, que entra no prompt como dado delimitado.
        3. `check_prescription_attempt` sobre a pergunta, antes da inferência. Aqui o aviso é
           usado como reforço dentro do próprio contexto: quando alguém pede posologia, o
           modelo recebe o limite escrito junto do dado, em vez de só levar o carimbo depois.
        4-5. Prompt e chain.
        6. `apply_guardrails` sobre pergunta e resposta juntas — é o passo que garante o
           rodapé de validação humana mesmo que o modelo tenha ignorado a instrução.
        7. Trilha de auditoria.

        `PacienteNaoEncontrado` e `ValueError` do retriever sobem para o chamador de
        propósito: pergunta sobre paciente inexistente é erro de quem chamou, e responder
        assim mesmo (sem contexto, mas parecendo que teve) é pior que falhar.
        """
        pergunta = sanitize_input(question)

        contexto = None
        if patient_id:
            contexto = self.retriever.get_patient_context(patient_id)["contexto"]

        pediu_prescricao, aviso = check_prescription_attempt(pergunta)
        if pediu_prescricao:
            contexto = f"{contexto or SEM_CONTEXTO}\n\n{aviso}"

        historico = self._historico_da_sessao(session_id)

        # Os marcadores de bloco são desarmados no dado que entra, aqui e no `build_prompt`:
        # a chain monta o prompt pelo `ChatPromptTemplate` e não passa pelo `build_prompt`,
        # então a proteção precisa existir nos dois caminhos ou vale só num deles.
        bruta = self.chain.invoke(
            {
                "system": SYSTEM_PROMPT,
                "patient_context": neutralizar_delimitadores(contexto or SEM_CONTEXTO),
                "history": neutralizar_delimitadores(
                    self._formatar_historico(historico) or "(primeira pergunta desta sessão)"
                ),
                "question": neutralizar_delimitadores(pergunta),
            }
        )

        # A pergunta guardada é a saneada, e a resposta é a crua: o rodapé de validação e o
        # aviso de prescrição são acrescentados pelo guardrail a cada turno, e realimentá-los
        # ensinaria o modelo a escrevê-los sozinho — o que faria a marca deixar de distinguir
        # o que o guardrail garantiu do que o modelo inventou.
        historico.add_user_message(pergunta)
        historico.add_ai_message(bruta)

        # O corte vem antes do guardrail: o rodapé de validação tem de ficar no fim do
        # texto que o médico lê, não enterrado no meio do trecho repetido.
        resultado = apply_guardrails(pergunta, cortar_repeticao(bruta))
        fonte = extrair_fonte(resultado.resposta)

        # Só a pergunta e o recorte da resposta vão para a trilha, ambos anonimizados pelo
        # PR 06. O contexto do paciente fica de fora de propósito: ele já está no banco, e
        # copiá-lo para um arquivo que é aberto no notebook e gravado no vídeo de entrega
        # espalharia dado clínico sem responder nenhuma pergunta de auditoria a mais.
        self.audit_logger.log(
            query=pergunta,
            response=resultado.resposta,
            patient_id=patient_id,
            source=fonte,
            guardrail_triggered=resultado.guardrail_triggered,
            session_id=session_id,
        )

        return {
            "response": resultado.resposta,
            "source": fonte,
            "guardrail_triggered": resultado.guardrail_triggered,
            "patient_context_used": bool(patient_id),
        }


_CANCELADO = object()


def _normalizar_escolha(escolha: str, disponiveis: list[str]) -> str | None:
    """Aceita `7`, `007`, `PACIENTE_007` ou `[PACIENTE_007]`. `None` se não bater com nenhum.

    A normalização fica na interface, e não no `PatientRetriever._validar_patient_id`, de
    propósito: a allowlist do retriever é a validação de verdade e continua exigindo o token
    exato, porque ela protege a consulta ao banco. O que se conserta aqui é só a distância
    entre o que a pessoa digita naturalmente e o formato que o banco guarda — obrigar a
    digitar os colchetes não deixa nada mais seguro, só faz errar.

    O resultado é sempre um item de `disponiveis`, nunca uma string montada aqui: assim uma
    entrada que não corresponde a paciente nenhum não vira um token com cara de válido.
    """
    limpo = escolha.strip().strip("[]").upper()
    if limpo.isdigit():
        limpo = f"PACIENTE_{int(limpo):03d}"
    alvo = f"[{limpo}]"
    return alvo if alvo in disponiveis else None


def _escolher_paciente(retriever: PatientRetriever) -> Any:
    """Lê o paciente e valida **na hora**, antes de deixar o usuário digitar a pergunta.

    Validar só dentro do `ask` deixava a interface pedir um token que o usuário não tem como
    conhecer e só reclamar depois de ele ter escrito a pergunta inteira — que aí é descartada.
    O erro estava no lugar certo (o retriever é quem valida), na hora errada.

    Devolve `None` para "nenhum paciente" e `_CANCELADO` para Ctrl-C, que são coisas
    diferentes: a primeira segue para perguntas de conhecimento geral, a segunda encerra.
    """
    disponiveis = retriever.listar_pacientes()

    print("\nPasso 1 de 2 — escolha o paciente. A pergunta vem no passo 2.")
    print(f"  aceita '7', '007' ou '[PACIENTE_007]'   |   '?' lista os {len(disponiveis)}")
    print("  Enter segue sem paciente")

    while True:
        try:
            escolha = input("paciente> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return _CANCELADO

        if not escolha:
            return None
        if escolha == "?":
            # Em colunas: uma linha só com 20 tokens quebra na largura do terminal e vira
            # um bloco ilegível justamente na hora de escolher.
            for inicio in range(0, len(disponiveis), 5):
                print("  " + "  ".join(disponiveis[inicio : inicio + 5]))
            continue

        normalizado = _normalizar_escolha(escolha, disponiveis)
        if normalizado:
            return normalizado

        # Erro específico para o engano mais provável, que é digitar a pergunta aqui. Dizer
        # só "não está no banco" faz a pessoa tentar outro nome de paciente, quando o
        # problema é ela estar no campo errado.
        if escolha.endswith("?") or len(escolha.split()) > 3:
            print("  Isso parece uma pergunta — ela vem no passo 2, depois de escolher o")
            print("  paciente. Aqui vai só o identificador, como [PACIENTE_007].")
        else:
            print(f"  '{escolha}' não está no banco. Use '?' para ver a lista.")


def _imprimir(resultado: dict[str, Any]) -> None:
    print(f"\n{resultado['response']}")
    rodape = [f"fonte: {resultado['source'] or 'nenhuma citada'}"]
    if resultado["guardrail_triggered"]:
        rodape.append("guardrail acionado")
    print(f"\n({' | '.join(rodape)})")


def main(argv: list[str] | None = None) -> None:
    """Interface de linha de comando: interativa por padrão, ou uma pergunta só.

    O modo de uma pergunta só (`--pergunta`) existe porque o interativo é ruim para testar: a
    cada rodada é preciso esperar o modelo carregar e digitar tudo de novo, o que não dá para
    repetir nem colar num relatório. Com o argumento, a mesma pergunta pode ser rodada contra
    vários pacientes e o resultado comparado.

    O interativo continua sendo o que vai no vídeo de entrega: é a forma mais curta de
    mostrar o assistente respondendo, o guardrail agindo e a linha caindo no `audit.jsonl`.
    """
    import argparse

    from dotenv import load_dotenv
    from langchain_core._api import LangChainDeprecationWarning

    from src.assistant.retriever import RAIZ

    parser = argparse.ArgumentParser(
        prog="python -m src.assistant.chain",
        description="Assistente clínico. Sem argumentos, entra no modo interativo.",
    )
    parser.add_argument("--paciente", help="identificador, ex.: [PACIENTE_007]")
    parser.add_argument("--pergunta", help="faz uma pergunta só e encerra, sem modo interativo")
    parser.add_argument("--listar", action="store_true", help="lista os pacientes e encerra")
    args = parser.parse_args(argv)

    # O aviso da `RunnableWithMessageHistory` é uma decisão já tomada e registrada no
    # checklist. Silenciar aqui, e não no `create_chain`, é o que mantém o aviso visível na
    # suíte de testes — que é onde ele serve para alguém lembrar de revisitar a decisão.
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
    load_dotenv(RAIZ / ".env")

    if args.listar:
        # Antes de carregar o modelo: listar paciente não precisa de 3B de pesos em memória.
        for nome in PatientRetriever.from_env().listar_pacientes():
            print(nome)
        return

    print("Carregando modelo e adapters (pode levar alguns segundos)...")
    assistente = MedicalAssistant.from_env()

    # Caminho relativo à raiz, como o resto do projeto faz no que é exibido: o absoluto
    # ocupa duas linhas largas na tela e cola o resultado à máquina de quem rodou.
    def _curto(caminho: Any) -> str:
        try:
            return str(Path(caminho).relative_to(RAIZ))
        except ValueError:
            return str(caminho)

    print(f"Banco:  {_curto(assistente.retriever.db_path)}")
    print(f"Trilha: {_curto(assistente.audit_logger.log_path)}")

    if args.pergunta:
        try:
            _imprimir(assistente.ask(args.pergunta, patient_id=args.paciente))
        except (PacienteNaoEncontrado, ValueError) as erro:
            raise SystemExit(f"\n{erro}")
        return

    paciente = args.paciente or _escolher_paciente(assistente.retriever)
    if paciente is _CANCELADO:
        return

    print(f"\nPasso 2 de 2 — pergunte. Paciente: {paciente or 'nenhum'}")
    print("  Enter vazio ou Ctrl-C encerra.")
    while True:
        try:
            pergunta = input("pergunta> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not pergunta:
            return
        try:
            _imprimir(assistente.ask(pergunta, patient_id=paciente))
        except (PacienteNaoEncontrado, ValueError) as erro:
            print(f"\n{erro}")


if __name__ == "__main__":
    main()
