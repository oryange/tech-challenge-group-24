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
import unicodedata
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

# A citação inteira, com um nível de aninhamento: o modelo escreve `[Fonte: protocolo asma
# [CID J45]]` e `[Fonte: exames do [PACIENTE_001]]`, e um `[^\]]*` pararia no colchete de
# dentro — mesma armadilha que o `extrair_fonte` evita contando profundidade. Um nível basta
# porque é o que as formas citáveis produzem; o que passar disso simplesmente não é
# deduplicado, que é o lado seguro de errar aqui.
_TAG_DE_FONTE = re.compile(
    r"\[Fonte:[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]",
    re.IGNORECASE,
)

# Corte da resposta quando o modelo entra em loop. Frases curtas não são comparadas por
# similaridade porque "Conduta:" ou "Achado esperado." parecidos não são degeneração, são
# estrutura — mas *idênticas* elas são, e ficar de fora da comparação inteira era um buraco:
# "Solicitar espirometria." tem 22 caracteres, nunca chegava a `comparaveis` e o modelo
# repetiu a frase 30 vezes numa resposta real, com o corte instalado e sem efeito. Por isso a
# regra é dupla: acima do piso vale semelhança, abaixo dele vale igualdade exata.
#
# O separador é capturado (e não descartado) porque ele carrega a formatação: a resposta vem
# com item de lista em linha própria, e um `split` que engole o `\n` seguido de `" ".join`
# devolve a lista inteira colada num parágrafo só — na tela que vai para a demonstração.
_FIM_DE_FRASE = re.compile(r"((?<=[.!?])\s+)")
FRASE_MINIMA = 25
SIMILARIDADE_DE_REPETICAO = 0.9

# Alerta de alergia. Vem do banco e é imposto pelo código, não pedido ao modelo, pela mesma
# razão que o rodapé de validação do PR 05 é imposto: medido com o assistente completo, a
# pergunta "o paciente pode receber dipirona?" para o [PACIENTE_001] — que tem dipirona
# registrada como alergia — não mencionou a alergia em 4 de 4 tentativas. O modelo respondia
# sobre o protocolo da condição de base e ignorava a pergunta. Um alerta que só aparece quando
# a geração colabora não é alerta, e este é o caso em que errar machuca o paciente.
#
# O alerta abre a resposta em vez de fechá-la: o rodapé de validação já ocupa o fim, e uma
# contraindicação que aparece depois de três parágrafos de protocolo é lida por último, ou não
# é lida. É afirmação sobre o prontuário, não conduta — não substitui a avaliação de quem
# prescreve, e por isso o rodapé de validação continua vindo depois.
ALERTA_ALERGIA = (
    "[ALERTA DE ALERGIA: o prontuário deste paciente registra alergia a {alergias}. "
    "Confira a contraindicação antes de qualquer conduta.]"
)

# Só letras e dígitos separam os termos: a comparação é feita sobre a forma normalizada, então
# "Dipirona," e "dipirona" viram o mesmo token e a pontuação da pergunta não atrapalha.
_TERMOS = re.compile(r"[^\W_]+", re.UNICODE)

# O que numa fonte é conferível contra o contexto: data (`01/08/2026`) e código CID (`J45`,
# `J45.0`). São os dois formatos que o `SYSTEM_PROMPT` manda o modelo citar e os dois que ele
# consegue inventar com aparência de legítimos — o resto da citação é prosa, que não dá para
# conferir por comparação literal sem barrar paráfrase correta.
#
# `IGNORECASE` porque a comparação adiante é feita sobre a forma de `_normalizar`, que baixa
# tudo para minúscula. Sem ele a detecção era sensível à caixa e a comparação não, e a fresta
# invertia o resultado da função: `[Fonte: protocolo cid j99]` não casava identificador nenhum,
# caía no `if not identificadores` e a citação fabricada entrava na trilha como conferida — sem
# nem o aviso que denunciaria. Errar para o lado de casar demais é o lado certo aqui: um falso
# identificador na prosa faz a fonte ser conferida contra o contexto, não descartada de saída.
_IDENTIFICADOR_DE_FONTE = re.compile(
    r"\d{2}/\d{2}/\d{4}|\b[A-Z]\d{2}(?:\.\d+)?\b",
    re.IGNORECASE,
)


def _normalizar(texto: str) -> str:
    """Minúsculas e sem acento, para comparar termo de alergia com o que o médico digitou.

    Sem isto, "Dipirona" na pergunta e "dipirona" no prontuário não casam, e o alerta deixaria
    de sair justamente por diferença de caixa — falha silenciosa e do lado errado.
    """
    decomposto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(caractere for caractere in decomposto if not unicodedata.combining(caractere))


def alergias_citadas(texto: str, alergias: list[str]) -> list[str]:
    """Alergias do prontuário mencionadas num texto, na grafia original do prontuário.

    O texto é a pergunta do médico **ou** a resposta do modelo: as duas passam por aqui, e o
    porquê está em `ask`. A função não sabe qual é qual de propósito — a pergunta que cita o
    fármaco e a resposta que o oferece são o mesmo problema visto de dois lados.

    A comparação é por termo inteiro e não por substring: "sulfa" como substring casaria
    dentro de "sulfametoxazol" — o que aqui até seria desejável — mas também dentro de
    palavras sem relação, e um alerta que dispara sozinho é ignorado depois da terceira vez.
    Alergia registrada com mais de uma palavra ("contraste iodado") casa quando todos os
    termos dela aparecem no texto.

    O alcance é o que o prontuário registra, literalmente: esta função **não** sabe que
    "novalgina" é dipirona nem que um paciente alérgico a penicilina pode reagir a
    cefalosporina. Cobrir isso exigiria uma base de sinônimos e de reatividade cruzada que o
    projeto não tem, e afirmar a garantia maior seria pior do que declarar esta — mesma razão
    pela qual o `audit_logger` declara que não cobre nome sem âncora.
    """
    no_texto = set(_TERMOS.findall(_normalizar(texto)))
    citadas = []
    for alergia in alergias:
        termos = set(_TERMOS.findall(_normalizar(alergia)))
        if termos and termos <= no_texto:
            citadas.append(alergia)
    return citadas


def cortar_repeticao(texto: str) -> str:
    """Corta a resposta na primeira frase que repete uma anterior.

    Este modelo degenera: acerta a primeira ou as duas primeiras frases e depois repete uma
    delas até esgotar o `max_tokens`. O conteúdo útil está antes do loop, então o corte não
    perde informação — só para de exibir a mesma frase catorze vezes.

    É paliativo e não conserta a geração: o modelo continua gastando o `max_tokens` no loop,
    e o que muda é só o que o médico vê. A correção na geração seria `repetition_penalty` no
    `mlx_lm.generate`, medida duas vezes e sem ganho nas duas — ver a entrada P1 do
    `CHECKLIST_FASE3.md`, que também registra por que a medição feita sobre a resposta já
    cortada não é a que decidiria isso.

    A comparação é por similaridade e não por igualdade porque o loop degrada junto: as
    repetições vêm com erro de digitação ("hemoglobria" no lugar de "hemoglobina"), e
    igualdade exata deixaria passar exatamente as piores.

    Frase curta não entra na comparação por similaridade — duas frases de estrutura curtas se
    parecem por acaso —, mas entra na comparação por igualdade: repetir "Solicitar
    espirometria." trinta vezes é loop com qualquer tamanho de frase.

    O espaçamento original entre as frases mantidas é preservado: o corte tira o trecho
    repetido e nada mais, então uma resposta em lista continua em lista.
    """
    # Com o separador capturado, o `split` alterna frase, separador, frase — o separador que
    # antecede a frase de índice `i` é o item `i - 1`.
    partes = _FIM_DE_FRASE.split(texto or "")
    mantidas: list[str] = []
    comparaveis: list[str] = []
    vistas: set[str] = set()
    for indice in range(0, len(partes), 2):
        frase = partes[indice]
        chave = " ".join(frase.lower().split())
        repetida = chave in vistas or (
            len(chave) >= FRASE_MINIMA
            and any(
                difflib.SequenceMatcher(None, chave, vista).ratio()
                >= SIMILARIDADE_DE_REPETICAO
                for vista in comparaveis
            )
        )
        if chave and repetida:
            break
        if indice:
            mantidas.append(partes[indice - 1])
        mantidas.append(frase)
        if chave:
            vistas.add(chave)
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


def deduplicar_fontes(resposta: str) -> str:
    """Mantém só a primeira ocorrência de cada citação `[Fonte: ...]` repetida.

    O modelo abre a resposta com a citação e a repete no fim, às vezes duas vezes seguidas —
    medido no assistente completo, três tags na mesma resposta de duas linhas. É ruído de
    apresentação: a informação é a mesma, e o médico lê a segunda tag como se apontasse para
    outra origem.

    Só a **repetição literal** é removida. Duas fontes diferentes na mesma resposta ficam as
    duas: elas podem estar cobrindo afirmações diferentes, e escolher uma seria decidir por
    conta própria de onde veio o quê.
    """
    vistas: set[str] = set()

    def manter(casamento: re.Match[str]) -> str:
        chave = _normalizar(casamento.group(0))
        if chave in vistas:
            return ""
        vistas.add(chave)
        return casamento.group(0)

    return _TAG_DE_FONTE.sub(manter, resposta or "").strip()


def fonte_confere(fonte: str | None, contexto: str | None) -> bool:
    """Diz se a fonte citada corresponde a algo que estava mesmo no contexto entregue.

    Existe porque a fonte vai para a trilha de auditoria e é o que o relatório mede como
    explainability: uma data ou um CID inventados pelo modelo entram no log com a mesma cara
    de uma citação legítima, e é a auditoria — que confere depois, sem o prompt em mãos — que
    fica sem como distinguir.

    A verificação é sobre os **identificadores** da citação (datas e códigos CID), não sobre a
    prosa: "consulta de 01/08/2026" confere se `01/08/2026` está no contexto. Termo genérico
    como "exames do paciente" não tem identificador e passa — não há o que conferir, e barrar
    a forma genérica só empurraria o modelo a citar menos.

    O que isto **não** faz, e importa não confundir: confere que a fonte *existe*, não que a
    afirmação saiu dela. Uma resposta que cita uma consulta real e atribui a ela um conteúdo
    que não estava lá continua passando. Verificar isso é atribuição de conteúdo à fonte, um
    problema de outra ordem — aqui se fecha só a citação fabricada.
    """
    if not fonte:
        return False
    identificadores = _IDENTIFICADOR_DE_FONTE.findall(fonte)
    if not identificadores:
        return True
    alvo = _normalizar(contexto or "")
    return all(_normalizar(identificador) in alvo for identificador in identificadores)


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

    def preload(self) -> None:
        """Adianta o carregamento do modelo, quando o LLM souber fazê-lo.

        Opcional de propósito: a chain funciona com qualquer `LLM` do LangChain, e a maioria
        (inclusive o falso da suíte de testes) não tem o que pré-carregar. Quem sabe, sabe —
        o porquê está no `MedicalMLXLLM.preload`.
        """
        carregar = getattr(self.llm, "preload", None)
        if callable(carregar):
            carregar()

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
        alergias_do_paciente: list[str] = []
        alergias_na_pergunta: list[str] = []
        if patient_id:
            dados = self.retriever.get_patient_context(patient_id)
            contexto = dados["contexto"]
            # A lista completa sobrevive ao `if` porque a resposta também é conferida contra
            # ela, adiante — só o reforço no contexto usa o lado da pergunta, porque ele é
            # montado antes da inferência.
            alergias_do_paciente = dados["alergias"]
            alergias_na_pergunta = alergias_citadas(pergunta, alergias_do_paciente)

        pediu_prescricao, aviso = check_prescription_attempt(pergunta)
        if pediu_prescricao:
            contexto = f"{contexto or SEM_CONTEXTO}\n\n{aviso}"

        # Reforço no contexto **e** carimbo na resposta, adiante. Os dois, e não um dos dois:
        # o reforço dá ao modelo a chance de responder a pergunta certa (é o que o aviso de
        # prescrição faz no passo acima), e o carimbo é o que garante o alerta quando ele não
        # aproveita a chance — que foi o comportamento medido.
        if alergias_na_pergunta:
            contexto = "{}\n\n{}".format(
                contexto or SEM_CONTEXTO,
                ALERTA_ALERGIA.format(alergias=", ".join(alergias_na_pergunta)),
            )

        historico = self._historico_da_sessao(session_id)

        # O contexto como o modelo de fato o recebe. Materializado numa variável porque é ele
        # que a conferência de fonte adiante usa: sem paciente selecionado o modelo lê o
        # `SEM_CONTEXTO`, e conferir a citação contra `None` seria conferir contra uma coisa
        # que não foi entregue a ninguém. Hoje as duas formas dão o mesmo veredito — o
        # `SEM_CONTEXTO` não contém data nem CID —, então isto não corrige comportamento: fixa
        # a referência certa antes que uma mudança naquele texto faça as duas divergirem.
        contexto_efetivo = contexto or SEM_CONTEXTO

        # Os marcadores de bloco são desarmados no dado que entra, aqui e no `build_prompt`:
        # a chain monta o prompt pelo `ChatPromptTemplate` e não passa pelo `build_prompt`,
        # então a proteção precisa existir nos dois caminhos ou vale só num deles.
        bruta = self.chain.invoke(
            {
                "system": SYSTEM_PROMPT,
                "patient_context": neutralizar_delimitadores(contexto_efetivo),
                "history": neutralizar_delimitadores(
                    self._formatar_historico(historico) or "(primeira pergunta desta sessão)"
                ),
                "question": neutralizar_delimitadores(pergunta),
            }
        )

        # O corte de repetição vem antes do guardrail: o rodapé de validação tem de ficar no
        # fim do texto que o médico lê, não enterrado no meio do trecho repetido.
        limpa = deduplicar_fontes(cortar_repeticao(bruta))

        # A pergunta guardada é a saneada, e a resposta é a já cortada, mas ainda sem o rodapé
        # do guardrail. As duas exclusões têm motivos diferentes:
        #
        # - O rodapé de validação e o aviso de prescrição são acrescentados a cada turno, e
        #   realimentá-los ensinaria o modelo a escrevê-los sozinho — o que faria a marca
        #   deixar de distinguir o que o guardrail garantiu do que o modelo inventou.
        # - O trecho repetido fica de fora porque `_formatar_historico` manda os primeiros
        #   `RESUMO_DA_RESPOSTA` caracteres de volta ao prompt: com o modelo em degeneração,
        #   guardar a resposta crua enchia esse recorte de uma frase repetida oito vezes e
        #   devolvia ao modelo exatamente o ancoramento que a tabela de similaridade do
        #   `create_chain` mede — e que o histórico-como-texto existe para evitar.
        historico.add_user_message(pergunta)
        historico.add_ai_message(limpa)

        resultado = apply_guardrails(pergunta, limpa)

        # O alerta é conferido nos **dois** lados, pergunta e resposta, e não só na pergunta:
        # checar só a pergunta faz o alerta sair em "o paciente pode receber dipirona?" e não
        # sair em "qual analgésico posso prescrever?" respondida com "sugiro dipirona 500mg" —
        # que é o caso perigoso, porque quem pergunta em aberto é justamente quem não tem o
        # alérgeno na cabeça. É o mesmo princípio que o `guardrails.py` já registra pelo lado
        # da posologia oferecida espontaneamente.
        #
        # O lado da resposta é conferido sobre `limpa` e não sobre `bruta`: alertar sobre um
        # fármaco que só aparece no trecho repetido, que o médico não chega a ver, é alarme sem
        # referente na tela — e alerta que dispara sozinho deixa de ser lido.
        alergias_alertadas = sorted(
            set(alergias_na_pergunta) | set(alergias_citadas(limpa, alergias_do_paciente))
        )

        # O carimbo é aplicado depois do guardrail e no topo do texto — ver `ALERTA_ALERGIA`.
        # Se o modelo já mencionou a alergia por conta própria, o carimbo continua vindo: o
        # médico precisa saber qual alerta veio do prontuário e qual veio da geração, e
        # suprimir o determinístico por semelhança devolveria a garantia ao modelo.
        resposta = resultado.resposta
        if alergias_alertadas:
            alerta = ALERTA_ALERGIA.format(alergias=", ".join(alergias_alertadas))
            resposta = f"{alerta}\n\n{resposta}"

        fonte = extrair_fonte(resposta)
        # Fonte citada que não corresponde ao contexto não vai para a trilha como se fosse
        # boa: ela é registrada como ausente, que é a conclusão correta para quem audita.
        if fonte and not fonte_confere(fonte, contexto_efetivo):
            warnings.warn(
                f"Fonte citada sem correspondência no contexto do paciente: {fonte!r}. "
                "Registrada na trilha como ausente.",
                stacklevel=2,
            )
            fonte = None

        # Só a pergunta e o recorte da resposta vão para a trilha, ambos anonimizados pelo
        # PR 06. O contexto do paciente fica de fora de propósito: ele já está no banco, e
        # copiá-lo para um arquivo que é aberto no notebook e gravado no vídeo de entrega
        # espalharia dado clínico sem responder nenhuma pergunta de auditoria a mais.
        #
        # A resposta gravada é a do guardrail, sem o carimbo de alergia. O recorte da trilha
        # tem 200 caracteres e o carimbo ocupa ~130 deles, sobrando um terço para o que o
        # modelo de fato respondeu — e gasto justamente com a parte determinística, que
        # `patient_id` mais o prontuário reconstroem inteira. As alergias alertadas vão em
        # campo próprio, que ainda deixa a trilha filtrável por "houve alerta".
        self.audit_logger.log(
            query=pergunta,
            response=resultado.resposta,
            patient_id=patient_id,
            source=fonte,
            guardrail_triggered=resultado.guardrail_triggered,
            session_id=session_id,
            # `fonte is not None`, e não o `resultado.tem_fonte` do PR 05: os dois medem
            # explainability, mas o do guardrail responde "citou alguma coisa?" e este
            # responde "citou algo que confere com o contexto?". Gravar o do guardrail deixaria
            # a trilha com `source: null` e `tem_fonte: true` na mesma linha — duas afirmações
            # contraditórias sobre a mesma resposta, e a auditoria sem como saber qual vale.
            tem_fonte=fonte is not None,
            motivos=resultado.motivos,
            alergias_alertadas=tuple(alergias_alertadas),
        )

        return {
            "response": resposta,
            "source": fonte,
            "guardrail_triggered": resultado.guardrail_triggered,
            "patient_context_used": bool(patient_id),
            "alergias_alertadas": alergias_alertadas,
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
    parser.add_argument("--paciente", help="identificador: 7, 007 ou [PACIENTE_007]")
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

    # `flush` porque o que vem depois demora: sem ele, a linha fica presa no buffer e a tela
    # passa a espera inteira em branco — exatamente o contrário do que a mensagem serve.
    print("Carregando modelo e adapters (pode levar alguns segundos)...", flush=True)
    assistente = MedicalAssistant.from_env()
    # Aqui, e não na primeira pergunta: a mensagem acima promete que o carregamento está
    # acontecendo agora. Sem isto ela é falsa — o prompt volta na hora e a espera cai dentro
    # da primeira pergunta, junto com o que o MLX imprime ao inicializar.
    assistente.preload()

    # Caminho relativo à raiz, como o resto do projeto faz no que é exibido: o absoluto
    # ocupa duas linhas largas na tela e cola o resultado à máquina de quem rodou.
    def _curto(caminho: Any) -> str:
        try:
            return str(Path(caminho).relative_to(RAIZ))
        except ValueError:
            return str(caminho)

    print(f"Banco:  {_curto(assistente.retriever.db_path)}")
    print(f"Trilha: {_curto(assistente.audit_logger.log_path)}")

    # O `--paciente` passa pela mesma normalização do modo interativo. Sem isto, `--paciente 7`
    # — a forma que o `--listar` e o passo 1 ensinam a usar — chegava cru ao retriever e batia
    # na allowlist, com uma mensagem pedindo o token completo. Duas formas de dizer o mesmo
    # paciente na mesma interface, uma delas recusada, é defeito da interface e não do usuário.
    paciente = args.paciente
    if paciente:
        paciente = _normalizar_escolha(paciente, assistente.retriever.listar_pacientes())
        if paciente is None:
            raise SystemExit(
                f"\n'{args.paciente}' não está no banco. Use --listar para ver os "
                "identificadores disponíveis."
            )

    if args.pergunta:
        try:
            _imprimir(assistente.ask(args.pergunta, patient_id=paciente))
        except (PacienteNaoEncontrado, ValueError) as erro:
            raise SystemExit(f"\n{erro}")
        return

    paciente = paciente or _escolher_paciente(assistente.retriever)
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
