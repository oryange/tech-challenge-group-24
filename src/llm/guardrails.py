"""Limites de atuação do assistente: o que entra no prompt e o que sai da LLM.

Uso como biblioteca:

    from src.llm.guardrails import apply_guardrails, sanitize_input

Cobre dois requisitos do enunciado, que são coisas diferentes e ficam em funções separadas:

* **"nunca prescrever diretamente, sem validação humana"** — `check_prescription_attempt`
  detecta intenção de prescrição e `validate_response` garante o rodapé de validação.
* **explainability** — `apply_guardrails` reporta se a resposta cita fonte, sem nunca
  inventar uma (ver `validate_response`).

Sobre o alcance real de `sanitize_input`: uma lista de padrões conhecidos é *denylist*, e
denylist de prompt injection é contornável por construção — tradução, encoding, sinônimo.
Ela está aqui como defesa em profundidade, não como controle principal. O que de fato
segura o sistema é estrutural e mora fora deste módulo:

1. a pergunta do médico entra no prompt dentro de um bloco delimitado e declarado como
   dado, não como instrução (`src/assistant/prompts.py`, PR 07);
2. o assistente não tem autoridade para prescrever em lugar nenhum do fluxo — mesmo uma
   injeção bem-sucedida devolve texto que sai daqui com o rodapé de validação humana.

Tratar a denylist como se fosse a proteção principal seria o erro perigoso: ela some com o
caso óbvio e dá a impressão de que o problema está resolvido.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Teto do que chega ao modelo. Não é economia de token: é o limite superior do trabalho que
# uma entrada hostil consegue impor às regex abaixo e ao contexto do prompt.
LIMITE_CARACTERES = 2000

MARCADOR_INSTRUCAO_REMOVIDA = "[instrução removida]"
RODAPE_VALIDACAO = "[Requer validação médica por profissional habilitado]"

# Prefixos, não strings inteiras: o rodapé completo é uma frase, e o que identifica a marca
# é o começo dela. Comparar com o texto exato faria uma resposta que já traz
# "[Requer validação médica antes da conduta]" receber um segundo rodapé.
MARCA_VALIDACAO = "[Requer validação médica"
MARCA_FONTE = "[Fonte:"

# A marca precisa *fechar* a resposta, não apenas aparecer nela. O modelo foi fine-tuned com
# textos que carregam essa frase, então ele pode citá-la no meio do que responde — e uma
# resposta como "o laudo trazia [Requer validação médica] na época" tem a marca sem ter o
# rodapé. Casar até o colchete que fecha, no fim do texto, mantém o motivo do prefixo de pé:
# a variante "[Requer validação médica antes da conduta]" continua valendo por si.
# `[^\]]*` não aninha quantificador — o casamento segue linear no tamanho da resposta.
_MARCA_VALIDACAO_NO_FIM = re.compile(rf"{re.escape(MARCA_VALIDACAO)}[^\]]*\]\s*\Z")

AVISO_PRESCRICAO = (
    "[Este assistente não emite prescrição. O conteúdo abaixo é apoio à decisão clínica e "
    "precisa de validação por profissional habilitado antes de qualquer conduta.]"
)

# Todos os quantificadores são simples e não aninhados (`\s+`, `\w*`), então o casamento é
# linear no tamanho da entrada. Um padrão do tipo `(\s+\w+)+` daria backtracking exponencial
# e transformaria a própria sanitização num vetor de negação de serviço — com entrada já
# truncada em 2000 caracteres, mas mesmo assim.
_PADROES_INJECAO: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(?:all\s+)?(?:previous|prior|above)\b", re.IGNORECASE),
    re.compile(r"\bignor[ae]\s+(?:todas\s+)?(?:as\s+)?instru[çc][õo]es\b", re.IGNORECASE),
    re.compile(r"\besque[çc]a\s+(?:todas\s+)?(?:as\s+)?instru[çc][õo]es\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"\bagora\s+voc[êe]\s+[ée]\b", re.IGNORECASE),
    re.compile(r"\bact\s+as\s+(?:if|though)\b", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"\bsystem\s+prompt\b", re.IGNORECASE),
    re.compile(r"\bprompt\s+do\s+sistema\b", re.IGNORECASE),
)

# Radicais, não palavras inteiras: "prescreva", "prescrever" e "prescrição" são a mesma
# intenção conjugada de formas diferentes, e listar cada flexão deixaria buraco na primeira
# que alguém escrevesse fora da lista.
_PADROES_PRESCRICAO: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("prescrição", re.compile(r"\bprescr(?:ev|iç|ic|it)\w*", re.IGNORECASE)),
    ("receita", re.compile(r"\breceit(?:a|ar|e|ei|em|ando)\w*", re.IGNORECASE)),
    ("posologia", re.compile(r"\bdose\s+de\b", re.IGNORECASE)),
    ("administração", re.compile(r"\badministr(?:a|e|ar|ando)\w*", re.IGNORECASE)),
)


@dataclass(frozen=True)
class ResultadoGuardrails:
    """Saída de `apply_guardrails`, já pronta para o retorno e o log do PR 07.

    `guardrail_triggered` e `tem_fonte` são separados de propósito. O primeiro responde
    "o limite de atuação precisou agir?" e é o que o audit log registra como evento; o
    segundo é a métrica de explainability, e uma resposta sem fonte não é um guardrail
    acionado — é uma resposta de qualidade pior.
    """

    resposta: str
    guardrail_triggered: bool
    tem_fonte: bool
    motivos: tuple[str, ...]


def sanitize_input(texto: str | None) -> str:
    """Trunca a entrada e neutraliza tentativas conhecidas de prompt injection.

    O corte acontece **duas vezes, e as duas são necessárias**. Antes das regex, porque uma
    entrada de dezenas de MB passaria inteira pelo motor de regex antes de ser cortada.
    Depois delas, porque o marcador é maior que os padrões que substitui: `"jailbreak"`
    (9 caracteres) vira `"[instrução removida]"` (20), e uma entrada hostil de 2000
    caracteres sai com mais de 4000 se o teto não for reaplicado. Sem o segundo corte a
    função também deixa de ser idempotente, já que a segunda passagem encurtaria o que a
    primeira devolveu.

    A neutralização substitui por um marcador em vez de apagar. Apagar reconstrói o ataque:
    `re.sub` percorre a entrada uma única vez e não reexamina o que já escreveu, então
    remover a ocorrência interna de `"ignignore previousore previous"` cola as pontas e
    devolve `"ignore previous"` — exatamente a instrução que se queria eliminar. Trocando
    por um marcador, as pontas não se encontram. O corte final não tem esse risco: truncar
    só descarta sufixo, nunca junta duas pontas.
    """
    if not texto:
        return ""
    limpo = texto[:LIMITE_CARACTERES]
    for padrao in _PADROES_INJECAO:
        limpo = padrao.sub(MARCADOR_INSTRUCAO_REMOVIDA, limpo)
    return limpo[:LIMITE_CARACTERES]


def check_prescription_attempt(texto: str | None) -> tuple[bool, str]:
    """Detecta intenção de prescrição. Devolve `(detectado, aviso)`.

    Vale para os dois lados da conversa: a pergunta do médico pode pedir uma prescrição, e o
    modelo pode oferecer posologia sem que ninguém tenha pedido.

    Os radicais são propositalmente amplos, e o custo de um falso positivo é baixo porque a
    ação do guardrail é **marcar**, não recusar: um protocolo que legitimamente diz
    "administrar 500 mg" continua sendo respondido, só que precedido do aviso. O erro caro
    seria o inverso — deixar passar posologia sem a marca de validação humana.
    """
    if not texto:
        return False, ""
    encontrados = [rotulo for rotulo, padrao in _PADROES_PRESCRICAO if padrao.search(texto)]
    if not encontrados:
        return False, ""
    return True, AVISO_PRESCRICAO


def validate_response(resposta: str | None) -> str:
    """Garante que a resposta carregue o rodapé de validação humana.

    O rodapé é o requisito central do enunciado ("nunca prescrever sem validação humana"), e
    por isso a marca só conta quando **fecha** a resposta: uma citação da frase no meio do
    texto deixaria a resposta sair sem nada no fim (ver `_MARCA_VALIDACAO_NO_FIM`).

    Não mexe na fonte de propósito. Se a resposta não cita `[Fonte: ...]`, o que falta é
    rastreabilidade, e acrescentar um `[Fonte:]` genérico aqui produziria uma citação que o
    modelo não fez — quebrando justamente a explainability que o campo existe para dar.
    A ausência é reportada em `ResultadoGuardrails.tem_fonte`, não remendada.
    """
    texto = (resposta or "").strip()
    if _MARCA_VALIDACAO_NO_FIM.search(texto):
        return texto
    return f"{texto}\n{RODAPE_VALIDACAO}".strip()


def apply_guardrails(query: str | None, response: str | None) -> ResultadoGuardrails:
    """Aplica todas as checagens de saída e devolve a resposta segura.

    Cobre os dois lados porque são vetores independentes: `query` diz se **pediram** uma
    prescrição, `response` diz se o modelo **entregou** uma. Checar só a pergunta deixaria
    passar a posologia oferecida espontaneamente, que é o caso mais perigoso justamente por
    não ter sido pedida.

    A sanitização de entrada não é chamada aqui: ela age antes, sobre o que vai *para* o
    prompt, enquanto esta função age sobre o que já voltou dele. Juntá-las obrigaria o
    chamador a passar a pergunta crua e a sanitizada, ou a sanitizar duas vezes.
    """
    na_pergunta, _ = check_prescription_attempt(query)
    na_resposta, _ = check_prescription_attempt(response)

    motivos: list[str] = []
    if na_pergunta:
        motivos.append("prescricao_na_pergunta")
    if na_resposta:
        motivos.append("prescricao_na_resposta")

    segura = validate_response(response)
    tem_fonte = MARCA_FONTE in segura
    if not tem_fonte:
        motivos.append("sem_fonte")

    if na_pergunta or na_resposta:
        segura = f"{AVISO_PRESCRICAO}\n\n{segura}"

    return ResultadoGuardrails(
        resposta=segura,
        guardrail_triggered=na_pergunta or na_resposta,
        tem_fonte=tem_fonte,
        motivos=tuple(motivos),
    )
