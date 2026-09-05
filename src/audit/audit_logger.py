"""Log estruturado de auditoria das interações com o assistente.

Uso como biblioteca:

    from src.audit.audit_logger import get_audit_logger

    get_audit_logger().log(
        query="Quais exames estão pendentes?",
        response="Hemograma e glicemia. [Fonte: exames do paciente]",
        patient_id="[PACIENTE_007]",
        source="exames do paciente",
        guardrail_triggered=False,
        session_id="sessao-01",
    )

Atende ao "implementar logging detalhado para rastreamento e auditoria" do enunciado. É
JSONL — uma linha por interação — porque o arquivo é lido de três formas diferentes ao longo
do projeto: filtrado por sessão, filtrado por paciente, e exibido cru na demonstração. JSON
único exigiria reescrever o arquivo inteiro a cada escrita; texto livre não sobreviveria ao
filtro.

O módulo se chama `audit` e não `logging` para não sombrear o `logging` da stdlib, como
registrado desde o PR 01.

Sobre PII: a pergunta do médico é texto livre digitado na hora e pode conter nome real de
paciente, telefone ou prontuário — dados que nenhum dos pipelines anteriores viu, porque
eles anonimizam dataset e banco, não a conversa. Este arquivo é o único artefato que
persiste esse texto em disco, e ainda por cima é exibido no notebook de demonstração e no
vídeo de entrega. Por isso tudo o que é texto livre passa por `_anonimizar_conversa` antes
de ser gravado: o `anonymize` do PR 02, mais as regras de `_PII_CONVERSA`.

O que a anonimização daqui deliberadamente **não** garante: **nome não ancorado passa**. As
regras do PR 02 são ancoradas em contexto ("paciente" + nome, "Dr." + nome) porque afrouxar
isso destruiria termos legítimos do PubMedQA no dataset de treino, e a pergunta digitada na
hora raramente traz a âncora — "Maria Silva está com febre" sai inteira no log. Detectar
nome próprio solto por regex tem falso positivo caro em texto clínico (nome de medicamento,
de escala, de sinal clínico), então não é resolvido aqui.

A anonimização deste módulo é, portanto, **best-effort**: cobre os formatos inequívocos
(datas, e-mail, CPF, telefone, prontuário ancorado) e não cobre nome livre. Registrar isso
importa pelo mesmo motivo que o `guardrails.py` do PR 05 registra o alcance da denylist —
uma garantia afirmada e não cumprida é pior que um limite declarado, porque some com a
vigilância de quem lê. Enquanto o limite valer, `logs/audit.jsonl` é dado sensível: não sai
do repositório, e o que for exibido no notebook ou no vídeo precisa ser conferido antes.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

from src.data.anonymizer import TOKEN_DATA, TOKEN_PACIENTE_ID, TOKEN_TELEFONE, anonymize

RAIZ = Path(__file__).resolve().parents[2]
CAMINHO_PADRAO = RAIZ / "logs" / "audit.jsonl"

# Recorte da resposta guardado no log. O log é trilha de auditoria, não cache de respostas:
# o que importa é reconstruir o que foi perguntado, se o guardrail agiu e de que fonte a
# resposta saiu. Guardar a resposta inteira multiplicaria o tamanho do arquivo sem responder
# nenhuma pergunta de auditoria que o recorte já não responda.
PREVIEW_CARACTERES = 200

# Teto do texto livre que chega ao `anonymize`, aplicado **antes** dele. Não é o recorte de
# auditoria (esse é o `PREVIEW_CARACTERES`, e vem depois): é o limite do trabalho que uma
# entrada gigante impõe às regex do PR 02, algumas das quais não são lineares. Pelo fluxo do
# assistente o texto já chega truncado em 2000 pelo `sanitize_input` do PR 05; aqui o mesmo
# número é repetido porque `log()` é API pública — o docstring deste módulo mostra a chamada
# direta — e um controle que depende de todo chamador ter passado por outro módulo não é
# controle. Folgado o bastante para não interferir nas âncoras do anonimizador, que casam
# dentro de poucas dezenas de caracteres.
LIMITE_TEXTO_LIVRE = 2000

# Quanto o teto pode recuar para não partir a última palavra. O corte de `LIMITE_TEXTO_LIVRE`
# vem antes da anonimização, então um dado pessoal a cavaleiro dele perde a cauda, deixa de
# casar com a regra correspondente e o pedaço da esquerda seria gravado em claro — mesma
# armadilha que o `PREVIEW_CARACTERES` evita anonimizando antes de recortar, e que aqui não
# dá para evitar pela ordem, porque o teto existe justamente para limitar o que chega às
# regex. A saída é não cortar no meio de um token. A margem cobre com folga a maior forma que
# as regras conhecem (CPF pontuado tem 14 caracteres) sem transformar o teto em sugestão:
# além dela o corte é seco, e uma sequência de 64 caracteres sem espaço não é PII conhecida.
MARGEM_TOKEN_PARTIDO = 64

# Permissões do que é criado aqui. O arquivo guarda pergunta de médico em texto livre e um
# recorte de resposta clínica; o padrão do umask (0644) o deixa legível por qualquer conta da
# máquina, sem que nada no fluxo denuncie isso.
MODO_DIRETORIO = 0o700
MODO_ARQUIVO = 0o600

# Regras próprias da conversa, complementares ao `anonymize` do PR 02 — que fica intocado de
# propósito: as âncoras de contexto existem lá para não destruir termos do PubMedQA, e
# afrouxá-las degradaria o dataset de treino. Aqui o dado tem outra forma (alguém digitando
# no chat) e outro destino (disco), então os formatos que dispensam âncora entram neste
# módulo, que é quem conhece essa diferença.
#
# Onze dígitos soltos são ambíguos entre celular e CPF, e nenhuma regex desfaz isso sem
# contexto. A primeira regra pega o formato do celular (DDD + 9 + oito dígitos); o que
# sobrar de 10 ou 11 dígitos cai no token de identificador. Os dois são PII e o que importa
# é que nenhum chegue ao disco — errar o rótulo entre eles é ruído, não vazamento.
_PII_CONVERSA: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<!\d)\d{2}9\d{8}(?!\d)"), TOKEN_TELEFONE),
    (re.compile(r"(?<!\d)\d{10,11}(?!\d)"), TOKEN_PACIENTE_ID),
)


def _limitar_texto_livre(texto: str) -> str:
    """Aplica `LIMITE_TEXTO_LIVRE` sem partir a última palavra.

    Recuar até o espaço anterior é o que impede que um CPF ou telefone a cavaleiro do teto
    chegue pela metade ao anonimizador — ver `MARGEM_TOKEN_PARTIDO`.
    """
    if len(texto) <= LIMITE_TEXTO_LIVRE:
        return texto
    cortado = texto[:LIMITE_TEXTO_LIVRE]
    if texto[LIMITE_TEXTO_LIVRE].isspace():
        return cortado
    cauda = re.search(r"\S+$", cortado)
    if cauda is not None and len(cauda.group()) <= MARGEM_TOKEN_PARTIDO:
        return cortado[: cauda.start()]
    return cortado


def _anonimizar_conversa(texto: str) -> str:
    """Anonimiza texto livre da conversa: as regras do PR 02 mais as de `_PII_CONVERSA`.

    Não cobre nome não ancorado — ver o limite declarado no topo do módulo.
    """
    resultado = anonymize(texto)
    for regex, token in _PII_CONVERSA:
        resultado = regex.sub(token, resultado)
    return resultado


def _e_diretorio_do_projeto(diretorio: Path) -> bool:
    """Diz se um diretório que já existia é do projeto — o único que pode ser apertado.

    "Nosso" precisa ser **dentro da `RAIZ` e não a própria `RAIZ`**, e não "do nosso uid e não
    gravável por todos", que foi a primeira tentativa. Essa heurística descrevia bem o `logs/`,
    mas descrevia igualmente bem a `$HOME` e a raiz do repositório — que também são nossas e
    também não são graváveis por todos, e são justamente os dois diretórios em que a nossa vida
    inteira mora. Medido: `AUDIT_LOG_PATH=~/audit.jsonl` apertava a home e
    `AUDIT_LOG_PATH=audit.jsonl` apertava a raiz do repositório, sem aviso, só por instanciar o
    assistente.

    O que estava faltando à exclusão não era "de outro dono", era "nosso e compartilhado com o
    resto da nossa vida". Um allowlist do que o projeto possui cobre as duas coisas de uma vez:
    o `logs/` continua sendo apertado, e nenhum diretório de propósito geral entra. Um denylist
    de `{home, raiz}` teria movido o defeito para `~/Documents` em vez de fechá-lo.

    A checagem de propriedade continua, por cima: dentro da `RAIZ` um diretório gravável por
    todos ou de outro `st_uid` é anomalia, e apertá-lo mexeria em algo que não montamos.
    """
    try:
        alvo = diretorio.resolve()
        info = diretorio.stat()
    except OSError:
        return False
    if alvo == RAIZ or not alvo.is_relative_to(RAIZ):
        return False
    compartilhado = bool(info.st_mode & 0o002)
    de_outro_dono = hasattr(os, "getuid") and info.st_uid != os.getuid()
    return not (compartilhado or de_outro_dono)


def _apertar_trilha(caminho: Path) -> None:
    """Tira a permissão de leitura alheia da trilha, inclusive de um arquivo preexistente.

    O `touch(mode=MODO_ARQUIVO)` só fixa o modo no instante da criação, e o caso comum do
    projeto é justamente o arquivo que já existe: quem rodou qualquer versão anterior a este
    controle tem um `logs/audit.jsonl` criado com o umask (0644), legível por qualquer conta
    da máquina, e nenhuma escrita posterior o promove. Medido neste repositório: o diretório
    estava em 0700 e o arquivo dentro dele, com 30 KB de trilha, em 0644 — o controle existia
    e nunca tinha valido para o arquivo que de fato existe.

    O aperto é **só quando há bit de grupo ou de outros** e é para `MODO_ARQUIVO`, não uma
    reaplicação cega: um arquivo já em 0600, ou em 0400 porque alguém o congelou de propósito,
    não é tocado. É a mesma ideia do `_apertar_diretorios` — corrigir o que ficou frouxo sem
    desfazer decisão de quem mexeu — e a mesma tolerância a falha, pela mesma razão: a trilha
    escrita vale mais que a trilha endurecida, e derrubar a consulta clínica porque o `chmod`
    não passou troca uma perda certa por uma incerta.
    """
    try:
        if caminho.stat().st_mode & 0o077:
            caminho.chmod(MODO_ARQUIVO)
    except OSError as erro:
        warnings.warn(
            f"Não foi possível restringir {caminho} a {MODO_ARQUIVO:o}: {erro}. "
            "A trilha continua sendo escrita, e pode estar legível por outras contas.",
            stacklevel=3,
        )


def _apertar_diretorios(folha: Path, criados: list[Path]) -> None:
    """Aplica `MODO_DIRETORIO` no diretório da trilha e nos que acabaram de ser criados.

    O `mode=` do `mkdir` sozinho não basta, e as duas limitações são exatamente as duas em que
    o caminho padrão do projeto cai: `exist_ok=True` devolve sem tocar na permissão de um
    diretório que já existia — e `logs/` é versionado (`logs/.gitkeep`), logo existe desde o
    clone com 0755 —, e `parents=True` cria os intermediários com o umask, deixando só o último
    em 0700. Sem este passo, `MODO_DIRETORIO` não valia justamente onde a trilha é escrita.

    **O que fica de fora, e é o ponto delicado.** Apertar um diretório que não é nosso é pior
    que o problema que este passo resolve: com `AUDIT_LOG_PATH=/tmp/audit.jsonl` a folha é o
    `/tmp`, e um `chmod 0700` ali derruba a máquina inteira se tiver privilégio para acontecer
    — e explode na construção se não tiver, quebrando uma configuração que antes funcionava.
    Então um diretório que já existia só é apertado quando é do projeto — ver
    `_e_diretorio_do_projeto`, que é onde mora o critério e o porquê dele. Os que este
    construtor criou não passam pela checagem, porque sobre eles não há dúvida de propriedade:
    não existiam um instante atrás.

    Falha de `chmod` avisa em vez de derrubar. A permissão do diretório é defesa em
    profundidade — o `touch(mode=MODO_ARQUIVO)` já protege o conteúdo —, e recusar a escrever
    a trilha porque não deu para endurecer a pasta troca uma perda certa (auditoria nenhuma)
    por uma incerta.
    """
    for diretorio in dict.fromkeys([*criados, folha]):
        if diretorio not in criados and not _e_diretorio_do_projeto(diretorio):
            continue
        try:
            diretorio.chmod(MODO_DIRETORIO)
        except OSError as erro:
            warnings.warn(
                f"Não foi possível restringir {diretorio} a {MODO_DIRETORIO:o}: {erro}. "
                "A trilha continua sendo criada em modo 0600.",
                stacklevel=3,
            )


# As duas formas numéricas de data. A extensa ("12 de março de 2024") fica de fora de
# propósito: o modelo cita a data no formato em que ela está no prontuário, e se a extensa
# aparecer o resultado é ela continuar redigida — o lado seguro de errar aqui.
_DATA_RASTREAVEL = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")

# Marca cada data numérica **antes** da anonimização, para que a restauração seja por posição
# e não por contagem de `[DATA]`. A contagem tinha uma coincidência que vazava PII:
#
#     in : consulta de 05 de maio de 2020, prontuario 12/03/2026
#     out: consulta de 12/03/2026, prontuario [PACIENTE_ID]     <- errado, e em claro
#
# A data extensa é consumida pela regra de data e vira `[DATA]`; a numérica é consumida pela
# regra de **prontuário**, que roda antes das de data, e vira `[PACIENTE_ID]`. Aí
# `count(TOKEN_DATA) == len(datas) == 1`, a guarda deixava passar, e o `replace` punha o número
# de prontuário — que a anonimização acabara de redigir — em claro na posição da outra data. Um
# identificador coberto reaparecia no disco, que é o oposto do que este módulo existe para
# fazer, e a trilha ainda passava a afirmar a data errada.
#
# Marcar antes resolve os dois lados de uma vez: a sentinela sobrevive à anonimização quando a
# data era livre (nenhuma regra a reconhece: são 12 dígitos seguidos, sem separador de data, e
# os lookarounds de `_PII_CONVERSA` exigem exatamente 10 ou 11), e **desaparece** quando a data
# servia de identificador ancorado, porque a regra da âncora a come junto com a pista. Sentinela
# que sumiu é a colisão se denunciando, e aí nada é restaurado.
_SENTINELA_DE_DATA = "400289220{:03d}"


def anonimizar_fonte(texto: str) -> str:
    """Anonimiza a citação de fonte, mas devolve as datas ao lugar.

    É pública, e não `_privada` como as outras deste bloco, porque o `chain.py` precisa dela
    **fora** do `log()`: quando a fonte citada não confere com o contexto, ela não é gravada
    (vira `None` na trilha) mas é mostrada num `warnings.warn` — e a fonte é texto livre
    gerado pelo modelo. Sem passar por aqui, o único registro que sobrava daquele texto era o
    do `stderr`, em claro, que é a tela do notebook de demonstração e do vídeo de entrega. A
    garantia deste módulo é sobre o que persiste em disco; o aviso caía fora dela por não ser
    escrita, e o efeito prático era o mesmo.

    O `source` é texto livre gerado pelo modelo e precisa da mesma anonimização dos outros
    campos — o modelo cita `[Fonte: consulta do paciente Joao Souza de 12/03/2026]`, que é
    justamente a forma **ancorada** que as regras do PR 02 sabem pegar. Sem isso o mesmo trecho
    saía anonimizado em `response_preview` e em claro em `source`, na mesma linha do arquivo.

    A data, porém, volta. O `anonymize` redige data junto com nome, e aplicá-lo inteiro trocava
    um vazamento por uma perda: `"consulta de [DATA]"` não diz de qual consulta a resposta
    saiu, que é a única pergunta que este campo existe para responder — e é o identificador que
    o `fonte_confere` acabou de validar contra o contexto. Medido no assistente completo, a
    trilha real saía com `source: "consulta de [DATA]"` e `tem_fonte: true`: a conferência
    acontecia e o resultado dela era jogado fora.

    Preservar a data não afrouxa nada, e é o mesmo raciocínio que este módulo já aplica ao
    `patient_id`: ele vai em claro na mesma linha, então a data não acrescenta poder de
    reidentificação nenhum a quem já tem o token do paciente e o banco. O que a anonimização
    precisa cobrir aqui é o nome, e esse continua coberto.

    A restauração é **posicional e falha fechado**, e as duas coisas dependem de a data ser
    marcada antes da anonimização, não reconhecida depois — ver `_SENTINELA_DE_DATA`.
    """
    datas = [casamento.group() for casamento in _DATA_RASTREAVEL.finditer(texto)]
    if not datas:
        return _anonimizar_conversa(texto)

    contador = itertools.count()
    marcado = _DATA_RASTREAVEL.sub(
        lambda _: _SENTINELA_DE_DATA.format(next(contador)), texto
    )
    anonimizado = _anonimizar_conversa(marcado)

    sentinelas = [_SENTINELA_DE_DATA.format(indice) for indice in range(len(datas))]
    if any(anonimizado.count(sentinela) != 1 for sentinela in sentinelas):
        # Alguma data foi consumida por uma regra que não é de data. Nada é restaurado, e o
        # texto volta a ser anonimizado a partir do original — o `marcado` não serve como saída,
        # porque devolveria a sentinela ao arquivo.
        return _anonimizar_conversa(texto)

    for sentinela, data in zip(sentinelas, datas):
        anonimizado = anonimizado.replace(sentinela, data)
    return anonimizado


def _caminho_do_ambiente(variavel: str, padrao: Path) -> Path:
    """Lê um caminho do `.env`, ancorando o relativo na raiz do repositório.

    Mesma semântica de `config._do_ambiente` (expanduser + âncora na raiz + resolve); o
    porquê de cada normalização está documentado lá.
    """
    valor = os.getenv(variavel)
    if not valor:
        return padrao
    caminho = Path(valor).expanduser()
    if not caminho.is_absolute():
        caminho = RAIZ / caminho
    return caminho.resolve()


class AuditLogger:
    """Escreve e consulta a trilha de auditoria em JSONL."""

    def __init__(self, log_path: str | os.PathLike[str] = CAMINHO_PADRAO) -> None:
        self.log_path = Path(log_path)
        # O diretório é criado aqui, e não na primeira escrita, para que um AUDIT_LOG_PATH
        # apontando para diretório inexistente falhe na construção — e não no meio de uma
        # consulta clínica, que é quando a primeira escrita acontece.
        pai = self.log_path.parent
        # Quem não existe agora é quem o `mkdir` abaixo vai criar. Precisa ser medido antes.
        criados = [d for d in (pai, *pai.parents) if not d.exists()]
        pai.mkdir(parents=True, exist_ok=True, mode=MODO_DIRETORIO)
        _apertar_diretorios(pai, criados)

    @classmethod
    def from_env(cls) -> "AuditLogger":
        """Constrói a partir de `AUDIT_LOG_PATH` no `.env`."""
        return cls(_caminho_do_ambiente("AUDIT_LOG_PATH", CAMINHO_PADRAO))

    def log(
        self,
        query: str,
        response: str,
        patient_id: str | None = None,
        source: str | None = None,
        guardrail_triggered: bool = False,
        session_id: str | None = None,
        tem_fonte: bool | None = None,
        motivos: tuple[str, ...] = (),
        alergias_alertadas: tuple[str, ...] = (),
    ) -> dict:
        """Registra uma interação e devolve a entrada gravada.

        A anonimização vem **antes** do recorte de 200 caracteres, nesta ordem de propósito.
        Recortar primeiro parte o texto num ponto arbitrário e pode cair no meio de um dado
        pessoal: as regras do `anonymizer` são ancoradas em contexto ("paciente" + nome,
        "prontuário" + número), então um corte entre a âncora e o dado faz a regra deixar de
        casar e o fragmento restante ser gravado em claro.

        `patient_id` não passa pela anonimização porque já é um token — `[PACIENTE_007]`,
        gerado pelo seed do PR 03. Anonimizá-lo destruiria a chave de filtro do
        `get_patient_logs` sem proteger dado nenhum.

        O teto de `LIMITE_TEXTO_LIVRE` é o único corte que vem **antes** da anonimização, e
        por isso está sujeito à mesma armadilha do parágrafo acima. Ser ordens de grandeza
        maior que o alcance das âncoras resolve o caso delas, mas não o do dado que cai
        exatamente em cima do corte: `_limitar_texto_livre` recua até o espaço anterior para
        que nenhum token chegue partido ao anonimizador.

        `source` também é texto livre e também é anonimizado, com a data preservada — o porquê
        de cada metade está em `_anonimizar_fonte`. O `or None` no fim mantém a distinção entre
        "não citou fonte" e "citou uma fonte vazia", que o `extrair_fonte` do PR 07 preserva.

        `tem_fonte` e `motivos` recebem o resto do `ResultadoGuardrails` do PR 05. Eles têm
        default e a assinatura do checklist continua valendo, mas sem eles a métrica de
        explainability — que é requisito do enunciado — seria calculada no PR 05 e não
        chegaria a nenhum lugar persistido. `tem_fonte=None` distingue "não foi medido" de
        "não tinha fonte", que numa auditoria são conclusões diferentes.

        `alergias_alertadas` é campo próprio, e não parte da resposta gravada, pelo mesmo
        motivo: o carimbo determinístico do PR 07 ocupa dois terços do recorte de auditoria com
        um texto reconstruível a partir de `patient_id` mais o prontuário. Em campo separado
        ele não come o recorte e ainda deixa a trilha filtrável por "houve alerta".

        O mesmo vale para as marcas do guardrail, e é o chamador quem decide: o `ask` do PR 07
        passa em `response` o texto do modelo **antes** do carimbo de alergia, do rodapé de
        validação e do `AVISO_PRESCRICAO`. Os três são determinísticos e este `log()` já recebe
        o que os reconstrói (`guardrail_triggered`, `motivos`, `alergias_alertadas`); só o
        `AVISO_PRESCRICAO` tem 161 dos 200 caracteres do recorte. Quem chamar direto e passar a
        resposta já marcada não erra nada — perde recorte.
        """
        entrada = {
            # Milissegundos, não segundos: duas interações da mesma sessão cabem no mesmo
            # segundo, e aí a trilha perde a ordem entre elas para quem lê o arquivo fora da
            # ordem de escrita — que é o caso de qualquer análise que ordene por timestamp.
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "session_id": session_id,
            "patient_id": patient_id,
            "query": _anonimizar_conversa(_limitar_texto_livre(query or "")),
            "response_preview": _anonimizar_conversa(
                _limitar_texto_livre(response or "")
            )[:PREVIEW_CARACTERES],
            "source": anonimizar_fonte(_limitar_texto_livre(source or "")) or None,
            "guardrail_triggered": guardrail_triggered,
            "tem_fonte": tem_fonte,
            "motivos": list(motivos),
            "alergias_alertadas": list(alergias_alertadas),
        }
        # `ensure_ascii=False` mantém o português legível no arquivo: com o padrão, "asmática"
        # vira "asmática" e a trilha fica ilegível justamente na hora de exibi-la.
        linha = json.dumps(entrada, ensure_ascii=False)
        # `touch` antes do append: o modo só vale no instante da criação.
        self.log_path.touch(mode=MODO_ARQUIVO, exist_ok=True)
        _apertar_trilha(self.log_path)
        with self.log_path.open("a", encoding="utf-8") as arquivo:
            arquivo.write(f"{linha}\n")
        return entrada

    def _ler(self) -> list[dict]:
        """Lê todas as entradas. Arquivo ausente é trilha vazia, não erro.

        Linhas que não decodificam são puladas em vez de derrubar a leitura. O caso real é o
        processo morrer no meio de uma escrita e deixar a última linha pela metade — e uma
        trilha de auditoria que se recusa a abrir por causa disso perde as entradas íntegras
        junto com a quebrada, que é o pior dos dois resultados.

        Puladas, mas não em silêncio. Descartar sem rastro faz "não consigo ler uma linha" e
        "não consigo ler nenhuma" terminarem no mesmo resultado mudo: com o arquivo inteiro
        corrompido, a consulta devolve `[]` e a leitura conclui "não houve interação" em vez
        de "a trilha está ilegível" — a conclusão oposta, e a mais perigosa numa auditoria.
        """
        if not self.log_path.exists():
            return []
        entradas = []
        descartadas = 0
        with self.log_path.open(encoding="utf-8") as arquivo:
            for linha in arquivo:
                linha = linha.strip()
                if not linha:
                    continue
                try:
                    entradas.append(json.loads(linha))
                except json.JSONDecodeError:
                    descartadas += 1
        if descartadas:
            # 3 = quem chamou `get_session_logs`/`get_patient_logs`. Com o default o aviso
            # apontaria para dentro deste arquivo, que não é onde alguém pode agir.
            warnings.warn(
                f"{descartadas} linha(s) ilegível(is) descartada(s) em {self.log_path}",
                stacklevel=3,
            )
        return entradas

    def get_session_logs(self, session_id: str) -> list[dict]:
        """Todas as interações de uma sessão, na ordem em que aconteceram."""
        return [e for e in self._ler() if e.get("session_id") == session_id]

    def get_patient_logs(self, patient_id: str) -> list[dict]:
        """Histórico de consultas ao assistente sobre um paciente, de todas as sessões."""
        return [e for e in self._ler() if e.get("patient_id") == patient_id]


_PADRAO: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Instância compartilhada, para quem não quer repetir a leitura do `.env`.

    Construída na primeira chamada e não no import, de propósito. Uma instância de módulo
    lê o ambiente no instante em que o módulo é importado — que num programa de linha de
    comando é **antes** do `load_dotenv`, porque os imports vêm antes de qualquer execução.
    O `AUDIT_LOG_PATH` do `.env` seria ignorado sem nada denunciar, e a trilha apareceria no
    caminho padrão enquanto o resto do sistema apontaria para o configurado. Adiando a
    construção, quem chama já está com o ambiente carregado.

    O mesmo vale para o `mkdir` do construtor, que deixa de ser efeito colateral de import.
    """
    global _PADRAO
    if _PADRAO is None:
        _PADRAO = AuditLogger.from_env()
    return _PADRAO
