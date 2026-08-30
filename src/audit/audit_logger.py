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

import json
import os
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

from src.data.anonymizer import TOKEN_PACIENTE_ID, TOKEN_TELEFONE, anonymize

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
        self.log_path.parent.mkdir(parents=True, exist_ok=True, mode=MODO_DIRETORIO)

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

        `tem_fonte` e `motivos` recebem o resto do `ResultadoGuardrails` do PR 05. Eles têm
        default e a assinatura do checklist continua valendo, mas sem eles a métrica de
        explainability — que é requisito do enunciado — seria calculada no PR 05 e não
        chegaria a nenhum lugar persistido. `tem_fonte=None` distingue "não foi medido" de
        "não tinha fonte", que numa auditoria são conclusões diferentes.
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
            "source": source,
            "guardrail_triggered": guardrail_triggered,
            "tem_fonte": tem_fonte,
            "motivos": list(motivos),
        }
        # `ensure_ascii=False` mantém o português legível no arquivo: com o padrão, "asmática"
        # vira "asmática" e a trilha fica ilegível justamente na hora de exibi-la.
        linha = json.dumps(entrada, ensure_ascii=False)
        # `touch` antes do append: o modo só vale no instante da criação, e é o único momento
        # em que dá para fixá-lo sem mexer na permissão de um arquivo que alguém já ajustou.
        self.log_path.touch(mode=MODO_ARQUIVO, exist_ok=True)
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
