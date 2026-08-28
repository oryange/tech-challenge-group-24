"""Log estruturado de auditoria das interações com o assistente.

Uso como biblioteca:

    from src.audit.audit_logger import audit_logger

    audit_logger.log(
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
vídeo de entrega. Por isso tudo o que é texto livre passa pelo `anonymize` do PR 02 antes de
ser gravado.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.data.anonymizer import anonymize

RAIZ = Path(__file__).resolve().parents[2]
CAMINHO_PADRAO = RAIZ / "logs" / "audit.jsonl"

# Recorte da resposta guardado no log. O log é trilha de auditoria, não cache de respostas:
# o que importa é reconstruir o que foi perguntado, se o guardrail agiu e de que fonte a
# resposta saiu. Guardar a resposta inteira multiplicaria o tamanho do arquivo sem responder
# nenhuma pergunta de auditoria que o recorte já não responda.
PREVIEW_CARACTERES = 200


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
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

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
        """
        entrada = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "session_id": session_id,
            "patient_id": patient_id,
            "query": anonymize(query),
            "response_preview": anonymize(response)[:PREVIEW_CARACTERES],
            "source": source,
            "guardrail_triggered": guardrail_triggered,
        }
        # `ensure_ascii=False` mantém o português legível no arquivo: com o padrão, "asmática"
        # vira "asmática" e a trilha fica ilegível justamente na hora de exibi-la.
        linha = json.dumps(entrada, ensure_ascii=False)
        with self.log_path.open("a", encoding="utf-8") as arquivo:
            arquivo.write(f"{linha}\n")
        return entrada

    def _ler(self) -> list[dict]:
        """Lê todas as entradas. Arquivo ausente é trilha vazia, não erro.

        Linhas que não decodificam são puladas em vez de derrubar a leitura. O caso real é o
        processo morrer no meio de uma escrita e deixar a última linha pela metade — e uma
        trilha de auditoria que se recusa a abrir por causa disso perde as entradas íntegras
        junto com a quebrada, que é o pior dos dois resultados.
        """
        if not self.log_path.exists():
            return []
        entradas = []
        with self.log_path.open(encoding="utf-8") as arquivo:
            for linha in arquivo:
                linha = linha.strip()
                if not linha:
                    continue
                try:
                    entradas.append(json.loads(linha))
                except json.JSONDecodeError:
                    continue
        return entradas

    def get_session_logs(self, session_id: str) -> list[dict]:
        """Todas as interações de uma sessão, na ordem em que aconteceram."""
        return [e for e in self._ler() if e.get("session_id") == session_id]

    def get_patient_logs(self, patient_id: str) -> list[dict]:
        """Histórico de consultas ao assistente sobre um paciente, de todas as sessões."""
        return [e for e in self._ler() if e.get("patient_id") == patient_id]


# Instância pronta para o PR 07 e o PR 08 usarem sem repetir a leitura do `.env`.
#
# Construída no import, o que cria o diretório do log nesse momento. É aceitável aqui: o
# `logs/` já existe versionado no repositório desde o PR 01, então no caminho padrão a
# criação é no-op, e quem aponta `AUDIT_LOG_PATH` para outro lugar quer o diretório criado.
audit_logger = AuditLogger.from_env()
