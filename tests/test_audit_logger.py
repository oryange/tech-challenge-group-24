"""Testes da trilha de auditoria."""

from __future__ import annotations

import json

import pytest

import src.audit.audit_logger as audit_logger_module
from src.audit.audit_logger import (
    LIMITE_TEXTO_LIVRE,
    MODO_ARQUIVO,
    MODO_DIRETORIO,
    PREVIEW_CARACTERES,
    AuditLogger,
)

CAMPOS_OBRIGATORIOS = {
    "timestamp",
    "session_id",
    "patient_id",
    "query",
    "response_preview",
    "source",
    "guardrail_triggered",
    "tem_fonte",
    "motivos",
    "alergias_alertadas",
}


@pytest.fixture
def logger(tmp_path) -> AuditLogger:
    return AuditLogger(tmp_path / "audit.jsonl")


def _log(logger: AuditLogger, **kwargs) -> dict:
    padrao = {
        "query": "Quais exames estão pendentes?",
        "response": "Hemograma e glicemia. [Fonte: exames do paciente]",
        "patient_id": "[PACIENTE_007]",
        "source": "exames do paciente",
        "guardrail_triggered": False,
        "session_id": "sessao-01",
    }
    return logger.log(**{**padrao, **kwargs})


def test_log_creates_file(logger):
    assert not logger.log_path.exists()

    _log(logger)

    assert logger.log_path.exists()


def test_cria_o_diretorio_pai(tmp_path):
    # Falha na construção, não na primeira consulta clínica.
    destino = tmp_path / "sem" / "nada" / "aqui" / "audit.jsonl"

    AuditLogger(destino)

    assert destino.parent.is_dir()


def test_log_entry_has_required_fields(logger):
    entrada = _log(logger)

    assert set(entrada) == CAMPOS_OBRIGATORIOS


def test_log_grava_uma_linha_json_por_interacao(logger):
    _log(logger, session_id="a")
    _log(logger, session_id="b")

    linhas = logger.log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(linhas) == 2
    assert [json.loads(linha)["session_id"] for linha in linhas] == ["a", "b"]


def test_log_truncates_response(logger):
    entrada = _log(logger, response="R" * (PREVIEW_CARACTERES * 3))

    assert len(entrada["response_preview"]) == PREVIEW_CARACTERES


def test_log_anonimiza_pii_da_pergunta(logger):
    entrada = _log(logger, query="O paciente João Silva relata febre há 3 dias.")

    assert "João Silva" not in entrada["query"]
    assert "[PACIENTE]" in entrada["query"]


def test_log_anonimiza_telefone_sem_formatacao(logger):
    # As regras do PR 02 exigem parênteses ou o +55; quem digita no chat escreve o número
    # corrido, e é esse que ia para o disco.
    entrada = _log(logger, query="Ligar para 11987654321.")

    assert "11987654321" not in entrada["query"]
    assert "[TELEFONE]" in entrada["query"]


def test_log_anonimiza_cpf_sem_pontuacao(logger):
    # Sem a âncora "CPF" na frente, a regra do PR 02 não casa o número solto.
    entrada = _log(logger, query="Confirmar cadastro 12345678901 antes da consulta.")

    assert "12345678901" not in entrada["query"]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Limite declarado no módulo: as regras do PR 02 são ancoradas em contexto para não "
        "destruir o dataset de treino, e detectar nome próprio solto por regex tem falso "
        "positivo caro em texto clínico. Registrado como falha esperada para não virar "
        "garantia implícita."
    ),
)
def test_log_anonimiza_nome_sem_ancora(logger):
    entrada = _log(logger, query="Maria Silva está com febre há 3 dias.")

    assert "Maria Silva" not in entrada["query"]


def test_log_anonimiza_antes_de_recortar(logger):
    # O CPF fica a cavaleiro do corte de 200 caracteres. Recortando primeiro, a regra do CPF
    # deixa de casar (ela exige os dois dígitos finais) e o pedaço da esquerda seria gravado
    # em claro. Anonimizando primeiro, o CPF inteiro vira token antes de qualquer corte.
    enchimento = "Evolução do quadro sem intercorrências relevantes. "
    prefixo = (enchimento * 5)[: PREVIEW_CARACTERES - 14]

    entrada = _log(logger, response=f"{prefixo}CPF 123.456.789-01 conforme cadastro.")

    assert "123.456" not in entrada["response_preview"]
    assert "123.456" not in logger.log_path.read_text(encoding="utf-8")


def test_log_limita_o_texto_livre_antes_de_anonimizar(logger):
    # `log()` é API pública e nem todo chamador passa pelo `sanitize_input` do PR 05. Sem o
    # teto, uma entrada de megabytes atravessa inteira as regex do anonimizador.
    entrada = _log(logger, query="a" * (LIMITE_TEXTO_LIVRE * 3))

    assert len(entrada["query"]) == LIMITE_TEXTO_LIVRE


@pytest.mark.parametrize(
    "pii",
    [
        "123.456.789-01",  # CPF pontuado
        "12345678901",  # CPF como quem digita no chat
        "11987654321",  # celular sem formatação
    ],
)
def test_o_teto_nao_parte_pii_no_corte(logger, pii):
    # Mesma armadilha do corte de 200, um andar acima: o teto de texto livre vem **antes** da
    # anonimização, então um dado a cavaleiro dele perderia a cauda, deixaria de casar com a
    # regra e o pedaço da esquerda iria em claro para o disco.
    enchimento = "a" * (LIMITE_TEXTO_LIVRE - len(pii) + 3)
    # Sete caracteres: cabe no que sobraria do dado à esquerda do corte em todas as formas
    # testadas, então o teste falha de verdade se o pedaço chegar ao disco.
    vazamento = pii[:7]

    entrada = _log(logger, query=f"{enchimento} {pii} conforme cadastro.")

    assert vazamento not in entrada["query"]
    assert vazamento not in logger.log_path.read_text(encoding="utf-8")


def test_o_teto_continua_valendo_para_token_gigante(logger):
    # O recuo é limitado: uma sequência longa sem espaço não é PII conhecida, e deixar o
    # recuo ilimitado transformaria o teto em sugestão — bastaria um texto sem espaço nenhum.
    entrada = _log(logger, query="a" * (LIMITE_TEXTO_LIVRE * 3))

    assert len(entrada["query"]) == LIMITE_TEXTO_LIVRE


def test_log_com_texto_livre_ausente_nao_quebra(logger):
    entrada = _log(logger, query="", response="")

    assert entrada["query"] == ""
    assert entrada["response_preview"] == ""


def test_o_teto_nao_atrapalha_a_anonimizacao(logger):
    # O corte é ordens de grandeza maior que o alcance das âncoras do PR 02: uma pergunta no
    # limite continua tendo o nome redigido normalmente.
    enchimento = "Evolução sem intercorrências. " * 40
    entrada = _log(logger, query=f"{enchimento}O paciente João Silva relata febre.")

    assert "João Silva" not in entrada["query"]


def test_trilha_e_diretorio_nao_ficam_legiveis_por_terceiros(tmp_path):
    destino = tmp_path / "nova" / "audit.jsonl"
    logger = AuditLogger(destino)

    _log(logger)

    # Só o dono. O arquivo guarda pergunta em texto livre e recorte de resposta clínica.
    assert destino.stat().st_mode & 0o777 == MODO_ARQUIVO
    assert destino.parent.stat().st_mode & 0o777 == MODO_DIRETORIO


def test_aperta_o_diretorio_do_projeto_que_ja_existia(tmp_path, monkeypatch):
    # O caso padrão do projeto: `logs/` é versionado (`logs/.gitkeep`) e existe desde o clone
    # com 0755. O `exist_ok=True` do `mkdir` devolve sem tocar na permissão, e `MODO_DIRETORIO`
    # não valia justamente onde a trilha de verdade é escrita.
    #
    # A `RAIZ` é reapontada porque o critério de "nosso" é estar dentro dela: sem isso o teste
    # mediria um diretório de propósito geral, que é exatamente o que **não** deve ser apertado.
    monkeypatch.setattr(audit_logger_module, "RAIZ", tmp_path.resolve())
    destino = tmp_path / "logs"
    destino.mkdir(mode=0o755)

    AuditLogger(destino / "audit.jsonl")

    assert destino.stat().st_mode & 0o777 == MODO_DIRETORIO


@pytest.mark.parametrize("nome_da_folha", ["home_do_usuario", "raiz_do_repo"])
def test_nao_aperta_diretorio_nosso_de_proposito_geral(tmp_path, monkeypatch, nome_da_folha):
    # A primeira versão excluía o gravável por todos e o de outro dono — e "nosso e não
    # gravável por todos" descreve tanto o `logs/` quanto a `$HOME` e a raiz do repositório.
    # Medido, `AUDIT_LOG_PATH=~/audit.jsonl` apertava a home e `AUDIT_LOG_PATH=audit.jsonl`
    # apertava a raiz do repo, sem aviso, só por instanciar o assistente.
    folha = tmp_path / nome_da_folha
    folha.mkdir(mode=0o755)
    (folha / "arquivo_sem_relacao.txt").write_text("nada a ver com a trilha", encoding="utf-8")
    # A folha é a própria `RAIZ` no caso da raiz do repo, e está fora dela no caso da home.
    raiz = folha if nome_da_folha == "raiz_do_repo" else tmp_path / "repo"
    monkeypatch.setattr(audit_logger_module, "RAIZ", raiz.resolve())

    AuditLogger(folha / "audit.jsonl")

    assert folha.stat().st_mode & 0o777 == 0o755


def test_aperta_a_trilha_que_ja_existia_com_o_umask(tmp_path):
    # O caso real do projeto: quem rodou qualquer versão anterior a este controle tem um
    # `logs/audit.jsonl` criado com o umask (0644) e legível por qualquer conta da máquina. O
    # `mode=` do `touch` só vale na criação, então nenhuma escrita posterior o promovia — o
    # diretório saía 0700 e o arquivo dentro dele continuava 0644, com a trilha inteira.
    destino = tmp_path / "audit.jsonl"
    destino.touch(mode=0o644)
    logger = AuditLogger(destino)

    _log(logger)

    assert destino.stat().st_mode & 0o777 == MODO_ARQUIVO


def test_nao_afrouxa_trilha_que_alguem_apertou_mais(tmp_path):
    # O aperto age só sobre bit de grupo ou de outros. Um arquivo congelado em 0400 de
    # propósito não é reaberto para escrita pelo `chmod` — corrigir o que ficou frouxo não é
    # licença para desfazer decisão de quem mexeu.
    # O helper é chamado direto porque uma trilha 0400 não é gravável nem pelo dono: passar
    # por `log()` mediria o `open("a")` falhando, não o critério do aperto.
    destino = tmp_path / "audit.jsonl"
    destino.touch(mode=0o400)

    audit_logger_module._apertar_trilha(destino)

    assert destino.stat().st_mode & 0o777 == 0o400


def test_aperta_tambem_os_diretorios_intermediarios(tmp_path):
    # `parents=True` cria os intermediários sem o `mode`: só o último saía 0700.
    destino = tmp_path / "a" / "b" / "c" / "audit.jsonl"

    AuditLogger(destino)

    for diretorio in (tmp_path / "a", tmp_path / "a" / "b", destino.parent):
        assert diretorio.stat().st_mode & 0o777 == MODO_DIRETORIO


def test_nao_aperta_diretorio_compartilhado_que_nao_criamos(tmp_path):
    # `AUDIT_LOG_PATH=/tmp/audit.jsonl` põe a folha num diretório de todo mundo. Apertá-lo
    # derruba a máquina se houver privilégio, e explode na construção se não houver — quebrando
    # uma configuração que funcionava. A trilha continua protegida pelo modo do arquivo.
    compartilhado = tmp_path / "tmp_de_todos"
    compartilhado.mkdir()
    # `chmod` e não `mkdir(mode=...)`: o umask come o bit de escrita de terceiros.
    compartilhado.chmod(0o777)

    AuditLogger(compartilhado / "audit.jsonl")

    assert compartilhado.stat().st_mode & 0o777 == 0o777


def test_aperta_a_folha_criada_dentro_de_um_compartilhado(tmp_path):
    # A dúvida de propriedade é só sobre o diretório que já existia. O que este construtor
    # criou é nosso, mesmo pendurado num compartilhado.
    compartilhado = tmp_path / "tmp_de_todos"
    compartilhado.mkdir()
    # `chmod` e não `mkdir(mode=...)`: o umask come o bit de escrita de terceiros.
    compartilhado.chmod(0o777)
    destino = compartilhado / "trilha" / "audit.jsonl"

    AuditLogger(destino)

    assert destino.parent.stat().st_mode & 0o777 == MODO_DIRETORIO
    assert compartilhado.stat().st_mode & 0o777 == 0o777


def test_chmod_impossivel_avisa_em_vez_de_derrubar(tmp_path, monkeypatch):
    # Perder a auditoria inteira porque não deu para endurecer a pasta troca uma perda certa
    # por uma incerta — o conteúdo já está protegido pelo modo do arquivo.
    def recusar(self, mode):
        raise PermissionError("sem permissão")

    monkeypatch.setattr(audit_logger_module.Path, "chmod", recusar)

    # Diretório que o construtor cria, para cair no ramo que aperta sem depender do critério
    # de propriedade — o que se mede aqui é o tratamento da falha, não o critério.
    with pytest.warns(UserWarning, match="restringir"):
        logger = AuditLogger(tmp_path / "nova" / "audit.jsonl")

    _log(logger)
    assert logger.log_path.exists()


def test_source_e_anonimizado_como_o_resto_do_texto_livre(logger):
    # O `source` é texto livre gerado pelo modelo, e a citação vem na forma ancorada que o
    # anonimizador sabe pegar. Sem isto o mesmo trecho saía anonimizado em `response_preview`
    # e em claro em `source`, na mesma linha do arquivo.
    entrada = _log(logger, source="consulta do paciente Joao Souza de 12/03/2026")

    assert "Joao Souza" not in entrada["source"]


def test_source_preserva_a_data_que_torna_a_fonte_rastreavel(logger):
    # Anonimizar a data também trocava um vazamento por uma perda: `"consulta de [DATA]"` não
    # diz de qual consulta a resposta saiu, que é a única pergunta que este campo responde — e
    # é o identificador que o `fonte_confere` acabou de validar. O `patient_id` já vai em claro
    # na mesma linha, então a data não acrescenta poder de reidentificação nenhum.
    entrada = _log(logger, source="consulta do paciente Joao Souza de 12/03/2026")

    assert entrada["source"] == "consulta do paciente [PACIENTE] de 12/03/2026"


def test_source_com_varias_datas_devolve_cada_uma_ao_seu_lugar(logger):
    entrada = _log(logger, source="consulta de 01/02/2026 e exame de 03/04/2026")

    assert entrada["source"] == "consulta de 01/02/2026 e exame de 03/04/2026"


def test_source_com_data_extensa_nao_desloca_a_numerica(logger):
    # A extensa continua redigida (é o alcance declarado de `_DATA_RASTREAVEL`) e a numérica
    # volta ao **seu** lugar. Marcando antes da anonimização a posição é a de origem; com a
    # guarda por contagem, ou nada era restaurado, ou a data ia para a posição errada.
    entrada = _log(logger, source="consulta de 12 de março de 2026 e exame de 03/04/2026")

    assert entrada["source"] == "consulta de [DATA] e exame de 03/04/2026"


def test_source_nao_restaura_data_que_era_identificador_ancorado(logger):
    # O defeito que este teste fecha era vazamento, não só data errada: a numérica é consumida
    # pela regra de **prontuário** e vira `[PACIENTE_ID]`; a extensa vira `[DATA]`. A contagem
    # coincidia em 1, a guarda deixava passar, e o `replace` punha o número de prontuário — que
    # a anonimização acabara de redigir — em claro na posição da outra data.
    entrada = _log(logger, source="consulta de 05 de maio de 2020, prontuario 12/03/2026")

    assert entrada["source"] == "consulta de [DATA], prontuario [PACIENTE_ID]"
    assert "12/03/2026" not in entrada["source"]


def test_source_nao_deixa_a_sentinela_no_arquivo(logger):
    # A marca é interna: nem quando a restauração acontece, nem quando ela falha fechado, a
    # sentinela pode chegar ao disco — ela tem cara de identificador para quem lê a trilha.
    for fonte in (
        "consulta de 12/03/2026",
        "prontuario 12/03/2026",
        "consulta de 12 de março de 2026 e exame de 03/04/2026",
    ):
        entrada = _log(logger, source=fonte)

        assert "40028922" not in entrada["source"]


def test_source_ausente_continua_distinto_de_source_vazio(logger):
    # `None` e `""` são conclusões diferentes numa auditoria — ver `extrair_fonte`.
    assert _log(logger, source=None)["source"] is None
    assert _log(logger, source="")["source"] is None


def test_reaperta_a_trilha_que_foi_afrouxada(logger):
    _log(logger)
    logger.log_path.chmod(0o640)

    _log(logger)

    # Decisão revista. Antes a trilha afrouxada era deixada como estava, sob o argumento de
    # não sobrescrever quem tinha mexido de propósito. O que a revisão mediu é que o caso real
    # não é esse: o `logs/audit.jsonl` do repositório foi criado com o umask antes de o
    # controle existir e ficou em 0644, e o `mode=` do `touch` nunca o promoveria. Não há uso
    # legítimo para bit de grupo ou de outros num arquivo que guarda pergunta de médico em
    # texto livre — o grupo de uma máquina corporativa é todo mundo. Apertar mais que
    # `MODO_ARQUIVO` continua sendo respeitado, que é o lado em que a intenção é plausível.
    assert logger.log_path.stat().st_mode & 0o777 == MODO_ARQUIVO


def test_patient_id_nao_e_anonimizado(logger):
    # Já é token do seed do PR 03; anonimizá-lo destruiria a chave de filtro.
    entrada = _log(logger, patient_id="[PACIENTE_007]")

    assert entrada["patient_id"] == "[PACIENTE_007]"
    assert logger.get_patient_logs("[PACIENTE_007]") == [entrada]


def test_preserva_acentuacao_no_arquivo(logger):
    _log(logger, query="Qual a conduta na crise asmática?")

    assert "asmática" in logger.log_path.read_text(encoding="utf-8")


def test_get_session_logs_filters_correctly(logger):
    _log(logger, session_id="sessao-A", query="Primeira da A.")
    _log(logger, session_id="sessao-B", query="Única da B.")
    _log(logger, session_id="sessao-A", query="Segunda da A.")

    da_sessao = logger.get_session_logs("sessao-A")

    assert len(da_sessao) == 2
    # Ordem de acontecimento preservada — é o que faz a trilha reconstruir a conversa.
    assert [e["query"] for e in da_sessao] == ["Primeira da A.", "Segunda da A."]


def test_get_patient_logs_filters_correctly(logger):
    _log(logger, patient_id="[PACIENTE_001]", session_id="s1")
    _log(logger, patient_id="[PACIENTE_002]", session_id="s1")
    _log(logger, patient_id="[PACIENTE_001]", session_id="s2")

    do_paciente = logger.get_patient_logs("[PACIENTE_001]")

    assert len(do_paciente) == 2
    # Atravessa sessões de propósito: é o histórico do paciente, não o da conversa.
    assert {e["session_id"] for e in do_paciente} == {"s1", "s2"}


def test_consulta_em_trilha_inexistente(logger):
    assert logger.get_session_logs("qualquer") == []
    assert logger.get_patient_logs("qualquer") == []


def test_linha_corrompida_nao_derruba_a_leitura(logger):
    _log(logger, session_id="s1", query="Íntegra antes.")
    with logger.log_path.open("a", encoding="utf-8") as arquivo:
        arquivo.write('{"session_id": "s1", "query": pela met\n')
    _log(logger, session_id="s1", query="Íntegra depois.")

    with pytest.warns(UserWarning, match="ilegível"):
        entradas = logger.get_session_logs("s1")

    assert [e["query"] for e in entradas] == ["Íntegra antes.", "Íntegra depois."]


def test_linha_corrompida_nao_e_descartada_em_silencio(logger):
    # Trilha inteira ilegível devolve [], que é indistinguível de "não houve interação" —
    # a conclusão oposta, e a pior das duas numa auditoria.
    logger.log_path.write_text("nem json é\ntampouco isto\n", encoding="utf-8")

    with pytest.warns(UserWarning, match="2 linha"):
        assert logger.get_session_logs("s1") == []


def test_log_registra_a_explainability_do_pr05(logger):
    # `tem_fonte` é a métrica de explainability do enunciado: calculada no PR 05, ela só
    # existe depois da entrega se alguém a persistir.
    entrada = _log(logger, tem_fonte=False, motivos=("sem_fonte", "prescricao_na_pergunta"))

    assert entrada["tem_fonte"] is False
    assert entrada["motivos"] == ["sem_fonte", "prescricao_na_pergunta"]


def test_explainability_ausente_nao_vira_negativa(logger):
    # None é "não foi medido"; False é "não tinha fonte". Numa auditoria são conclusões
    # diferentes, e achatar as duas em False inventaria um resultado.
    entrada = _log(logger)

    assert entrada["tem_fonte"] is None
    assert entrada["motivos"] == []


def test_from_env_le_o_caminho(monkeypatch, tmp_path):
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "outro" / "audit.jsonl"))

    logger = AuditLogger.from_env()

    assert logger.log_path == tmp_path / "outro" / "audit.jsonl"


def test_from_env_ancora_relativo_na_raiz(monkeypatch):
    monkeypatch.setenv("AUDIT_LOG_PATH", "logs/audit.jsonl")

    logger = AuditLogger.from_env()

    assert logger.log_path.is_absolute()
    assert logger.log_path.parts[-2:] == ("logs", "audit.jsonl")


def test_get_audit_logger_le_o_ambiente_na_chamada_e_nao_no_import(monkeypatch, tmp_path):
    # A regressão que este teste fixa: instância construída no import lê o ambiente antes do
    # `load_dotenv` de um programa de linha de comando, e o AUDIT_LOG_PATH do `.env` é
    # ignorado sem nada denunciar.
    monkeypatch.setattr(audit_logger_module, "_PADRAO", None)
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "tardio" / "audit.jsonl"))

    logger = audit_logger_module.get_audit_logger()

    assert logger.log_path == tmp_path / "tardio" / "audit.jsonl"


def test_get_audit_logger_reaproveita_a_mesma_instancia(monkeypatch):
    monkeypatch.setattr(audit_logger_module, "_PADRAO", None)

    assert audit_logger_module.get_audit_logger() is audit_logger_module.get_audit_logger()
