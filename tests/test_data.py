"""Testes do pipeline de dados."""

from __future__ import annotations

import json

import pytest

from src.data.anonymizer import anonymize, anonymize_record
from src.data.synthetic_generator import (
    CONDICOES,
    TOTAL_PADRAO,
    generate_synthetic,
)
from src.data.curator import curate, deduplicate, filter_by_quality
from src.data.loader import (
    format_context,
    load_jsonl,
    load_pubmedqa,
    normalize_text,
    save_jsonl,
    to_instruction_format,
)

REGISTRO_CRU = {
    "pubid": 21645374,
    "question": "Do mitochondria play a role in remodelling lace plant leaves?",
    "context": {
        "contexts": ["Programmed cell death is regulated.", "Mitochondria were observed."],
        "labels": ["BACKGROUND", "RESULTS"],
    },
    "long_answer": "Results depicted mitochondrial dynamics in vivo as PCD progresses.",
    "final_decision": "yes",
}


def test_to_instruction_format():
    convertido = to_instruction_format(REGISTRO_CRU)

    assert set(convertido) == {"instruction", "input", "output", "source"}
    assert convertido["instruction"] == REGISTRO_CRU["question"]
    assert convertido["output"] == REGISTRO_CRU["long_answer"]
    assert all(isinstance(v, str) for v in convertido.values())


def test_source_carrega_pubid():
    # O source é o que o assistente cita como fonte da informação (explainability).
    assert to_instruction_format(REGISTRO_CRU)["source"] == "PubMedQA:21645374"


def test_format_context_preserva_rotulos():
    formatado = format_context(REGISTRO_CRU["context"])

    assert "BACKGROUND: Programmed cell death is regulated." in formatado
    assert "RESULTS: Mitochondria were observed." in formatado


def test_format_context_sem_rotulos_alinhados():
    # Se labels e contexts vierem desalinhados, nenhum trecho pode ser perdido.
    context = {"contexts": ["Primeiro trecho.", "Segundo trecho."], "labels": ["SO_UM"]}

    formatado = format_context(context)

    assert "Primeiro trecho." in formatado
    assert "Segundo trecho." in formatado


def test_to_instruction_format_campos_ausentes():
    convertido = to_instruction_format({"pubid": 1})

    assert convertido["instruction"] == ""
    assert convertido["output"] == ""
    assert convertido["input"] == ""


def test_save_jsonl_uma_linha_por_registro(tmp_path):
    registros = [
        {"instruction": "p1", "input": "c1", "output": "r1", "source": "PubMedQA:1"},
        {"instruction": "p2", "input": "c2", "output": "r2", "source": "PubMedQA:2"},
    ]

    destino = tmp_path / "sub" / "saida.jsonl"  # diretório inexistente de propósito
    total = save_jsonl(registros, destino)

    linhas = destino.read_text(encoding="utf-8").strip().split("\n")
    assert total == 2
    assert len(linhas) == 2
    assert json.loads(linhas[0])["source"] == "PubMedQA:1"


def test_save_jsonl_preserva_acentuacao(tmp_path):
    destino = tmp_path / "acentos.jsonl"
    save_jsonl([{"instruction": "hipertensão", "input": "", "output": "avaliação", "source": "x"}], destino)

    conteudo = destino.read_text(encoding="utf-8")
    assert "hipertensão" in conteudo  # ensure_ascii=False, não ã


def test_normalize_text_remove_quebras_nao_escapadas():
    # U+2029 existe de verdade no pqa_labeled e o json.dumps(ensure_ascii=False) nao escapa:
    # cru no texto, ele parte a linha do JSONL em duas.
    assert normalize_text("remissao pode ocorrer. .") == "remissao pode ocorrer. ."
    assert normalize_text("a b") == "a b"
    assert normalize_text("  espacos  ") == "espacos"
    assert normalize_text(None) == ""


def test_save_jsonl_uma_linha_fisica_por_registro(tmp_path):
    # Regressao: cada registro tem de ocupar exatamente uma linha, inclusive sob splitlines().
    registros = [
        {"instruction": "p q", "input": "", "output": "r", "source": "PubMedQA:1"},
        {"instruction": "p2", "input": "", "output": "r2", "source": "PubMedQA:2"},
    ]
    convertidos = [
        to_instruction_format(
            {"pubid": i + 1, "question": r["instruction"], "long_answer": r["output"]}
        )
        for i, r in enumerate(registros)
    ]

    destino = tmp_path / "s.jsonl"
    save_jsonl(convertidos, destino)
    texto = destino.read_text(encoding="utf-8")

    assert len(texto.splitlines()) == 2
    assert len(texto.rstrip("\n").split("\n")) == 2
    for linha in texto.splitlines():
        json.loads(linha)  # nao levanta


def test_load_pubmedqa_descarta_registros_incompletos(mocker):
    # Sem tocar a rede: o dataset é mockado.
    mocker.patch(
        "src.data.loader.load_dataset",
        return_value={"train": [REGISTRO_CRU, {"pubid": 2, "question": "", "long_answer": "x"}]},
    )

    registros = list(load_pubmedqa())

    assert len(registros) == 1
    assert registros[0]["source"] == "PubMedQA:21645374"


# --------------------------------------------------------------------------------------
# anonymizer.py
# --------------------------------------------------------------------------------------


def test_anonymize_removes_name():
    texto = "Paciente atendido pelo Dr. João Carlos Silva na unidade central."

    resultado = anonymize(texto)

    assert "João" not in resultado
    assert "Silva" not in resultado
    assert "[MÉDICO]" in resultado


def test_anonymize_removes_date():
    for texto, _ in (
        ("Consulta em 12/03/2024.", "barra"),
        ("Retorno em 2024-05-01.", "iso"),
        ("Internado em 12 de março de 2024.", "extenso"),
    ):
        resultado = anonymize(texto)
        assert "[DATA]" in resultado, texto
        assert "2024" not in resultado, texto


def test_anonymize_record_keys_preserved():
    registro = {
        "instruction": "Qual a conduta para o paciente Pedro Alves?",
        "input": "CPF 123.456.789-00",
        "output": "Encaminhar ao Dr. Silva.",
        "source": "PubMedQA:21645374",
    }

    anonimizado = anonymize_record(registro)

    assert set(anonimizado) == set(registro)
    assert "[PACIENTE]" in anonimizado["instruction"]
    assert "[PACIENTE_ID]" in anonimizado["input"]
    assert "[MÉDICO]" in anonimizado["output"]


def test_anonymize_record_nao_toca_source():
    # O source é a referência citada na resposta (explainability); anonimizá-lo destruiria
    # a rastreabilidade sem proteger nenhum paciente.
    registro = {"instruction": "x", "input": "", "output": "", "source": "PubMedQA:21645374"}

    assert anonymize_record(registro)["source"] == "PubMedQA:21645374"


def test_anonymize_remove_cpf_telefone_email_prontuario():
    texto = (
        "A paciente Maria Souza, CPF 123.456.789-00, telefone (11) 98765-4321, "
        "e-mail maria@hospital.com.br, prontuário nº 4455667."
    )

    resultado = anonymize(texto)

    for vazado in ("Maria", "123.456.789-00", "98765-4321", "maria@hospital.com.br", "4455667"):
        assert vazado not in resultado, vazado
    assert resultado.endswith(".")  # regressão: o ID não pode engolir a pontuação final


@pytest.mark.parametrize(
    "cientifico",
    [
        "The study covered 2000-2012 in three centers.",  # intervalo de anos, não telefone
        "Mortality was lower (P = .04) in the treated group.",  # espaçamento preservado
        "Effects on children: a randomized trial.",
        "Programmed Cell Death in Aponogeton madagascariensis.",  # capitalizadas, não nomes
    ],
)
def test_anonymize_preserva_texto_cientifico(cientifico):
    # Regras agressivas destruiriam o PubMedQA: medido em 1.000 registros, apenas 1 campo
    # é alterado — e nele há datas reais (8/1/97), então a substituição é correta.
    assert anonymize(cientifico) == cientifico


def test_anonymize_idempotente():
    texto = "Sr. Pedro Alves, CNS 123456789012345, retorno 01/02/2024, tel (11) 98765-4321."

    uma = anonymize(texto)

    assert anonymize(uma) == uma


def test_anonymize_texto_vazio_ou_none():
    assert anonymize("") == ""
    assert anonymize(None) == ""


# --------------------------------------------------------------------------------------
# synthetic_generator.py
# --------------------------------------------------------------------------------------


def test_synthetic_generator_output_count():
    assert len(generate_synthetic()) == TOTAL_PADRAO
    assert len(generate_synthetic(total=25)) == 25


def test_synthetic_cobre_os_quatro_tipos_do_enunciado():
    # O enunciado pede protocolos, FAQs de médicos, laudos e receitas/procedimentos.
    tipos = {r["source"].split(":")[0] for r in generate_synthetic()}

    assert tipos == {"Protocolo", "FAQ", "Laudo", "Receita", "Procedimento"}


def test_synthetic_cobre_todas_as_condicoes_cid10():
    cids = {r["source"].split(":")[1] for r in generate_synthetic()}

    assert cids == {c["cid"] for c in CONDICOES}


def test_synthetic_toda_resposta_cita_fonte_e_exige_validacao():
    # Explainability e limite de atuação vêm dos dados, não só do guardrail.
    for registro in generate_synthetic():
        assert f"[Fonte: {registro['source']}]" in registro["output"]
        assert "[Requer validação médica" in registro["output"]


def test_synthetic_nao_contem_pii():
    # Anonimizar os dados sintéticos tem de ser operação nula: eles já nascem com tokens.
    for registro in generate_synthetic():
        for campo in ("instruction", "input", "output"):
            assert anonymize(registro[campo]) == registro[campo], registro["source"]


def test_synthetic_registros_sao_unicos():
    # Regressão: sortear (condição, pergunta) repetia pares e o curator, que remove
    # duplicatas, descartava ~20% do que este módulo gerava.
    registros = generate_synthetic()

    assert len({r["instruction"] for r in registros}) == len(registros)


def test_synthetic_respostas_sobrevivem_ao_filtro_de_qualidade():
    # O curator descarta respostas com menos de 20 palavras.
    assert all(len(r["output"].split()) >= 20 for r in generate_synthetic())


def test_synthetic_e_deterministico_por_seed():
    assert generate_synthetic(seed=42) == generate_synthetic(seed=42)
    assert generate_synthetic(seed=42) != generate_synthetic(seed=7)


def test_synthetic_formato_igual_ao_loader():
    for registro in generate_synthetic(total=10):
        assert set(registro) == {"instruction", "input", "output", "source"}
        assert all(isinstance(v, str) and v for v in registro.values())


# --------------------------------------------------------------------------------------
# curator.py
# --------------------------------------------------------------------------------------


def _registro(instruction="pergunta", input="contexto", output=None, source="PubMedQA:1"):
    return {
        "instruction": instruction,
        "input": input,
        "output": output if output is not None else " ".join(["palavra"] * 25),
        "source": source,
    }


def test_curator_deduplication():
    registros = [
        _registro(instruction="Qual a conduta?", source="PubMedQA:1"),
        _registro(instruction="Qual a conduta?", source="PubMedQA:2"),  # duplicata
        _registro(instruction="Outra pergunta", source="PubMedQA:3"),
    ]

    unicos = deduplicate(registros)

    assert len(unicos) == 2
    assert unicos[0]["source"] == "PubMedQA:1"  # primeira ocorrência vence


def test_curator_deduplication_ignora_caixa_e_espacos():
    registros = [
        _registro(instruction="Qual a  CONDUTA?"),
        _registro(instruction="qual a conduta?"),
    ]

    assert len(deduplicate(registros)) == 1


def test_curator_quality_filter():
    curto = _registro(output="Resposta curta.")
    longo = _registro(instruction="outra", output=" ".join(["palavra"] * 25))

    filtrados = filter_by_quality([curto, longo])

    assert len(filtrados) == 1
    assert filtrados[0]["output"].split()[0] == "palavra"


def test_curator_quality_filter_limite_inclusivo():
    # Exatamente 20 palavras deve passar; 19 não.
    assert len(filter_by_quality([_registro(output=" ".join(["p"] * 20))])) == 1
    assert len(filter_by_quality([_registro(output=" ".join(["p"] * 19))])) == 0


def test_curator_quality_filter_descarta_campos_vazios():
    assert filter_by_quality([_registro(instruction="", output=" ".join(["p"] * 30))]) == []
    assert filter_by_quality([_registro(output="")]) == []


def test_curate_embaralha_para_o_split_do_fine_tuning(tmp_path):
    # Sem embaralhar, o split sequencial 90/10 do fine-tuning jogaria TODOS os registros
    # hospitalares na validação e o modelo nunca treinaria com os dados do hospital.
    pubmed = [_registro(instruction=f"p{i}", source=f"PubMedQA:{i}") for i in range(90)]
    hospital = [_registro(instruction=f"h{i}", source=f"Protocolo:{i}") for i in range(10)]
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    save_jsonl(pubmed, a)
    save_jsonl(hospital, b)

    registros, _ = curate(caminhos=(a, b))
    treino = registros[: int(len(registros) * 0.9)]

    assert any(not r["source"].startswith("PubMedQA") for r in treino)


def test_curate_estatisticas_e_deterministico(tmp_path):
    arquivo = tmp_path / "e.jsonl"
    save_jsonl(
        [
            _registro(instruction="a"),
            _registro(instruction="a"),  # duplicata
            _registro(instruction="b", output="curto"),  # filtrado
            _registro(instruction="c"),
        ],
        arquivo,
    )

    registros, stats = curate(caminhos=(arquivo,))

    assert stats == {"lidos": 4, "duplicatas_removidas": 1, "curtos_removidos": 1, "final": 2}
    assert [r["instruction"] for r in curate(caminhos=(arquivo,))[0]] == [
        r["instruction"] for r in registros
    ]


def test_curate_anonimiza_antes_de_gravar(tmp_path):
    arquivo = tmp_path / "pii.jsonl"
    save_jsonl(
        [_registro(instruction="Conduta para o paciente Pedro Alves?", input="CPF 123.456.789-00")],
        arquivo,
    )

    registros, _ = curate(caminhos=(arquivo,))

    assert "Pedro" not in registros[0]["instruction"]
    assert "123.456.789-00" not in registros[0]["input"]


def test_load_jsonl_arquivo_ausente_erro_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="Rode as etapas anteriores"):
        load_jsonl(tmp_path / "nao_existe.jsonl")
