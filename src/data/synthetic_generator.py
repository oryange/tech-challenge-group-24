"""Gera os dados hospitalares sintéticos usados no fine-tuning.

Uso, a partir da raiz do repositório:

    python -m src.data.synthetic_generator

O enunciado da Fase 3 pede fine-tuning com "protocolos médicos do hospital; exemplos de
perguntas frequentes feitas por médicos; modelos de laudos, receitas e procedimentos
internos". O PubMedQA cobre literatura científica, não o hospital — este módulo cobre a
parte "dados próprios", com conteúdo inteiramente fabricado a partir de templates.

Três decisões que valem registro:

* **Nenhum dado real, e nenhum dado a anonimizar.** Nomes já saem como os tokens que o
  `anonymizer` produziria (`[PACIENTE]`, `[MÉDICO]`, `[DATA]`), então rodar a anonimização
  em cima destes registros é uma operação nula — o `curator` pode aplicá-la sem risco.
* **Toda resposta cita a fonte** (`[Fonte: Protocolo:J45]`). Isso ensina o modelo a citar,
  que é o requisito de explainability; o guardrail valida a presença do marcador em vez de
  precisar inventá-lo.
* **Toda resposta carrega o disclaimer de validação humana.** A posologia é fictícia e o
  assistente nunca prescreve por conta própria — é o limite de atuação exigido pelo
  enunciado, aprendido dos dados e não só imposto por filtro.
"""

from __future__ import annotations

import random
from pathlib import Path

from src.data.loader import save_jsonl

RAIZ = Path(__file__).resolve().parents[2]
SAIDA_PADRAO = RAIZ / "data" / "synthetic" / "synthetic_hospital.jsonl"

TOTAL_PADRAO = 100
SEED_PADRAO = 42

DISCLAIMER = "[Requer validação médica por profissional habilitado]"

# Condições clínicas com código CID-10. Os campos alimentam protocolos, laudos, receitas e
# procedimentos, de forma que o mesmo quadro apareça de ângulos diferentes no dataset.
CONDICOES: tuple[dict[str, str], ...] = (
    {
        "cid": "J45",
        "nome": "asma",
        "exame": "espirometria",
        "achado": "distúrbio ventilatório obstrutivo com resposta ao broncodilatador",
        "conduta": "broncodilatador inalatório de resgate e corticoide inalatório de manutenção",
        "medicamento": "corticoide inalatório",
        "posologia": "1 inalação a cada 12 horas por 30 dias",
        "procedimento": "nebulização assistida",
    },
    {
        "cid": "I10",
        "nome": "hipertensão arterial essencial",
        "exame": "monitorização ambulatorial da pressão arterial",
        "achado": "médias pressóricas acima do limite para o período de vigília",
        "conduta": "restrição de sódio, atividade física regular e anti-hipertensivo em dose inicial",
        "medicamento": "anti-hipertensivo de primeira linha",
        "posologia": "1 comprimido ao dia, pela manhã",
        "procedimento": "aferição seriada de pressão arterial",
    },
    {
        "cid": "E11",
        "nome": "diabetes mellitus tipo 2",
        "exame": "hemoglobina glicada",
        "achado": "controle glicêmico acima da meta terapêutica individualizada",
        "conduta": "orientação nutricional, atividade física e antidiabético oral",
        "medicamento": "antidiabético oral",
        "posologia": "1 comprimido a cada 12 horas, junto às refeições",
        "procedimento": "coleta de glicemia capilar",
    },
    {
        "cid": "J18",
        "nome": "pneumonia adquirida na comunidade",
        "exame": "radiografia de tórax",
        "achado": "opacidade heterogênea em lobo inferior direito",
        "conduta": "antibioticoterapia empírica e reavaliação clínica em 48 a 72 horas",
        "medicamento": "antibiótico de amplo espectro",
        "posologia": "1 comprimido a cada 8 horas por 7 dias",
        "procedimento": "oximetria de pulso seriada",
    },
    {
        "cid": "N39.0",
        "nome": "infecção do trato urinário",
        "exame": "urocultura com antibiograma",
        "achado": "crescimento bacteriano significativo com perfil de sensibilidade definido",
        "conduta": "antibioticoterapia guiada pelo antibiograma e hidratação",
        "medicamento": "antibiótico conforme antibiograma",
        "posologia": "1 comprimido a cada 12 horas por 5 dias",
        "procedimento": "coleta de urina de jato médio",
    },
    {
        "cid": "K29.7",
        "nome": "gastrite não especificada",
        "exame": "endoscopia digestiva alta",
        "achado": "mucosa gástrica com enantema difuso, sem sinais de sangramento ativo",
        "conduta": "inibidor de bomba de prótons e orientação dietética",
        "medicamento": "inibidor de bomba de prótons",
        "posologia": "1 comprimido ao dia, em jejum, por 28 dias",
        "procedimento": "preparo para endoscopia",
    },
    {
        "cid": "J44",
        "nome": "doença pulmonar obstrutiva crônica",
        "exame": "espirometria com prova broncodilatadora",
        "achado": "obstrução ao fluxo aéreo pouco reversível",
        "conduta": "cessação do tabagismo, broncodilatador de longa duração e reabilitação pulmonar",
        "medicamento": "broncodilatador de longa duração",
        "posologia": "1 inalação ao dia, sempre no mesmo horário",
        "procedimento": "treinamento de técnica inalatória",
    },
    {
        "cid": "G43",
        "nome": "migrânea",
        "exame": "avaliação neurológica clínica",
        "achado": "exame neurológico sem sinais focais, padrão compatível com cefaleia primária",
        "conduta": "analgesia na crise, diário de cefaleia e profilaxia se alta frequência",
        "medicamento": "analgésico simples",
        "posologia": "1 comprimido na crise, no máximo 3 doses ao dia",
        "procedimento": "aplicação de escala de intensidade de dor",
    },
    {
        "cid": "A09",
        "nome": "gastroenterite presumivelmente infecciosa",
        "exame": "avaliação clínica de hidratação",
        "achado": "desidratação leve, sem sinais de choque",
        "conduta": "hidratação oral, dieta leve e sinais de alerta orientados",
        "medicamento": "sais de reidratação oral",
        "posologia": "1 sachê diluído após cada episódio de perda",
        "procedimento": "reidratação oral supervisionada",
    },
    {
        "cid": "I21",
        "nome": "infarto agudo do miocárdio",
        "exame": "eletrocardiograma e troponina seriada",
        "achado": "alteração de segmento ST com curva de troponina ascendente",
        "conduta": "acionamento imediato do protocolo de dor torácica e transferência para hemodinâmica",
        "medicamento": "antiagregante plaquetário",
        "posologia": "dose de ataque conforme protocolo institucional",
        "procedimento": "protocolo de dor torácica",
    },
)

PERGUNTAS_FAQ: tuple[str, ...] = (
    "Quando devo solicitar {exame} em um paciente com suspeita de {nome}?",
    "Qual a conduta inicial para {nome} no pronto atendimento?",
    "Quais critérios indicam internação em um quadro de {nome}?",
    "Em quanto tempo devo reavaliar um paciente com {nome} após iniciar o tratamento?",
    "Que orientações de alta devo dar ao paciente com {nome}?",
    "Quais sinais de alerta justificam retorno imediato em {nome}?",
)

RESPOSTAS_FAQ: tuple[str, ...] = (
    "Conforme o protocolo interno de {nome} (CID-10 {cid}), a recomendação é {conduta}. "
    "O exame de referência é {exame}, e a reavaliação deve ser registrada no prontuário do "
    "[PACIENTE] pelo [MÉDICO] responsável.",
    "No protocolo institucional de {nome} (CID-10 {cid}), a conduta esperada é {conduta}. "
    "Solicitar {exame} quando houver dúvida diagnóstica e documentar a decisão clínica.",
    "Para {nome} (CID-10 {cid}), o serviço orienta {conduta}. Casos sem melhora após a "
    "reavaliação devem ser discutidos com a equipe assistencial antes de mudar a estratégia.",
)


def _finalizar(texto: str, fonte: str) -> str:
    """Acrescenta a citação de fonte e o disclaimer de validação humana."""
    return f"{texto}\n\n[Fonte: {fonte}] {DISCLAIMER}"


def gerar_protocolo(condicao: dict[str, str]) -> dict[str, str]:
    fonte = f"Protocolo:{condicao['cid']}"
    return {
        "instruction": f"Descreva o protocolo institucional para {condicao['nome']}.",
        "input": f"Condição: {condicao['nome']} (CID-10 {condicao['cid']}).",
        "output": _finalizar(
            f"Protocolo de {condicao['nome']} (CID-10 {condicao['cid']}). "
            f"Avaliação: solicitar {condicao['exame']} para confirmação diagnóstica. "
            f"Conduta: {condicao['conduta']}. "
            f"Procedimento associado: {condicao['procedimento']}. "
            f"Registrar a evolução no prontuário do [PACIENTE] a cada reavaliação.",
            fonte,
        ),
        "source": fonte,
    }


def gerar_faq(
    condicao: dict[str, str], template_pergunta: str, rng: random.Random
) -> dict[str, str]:
    fonte = f"FAQ:{condicao['cid']}"
    pergunta = template_pergunta.format(**condicao)
    resposta = rng.choice(RESPOSTAS_FAQ).format(**condicao)
    return {
        "instruction": pergunta,
        "input": f"Dúvida frequente da equipe médica sobre {condicao['nome']}.",
        "output": _finalizar(resposta, fonte),
        "source": fonte,
    }


def gerar_laudo(condicao: dict[str, str]) -> dict[str, str]:
    fonte = f"Laudo:{condicao['cid']}"
    return {
        "instruction": f"Redija um modelo de laudo de {condicao['exame']}.",
        "input": f"Exame: {condicao['exame']}. Hipótese: {condicao['nome']} (CID-10 {condicao['cid']}).",
        "output": _finalizar(
            f"Laudo de {condicao['exame']} realizado em [DATA]. "
            f"Paciente: [PACIENTE]. Solicitante: [MÉDICO]. "
            f"Achado: {condicao['achado']}. "
            f"Conclusão: achados compatíveis com {condicao['nome']} (CID-10 {condicao['cid']}), "
            f"a correlacionar com o quadro clínico.",
            fonte,
        ),
        "source": fonte,
    }


def gerar_receita(condicao: dict[str, str]) -> dict[str, str]:
    fonte = f"Receita:{condicao['cid']}"
    return {
        "instruction": f"Apresente o modelo de receita usado no tratamento de {condicao['nome']}.",
        "input": f"Condição: {condicao['nome']} (CID-10 {condicao['cid']}).",
        "output": _finalizar(
            f"Modelo de receita para {condicao['nome']} (CID-10 {condicao['cid']}). "
            f"Paciente: [PACIENTE]. Prescritor: [MÉDICO]. Data: [DATA]. "
            f"Item: {condicao['medicamento']} — uso conforme orientação, {condicao['posologia']}. "
            f"Este é um modelo de documento: a escolha do fármaco, a dose e a duração são "
            f"definidas pelo profissional responsável no atendimento.",
            fonte,
        ),
        "source": fonte,
    }


def gerar_procedimento(condicao: dict[str, str]) -> dict[str, str]:
    fonte = f"Procedimento:{condicao['cid']}"
    return {
        "instruction": f"Descreva o procedimento interno de {condicao['procedimento']}.",
        "input": f"Procedimento: {condicao['procedimento']}. Contexto: {condicao['nome']}.",
        "output": _finalizar(
            f"Procedimento interno: {condicao['procedimento']}, indicado no manejo de "
            f"{condicao['nome']} (CID-10 {condicao['cid']}). "
            f"Preparo: conferir identificação do [PACIENTE] e a prescrição do [MÉDICO]. "
            f"Execução: seguir a técnica padronizada pelo serviço e registrar intercorrências. "
            f"Registro: anotar data, hora e responsável no prontuário.",
            fonte,
        ),
        "source": fonte,
    }


def generate_synthetic(
    total: int = TOTAL_PADRAO, seed: int = SEED_PADRAO
) -> list[dict[str, str]]:
    """Gera `total` registros distribuídos entre os quatro tipos exigidos pelo enunciado.

    A geração é determinística por `seed` — sem isso, cada execução mudaria o dataset e as
    métricas de avaliação do modelo não seriam comparáveis entre rodadas.
    """
    rng = random.Random(seed)
    registros: list[dict[str, str]] = []

    # Um registro de cada tipo por condição garante cobertura de todos os CID-10 do catálogo.
    for condicao in CONDICOES:
        registros.append(gerar_protocolo(condicao))
        registros.append(gerar_laudo(condicao))
        registros.append(gerar_receita(condicao))
        registros.append(gerar_procedimento(condicao))

    # O restante vira FAQ. As combinações (condição, pergunta) são ENUMERADAS, não sorteadas:
    # sortear repetia pares e o `curator`, que remove duplicatas, descartaria ~20% do que
    # este módulo gera. Enumerar mantém todo registro gerado sobrevivendo à curadoria.
    combinacoes = [(c, p) for c in CONDICOES for p in PERGUNTAS_FAQ]
    rng.shuffle(combinacoes)
    for condicao, pergunta in combinacoes:
        if len(registros) >= total:
            break
        registros.append(gerar_faq(condicao, pergunta, rng))

    return registros[:total]


def main(caminho: Path = SAIDA_PADRAO, total: int = TOTAL_PADRAO) -> int:
    from collections import Counter

    registros = generate_synthetic(total=total)
    gravados = save_jsonl(registros, caminho)

    tipos = Counter(r["source"].split(":")[0] for r in registros)
    palavras = [len(r["output"].split()) for r in registros]
    print(f"Registros gravados: {gravados}")
    print(f"Arquivo: {caminho.relative_to(RAIZ)}")
    print(f"Tipos: {dict(tipos)}")
    print(f"Condições CID-10 cobertas: {len({r['source'].split(':')[1] for r in registros})}")
    print(f"Palavras na resposta: min={min(palavras)} media={sum(palavras)/len(palavras):.0f} max={max(palavras)}")
    return gravados


if __name__ == "__main__":
    main()
