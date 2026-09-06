"""
Experimento 3: pipeline neuro-simbólico (LLM -> FOL -> Prover9).

Cada modelo lê as premissas e a conclusão em linguagem natural e produz a
formalização em lógica de primeira ordem. Essa formalização — e não a do
dataset — é submetida ao Prover9. É o contraponto direto do experimento 1:
mesmos modelos, mesmos exemplos, mudando só quem decide.

Em cada exemplo o modelo é consultado duas vezes:
    1. braço COM solver .. gera o FOL, que o Prover9 resolve
    2. braço CONTROLE ..... responde direto, exatamente como no experimento 1 (reaproveita o mesmo prompt e a mesma função de consulta, para o controle ser de fato idêntico e não apenas parecido)

A execução é retomável: se cair no meio da madrugada, basta rodar de novo
que ela continua de onde parou.

Uso:
    python analisar_fol_llm_prover9.py                 # os 3 modelos
    python analisar_fol_llm_prover9.py --limite 5      # teste rápido
    python analisar_fol_llm_prover9.py --modelos gemma4:12b
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple

import ollama

from fol_comum import (
    FIELD_MAP,
    calcular_acertou,
    iter_json_records,
    salvar_resumo_simulacao,
    sanitizar_nome_arquivo,
)
from analisar_fol_prover9 import (
    ErroDeTraducao,
    _localizar_solvers,
    configurar_saida_utf8,
    resolver_exemplo,
    traduzir_formula,
)
# O braço de controle reaproveita as funções do experimento 1 tal como
# estão: é isso que garante que a comparação com/sem solver seja legítima.
from analisar_fol_ollama import build_prompt, parse_model_output, query_ollama


MODELOS: List[str] = [
    "gemma4:12b",
    "qwen3.5:9b",
    "llama3.2:3b",
]

# Pode ser sobrescrito por variável de ambiente ou por --host. Útil quando o
# Ollama roda no Windows e o script roda no WSL.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
INPUT_FILE = "dataset.jsonl"
RESUMO_SIMULACAO_FILE = "resumo_with_solver.jsonl"

INTERVALO_ENTRE_MODELOS_SEGUNDOS = 10 * 60

# Tentativas de formalização por exemplo. A cada falha o erro é devolvido
# ao modelo para ele corrigir. Esgotadas as tentativas, conta como erro —
# nada de cair para a resposta em linguagem natural.
MAX_TENTATIVAS_FOL = 3

TIMEOUT_SOLVER_SEGUNDOS = 10

# Retentativas por falha de comunicação com o Ollama (independentes das
# tentativas de formalização).
MAX_RETENTATIVAS_REDE = 3
RETENTATIVA_REDE_SEGUNDOS = 5

TEMPERATURE = 0.0
NUM_PREDICT = 2048
THINK = False


SYSTEM_PROMPT_FOL = (
    "Você é um especialista em lógica de primeira ordem. Sua tarefa é "
    "traduzir enunciados em linguagem natural para fórmulas de lógica de "
    "primeira ordem, de forma precisa e literal. Você não resolve o "
    "problema nem opina sobre a resposta: apenas formaliza."
)

PROMPT_FOL = """\
Traduza as PREMISSAS e a CONCLUSÃO abaixo para lógica de primeira ordem.

Use EXATAMENTE esta notação:
    ∀x  para todo          ∃x  existe
    →   implica            ↔   se e somente se
    ∧   e                  ∨   ou
    ¬   não                ⊕   ou exclusivo

Regras:
- Predicados em CamelCase, iniciando com letra maiúscula: Estudante(x)
- Constantes (nomes próprios) em minúscula: joao
- Variáveis: apenas x, y, z
- Uma fórmula para cada premissa, na mesma ordem das premissas
- Use os MESMOS nomes de predicado nas premissas e na conclusão
- Não escreva explicações, apenas as fórmulas

Exemplo:
    Premissas:
    1. Todo pássaro voa.
    2. Piu é um pássaro.
    Conclusão: Piu voa.
    Resposta:
    {{"premissas_fol": ["∀x (Passaro(x) → Voa(x))", "Passaro(piu)"], \
"conclusao_fol": "Voa(piu)"}}

PREMISSAS:
{premises}

CONCLUSÃO:
{conclusion}

Responda ESTRITAMENTE no formato JSON abaixo, sem nenhum texto antes ou depois:

{{"premissas_fol": ["<fórmula 1>", "<fórmula 2>", ...], "conclusao_fol": "<fórmula>"}}
"""

PROMPT_CORRECAO = """\
A formalização que você produziu falhou ao ser processada.

Erro: {erro}
Fórmula com problema: {formula}

Corrija o problema e responda de novo, no mesmo formato JSON, com todas as
premissas e a conclusão.
"""


def listar_modelos(client: ollama.Client) -> set:
    """
    Nomes dos modelos disponíveis.

    O formato da resposta mudou entre versões da biblioteca (dicionário nas
    antigas, objeto Pydantic a partir da 0.4), então os dois são aceitos.
    """
    resposta = client.list()
    modelos = resposta.get("models", []) if isinstance(resposta, dict) \
        else getattr(resposta, "models", [])

    nomes = set()
    for item in modelos:
        if isinstance(item, dict):
            nome = item.get("name") or item.get("model")
        else:
            nome = getattr(item, "model", None) or getattr(item, "name", None)
        if nome:
            nomes.add(nome)
    return nomes


def _formatar_lista(valor: Any) -> str:
    """Formata premissas como itens numerados (mesmo estilo do experimento 1)."""
    if valor in (None, "", []):
        return "(não informado)"
    if isinstance(valor, list):
        return "\n".join(f"{i}. {item}" for i, item in enumerate(valor, start=1))
    return str(valor)


def montar_prompt_fol(exemplo: Dict[str, Any]) -> str:
    return PROMPT_FOL.format(
        premises=_formatar_lista(exemplo.get(FIELD_MAP["premises"])),
        conclusion=_formatar_lista(exemplo.get(FIELD_MAP["conclusion"])),
    )


def consultar_modelo(
    client: ollama.Client, modelo: str, mensagens: List[Dict[str, str]]
) -> str:
    """
    Envia uma conversa ao modelo e devolve o texto da resposta.

    Repete em caso de falha de comunicação: numa execução que passa a noite
    inteira, um soluço do servidor não pode derrubar o registro.
    """
    ultimo_erro: Optional[Exception] = None

    for tentativa in range(1, MAX_RETENTATIVAS_REDE + 1):
        try:
            resposta = client.chat(
                model=modelo,
                messages=mensagens,
                format="json",
                think=THINK,
                options={"temperature": TEMPERATURE, "num_predict": NUM_PREDICT},
            )
            conteudo = resposta["message"]["content"]
            if not conteudo.strip():
                conteudo = getattr(resposta["message"], "thinking", None) or ""
            return conteudo
        except Exception as exc:
            ultimo_erro = exc
            print(f"    [rede {tentativa}/{MAX_RETENTATIVAS_REDE}] {exc}",
                file=sys.stderr, flush=True)
            if tentativa < MAX_RETENTATIVAS_REDE:
                time.sleep(RETENTATIVA_REDE_SEGUNDOS)

    raise RuntimeError(f"Ollama não respondeu após {MAX_RETENTATIVAS_REDE} tentativas: {ultimo_erro}")


def extrair_fol(texto: str) -> Tuple[List[str], str]:
    """Lê o JSON devolvido pelo modelo e valida o formato mínimo."""
    limpo = texto.strip()
    if limpo.startswith("```"):
        limpo = limpo.strip("`")
        if limpo.lower().startswith("json"):
            limpo = limpo[4:].strip()

    try:
        dados = json.loads(limpo)
    except json.JSONDecodeError as exc:
        raise ErroDeTraducao(f"resposta não é JSON válido: {exc}") from exc

    premissas = dados.get("premissas_fol")
    conclusao = dados.get("conclusao_fol")

    if isinstance(premissas, str):
        premissas = [premissas]
    if not isinstance(premissas, list) or not premissas:
        raise ErroDeTraducao("'premissas_fol' ausente ou vazio")
    if not isinstance(conclusao, str) or not conclusao.strip():
        raise ErroDeTraducao("'conclusao_fol' ausente ou vazio")

    return [str(p) for p in premissas], conclusao


def traduzir_saida_do_modelo(premissas: List[str], conclusao: str) -> Dict[str, Any]:
    """
    Passa o FOL gerado pelo modelo pelo mesmo tradutor usado no FOL de
    ouro. Em caso de erro, informa qual fórmula falhou, para o retorno
    ao modelo ser específico.
    """
    traduzidas = []
    for formula in premissas:
        try:
            traduzidas.append(traduzir_formula(formula))
        except ErroDeTraducao as exc:
            raise ErroDeTraducao(f"{exc} | fórmula: {formula}") from exc

    try:
        conclusao_traduzida = traduzir_formula(conclusao)
    except ErroDeTraducao as exc:
        raise ErroDeTraducao(f"{exc} | fórmula: {conclusao}") from exc

    return {"premissas": traduzidas, "conclusao": conclusao_traduzida}


def _houve_erro_de_sintaxe(analise: Dict[str, Any]) -> Optional[str]:
    """Detecta erro fatal do solver (código 1), que quase sempre é sintaxe."""
    for chave in ("prover9_conclusao", "prover9_negacao"):
        resultado = analise.get(chave)
        if resultado and resultado.get("status") == "erro_fatal":
            return resultado.get("erro", "erro fatal do Prover9")
    return None


def resolver_com_llm(
    client: ollama.Client,
    modelo: str,
    exemplo: Dict[str, Any],
    diretorio: Path,
    timeout: int,
    prover9_bin: str,
    mace4_bin: str,
) -> Dict[str, Any]:
    """
    Braço COM solver: o modelo formaliza, o Prover9 decide.

    Um veredito 'Indeterminado' NÃO dispara nova tentativa: 'Uncertain' é
    um rótulo legítimo em cerca de um terço do dataset, e reperguntar
    nesses casos empurraria o pipeline para Válido/Inválido.
    """
    mensagens = [
        {"role": "system", "content": SYSTEM_PROMPT_FOL},
        {"role": "user", "content": montar_prompt_fol(exemplo)},
    ]

    ultimo_erro = None
    bruto = ""

    for tentativa in range(1, MAX_TENTATIVAS_FOL + 1):
        try:
            bruto = consultar_modelo(client, modelo, mensagens)
            premissas_fol, conclusao_fol = extrair_fol(bruto)
            traducao = traduzir_saida_do_modelo(premissas_fol, conclusao_fol)

            analise = resolver_exemplo(
                traducao, diretorio, timeout, prover9_bin, mace4_bin
            )

            erro_sintaxe = _houve_erro_de_sintaxe(analise)
            if erro_sintaxe:
                raise ErroDeTraducao(f"Prover9 rejeitou a fórmula: {erro_sintaxe}")

            analise["tentativas"] = tentativa
            analise["fol_gerado"] = {
                "premissas": premissas_fol,
                "conclusao": conclusao_fol,
            }
            analise["traducao"] = traducao
            return analise

        except ErroDeTraducao as exc:
            ultimo_erro = str(exc)
            print(f"    [tentativa {tentativa}/{MAX_TENTATIVAS_FOL}] {exc}",
                file=sys.stderr, flush=True)
            if tentativa < MAX_TENTATIVAS_FOL:
                formula = ultimo_erro.split("| fórmula:")[-1].strip()
                mensagens.append({"role": "assistant", "content": bruto})
                mensagens.append({"role": "user", "content": PROMPT_CORRECAO.format(
                    erro=ultimo_erro, formula=formula)})

    return {
        "veredito": None,
        "motivo": "falha_de_formalizacao",
        "tentativas": MAX_TENTATIVAS_FOL,
        "erro": ultimo_erro,
    }


def resolver_controle(
    client: ollama.Client, modelo: str, exemplo: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Braço CONTROLE: pergunta direta em linguagem natural, com o mesmo
    prompt e a mesma função de consulta do experimento 1.
    """
    bruto = query_ollama(client, modelo, build_prompt(exemplo))
    return parse_model_output(bruto)


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

def _linhas_ja_processadas(caminho: Path) -> int:
    """Conta registros já gravados, ignorando uma última linha truncada."""
    if not caminho.exists():
        return 0
    validas = 0
    with caminho.open("r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            try:
                json.loads(linha)
            except json.JSONDecodeError:
                break
            validas += 1
    return validas


def processar_modelo(
    client: ollama.Client,
    modelo: str,
    registros: List[Dict[str, Any]],
    timeout: int,
    prover9_bin: str,
    mace4_bin: str,
    sufixo: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Roda os dois braços para um modelo e devolve (resumo_solver, resumo_controle).

    `sufixo` separa os arquivos de teste dos de valer: sem isso, um
    `--limite 5` deixaria 5 linhas no arquivo da execução real e a retomada
    pularia esses registros achando que já estavam prontos.
    """
    saida = Path(f"{sanitizar_nome_arquivo(modelo)}_with_solver{sufixo}.jsonl")
    ja_feitos = _linhas_ja_processadas(saida)

    if ja_feitos >= len(registros):
        print(f"[{modelo}] já concluído ({ja_feitos} registros) — pulando.", flush=True)
    elif ja_feitos:
        print(f"[{modelo}] retomando a partir do registro {ja_feitos + 1}.", flush=True)

    contagem = {
        "solver": {"acertos": 0, "erros": 0, "nao_comparaveis": 0},
        "controle": {"acertos": 0, "erros": 0, "nao_comparaveis": 0},
    }
    falhas_formalizacao = 0
    falhas_processamento = 0
    timeouts = 0
    tentativas_extras = 0
    tempos: List[float] = []

    horario_inicio = datetime.now()

    with saida.open("a", encoding="utf-8") as fout, \
            TemporaryDirectory(prefix="llm_prover9_") as tmp:

        diretorio = Path(tmp)

        for indice, exemplo in enumerate(registros, start=1):
            if indice <= ja_feitos:
                continue

            inicio = time.time()
            registro: Dict[str, Any] = dict(exemplo)
            label = exemplo.get(FIELD_MAP["label"])

            try:
                analise = resolver_com_llm(
                    client, modelo, exemplo, diretorio, timeout, prover9_bin, mace4_bin
                )
                acertou = calcular_acertou(analise.get("veredito"), label)
                analise["acertou"] = acertou

                if analise.get("motivo") == "falha_de_formalizacao":
                    falhas_formalizacao += 1
                if analise.get("motivo") == "timeout":
                    timeouts += 1
                tentativas_extras += analise.get("tentativas", 1) - 1

                if acertou is True:
                    contagem["solver"]["acertos"] += 1
                elif acertou is False:
                    contagem["solver"]["erros"] += 1
                else:
                    contagem["solver"]["nao_comparaveis"] += 1

                controle = resolver_controle(client, modelo, exemplo)
                acertou_controle = calcular_acertou(controle.get("veredito"), label)
                controle["acertou"] = acertou_controle

                if acertou_controle is True:
                    contagem["controle"]["acertos"] += 1
                elif acertou_controle is False:
                    contagem["controle"]["erros"] += 1
                else:
                    contagem["controle"]["nao_comparaveis"] += 1

                registro["analise_solver"] = analise
                registro["analise_controle"] = controle
                registro["erro"] = None

            except Exception as exc:
                print(f"  [erro] registro {indice}: {exc}", file=sys.stderr, flush=True)
                registro["analise_solver"] = None
                registro["analise_controle"] = None
                registro["erro"] = str(exc)
                falhas_processamento += 1

            decorrido = time.time() - inicio
            tempos.append(decorrido)
            fout.write(json.dumps(registro, ensure_ascii=False) + "\n")
            fout.flush()

            media = sum(tempos) / len(tempos)
            restantes = len(registros) - indice
            eta = timedelta(seconds=int(media * restantes))
            print(
                f"[{modelo}] [{indice}/{len(registros)}] {decorrido:.1f}s "
                f"| média {media:.1f}s | falta ~{eta}",
                flush=True,
            )

    horario_fim = datetime.now()

    def montar_resumo(chave: str, sufixo: str) -> Dict[str, Any]:
        c = contagem[chave]
        comparaveis = c["acertos"] + c["erros"]
        acuracia = (c["acertos"] / comparaveis) if comparaveis else None
        resumo = {
            "modelo": f"{sanitizar_nome_arquivo(modelo)}_{sufixo}",
            "horario_inicio": horario_inicio.isoformat(timespec="seconds"),
            "horario_fim": horario_fim.isoformat(timespec="seconds"),
            "duracao_total_segundos": round((horario_fim - horario_inicio).total_seconds(), 2),
            "total_exemplos": len(registros),
            "acertos": c["acertos"],
            "erros": c["erros"],
            "nao_comparaveis": c["nao_comparaveis"],
            "falhas_processamento": falhas_processamento,
            "acuracia": round(acuracia, 4) if acuracia is not None else None,
        }
        if chave == "solver":
            resumo.update({
                "falhas_formalizacao": falhas_formalizacao,
                "timeouts": timeouts,
                "tentativas_extras": tentativas_extras,
            })
        return resumo

    return montar_resumo("solver", "with_solver"), \
        montar_resumo("controle", "sem_solver_controle")


def main() -> None:
    # No Windows, sem isto, uma mensagem de erro citando uma fórmula com
    # ∀ ou → derruba a execução quando a saída vai para um arquivo.
    configurar_saida_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelos", nargs="+", help="sobrescreve a lista de modelos")
    parser.add_argument("--limite", type=int, help="processa apenas os N primeiros registros")
    parser.add_argument("--timeout", type=int, default=TIMEOUT_SOLVER_SEGUNDOS,
                        help="segundos por chamada de solver")
    parser.add_argument("--intervalo", type=int, default=INTERVALO_ENTRE_MODELOS_SEGUNDOS,
                        help="segundos de descanso entre modelos")
    parser.add_argument("--host", help=f"endereço do Ollama (padrão: {OLLAMA_HOST})")
    args = parser.parse_args()

    modelos = args.modelos or MODELOS
    prover9_bin, mace4_bin = _localizar_solvers()

    arquivo = Path(INPUT_FILE)
    if not arquivo.exists():
        print(f"[erro] arquivo de entrada não encontrado: {arquivo}", file=sys.stderr)
        sys.exit(1)
    with arquivo.open("r", encoding="utf-8") as f:
        registros = [obj for _, obj, _ in iter_json_records(f)]
    if args.limite:
        registros = registros[: args.limite]

    host = args.host or OLLAMA_HOST
    client = ollama.Client(host=host)
    try:
        disponiveis = listar_modelos(client)
    except Exception as exc:
        print(f"[erro] não consegui falar com o Ollama em {host}: {exc}\n"
            "Verifique se o servidor está no ar ('ollama serve') e, se ele\n"
            "roda no Windows enquanto este script roda no WSL, use --host\n"
            "ou a variável de ambiente OLLAMA_HOST.", file=sys.stderr)
        sys.exit(1)

    faltando = [m for m in modelos if m not in disponiveis]
    if faltando:
        print(f"[erro] modelos não encontrados no Ollama: {', '.join(faltando)}\n"
            f"Disponíveis: {', '.join(sorted(disponiveis)) or '(nenhum)'}\n"
            "Baixe com 'ollama pull <modelo>' ou ajuste --modelos.", file=sys.stderr)
        sys.exit(1)

    sufixo = f"_teste{args.limite}" if args.limite else ""
    if args.limite:
        print(f"[teste] arquivos com sufixo '{sufixo}'; o resumo não será gravado.",
            flush=True)

    print(f"=== {len(modelos)} modelo(s), {len(registros)} exemplos cada ===", flush=True)
    print(f"Início: {datetime.now():%Y-%m-%d %H:%M:%S}\n", flush=True)

    for indice, modelo in enumerate(modelos):
        print(f"\n=== {modelo} ({indice + 1}/{len(modelos)}) ===", flush=True)
        resumo_solver, resumo_controle = processar_modelo(
            client, modelo, registros, args.timeout, prover9_bin, mace4_bin,
            sufixo=sufixo
        )
        if args.limite:
            print("  (execução de teste: resumo não gravado)", flush=True)
        else:
            salvar_resumo_simulacao(resumo_solver, RESUMO_SIMULACAO_FILE)
            salvar_resumo_simulacao(resumo_controle, RESUMO_SIMULACAO_FILE)

        print(f"\n  com solver .. {resumo_solver['acuracia']}  "
            f"(falhas de formalização: {resumo_solver['falhas_formalizacao']})", flush=True)
        print(f"  controle .... {resumo_controle['acuracia']}", flush=True)

        if indice < len(modelos) - 1 and args.intervalo:
            print(f"\nDescanso de {args.intervalo // 60} min antes do próximo modelo...",
                flush=True)
            time.sleep(args.intervalo)

    print(f"\n=== Concluído em {datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    print(f"Resumos em: {Path(RESUMO_SIMULACAO_FILE).resolve()}", flush=True)


if __name__ == "__main__":
    main()
