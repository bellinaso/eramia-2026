"""
Experimento 2: linha de base simbólica com Prover9 sobre o FOL de ouro.

Diferente do experimento 1 (`analisar_fol_ollama.py`), aqui NENHUM LLM é
consultado. As fórmulas em lógica de primeira ordem que já vêm no dataset
(`premises-FOL` / `conclusion-FOL`) são traduzidas para a sintaxe do
Prover9 e submetidas a um provador de teoremas de verdade.

O resultado mede o "teto simbólico" da tarefa: quanto dela é resolvível
quando a formalização já está pronta e correta.

Uso:
    python analisar_fol_prover9.py --apenas-traduzir   # valida o tradutor
    python analisar_fol_prover9.py --limite 5          # teste rápido
    python analisar_fol_prover9.py                     # execução completa
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fol_comum import (
    FIELD_MAP,
    calcular_acertou,
    iter_json_records,
    salvar_resumo_simulacao,
)


INPUT_FILE = "dataset.jsonl"
OUTPUT_FILE = "prover9_gold_fol.jsonl"
RESUMO_SIMULACAO_FILE = "resumo_with_solver.jsonl"

# Identificação desta execução no arquivo de resumo (não há modelo de LLM
# envolvido: o "modelo" aqui é o próprio provador sobre o FOL de ouro).
NOME_EXECUCAO = "gold_fol_prover9"

# Segundos por chamada de solver. Cada exemplo faz até 4 chamadas
# (2x Prover9 + 2x Mace4), então o pior caso é ~4x este valor.
TIMEOUT_SOLVER_SEGUNDOS = 10

# Maior tamanho de domínio que o Mace4 tenta ao procurar um contramodelo.
MACE4_TAMANHO_MAXIMO_DOMINIO = 10


# ---------------------------------------------------------------------------
# Tradução: Unicode FOL (FOLIO) -> sintaxe do Prover9
# ---------------------------------------------------------------------------
#
# O dataset usa notação matemática com operadores Unicode, que o Prover9 não
# entende. A tradução NÃO pode ser uma simples troca de caracteres, por três
# motivos levantados na análise das 1288 fórmulas do dataset:
#
#   1. `⊕` (ou-exclusivo) não existe no Prover9 e precisa virar
#      `-(A <-> B)`. Como ele aninha e se mistura com `→`, os limites dos
#      operandos dependem de precedência — daí o parser de verdade.
#   2. O Prover9 trata QUALQUER símbolo iniciado por `u`–`z` como variável.
#      17 constantes do dataset (yale, yuri, y1984, winter, ...) caem nessa
#      faixa e, sem renomeação, viram variáveis livres quantificadas
#      universalmente: o provador responde com confiança, e errado.
#   3. Alguns predicados contêm `’` (U+2019), que não é um caractere válido
#      de identificador.

QUANTIFICADORES = {"∀": "all", "∃": "exists"}

# Conectivos binários, do menor para o maior nível de precedência.
# `⊕` fica entre `∨` e `→`: é isso que faz `A ⊕ B → C ⊕ D` ser lido como
# `(A⊕B) → (C⊕D)`, que é o sentido pretendido no dataset.
IFF = {"↔", "⟷", "≡"}
IMP = {"→", "⇒"}
XOR = {"⊕"}
OR = {"∨"}
AND = {"∧"}
NOT = {"¬", "~"}

# Faixa de iniciais que o Prover9 reserva para variáveis.
INICIAIS_DE_VARIAVEL = "uvwxyz"

# Prefixo aplicado a constantes que cairiam na faixa de variáveis.
# `c` está fora de `u`–`z`, então a constante continua sendo constante.
PREFIXO_CONSTANTE = "c_"


class ErroDeTraducao(Exception):
    """Fórmula que não pôde ser interpretada nem reparada."""


def _tokenizar(formula: str) -> List[str]:
    """Quebra a fórmula em tokens (símbolos, identificadores, parênteses)."""
    tokens: List[str] = []
    i = 0
    n = len(formula)

    while i < n:
        c = formula[i]

        if c.isspace():
            i += 1
            continue

        if c in "(),":
            tokens.append(c)
            i += 1
            continue

        if c in QUANTIFICADORES or c in IFF or c in IMP or c in XOR \
                or c in OR or c in AND or c in NOT:
            tokens.append(c)
            i += 1
            continue

        # Identificador: letras, dígitos, `_` e também os caracteres
        # "sujos" (como `’`) que serão removidos na etapa de limpeza.
        if c.isalnum() or c == "_" or not c.isascii():
            inicio = i
            while i < n:
                d = formula[i]
                if d.isspace() or d in "()," or d in QUANTIFICADORES or d in IFF \
                        or d in IMP or d in XOR or d in OR or d in AND or d in NOT:
                    break
                i += 1
            tokens.append(formula[inicio:i])
            continue

        raise ErroDeTraducao(f"caractere inesperado {c!r} na posição {i}")

    return tokens


class _Parser:
    """
    Parser descendente recursivo. Precedência, do menor para o maior:
        ↔  <  →  <  ⊕  <  ∨  <  ∧  <  ¬  <  quantificador  <  átomo
    """

    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.pos = 0
        # Marca se foi preciso ler uma vírgula solta como conjunção.
        self.usou_virgula_como_conjuncao = False

    def _atual(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consumir(self, esperado: Optional[str] = None) -> str:
        token = self._atual()
        if token is None:
            raise ErroDeTraducao("fim inesperado da fórmula")
        if esperado is not None and token != esperado:
            raise ErroDeTraducao(f"esperava {esperado!r}, encontrei {token!r}")
        self.pos += 1
        return token

    def analisar(self) -> Tuple:
        arvore = self._iff()
        if self.pos != len(self.tokens):
            raise ErroDeTraducao(f"sobrou conteúdo a partir de {self._atual()!r}")
        return arvore

    def _iff(self) -> Tuple:
        esquerda = self._imp()
        while self._atual() in IFF:
            self._consumir()
            esquerda = ("iff", esquerda, self._imp())
        return esquerda

    def _imp(self) -> Tuple:
        esquerda = self._xor()
        if self._atual() in IMP:
            self._consumir()
            # implicação associa à direita: A → B → C  ==  A → (B → C)
            return ("imp", esquerda, self._imp())
        return esquerda

    def _xor(self) -> Tuple:
        esquerda = self._or()
        while self._atual() in XOR:
            self._consumir()
            esquerda = ("xor", esquerda, self._or())
        return esquerda

    def _or(self) -> Tuple:
        esquerda = self._and()
        while self._atual() in OR:
            self._consumir()
            esquerda = ("or", esquerda, self._and())
        return esquerda

    def _and(self) -> Tuple:
        esquerda = self._unario()
        while self._atual() in AND or self._atual() == ",":
            # Outro defeito conhecido do FOLIO: vírgula no lugar de `∧`
            # (ex.: `∀x ∀y (SuperheroMovie(x), NamedAfter(x, y) → ...)`).
            # Vírgulas legítimas de lista de argumentos são consumidas em
            # `_argumentos`, que nunca chega até aqui — então uma vírgula
            # neste ponto só pode ser conjunção.
            if self._consumir() == ",":
                self.usou_virgula_como_conjuncao = True
            esquerda = ("and", esquerda, self._unario())
        return esquerda

    def _unario(self) -> Tuple:
        token = self._atual()

        if token in NOT:
            self._consumir()
            return ("not", self._unario())

        if token in QUANTIFICADORES:
            self._consumir()
            variaveis = []
            # `∀x y (...)` — pode haver mais de uma variável antes do corpo
            while self._atual() is not None and self._e_identificador(self._atual()):
                variaveis.append(self._consumir())
            if not variaveis:
                raise ErroDeTraducao(f"quantificador {token!r} sem variável")
            corpo = self._unario()
            for variavel in reversed(variaveis):
                corpo = (QUANTIFICADORES[token], variavel, corpo)
            return corpo

        if token == "(":
            self._consumir("(")
            interno = self._iff()
            self._consumir(")")
            return interno

        return self._atomo()

    @staticmethod
    def _e_identificador(token: str) -> bool:
        return token not in "()," and token[0] not in "()"

    def _atomo(self) -> Tuple:
        nome = self._consumir()
        if not self._e_identificador(nome):
            raise ErroDeTraducao(f"esperava um átomo, encontrei {nome!r}")

        argumentos = self._argumentos()
        return ("pred", nome, argumentos)

    def _argumentos(self) -> List[Tuple]:
        """Lê `(t1, t2, ...)` se houver; devolve [] para símbolos de aridade 0."""
        if self._atual() != "(":
            return []
        self._consumir("(")
        argumentos = []
        if self._atual() != ")":
            argumentos.append(self._termo())
            while self._atual() == ",":
                self._consumir(",")
                argumentos.append(self._termo())
        self._consumir(")")
        return argumentos

    def _termo(self) -> Tuple:
        nome = self._consumir()
        if not self._e_identificador(nome):
            raise ErroDeTraducao(f"esperava um termo, encontrei {nome!r}")
        return ("termo", nome, self._argumentos())


def _limpar_identificador(nome: str) -> str:
    """Remove tudo que não é `[A-Za-z0-9_]` (mata o `’` dos predicados)."""
    limpo = "".join(c for c in nome if c.isascii() and (c.isalnum() or c == "_"))
    if not limpo:
        raise ErroDeTraducao(f"identificador vazio após limpeza: {nome!r}")
    if limpo[0].isdigit():
        limpo = "s_" + limpo
    return limpo


def _nome_de_variavel(profundidade: int) -> str:
    """Nomes garantidamente dentro da faixa de variáveis do Prover9."""
    base = ["x", "y", "z"]
    if profundidade < len(base):
        return base[profundidade]
    return f"x{profundidade}"


def _sanitizar(no: Tuple, escopo: Dict[str, str]) -> Tuple:
    """
    Reescreve a árvore deixando todos os identificadores válidos para o
    Prover9. `escopo` mapeia variável original -> variável renomeada, e é
    o que distingue uma variável ligada de uma constante.
    """
    tipo = no[0]

    if tipo in ("all", "exists"):
        _, variavel, corpo = no
        novo_nome = _nome_de_variavel(len(escopo))
        novo_escopo = dict(escopo)
        novo_escopo[variavel] = novo_nome
        return (tipo, novo_nome, _sanitizar(corpo, novo_escopo))

    if tipo == "not":
        return ("not", _sanitizar(no[1], escopo))

    if tipo in ("and", "or", "imp", "iff", "xor"):
        return (tipo, _sanitizar(no[1], escopo), _sanitizar(no[2], escopo))

    if tipo == "pred":
        _, nome, argumentos = no
        # Um símbolo seguido de `(` nunca é lido como variável pelo
        # Prover9, então só predicados de aridade 0 correm risco.
        nome_limpo = _limpar_identificador(nome)
        if not argumentos and nome_limpo[0] in INICIAIS_DE_VARIAVEL:
            nome_limpo = PREFIXO_CONSTANTE + nome_limpo
        return ("pred", nome_limpo, [_sanitizar(a, escopo) for a in argumentos])

    if tipo == "termo":
        _, nome, argumentos = no
        if not argumentos and nome in escopo:
            return ("termo", escopo[nome], [])
        nome_limpo = _limpar_identificador(nome)
        if not argumentos and nome_limpo[0] in INICIAIS_DE_VARIAVEL:
            # Constante que cairia na faixa de variáveis do Prover9.
            nome_limpo = PREFIXO_CONSTANTE + nome_limpo
        return ("termo", nome_limpo, [_sanitizar(a, escopo) for a in argumentos])

    raise ErroDeTraducao(f"nó desconhecido: {tipo!r}")


def _emitir(no: Tuple) -> str:
    """Converte a árvore já sanitizada em texto na sintaxe do Prover9."""
    tipo = no[0]

    if tipo in ("pred", "termo"):
        _, nome, argumentos = no
        if not argumentos:
            return nome
        return f"{nome}({', '.join(_emitir(a) for a in argumentos)})"

    if tipo == "not":
        return f"-({_emitir(no[1])})"

    if tipo == "and":
        return f"({_emitir(no[1])} & {_emitir(no[2])})"

    if tipo == "or":
        return f"({_emitir(no[1])} | {_emitir(no[2])})"

    if tipo == "imp":
        return f"({_emitir(no[1])} -> {_emitir(no[2])})"

    if tipo == "iff":
        return f"({_emitir(no[1])} <-> {_emitir(no[2])})"

    if tipo == "xor":
        # O Prover9 não tem ou-exclusivo: A ⊕ B  ==  -(A <-> B)
        return f"-(({_emitir(no[1])}) <-> ({_emitir(no[2])}))"

    if tipo == "all":
        return f"(all {no[1]} ({_emitir(no[2])}))"

    if tipo == "exists":
        return f"(exists {no[1]} ({_emitir(no[2])}))"

    raise ErroDeTraducao(f"nó desconhecido na emissão: {tipo!r}")


def _reparar_parenteses(formula: str) -> Tuple[str, bool]:
    """
    Conserta o defeito conhecido do FOLIO: um `)` sobrando no fim.
    Reparo deliberadamente conservador — qualquer outro desequilíbrio é
    deixado para falhar, em vez de adivinhado.
    """
    texto = formula.strip()
    if texto.count(")") == texto.count("(") + 1 and texto.endswith(")"):
        return texto[:-1].strip(), True
    return texto, False


def traduzir_formula(formula: str) -> Dict[str, Any]:
    """
    Traduz uma fórmula FOL Unicode para a sintaxe do Prover9.

    Devolve {"original", "prover9", "reparado", "reparos"}. `reparos` lista
    os defeitos do dataset que precisaram ser contornados, para que nenhum
    conserto fique escondido.

    Levanta ErroDeTraducao se a fórmula não puder ser interpretada.
    """
    reparos: List[str] = []

    texto, parentese_reparado = _reparar_parenteses(formula)
    if parentese_reparado:
        reparos.append("parentese_extra")

    parser = _Parser(_tokenizar(texto))
    arvore = parser.analisar()
    if parser.usou_virgula_como_conjuncao:
        reparos.append("virgula_como_conjuncao")

    return {
        "original": formula,
        "prover9": _emitir(_sanitizar(arvore, {})),
        "reparado": bool(reparos),
        "reparos": reparos,
    }


def traduzir_exemplo(exemplo: Dict[str, Any]) -> Dict[str, Any]:
    """Traduz as premissas e a conclusão de um registro do dataset."""
    premissas = exemplo.get(FIELD_MAP["premises_fol"]) or []
    if isinstance(premissas, str):
        premissas = [premissas]

    conclusao = exemplo.get(FIELD_MAP["conclusion_fol"])
    if not conclusao:
        raise ErroDeTraducao("registro sem 'conclusion-FOL'")

    return {
        "premissas": [traduzir_formula(p) for p in premissas],
        "conclusao": traduzir_formula(conclusao),
    }


# ---------------------------------------------------------------------------
# Execução dos solvers
# ---------------------------------------------------------------------------

def _montar_entrada(
    premissas: List[str],
    objetivo: Optional[str],
    timeout: int,
    para_mace4: bool = False,
) -> str:
    """Monta o arquivo de entrada no formato LADR (Prover9/Mace4)."""
    linhas = [f"assign(max_seconds, {timeout})."]
    if para_mace4:
        linhas.append(f"assign(end_size, {MACE4_TAMANHO_MAXIMO_DOMINIO}).")
    linhas.append("")
    linhas.append("formulas(assumptions).")
    linhas.extend(f"  {p}." for p in premissas)
    linhas.append("end_of_list.")

    if objetivo is not None:
        linhas.append("")
        linhas.append("formulas(goals).")
        linhas.append(f"  {objetivo}.")
        linhas.append("end_of_list.")

    return "\n".join(linhas) + "\n"


def _executar_solver(
    binario: str,
    conteudo: str,
    diretorio: Path,
    nome: str,
    timeout: int,
) -> Dict[str, Any]:
    """
    Roda Prover9 ou Mace4 sobre `conteudo` e devolve o código de saída.

    Códigos do LADR usados aqui:
        0 -> prova encontrada (Prover9) / modelo encontrado (Mace4)
        1 -> erro fatal, normalmente sintaxe inválida
        2 -> busca esgotada sem sucesso (resultado definitivo)
        4 -> estourou max_seconds
    """
    arquivo = diretorio / f"{nome}.in"
    arquivo.write_text(conteudo, encoding="utf-8")

    try:
        processo = subprocess.run(
            [binario, "-f", str(arquivo)],
            capture_output=True,
            text=True,
            # margem sobre o max_seconds interno, como rede de segurança
            timeout=timeout + 15,
        )
        codigo = processo.returncode
        saida = processo.stdout
    except subprocess.TimeoutExpired:
        codigo = 4
        saida = ""

    status = {
        0: "sucesso",
        1: "erro_fatal",
        2: "esgotado",
        3: "limite_memoria",
        4: "timeout",
    }.get(codigo, f"codigo_{codigo}")

    resultado = {"codigo_saida": codigo, "status": status}

    if codigo == 1:
        # Sintaxe inválida quase sempre significa bug no tradutor, não uma
        # propriedade do dataset — por isso o erro é preservado.
        resultado["erro"] = saida.strip()[-500:]

    return resultado


def resolver_exemplo(
    traducao: Dict[str, Any],
    diretorio: Path,
    timeout: int,
    prover9_bin: str,
    mace4_bin: str,
) -> Dict[str, Any]:
    """
    Decide o veredito de 3 vias para um exemplo já traduzido.

    Protocolo:
        1. Prover9 tenta provar a conclusão          -> Válido
        2. Prover9 tenta provar a negação dela       -> Inválido
        3. nenhum provou: Mace4 procura contramodelo -> Indeterminado
    """
    premissas = [p["prover9"] for p in traducao["premissas"]]
    conclusao = traducao["conclusao"]["prover9"]
    negacao = f"-({conclusao})"

    prova_conclusao = _executar_solver(
        prover9_bin,
        _montar_entrada(premissas, conclusao, timeout),
        diretorio, "p9_conclusao", timeout,
    )
    prova_negacao = _executar_solver(
        prover9_bin,
        _montar_entrada(premissas, negacao, timeout),
        diretorio, "p9_negacao", timeout,
    )

    provou_conclusao = prova_conclusao["codigo_saida"] == 0
    provou_negacao = prova_negacao["codigo_saida"] == 0

    analise: Dict[str, Any] = {
        "prover9_conclusao": prova_conclusao,
        "prover9_negacao": prova_negacao,
        "mace4_conclusao": None,
        "mace4_negacao": None,
        "premissas_inconsistentes": False,
        "confianca": "alta",
    }

    if provou_conclusao and provou_negacao:
        # As premissas provam tudo: o problema é o dataset, não a resposta.
        analise["veredito"] = None
        analise["motivo"] = "premissas_inconsistentes"
        analise["premissas_inconsistentes"] = True
        return analise

    if provou_conclusao:
        analise["veredito"] = "Válido"
        analise["motivo"] = "prova_encontrada"
        return analise

    if provou_negacao:
        analise["veredito"] = "Inválido"
        analise["motivo"] = "prova_da_negacao"
        return analise

    # Nenhuma direção foi provada. O Mace4 confirma se "Indeterminado" é
    # real (existe modelo para os dois lados) ou apenas falta de tempo.
    modelo_conclusao = _executar_solver(
        mace4_bin,
        _montar_entrada(premissas + [conclusao], None, timeout, para_mace4=True),
        diretorio, "m4_conclusao", timeout,
    )
    modelo_negacao = _executar_solver(
        mace4_bin,
        _montar_entrada(premissas + [negacao], None, timeout, para_mace4=True),
        diretorio, "m4_negacao", timeout,
    )
    analise["mace4_conclusao"] = modelo_conclusao
    analise["mace4_negacao"] = modelo_negacao

    achou_os_dois_modelos = (
        modelo_conclusao["codigo_saida"] == 0 and modelo_negacao["codigo_saida"] == 0
    )
    buscas_esgotadas = (
        prova_conclusao["codigo_saida"] == 2 and prova_negacao["codigo_saida"] == 2
    )

    analise["veredito"] = "Indeterminado"
    if achou_os_dois_modelos:
        analise["motivo"] = "contramodelos_confirmados"
    elif buscas_esgotadas:
        analise["motivo"] = "busca_esgotada"
    else:
        # Pode ser indeterminado de verdade, mas não foi demonstrado.
        analise["motivo"] = "timeout"
        analise["confianca"] = "baixa"

    return analise


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------

# Pastas onde o Prover9 costuma ficar, além do PATH. A variável de ambiente
# PROVER9_BIN tem prioridade sobre todas.
DIRETORIOS_CANDIDATOS = [
    r"C:\Program Files (x86)\Prover9-Mace4\bin-win32",
    r"C:\Program Files\Prover9-Mace4\bin-win32",
    r"C:\Program Files (x86)\Prover9-Mace4\bin",
    r"C:\Program Files\Prover9-Mace4\bin",
    os.path.expanduser("~/LADR-2009-11A/bin"),
    "/usr/local/bin",
]


def _procurar_binario(nome: str) -> Optional[str]:
    """Procura o executável no PROVER9_BIN, no PATH e nas pastas conhecidas."""
    pasta_env = os.environ.get("PROVER9_BIN")
    diretorios = ([pasta_env] if pasta_env else []) + DIRETORIOS_CANDIDATOS

    for diretorio in diretorios:
        if not diretorio:
            continue
        achado = shutil.which(nome, path=diretorio)
        if achado:
            return achado

    # shutil.which já resolve o sufixo .exe no Windows via PATHEXT
    return shutil.which(nome)


def _localizar_solvers() -> Tuple[str, str]:
    """Garante que prover9 e mace4 estão acessíveis antes de começar."""
    prover9 = _procurar_binario("prover9")
    mace4 = _procurar_binario("mace4")

    faltando = [n for n, b in (("prover9", prover9), ("mace4", mace4)) if b is None]
    if faltando:
        if os.name == "nt":
            instrucoes = (
                "Instale o pacote Windows do Prover9-Mace4:\n\n"
                "  http://www.cs.unm.edu/~mccune/prover9/gui/"
                "Prover9-Mace4-v05-setup.exe\n\n"
                "Depois aponte para a pasta dos executáveis, por exemplo:\n\n"
                '  $env:PROVER9_BIN = "C:\\Program Files (x86)\\'
                'Prover9-Mace4\\bin-win32"\n'
            )
        else:
            instrucoes = (
                "O Prover9 não é mais empacotado no Ubuntu. Compile o LADR:\n\n"
                "  wget https://www.cs.unm.edu/~mccune/prover9/download/"
                "LADR-2009-11A.tar.gz\n"
                "  tar xzf LADR-2009-11A.tar.gz && cd LADR-2009-11A\n"
                '  make all XFLAGS="-std=gnu89 -w"\n'
                '  export PATH="$PWD/bin:$PATH"\n'
            )

        print(
            f"[erro] não encontrei: {', '.join(faltando)}\n\n{instrucoes}\n"
            "Para validar o tradutor sem instalar nada, use --apenas-traduzir.",
            file=sys.stderr,
        )
        sys.exit(1)

    return prover9, mace4


def _carregar_registros(caminho: str) -> List[Dict[str, Any]]:
    arquivo = Path(caminho)
    if not arquivo.exists():
        print(f"[erro] arquivo de entrada não encontrado: {arquivo}", file=sys.stderr)
        sys.exit(1)

    with arquivo.open("r", encoding="utf-8") as f:
        try:
            return [obj for _, obj, _ in iter_json_records(f)]
        except RuntimeError as exc:
            print(f"[erro] falha ao ler o arquivo de entrada: {exc}", file=sys.stderr)
            sys.exit(1)


def apenas_traduzir(registros: List[Dict[str, Any]], destino: Path) -> int:
    """
    Traduz tudo sem chamar solver algum. Serve para validar o tradutor
    antes de instalar o Prover9.
    """
    destino.mkdir(parents=True, exist_ok=True)
    caminho_jsonl = destino / "traducao_prover9.jsonl"

    total_formulas = 0
    reparadas = 0
    falhas = 0

    with caminho_jsonl.open("w", encoding="utf-8") as fout:
        for numero, exemplo in enumerate(registros, start=1):
            try:
                traducao = traduzir_exemplo(exemplo)
            except ErroDeTraducao as exc:
                falhas += 1
                print(f"  [falha] registro {numero}: {exc}", file=sys.stderr)
                fout.write(json.dumps(
                    {"registro": numero, "erro": str(exc)}, ensure_ascii=False
                ) + "\n")
                continue

            formulas = traducao["premissas"] + [traducao["conclusao"]]
            total_formulas += len(formulas)
            reparadas += sum(1 for f in formulas if f["reparado"])

            (destino / f"registro_{numero:03d}.in").write_text(
                _montar_entrada(
                    [p["prover9"] for p in traducao["premissas"]],
                    traducao["conclusao"]["prover9"],
                    TIMEOUT_SOLVER_SEGUNDOS,
                ),
                encoding="utf-8",
            )
            fout.write(json.dumps(
                {"registro": numero, **traducao}, ensure_ascii=False
            ) + "\n")

    print("\n=== Tradução (sem solver) ===")
    print(f"Registros:                 {len(registros)}")
    print(f"Fórmulas traduzidas:       {total_formulas}")
    print(f"Fórmulas reparadas:        {reparadas}")
    print(f"Falhas de tradução:        {falhas}")
    print(f"Traduções salvas em:       {caminho_jsonl}")
    print(f"Arquivos .in salvos em:    {destino}")

    return 1 if falhas else 0


def processar(
    registros: List[Dict[str, Any]],
    output_path: str,
    timeout: int,
    prover9_bin: str,
    mace4_bin: str,
) -> Dict[str, Any]:
    """Roda o protocolo completo sobre todos os registros."""
    total = 0
    acertos = 0
    erros = 0
    nao_comparaveis = 0
    falhas_processamento = 0
    falhas_traducao = 0
    timeouts = 0
    inconsistentes = 0

    inicio = datetime.now()

    with Path(output_path).open("w", encoding="utf-8") as fout, \
            tempfile.TemporaryDirectory(prefix="prover9_") as tmp:

        diretorio = Path(tmp)

        for numero, exemplo in enumerate(registros, start=1):
            total += 1
            print(f"[{numero}/{len(registros)}] resolvendo...")

            registro: Dict[str, Any] = dict(exemplo)

            try:
                traducao = traduzir_exemplo(exemplo)
                analise = resolver_exemplo(
                    traducao, diretorio, timeout, prover9_bin, mace4_bin
                )

                label = exemplo.get(FIELD_MAP["label"])
                acertou = calcular_acertou(analise.get("veredito"), label)
                analise["acertou"] = acertou
                analise["traducao"] = traducao

                if acertou is True:
                    acertos += 1
                elif acertou is False:
                    erros += 1
                else:
                    nao_comparaveis += 1

                if analise.get("motivo") == "timeout":
                    timeouts += 1
                if analise.get("premissas_inconsistentes"):
                    inconsistentes += 1

                registro["analise_solver"] = analise
                registro["erro"] = None

            except ErroDeTraducao as exc:
                print(f"  [erro] tradução do registro {numero}: {exc}", file=sys.stderr)
                registro["analise_solver"] = None
                registro["erro"] = f"traducao: {exc}"
                falhas_traducao += 1
                falhas_processamento += 1
            except Exception as exc:
                print(f"  [erro] registro {numero}: {exc}", file=sys.stderr)
                registro["analise_solver"] = None
                registro["erro"] = str(exc)
                falhas_processamento += 1

            fout.write(json.dumps(registro, ensure_ascii=False) + "\n")
            fout.flush()

    fim = datetime.now()
    duracao = (fim - inicio).total_seconds()
    comparaveis = acertos + erros
    acuracia = (acertos / comparaveis) if comparaveis > 0 else None

    # As chaves seguintes repetem, na mesma ordem, o esquema do resumo
    # do experimento 1, para as linhas ficarem comparáveis.
    resumo = {
        "modelo": NOME_EXECUCAO,
        "horario_inicio": inicio.isoformat(timespec="seconds"),
        "horario_fim": fim.isoformat(timespec="seconds"),
        "duracao_total_segundos": round(duracao, 2),
        "total_exemplos": total,
        "acertos": acertos,
        "erros": erros,
        "nao_comparaveis": nao_comparaveis,
        "falhas_processamento": falhas_processamento,
        "acuracia": round(acuracia, 4) if acuracia is not None else None,
        "falhas_traducao": falhas_traducao,
        "timeouts": timeouts,
        "premissas_inconsistentes": inconsistentes,
    }

    print("\n=== Resumo: Prover9 sobre o FOL de ouro ===")
    print(f"Total de exemplos:         {total}")
    print(f"Acertos:                   {acertos}")
    print(f"Erros:                     {erros}")
    print(f"Não comparáveis:           {nao_comparaveis}")
    print(f"Acurácia (comparáveis):    {acuracia:.2%}" if acuracia is not None else "Acurácia: N/A")
    print(f"Falhas de tradução:        {falhas_traducao}")
    print(f"Indeterminados por timeout:{timeouts:>4}")
    print(f"Premissas inconsistentes:  {inconsistentes}")
    print(f"Tempo total:               {duracao:.1f}s")
    print(f"Resultados salvos em:      {Path(output_path).resolve()}")

    return resumo


def configurar_saida_utf8() -> None:
    """
    Força UTF-8 em stdout/stderr.

    No Windows a saída redirecionada usa a codificação local (cp1252), que
    não tem ∀, → nem ∧. Uma mensagem de erro citando a fórmula original
    derrubaria a execução inteira — justamente o que não pode acontecer
    numa rodada que passa a noite.
    """
    for fluxo in (sys.stdout, sys.stderr):
        try:
            fluxo.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main() -> None:
    configurar_saida_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apenas-traduzir", action="store_true",
        help="traduz tudo e escreve os .in, sem chamar o solver",
    )
    parser.add_argument("--limite", type=int, help="processa apenas os N primeiros registros")
    parser.add_argument(
        "--timeout", type=int, default=TIMEOUT_SOLVER_SEGUNDOS,
        help=f"segundos por chamada de solver (padrão: {TIMEOUT_SOLVER_SEGUNDOS})",
    )
    parser.add_argument(
        "--saida-traducao", default="traducao_prover9",
        help="diretório para os arquivos de --apenas-traduzir",
    )
    args = parser.parse_args()

    registros = _carregar_registros(INPUT_FILE)
    if args.limite:
        registros = registros[: args.limite]

    if args.apenas_traduzir:
        sys.exit(apenas_traduzir(registros, Path(args.saida_traducao)))

    prover9_bin, mace4_bin = _localizar_solvers()

    # Execução de teste (--limite) não sobrescreve nem o arquivo de
    # resultados nem o resumo da execução real.
    saida = OUTPUT_FILE if not args.limite else \
        OUTPUT_FILE.replace(".jsonl", f"_teste{args.limite}.jsonl")

    resumo = processar(registros, saida, args.timeout, prover9_bin, mace4_bin)

    if args.limite:
        print("(execução de teste: resumo não gravado)")
    else:
        salvar_resumo_simulacao(resumo, RESUMO_SIMULACAO_FILE)
        print(f"Resumo anexado em:         {Path(RESUMO_SIMULACAO_FILE).resolve()}")


if __name__ == "__main__":
    main()
