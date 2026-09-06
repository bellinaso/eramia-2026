"""
Funções compartilhadas entre os experimentos (com e sem solver).

Este módulo não importa `ollama` nem nenhuma outra dependência pesada:
serve tanto para o experimento que consulta LLMs quanto para o que usa
apenas o provador de teoremas.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


FIELD_MAP = {
    "premises": "premises",
    "premises_fol": "premises-FOL",
    "conclusion": "conclusion",
    "conclusion_fol": "conclusion-FOL",
    "label": "label",
}


def _normalizar_texto(valor: Any) -> str:
    """Normaliza texto para comparação: minúsculas, sem acentos, sem espaços nas pontas."""
    if valor is None:
        return ""
    texto = str(valor).strip().lower()
    substituicoes = {
        "á": "a", "à": "a", "ã": "a", "â": "a",
        "é": "e", "ê": "e",
        "í": "i",
        "ó": "o", "õ": "o", "ô": "o",
        "ú": "u",
    }
    for original, substituto in substituicoes.items():
        texto = texto.replace(original, substituto)
    return texto


def _mapear_para_categoria(valor: Any) -> Optional[str]:
    """
    Mapeia diferentes representações de veredito/label para uma categoria
    canônica de 3 vias:
        "valido"        -> conclusão válida / label positivo
        "invalido"      -> conclusão inválida / label negativo
        "indeterminado" -> conclusão indeterminada / label "incerto"
        None            -> formato não reconhecido (nem sequer uma das
                            3 categorias acima) — só isso conta como
                            "não foi possível comparar".
    """
    texto = _normalizar_texto(valor)

    validos = {"valido", "true", "verdadeiro", "1", "entailment", "yes", "sim"}
    invalidos = {"invalido", "false", "falso", "0", "contradiction", "no", "nao"}
    indeterminados = {
        "indeterminado", "incerto", "uncertain", "unknown",
        "neutral", "desconhecido",
    }

    if texto in validos:
        return "valido"
    if texto in invalidos:
        return "invalido"
    if texto in indeterminados:
        return "indeterminado"
    return None


def calcular_acertou(veredito: Any, label: Any) -> Optional[bool]:
    """
    Compara o veredito com o label de referência do dataset e retorna:
        True  -> acertou (inclui ambos caindo em "indeterminado")
        False -> errou
        None  -> veredito e/ou label em formato não reconhecido
    """
    categoria_veredito = _mapear_para_categoria(veredito)
    categoria_label = _mapear_para_categoria(label)

    if categoria_veredito is None or categoria_label is None:
        return None

    return categoria_veredito == categoria_label


def sanitizar_nome_arquivo(nome_modelo: str) -> str:
    """Transforma o nome do modelo (ex.: 'qwen3.5:9b') em um nome de arquivo seguro."""
    nome = nome_modelo.strip().lower()
    for caractere in [":", "/", "\\", " "]:
        nome = nome.replace(caractere, "_")
    return nome


def iter_json_records(file_obj):
    """
    Lê registros JSON de um arquivo de forma tolerante ao formato: aceita
    tanto .jsonl "de verdade" (um objeto compacto por linha) quanto
    arquivos com objetos JSON formatados em várias linhas.

    Gera tuplas (numero_do_registro, objeto_python, texto_bruto_do_registro).
    """
    text = file_obj.read()
    decoder = json.JSONDecoder()
    idx = 0
    length = len(text)
    record_number = 0

    while idx < length:
        # pula espaços em branco / quebras de linha entre registros
        while idx < length and text[idx] in " \t\r\n":
            idx += 1
        if idx >= length:
            break

        start = idx
        try:
            obj, end_idx = decoder.raw_decode(text, idx)
        except json.JSONDecodeError as exc:
            record_number += 1
            raise RuntimeError(
                f"Não foi possível interpretar o JSON a partir do caractere "
                f"{start} (registro aproximado {record_number}): {exc}"
            ) from exc

        record_number += 1
        yield record_number, obj, text[start:end_idx]
        idx = end_idx


def salvar_resumo_simulacao(resumo: Dict[str, Any], resumo_path: str) -> None:
    """Anexa (append) uma linha de resumo ao arquivo de resumo da simulação."""
    with Path(resumo_path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(resumo, ensure_ascii=False) + "\n")
