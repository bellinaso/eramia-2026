#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analisar_fol_ollama.py

Lê um arquivo .jsonl contendo exemplos de raciocínio lógico (premissas,
versão em FOL das premissas, conclusão, versão em FOL da conclusão e
label), envia cada exemplo para um modelo local rodando no Ollama e pede
uma análise sobre a validade lógica da conclusão. Os resultados são
salvos, de forma incremental, em um novo arquivo .jsonl.

Requisitos:
    pip install ollama

Uso:
    python analisar_fol_ollama.py
    (ajuste as constantes de configuração abaixo conforme necessário)
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional
import ollama


MODEL_NAME = "qwen3.5:9b"
OLLAMA_HOST = "http://localhost:11434"

INPUT_FILE = "dataset.jsonl"
OUTPUT_FILE = "resultados_fol.jsonl"

FIELD_MAP = {
    "premises": "premises",
    "premises_fol": "premises-FOL",
    "conclusion": "conclusion",
    "conclusion_fol": "conclusion-FOL",
    "label": "label",
}

# Parâmetros de geração do modelo
TEMPERATURE = 0.0
NUM_PREDICT = 2048
REQUEST_TIMEOUT = 120

THINK = False

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 3

DEBUG = False

SYSTEM_PROMPT = (
    "Você é um assistente especialista em lógica formal e lógica de primeira "
    "ordem (FOL). Sua tarefa é analisar se uma conclusão decorre "
    "logicamente de um conjunto de premissas, usando tanto a versão em "
    "linguagem natural quanto a versão formal (FOL) quando disponíveis. "
    "Seja rigoroso, objetivo e sempre responda no formato solicitado."
)

PROMPT_TEMPLATE = """
Analise o exemplo de raciocínio lógico abaixo e determine se a CONCLUSÃO é
válida a partir das PREMISSAS fornecidas.

PREMISSAS (linguagem natural):
{premises}

PREMISSAS (FOL):
{premises_fol}

CONCLUSÃO (linguagem natural):
{conclusion}

CONCLUSÃO (FOL):
{conclusion_fol}

LABEL fornecido no dataset (gabarito de referência, pode ser usado como
ponto de checagem, mas sua análise deve ser feita de forma independente):
{label}

Responda ESTRITAMENTE no seguinte formato JSON, sem nenhum texto antes ou
depois:

{{
    "veredito": "<Válido | Inválido | Indeterminado>",
    "concorda_com_label": <true | false | null>,
    "justificativa": "<explicação lógica objetiva, citando as regras de "
                        "inferência ou contraexemplos relevantes>"
}}
"""


def build_prompt(example: Dict[str, Any]) -> str:
    """Monta o prompt final substituindo os campos do exemplo."""
    def get(field_key: str) -> str:
        value = example.get(FIELD_MAP[field_key], "")
        if value in (None, "", []):
            return "(não informado)"
        if isinstance(value, list):
            # formata listas (ex.: várias premissas) como itens numerados,
            # em vez do repr cru do Python (['a', 'b'])
            return "\n".join(f"{i}. {item}" for i, item in enumerate(value, start=1))
        return str(value)

    return PROMPT_TEMPLATE.format(
        premises=get("premises"),
        premises_fol=get("premises_fol"),
        conclusion=get("conclusion"),
        conclusion_fol=get("conclusion_fol"),
        label=get("label"),
    )


def query_ollama(client: ollama.Client, prompt: str) -> str:
    """
    Envia o prompt ao modelo via Ollama e retorna o texto bruto da resposta.
    Faz algumas tentativas em caso de falha de comunicação.
    """
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                format="json", # força o Ollama a retornar JSON sintaticamente válido
                think=THINK, # desliga raciocínio estendido em modelos "thinking"
                options={
                    "temperature": TEMPERATURE,
                    "num_predict": NUM_PREDICT,
                },
            )
            if DEBUG:
                print(f"  [debug] resposta completa do Ollama: {response}", file=sys.stderr)

            content = response["message"]["content"]

            # Fallback: alguns modelos "thinking" ainda deixam o content vazio
            # mesmo com think=False, ou colocam a resposta dentro de
            # 'thinking' quando o cliente/versão do Ollama não suporta o
            # parâmetro. Nesse caso, tentamos aproveitar o campo thinking.
            if not content.strip():
                thinking_text = getattr(response["message"], "thinking", None) or ""
                if thinking_text.strip():
                    if DEBUG:
                        print(
                            "  [debug] 'content' veio vazio; usando 'thinking' "
                            "como texto de resposta (fallback).",
                            file=sys.stderr,
                        )
                    content = thinking_text

            return content
        except Exception as exc: # captura erros de conexão, timeout, etc.
            last_error = exc
            print(
                f"  [aviso] tentativa {attempt}/{MAX_RETRIES} falhou "
                f"ao comunicar com o Ollama: {exc}",
                file=sys.stderr,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

    # Se chegou aqui, todas as tentativas falharam
    raise RuntimeError(f"Falha ao consultar o Ollama após {MAX_RETRIES} tentativas: {last_error}")


def parse_model_output(raw_text: str) -> Dict[str, Any]:
    """
    Tenta interpretar a saída do modelo como JSON estruturado.
    Se falhar, devolve o texto bruto em um campo de fallback.
    """
    text = raw_text.strip()

    # Remove possíveis cercas de código (```json ... ```) que alguns modelos adicionam
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        parsed = json.loads(text)
        return {
            "veredito": parsed.get("veredito"),
            "concorda_com_label": parsed.get("concorda_com_label"),
            "justificativa": parsed.get("justificativa"),
            "resposta_bruta": raw_text,
            "parse_ok": True,
        }
    except json.JSONDecodeError:
        return {
            "veredito": None,
            "concorda_com_label": None,
            "justificativa": None,
            "resposta_bruta": raw_text,
            "parse_ok": False,
        }


def iter_json_records(file_obj):
    """
    Lê registros JSON de um arquivo de forma tolerante ao formato:
    aceita tanto .jsonl "de verdade" (um objeto compacto por linha)
    quanto arquivos com objetos JSON formatados/"pretty-printed" em
    várias linhas (como o exemplo que você enviou, onde cada registro
    começa com "{" numa linha e termina com "}" várias linhas depois).

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
            # não foi possível decodificar a partir daqui: reporta e para,
            # pois não há como saber com segurança onde o próximo registro começa
            record_number += 1
            raise RuntimeError(
                f"Não foi possível interpretar o JSON a partir do caractere "
                f"{start} (registro aproximado {record_number}): {exc}"
            ) from exc

        record_number += 1
        yield record_number, obj, text[start:end_idx]
        idx = end_idx


def process_file(input_path: str, output_path: str) -> None:
    in_file = Path(input_path)
    out_file = Path(output_path)

    if not in_file.exists():
        print(f"[erro] arquivo de entrada não encontrado: {in_file}", file=sys.stderr)
        sys.exit(1)

    client = ollama.Client(host=OLLAMA_HOST)

    total = 0
    ok = 0
    failed = 0

    with in_file.open("r", encoding="utf-8") as fin, \
        out_file.open("w", encoding="utf-8") as fout:

        try:
            records = list(iter_json_records(fin))
        except RuntimeError as exc:
            print(f"[erro] falha ao ler o arquivo de entrada: {exc}", file=sys.stderr)
            sys.exit(1)

        for record_number, example, _raw_text in records:
            total += 1
            print(f"[{record_number}] processando exemplo...")

            prompt = build_prompt(example)
            result_record: Dict[str, Any] = dict(example) # preserva os campos originais

            try:
                raw_output = query_ollama(client, prompt)
                analysis = parse_model_output(raw_output)
                if not analysis["parse_ok"]:
                    print(
                        f"  [aviso] registro {record_number}: não foi possível "
                        f"interpretar a saída do modelo como JSON. "
                        f"Veja 'resposta_bruta' no resultado.",
                        file=sys.stderr,
                    )
                result_record["analise_modelo"] = analysis
                result_record["erro"] = None
                ok += 1
            except Exception as exc:
                print(f"  [erro] falha ao processar registro {record_number}: {exc}", file=sys.stderr)
                result_record["analise_modelo"] = None
                result_record["erro"] = str(exc)
                failed += 1

            fout.write(json.dumps(result_record, ensure_ascii=False) + "\n")
            fout.flush()

    print("\n=== Resumo ===")
    print(f"Total de exemplos lidos:   {total}")
    print(f"Processados com sucesso:   {ok}")
    print(f"Falharam:                  {failed}")
    print(f"Resultados salvos em:      {out_file.resolve()}")


if __name__ == "__main__":
    process_file(INPUT_FILE, OUTPUT_FILE)