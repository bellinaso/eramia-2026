import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import ollama


# Lista de modelos que serão testados, em sequência, contra o dataset
# inteiro. Ajuste os nomes conforme os modelos que você tem disponíveis
# no seu Ollama local (ollama list).
MODELOS: List[str] = [
    "gemma4:12b",
    "qwen3.5:9b",
    "lamma3.2:3b"
]

OLLAMA_HOST = "http://localhost:11434"

INPUT_FILE = "dataset.jsonl"

# Intervalo de descanso entre a execução de um modelo e o início do
# próximo (em segundos). 10 minutos = 600 segundos.
INTERVALO_ENTRE_MODELOS_SEGUNDOS = 10 * 60

# Arquivo único onde é anexado (append) um resumo por modelo executado.
RESUMO_SIMULACAO_FILE = "resumo_simulacao.jsonl"

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
    "logicamente de um conjunto de premissas em linguagem natural. "
    "Seja rigoroso, objetivo e sempre responda no formato solicitado."
)

# O prompt agora contém SOMENTE premissas e conclusão em linguagem
# natural: nada de FOL e nada do label de referência, para que o modelo
# não seja influenciado por nenhum contexto ou "resposta pronta".
PROMPT_TEMPLATE = """
Analise o exemplo de raciocínio lógico abaixo e determine se a CONCLUSÃO é
válida a partir das PREMISSAS fornecidas.

PREMISSAS (linguagem natural):
{premises}

CONCLUSÃO (linguagem natural):
{conclusion}

Responda ESTRITAMENTE no seguinte formato JSON, sem nenhum texto antes ou
depois:

{{
    "veredito": "<Válido | Inválido | Indeterminado>",
    "justificativa": "<explicação lógica objetiva, citando as regras de "
                        "inferência ou contraexemplos relevantes>"
}}
"""


def build_prompt(example: Dict[str, Any]) -> str:
    """Monta o prompt final substituindo os campos do exemplo.

    Usa apenas as premissas e a conclusão em linguagem natural — as
    versões em FOL e o label não entram no prompt.
    """
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
        conclusion=get("conclusion"),
    )


def query_ollama(client: ollama.Client, model_name: str, prompt: str) -> str:
    """
    Envia o prompt ao modelo `model_name` via Ollama e retorna o texto
    bruto da resposta. Faz algumas tentativas em caso de falha de
    comunicação.
    """
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat(
                model=model_name,
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
            "justificativa": parsed.get("justificativa"),
            "resposta_bruta": raw_text,
            "parse_ok": True,
        }
    except json.JSONDecodeError:
        return {
            "veredito": None,
            "justificativa": None,
            "resposta_bruta": raw_text,
            "parse_ok": False,
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

    IMPORTANTE: ajuste os conjuntos abaixo caso o seu dataset use outras
    convenções de rótulo (por exemplo "entailment" / "contradiction" /
    "neutral", ou algum outro esquema específico do seu dataset).
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
    Compara o veredito dado pelo modelo com o label de referência do
    dataset (que NÃO foi mostrado ao modelo no prompt) e retorna:
        True  -> o modelo acertou (inclui o caso em que ambos caem na
                categoria "indeterminado"/"uncertain" — isso é um
                acerto, não uma comparação impossível)
        False -> o modelo errou
        None  -> não foi possível determinar, porque o veredito e/ou o
                label vieram em um formato não reconhecido por
                _mapear_para_categoria (nenhuma das 3 categorias)
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


def process_file(client: ollama.Client, input_path: str, output_path: str, model_name: str) -> Dict[str, Any]:
    """
    Processa o dataset inteiro para um único modelo, salvando o arquivo
    de resultados individual e retornando um dicionário de resumo com
    as estatísticas da execução.
    """
    in_file = Path(input_path)
    out_file = Path(output_path)

    if not in_file.exists():
        print(f"[erro] arquivo de entrada não encontrado: {in_file}", file=sys.stderr)
        sys.exit(1)

    total = 0
    falhas_processamento = 0
    acertos = 0
    erros = 0
    nao_comparaveis = 0
    tempos_por_exemplo: List[float] = []

    horario_inicio = datetime.now()

    with in_file.open("r", encoding="utf-8") as fin, \
        out_file.open("w", encoding="utf-8") as fout:

        try:
            records = list(iter_json_records(fin))
        except RuntimeError as exc:
            print(f"[erro] falha ao ler o arquivo de entrada: {exc}", file=sys.stderr)
            sys.exit(1)

        for record_number, example, _raw_text in records:
            total += 1
            print(f"[{model_name}] [{record_number}] processando exemplo...")

            prompt = build_prompt(example)
            result_record: Dict[str, Any] = dict(example) # preserva os campos originais

            inicio_exemplo = time.time()
            try:
                raw_output = query_ollama(client, model_name, prompt)
                analysis = parse_model_output(raw_output)

                if not analysis["parse_ok"]:
                    print(
                        f"  [aviso] registro {record_number}: não foi possível "
                        f"interpretar a saída do modelo como JSON. "
                        f"Veja 'resposta_bruta' no resultado.",
                        file=sys.stderr,
                    )

                label_original = example.get(FIELD_MAP["label"])
                acertou = calcular_acertou(analysis.get("veredito"), label_original)
                analysis["acertou"] = acertou

                if acertou is True:
                    acertos += 1
                elif acertou is False:
                    erros += 1
                else:
                    # acertou is None: veredito e/ou label vieram em um
                    # formato que _mapear_para_categoria não reconhece
                    # (não é "indeterminado/uncertain" — isso já conta
                    # como acerto ou erro acima; aqui é caso raro de
                    # texto fora do esperado).
                    nao_comparaveis += 1

                result_record["analise_modelo"] = analysis
                result_record["erro"] = None
            except Exception as exc:
                print(f"  [erro] falha ao processar registro {record_number}: {exc}", file=sys.stderr)
                result_record["analise_modelo"] = None
                result_record["erro"] = str(exc)
                falhas_processamento += 1
            finally:
                tempos_por_exemplo.append(time.time() - inicio_exemplo)

            fout.write(json.dumps(result_record, ensure_ascii=False) + "\n")
            fout.flush()

    horario_fim = datetime.now()
    tempo_total_segundos = sum(tempos_por_exemplo)
    tempo_medio_por_exemplo = (
        tempo_total_segundos / len(tempos_por_exemplo) if tempos_por_exemplo else 0.0
    )

    comparaveis = acertos + erros
    acuracia = (acertos / comparaveis) if comparaveis > 0 else None

    # Esquema fixo e enxuto salvo no arquivo de resumo — sempre estas
    # chaves, nesta ordem, com horário/duração da execução mas sem
    # caminhos de arquivo.
    resumo = {
        "modelo": f"{sanitizar_nome_arquivo(model_name)}_without_solver",
        "horario_inicio": horario_inicio.isoformat(timespec="seconds"),
        "horario_fim": horario_fim.isoformat(timespec="seconds"),
        "duracao_total_segundos": round((horario_fim - horario_inicio).total_seconds(), 2),
        "total_exemplos": total,
        "acertos": acertos,
        "erros": erros,
        "nao_comparaveis": nao_comparaveis,
        "falhas_processamento": falhas_processamento,
        "acuracia": round(acuracia, 4) if acuracia is not None else None,
    }

    print("\n=== Resumo do modelo:", model_name, "===")
    print(f"Total de exemplos lidos:   {total}")
    print(f"Acertos:                   {acertos}")
    print(f"Erros:                     {erros}")
    print(f"Não comparáveis (formato): {nao_comparaveis}")
    print(f"Acurácia (sobre comparáveis): {acuracia:.2%}" if acuracia is not None else "Acurácia: N/A")
    print(f"Falhas de processamento:   {falhas_processamento}")
    print(f"Tempo médio por exemplo:   {tempo_medio_por_exemplo:.2f}s")
    print(f"Resultados salvos em:      {out_file.resolve()}")

    return resumo


def salvar_resumo_simulacao(resumo: Dict[str, Any], resumo_path: str) -> None:
    """Anexa (append) uma linha de resumo ao arquivo de resumo da simulação."""
    with Path(resumo_path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(resumo, ensure_ascii=False) + "\n")


def executar_para_todos_modelos() -> None:
    """
    Executa o dataset inteiro para cada modelo em MODELOS, salvando um
    arquivo de resultados por modelo (com timestamp único desta execução,
    para nunca sobrescrever resultados de simulações anteriores mesmo que
    o mesmo modelo seja testado de novo no futuro) e aguardando
    INTERVALO_ENTRE_MODELOS_SEGUNDOS entre uma execução e outra.

    A lista MODELOS pode ser livremente editada (remover modelos, trocar
    por outros) entre uma chamada e outra do script: o arquivo de resumo
    (RESUMO_SIMULACAO_FILE) é sempre aberto em modo append, então cada
    simulação apenas adiciona novas linhas, nunca apaga o histórico
    anterior.
    """
    if not MODELOS:
        print("[erro] a lista MODELOS está vazia. Adicione ao menos um modelo.", file=sys.stderr)
        sys.exit(1)

    client = ollama.Client(host=OLLAMA_HOST)

    # Identificador único desta chamada do script (mesmo para todos os
    # modelos executados nela), usado para nunca sobrescrever arquivos de
    # execuções passadas ou futuras, mesmo que o mesmo modelo seja
    # testado novamente em outra simulação.
    execucao_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for indice, modelo in enumerate(MODELOS):
        print(f"\n=== Iniciando execução para o modelo: {modelo} ({indice + 1}/{len(MODELOS)}) ===")

        nome_arquivo_saida = f"{sanitizar_nome_arquivo(modelo)}_{execucao_id}_without_solver.jsonl"
        resumo = process_file(client, INPUT_FILE, nome_arquivo_saida, modelo)
        salvar_resumo_simulacao(resumo, RESUMO_SIMULACAO_FILE)

        ultimo_modelo = indice == len(MODELOS) - 1
        if not ultimo_modelo:
            minutos = INTERVALO_ENTRE_MODELOS_SEGUNDOS // 60
            print(f"\nAguardando {minutos} minutos antes de iniciar o próximo modelo...")
            time.sleep(INTERVALO_ENTRE_MODELOS_SEGUNDOS)

    print("\n=== Execução concluída para todos os modelos ===")
    print(f"Resumo geral salvo em: {Path(RESUMO_SIMULACAO_FILE).resolve()}")


if __name__ == "__main__":
    executar_para_todos_modelos()