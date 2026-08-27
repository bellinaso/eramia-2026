import pandas as pd
import ollama
from sklearn.metrics import classification_report

# 1. Carrega o dataset de testes lógicos
df = pd.read_csv("dataset_logica.csv") 

def avaliar_proposicao(texto_logico):
    prompt = f"""
    Analise a seguinte proposição e responda apenas com 'Verdadeiro' ou 'Falso' no final.
    
    Proposição: {texto_logico}
    
    Análise passo a passo:
    """
    
    resposta = ollama.generate(model="llama3.2", prompt=prompt)
    texto_ia = resposta['response'].upper()
    
    # Extração simples da resposta da IA
    if "VERDADEIRO" in texto_ia:
        return "Verdadeiro"
    elif "FALSO" in texto_ia:
        return "Falso"
    return "Inconclusivo"

# 2. Executa a IA no dataset
print("Avaliando preposições complexas...")
df['resposta_ia'] = df['Proposicao_Complexa'].apply(avaliar_proposicao)

# 3. Calcula as métricas de Precisão
print("\n--- Relatório de Precisão da IA ---")
print(classification_report(df['Resposta_Esperada'], df['resposta_ia']))
