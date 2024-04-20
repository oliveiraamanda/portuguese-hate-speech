import requests
import argparse
import pandas as pd
import time

# Define constantes
DEFAULT_API_KEY = '103274251389094881747$f34de79813883385e129ca9cc225bfde235e75556bc5bdde7f5e4a56d37fc91f'
URL = "https://chat.maritaca.ai/api/chat/inference"
HEADERS = {
    "authorization": f"Key {DEFAULT_API_KEY}"
}
SEMENTE = 42

# Leitura dos conjuntos de dados
arquivo_treinamento = './data/train_toldbr.csv'
arquivo_teste = './data/test_toldbr.csv'
dados_treinamento = pd.read_csv(arquivo_treinamento)
dados_teste = pd.read_csv(arquivo_teste)


def create_prompt(texto, method, n_instances):
    if method == "zero-shot":
        return ("Você está realizando análise de texto. O texto a seguir foi utilizado em um contexto de uma rede social. "
                "Responda apenas 'sim' ou 'não' caso o texto abaixo contenha discurso de ódio, ofensa, agressividade ou outras formas de toxicidade: " + texto)
    else:
        exemplos_toxicos = dados_treinamento[dados_treinamento['Toxic'] == 1.0].sample(n_instances//2, random_state=SEMENTE)
        exemplos_nao_toxicos = dados_treinamento[dados_treinamento['Toxic'] == 0.0].sample(n_instances//2, random_state=SEMENTE)

        messages = []
        for _, row in exemplos_toxicos.iterrows():
            messages.append({"role": "user", "content": "Você está realizando análise de texto. O texto a seguir foi utilizado em um contexto de uma rede social. Responda apenas 'sim' ou 'não' caso o texto abaixo contenha discurso de ódio, ofensa, agressividade ou outras formas de toxicidade: " + row['clean_text']})
            messages.append({"role": "assistant", "content": "sim, é tóxico."})

        for _, row in exemplos_nao_toxicos.iterrows():
            messages.append({"role": "user", "content": "Você está realizando análise de texto. O texto a seguir foi utilizado em um contexto de uma rede social. Responda apenas 'sim' ou 'não' caso o texto abaixo contenha discurso de ódio, ofensa, agressividade ou outras formas de toxicidade: "+ row['clean_text']})
            messages.append({"role": "assistant", "content": "não é tóxico."})

        messages.append({"role": "user", "content": "Você está realizando análise de texto. O texto a seguir foi utilizado em um contexto de uma rede social. Responda apenas 'sim' ou 'não' caso o texto abaixo contenha discurso de ódio, ofensa, agressividade ou outras formas de toxicidade: " + texto})
        return messages


def get_response(text, headers, method="zero-shot", n_instances=50):
    prompt = create_prompt(text, method, n_instances)
    data = {
        "messages": prompt,
        "do_sample": False,
        "temperature": 0,
        "model": "maritalk",
        "repetition_penalty": 1.2
    }

    max_retries = 5

    for i in range(max_retries):
        resposta = requests.post(URL, headers=headers, json=data)

        if resposta.status_code == 200:
            # Dependendo da estrutura da resposta, isso pode precisar de ajustes.
            return resposta.json()['answer']

        elif resposta.status_code == 429:  # Rate limited
            sleep_time = min(5.0, 0.2 * (2 ** i))  # Exponential backoff starting from 200ms, max 5 seconds
            print(f"Rate limited! Tentando novamente em {sleep_time} segundos...")
            time.sleep(sleep_time)

        else:
            print(f"Erro na requisição: {resposta.status_code} - {resposta.text}")
            return None

    # Se o código chegar até aqui, significa que todas as tentativas falharam.
    print("Número máximo de tentativas alcançado!")
    return None


def main(args):
    HEADERS['Authorization'] = f'Key {args.api_key}'
    texts = dados_teste['clean_text'].tolist()

    results = []
    for text in texts:
        print(text)
        result = get_response(text, HEADERS, method=args.method, n_instances=args.n_instances)
        if result:
            print(result + '\n')
            results.append(result)
        else:
            print("Failed to get a result!\n")

        time.sleep(6)  # Delay to respect API's rate limit

    # Salva os resultados
    output_filename = f"./results/Maritaca_ToldBr_{args.method}_{args.n_instances if args.method == 'few-shot' else 'zero-shot'}.csv"
    
    #with open(output_filename, 'w') as file:
    #    file.write("\n".join(results))
    # Salvar resultados
    dados_teste['predictions'] = results
    dados_teste.to_csv(output_filename, index=False)
    print("Classificação concluída e resultados salvos.")

if __name__ == "__main__":
    #python seu_script.py --method few-shot --n_instances 100 --api_key SUA_CHAVE

    parser = argparse.ArgumentParser(description='Send texts to OpenAI API for classification.')
    parser.add_argument('--method', type=str, choices=['zero-shot', 'few-shot'], default='zero-shot', help='Learning method to use.')
    parser.add_argument('--n_instances', type=int, default=50, help='Number of instances to be used for few-shot learning.')
    parser.add_argument('--api_key', type=str, default=DEFAULT_API_KEY, help='OpenAI API key.')

    args = parser.parse_args()
    main(args)
