# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
import pandas as pd

# Substitua o 'caminho_do_arquivo.csv' pelo caminho real do seu arquivo CSV
arquivo_csv = 'data/train_toldbr.csv'

# Ler o arquivo CSV
dados = pd.read_csv(arquivo_csv)

# Mostrar as primeiras linhas do DataFrame
print(dados.head())

"""Este código carrega os dados de treino e teste, cria um modelo de classificação BERTimbau, treina-o no conjunto de dados de treino e avalia-o no conjunto de dados de teste. Ele exibe F1-Score, precisão, recall e matriz de confusão como resultado. Deve-se instalar o pacote simpletransformers."""

#!pip install simpletransformers

# baseado em https://github.com/ThilinaRajapakse/simpletransformers
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from simpletransformers.classification import ClassificationModel

# Carregar os dados de treino e teste
arquivo_csv_treino = 'data/train_toldbr.csv'
arquivo_csv_teste = 'data/test_toldbr.csv'

dados_treino = pd.read_csv(arquivo_csv_treino)
dados_teste = pd.read_csv(arquivo_csv_teste)

train_df = pd.DataFrame({'text': dados_treino.iloc[:, 2], 'labels': dados_treino.iloc[:, 1]})
eval_df = pd.DataFrame({'text': dados_teste.iloc[:, 2], 'labels': dados_teste.iloc[:, 1]})

print(train_df.head())

# Criar o modelo de classificação
model_args = {
    'num_train_epochs': 10,
    'train_batch_size': 8,
    'eval_batch_size': 8,
    'overwrite_output_dir': True,
    'save_steps': -1,
    'save_model_every_epoch': False,
    'learning_rate': 3e-5,
    'fp16': True,
}

model = ClassificationModel(
    'bert',
    #'neuralmind/bert-base-portuguese-cased',
    'neuralmind/bert-large-portuguese-cased', 
    num_labels=2,
    args=model_args,
    use_cuda=True,  # Se estiver usando uma GPU, você pode mudar para True
)


# Treinar o modelo nos dados de treino
model.train_model(train_df)

# Avaliar o modelo nos dados de teste
# Avaliar o modelo nos dados de teste
predictions, raw_outputs = model.predict(eval_df['text'].tolist())

# Converter raw_outputs em probabilidades usando a função softmax
probabilities = np.exp(raw_outputs) / np.sum(np.exp(raw_outputs), axis=1, keepdims=True)
predicted_probs = probabilities[:, 1]  # Probabilidades para a classe 1 (tóxico)

# Adicionar 'predicts' e 'probabilities' ao DataFrame de avaliação
eval_df['predictions'] = predictions
eval_df['probabilities'] = predicted_probs

# Salvar o DataFrame atualizado em um arquivo CSV
eval_df.to_csv('./results/bertimbau_test_toldbr.csv', index=False)

#predicted_labels = np.argmax(raw_outputs, axis=1)
#print("Rótulos únicos em predicted_labels:", np.unique(predicted_labels))

# # Gerar o relatório de classificação
# report = classification_report(eval_df['labels'], predicted_labels, output_dict=True)

# print("Classificação Report:")
# for label, metrics in report.items():
#     if label in ['accuracy', 'macro avg', 'weighted avg']:
#         continue
#     print(f"Class: {label}")
#     print(f"\tPrecision: {metrics['precision']}")
#     print(f"\tRecall: {metrics['recall']}")
#     print(f"\tF1-score: {metrics['f1-score']}")
# print(f"Macro Avg: {report['macro avg']}")
# print(f"Weighted Avg: {report['weighted avg']}")

# # Plotar a matriz de confusão
# conf_matrix = confusion_matrix(eval_df['labels'], predicted_labels)
# plt.figure(figsize=(6, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')

# # Salvar a figura
# plt.savefig('./results/bertimbau_confusion_matrix.png', dpi=300, bbox_inches='tight')

# plt.show()



