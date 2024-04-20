from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import pandas as pd
import numpy as np
import torch

# Verificar se CUDA está disponível e definir o dispositivo padrão como cuda
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Define a primeira GPU disponível como padrão

# Modelo e tokenizer
model_name = "PORTULAN/albertina-ptpt"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Carregar o seu dataset
dataset = load_dataset('csv', data_files={
    'train': 'data/train_toldbr.csv',
    'test': 'data/test_toldbr.csv'
})

# Função de tokenização para a coluna "text"
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Aplicar a função de tokenização no dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Adicionando a coluna 'Toxic' como 'labels'
tokenized_datasets["train"] = tokenized_datasets["train"].add_column("labels", tokenized_datasets["train"]["Toxic"])
tokenized_datasets["test"] = tokenized_datasets["test"].add_column("labels", tokenized_datasets["test"]["Toxic"])

# Configurações de treinamento
training_args = TrainingArguments(
    output_dir="results/albertina-ptbr-toldbr",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    save_strategy="epoch",
    logging_dir='./logs',
)

# Definir o Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)

# Treinar
trainer.train()

# Predições
predictions, raw_outputs = trainer.predict(tokenized_datasets["test"])

# Converter raw_outputs em probabilidades usando a função softmax
probabilities = np.exp(raw_outputs) / np.sum(np.exp(raw_outputs), axis=1, keepdims=True)
predicted_probs = probabilities[:, 1]  # Probabilidades para a classe 1 (tóxico)

# Convertendo tokenized_datasets["test"] para DataFrame e adicionando 'predictions' e 'probabilities'
eval_df = pd.DataFrame(tokenized_datasets["test"])
eval_df['predictions'] = predictions
eval_df['probabilities'] = predicted_probs

# Salvar o DataFrame atualizado em um arquivo CSV
eval_df.to_csv('./results/albertina-ptpt_test_toldbr.csv', index=False)
