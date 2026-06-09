import os
import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    if not os.path.exists('train_dataset.csv') or not os.path.exists('val_dataset.csv'):
        print("Dataset not found. Please run prepare_data.py first.")
        return

    # Load label mapping
    with open('label_mapping.json', 'r', encoding='utf-8') as f:
        label_to_id = json.load(f)
    id_to_label = {int(v): k for k, v in label_to_id.items()}
    num_labels = len(label_to_id)

    # Load data
    train_df = pd.read_csv('train_dataset.csv')
    val_df = pd.read_csv('val_dataset.csv')
    
    # Fill NAs just in case
    train_df['text'] = train_df['text'].fillna('')
    val_df['text'] = val_df['text'].fillna('')

    train_dataset = Dataset.from_pandas(train_df.rename(columns={'label_id': 'labels'}))
    val_dataset = Dataset.from_pandas(val_df.rename(columns={'label_id': 'labels'}))

    model_name = "Ransaka/sinhala-bert-medium-v2"
    
    print(f"Loading tokenizer {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=False, truncation=True, max_length=256)

    print("Tokenizing data. This may take a while...")
    encoded_train = train_dataset.map(tokenize_function, batched=True)
    encoded_val = val_dataset.map(tokenize_function, batched=True)

    print(f"Loading model {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id_to_label,
        label2id=label_to_id
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir='./logs',
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to="none" # Disable wandb and other reporting tools for simple execution
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train,
        eval_dataset=encoded_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # Save final model
    output_dir = './sinhala-news-classifier-model'
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
