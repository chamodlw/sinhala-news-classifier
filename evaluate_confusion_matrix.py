"""
Figure 7.2.1 (Appendix B, Figure B.4)
=====================================
Generates the 6x6 confusion matrix for the fine-tuned Sinhala-BERT
categorisation model on the held-out validation/test set
(Politics / Business / Sports / Crime / Education / Health).

Saves two versions of the figure:
    confusion_matrix_counts.png   - raw counts
    confusion_matrix_percent.png  - row-normalised percentages

Usage:
    python evaluate_confusion_matrix.py
"""

import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = './sinhala-news-classifier-model'
TEST_CSV  = 'val_dataset.csv'          # held-out split produced by prepare_data.py
BATCH_SIZE = 16
MAX_LENGTH = 256


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return tokenizer, model, device


@torch.no_grad()
def predict_all(texts, tokenizer, model, device):
    preds = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return preds


def plot_confusion_matrix(cm, labels, normalize, out_path, title):
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
        fmt, vmax = '.1f', 100
    else:
        cm_display = cm
        fmt, vmax = 'd', None

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_display, cmap='Blues', vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, label='% of true class' if normalize else 'count')

    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title)

    thresh = cm_display.max() / 2.0
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            value = cm_display[i, j]
            text = f'{value:{fmt}}' + ('%' if normalize else '')
            ax.text(j, i, text, ha='center', va='center',
                     color='white' if value > thresh else 'black', fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def main():
    with open('label_mapping.json', encoding='utf-8') as f:
        label_to_id = json.load(f)
    labels = [lbl for lbl, _ in sorted(label_to_id.items(), key=lambda kv: kv[1])]

    df = pd.read_csv(TEST_CSV)
    df['text'] = df['text'].fillna('')
    texts, y_true = df['text'].tolist(), df['label_id'].tolist()

    print(f"Loaded {len(texts)} held-out samples from '{TEST_CSV}'")
    print("Loading fine-tuned model...")
    tokenizer, model, device = load_model()

    print("Running inference...")
    y_pred = predict_all(texts, tokenizer, model, device)

    print("\n" + classification_report(y_true, y_pred, target_names=labels, digits=3))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    plot_confusion_matrix(cm, labels, normalize=False,
                           out_path='confusion_matrix_counts.png',
                           title='Confusion Matrix - Sinhala-BERT Categorisation (counts)')
    plot_confusion_matrix(cm, labels, normalize=True,
                           out_path='confusion_matrix_percent.png',
                           title='Confusion Matrix - Sinhala-BERT Categorisation (row %)')


if __name__ == '__main__':
    main()
