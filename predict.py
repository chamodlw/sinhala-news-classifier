import os
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="Sinhala News Category Predictor")
    parser.add_argument("--text", type=str, help="Sinhala text to categorize")
    args = parser.parse_args()

    model_dir = './sinhala-news-classifier-model'
    
    if not os.path.exists(model_dir):
        print(f"Model directory '{model_dir}' not found. Have you trained the model yet?")
        return

    print("Loading model. Please wait...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    if args.text:
        text = args.text
    else:
        print("\nPlease enter a Sinhala news excerpt to classify (or type 'quit' to exit):")
        text = input("> ")

    while text.lower() not in ['quit', 'exit']:
        if text.strip():
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits
            predicted_class_id = logits.argmax().item()
            predicted_label = model.config.id2label[predicted_class_id]

            print(f"\nPredicted Category: {predicted_label}")
            probs = torch.nn.functional.softmax(logits, dim=-1)[0].tolist()
            
            print("Confidence Scores:")
            for i, p in enumerate(probs):
                label = model.config.id2label[i]
                print(f"  {label}: {p*100:.2f}%")
            print("-" * 50)

        if args.text:
            break

        print("\nEnter another news excerpt (or type 'quit' to exit):")
        text = input("> ")

if __name__ == "__main__":
    main()
