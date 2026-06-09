import os
import threading
import warnings

import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

_model = None
_tokenizer = None
_device = None
_model_lock = threading.Lock()

MODEL_DIR = './sinhala-news-classifier-model'


def _load_model():
    global _model, _tokenizer, _device
    with _model_lock:
        if _model is not None:
            return True
        if not os.path.exists(MODEL_DIR):
            return False
        print("Loading classifier model...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _model.to(_device)
        _model.eval()
        print("Classifier model loaded!")
        return True


def _predict(text: str):
    inputs = _tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0].tolist()
    predicted_id = logits.argmax().item()
    category = _model.config.id2label[predicted_id].lower()
    all_scores = {_model.config.id2label[i].lower(): round(p, 4) for i, p in enumerate(probs)}
    return category, all_scores


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': _model is not None})


@app.route('/categorize', methods=['POST'])
def categorize():
    data = request.get_json(force=True)
    articles = data.get('articles', [])

    if not _load_model():
        return jsonify({'error': 'Model not found. Train the model first (run train.py).'}), 503

    results = []
    stats: dict = {}

    for article in articles:
        text = (article.get('title', '') + ' ' + article.get('content', '')).strip()
        category, all_scores = _predict(text)
        confidence = round(all_scores.get(category, 0.0), 3)

        results.append({
            'id': article.get('id'),
            'title': article.get('title'),
            'content': article.get('content'),
            'source': article.get('source'),
            'date': article.get('date'),
            'category': category,
            'confidence': confidence,
            'all_scores': all_scores,
        })
        stats[category] = stats.get(category, 0) + 1

    return jsonify({'success': True, 'results': results, 'stats': stats, 'total': len(results)})


if __name__ == '__main__':
    threading.Thread(target=_load_model, daemon=True).start()
    print("Starting classifier API on http://0.0.0.0:5001 ...")
    app.run(host='0.0.0.0', port=5001, debug=False)
