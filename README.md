# Sinhala News Classifier Usage Guide

This project leverages a fine-tuned `Ransaka/sinhala-bert-medium-v2` transformer model to automatically categorize Sinhala news excerpts into one of 6 distinct categories: `Politics`, `Business`, `Sports`, `Crime`, `Education`, and `Health`.

Here is how you can use this model to categorize new Sinhala news content.

## 1. Interactive Console Mode

The easiest way to test the model manually is by running `predict.py` in interactive mode. This will open a prompt where you can paste any Sinhala text and immediately get the prediction.

```powershell
.\venv\Scripts\Activate.ps1 
python predict.py 
```

**Example Session:**
```text
Loading model. Please wait...

Please enter a Sinhala news excerpt to classify (or type 'quit' to exit):
> ලෝක කුසලාන ක්‍රිකට් තරඟාවලියේ ශ්‍රී ලංකා කණ්ඩායම අද තරඟ බිමට පිවිසෙනවා.

Predicted Category: Sports
Confidence Scores:
  Politics: 2.10%
  Business: 1.05%
  Sports: 92.30%
...
```
*(Type `quit` or `exit` to close the prompt)*

## 2. Command Line Argument Mode (Automated)

If you have a batch script or want to classify a single text snippet quickly from the terminal, pass the text via the `--text` argument.

```powershell
python predict.py --text "ඔබේ ප්‍රවෘත්ති පෙළ මෙහි ඇතුළත් කරන්න"
```

## 3. Python Integration (Using the Model in Your Own App)

To integrate this categorizer into a larger application (like a web app, Discord bot, or a larger news scraper), you can import the model and tokenizer directly into your own Python code.

Here is a snippet you can copy/paste:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Provide the path to the trained model directory
model_dir = './sinhala-news-classifier-model'

# 2. Load the Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Ensure CPU/GPU compatibility
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

def predict_category(news_text: str) -> str:
    # 3. Tokenize the input text
    inputs = tokenizer(news_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 4. Predict
    with torch.no_grad():
        outputs = model(**inputs)

    # 5. Extract category
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    predicted_label = model.config.id2label[predicted_class_id]
    
    return predicted_label

# --- Usage Example ---
my_news = "පාසල් ළමුන් සඳහා නව අධ්‍යාපනික වැඩසටහනක් ආරම්භ කෙරේ"
category = predict_category(my_news)
print(f"The predicted category is: {category}")
# Output: The predicted category is: Education
```

### Best Practices

- **Length of Text:** The model crops texts to 256 tokens (`max_length=256`). For best categorization, make sure the most important, context-heavy parts of the news article (like the headline or the first paragraph) are passed.
- **Hardware:** It will run on `cpu` or `cuda` (NVIDIA GPU). Using a GPU is ~20x faster.

## 4. How to Enhance with More Data and Re-Train

If you collect more news data in the future and want to make the model even smarter, follow these 4 simple steps:

1. **Add Your New Data:**
   - For `Crime`, `Health`, or `Education`, append your new CSV files/rows into the `Categorized data set` root folder (with a `content` or `news_content` column).
   - For `Business`, `Politics`, or `Sports`, just throw your new `.json` files inside their respective folders within `Categorized data set`.

2. **Re-Run Data Preparation:**
   Execute the processing script to absorb your new data into the master `.csv` sets:
   ```powershell
   python prepare_data.py
   ```

3. **Train the Model:**
   Start the training script so the transformer learns your new data:
   ```powershell
   python train.py
   ```
   *(Note: This defaults to 3 epochs. Ensure your GPU is not being used by heavy games or software while this runs!)*

4. **Test!**
   That's it! Your `./sinhala-news-classifier-model` folder is now upgraded with your new data. Run `python predict.py` to test it.
