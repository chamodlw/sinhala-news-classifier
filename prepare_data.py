import os
import glob
import json
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    data = []
    base_dir = "Categorized data set"
    
    # 1. Load CSV files
    csv_mapping = {
        "sinhala_crime_news_dataset.csv": "Crime",
        "sinhala_education_news_dataset.csv": "Education",
        "sinhala_health_news_dataset.csv": "Health"
    }
    
    for filename, category in csv_mapping.items():
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            for idx, row in df.iterrows():
                if 'news_content' in row:
                    content = str(row['news_content']).strip()
                elif 'content' in row:
                    content = str(row['content']).strip()
                else:
                    content = ""
                if content and str(content) != 'nan':
                    data.append({
                        "text": content,
                        "label": category
                    })
                    
    # 2. Load JSON files from directories
    json_dirs = {
        "business": "Business",
        "politics": "Politics",
        "sports": "Sports"
    }
    
    for dir_name, category in json_dirs.items():
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            json_files = glob.glob(os.path.join(dir_path, "*.json"))
            for filepath in json_files:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        article = json.load(f)
                        # According to the sample, the key is "News Content"
                        content = str(article.get("News Content", '')).strip()
                        if content:
                            data.append({
                                "text": content,
                                "label": category
                            })
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
                    
    return pd.DataFrame(data)

def main():
    print("Loading data...")
    df = load_data()
    print(f"Total initially loaded rows: {len(df)}")
    
    # Drop duplicates and NAs
    df = df.dropna(subset=['text']).drop_duplicates(subset=['text'])
    print(f"Total rows after cleaning: {len(df)}")
    
    print("\nClass distribution:")
    print(df['label'].value_counts())
    
    # Map labels to integers
    categories = ['Politics', 'Business', 'Sports', 'Crime', 'Education', 'Health']
    label_to_id = {cat: i for i, cat in enumerate(categories)}
    
    df['label_id'] = df['label'].map(label_to_id)
    
    # Save the label mapping
    with open('label_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(label_to_id, f, ensure_ascii=False, indent=4)
        
    # Split the dataset
    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df['label_id'], random_state=42)
    
    # Save the prepared datasets
    train_df[['text', 'label_id']].to_csv('train_dataset.csv', index=False)
    val_df[['text', 'label_id']].to_csv('val_dataset.csv', index=False)
    
    print(f"\nTrain set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print("Data preparation complete. Output saved to 'train_dataset.csv' and 'val_dataset.csv'.")

if __name__ == "__main__":
    main()
