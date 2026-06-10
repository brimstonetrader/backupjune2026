# pip install transformers torch datasets accelerate pandas openpyxl
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)

from datasets import Dataset
import pandas as pd
import numpy as np
import math
import re
import torch

# ── Load & clean training data ──────────────────────────────────────────────
df = pd.read_excel(
    'training-Obama-Romney-tweets.xlsx',
    sheet_name='Romney',
    header=0,
    skiprows=[1],
)
df = df[df['Unnamed: 4'].isin([1, 0, -1]) & df['Anootated tweet'].notna()].copy()

def strip_tags(text):
    return re.sub(r'<[^>]+>', '', str(text))

# Map -1/0/1 → 0/1/2 for HuggingFace (needs 0-indexed integer labels)
label_map_to_hf  = {-1: 0, 0: 1, 1: 2}
label_map_from_hf = {0: -1, 1: 0, 2: 1}

df['text']  = df['Anootated tweet'].apply(strip_tags)
df['label'] = df['Unnamed: 4'].map(label_map_to_hf)

# Oversample to balance classes
train_df, test_df = df.iloc[:-250], df.iloc[-250:]
max_count = train_df['label'].value_counts().max()
train_balanced = pd.concat([
    grp.sample(max_count, replace=True, random_state=42)
    for _, grp in train_df.groupby('label')
]).sample(frac=1, random_state=42).reset_index(drop=True)

# ── Tokenize ─────────────────────────────────────────────────────────────────
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)

train_ds = Dataset.from_pandas(train_balanced[['text', 'label']]).map(tokenize, batched=True)
test_ds  = Dataset.from_pandas(test_df[['text', 'label']].reset_index(drop=True)).map(tokenize, batched=True)
train_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# ── Fine-tune ────────────────────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3, ignore_mismatched_sizes=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {'accuracy': acc}

args = TrainingArguments(
    output_dir='./romney-roberta',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=25,
    weight_decay=0.01,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    logging_steps=20,
    fp16=torch.cuda.is_available(),  # use fp16 if GPU available
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

print("Fine-tuning...")
trainer.train()

# ── Evaluate on held-out test set ────────────────────────────────────────────
preds_raw = trainer.predict(test_ds).predictions
preds  = [label_map_from_hf[p] for p in np.argmax(preds_raw, axis=-1)]
actuals = list(test_df['Unnamed: 4'])

correct = sum(p == a for p, a in zip(preds, actuals))
print(f'Accuracy: {correct / len(actuals):.3f}')

for label, name in [(1, 'Positive'), (-1, 'Negative')]:
    tp = sum(p == label == a for p, a in zip(preds, actuals))
    fp = sum(p == label != a for p, a in zip(preds, actuals))
    fn = sum(p != label == a for p, a in zip(preds, actuals))
    p, r = tp/(tp+fp or 1), tp/(tp+fn or 1)
    print(f"{name}: P={p:.2f} R={r:.2f} F1={2*p*r/(p+r or 1):.2f}")

# ── Predict on unlabeled test data ───────────────────────────────────────────
def classify(text):
    inputs = tokenizer(
        strip_tags(text), return_tensors='pt',
        truncation=True, padding=True, max_length=128
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    return label_map_from_hf[logits.argmax(-1).item()]

dg = pd.read_excel('final-testData-no-label-Romney-tweets.xlsx', sheet_name='Romney')

with open('romney_predictions.txt', 'w') as f:
    f.write('(setf x *(\n')
    for i, row in dg.iterrows():
        pred = classify(str(row[1]))
        f.write(f'({i} {pred})\n')
    f.write(') )')

print('Predictions written to romney_predictions.txt')