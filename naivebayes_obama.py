import pandas as pd
import math
import numpy as np
import re

df = pd.read_excel(
    'training-Obama-Romney-tweets.xlsx',
    sheet_name='Obama',
    header=0,
    skiprows=[1],
)
df = df[df['Unnamed: 4'].isin([1, 0, -1]) & df['Anootated tweet'].notna()].copy()
df['tweet_num'] = df.index + 1  # 1-based tweet number

train, test_df = df.iloc[:-250], df.iloc[-250:]

train_pos = train[train['Unnamed: 4'] == 1]
train_neu = train[train['Unnamed: 4'] == 0]
train_neg = train[train['Unnamed: 4'] == -1]
max_count = max(len(train_pos), len(train_neu), len(train_neg))

train = pd.concat([
    train_pos.sample(max_count, replace=True, random_state=42),
    train_neu.sample(max_count, replace=True, random_state=42),
    train_neg.sample(max_count, replace=True, random_state=42),
])


def strip_tags(text):
    return re.sub(r'<[^>]+>', '', text)


dpos = {}
dneu = {}
dneg = {}
for _, row in train.iterrows():
    tweet = row['Anootated tweet']
    label = row['Unnamed: 4']
    for word in strip_tags(str(tweet)).split():
        if label == 1:  dpos[word] = dpos.get(word, 0) + 1
        if label == 0:  dneu[word] = dneu.get(word, 0) + 1
        if label == -1: dneg[word] = dneg.get(word, 0) + 1

vocab = set(dpos) | set(dneu) | set(dneg)
possum, neusum, negsum = sum(dpos.values()), sum(dneu.values()), sum(dneg.values())

common = set()

def classify(text):
    p_pos = p_neu = p_neg = 0
    for word in strip_tags(str(text)).split():
        if word not in common:
            p_pos += math.log(dpos.get(word, 0) + 1) - math.log(possum + len(vocab))
            p_neu += math.log(dneu.get(word, 0) + 1) - math.log(neusum + len(vocab))
            p_neg += math.log(dneg.get(word, 0) + 1) - math.log(negsum + len(vocab))

    if p_pos > p_neu and p_pos > p_neg: return 1
    elif p_neu > p_neg: return 0
    else: return -1

rate = [0, 0]
for _, row in test_df.iterrows():
    if classify(str(row['Anootated tweet'])) == row['Unnamed: 4']: rate[0] += 1
    rate[1] += 1
print('Accuracy: ', rate[0] / rate[1])

preds = [classify(str(r['Anootated tweet'])) for _, r in test_df.iterrows()]
actuals = list(test_df['Unnamed: 4'])

for label, name in [(1, 'Positive'), (-1, 'Negative')]:
    tp = sum(p == label == a for p, a in zip(preds, actuals))
    fp = sum(p == label != a for p, a in zip(preds, actuals))
    fn = sum(p != label == a for p, a in zip(preds, actuals))
    p, r = tp/(tp+fp or 1), tp/(tp+fn or 1)
    print(f"{name}: P={p:.2f} R={r:.2f} F1={2*p*r/(p+r or 1):.2f}")


dg = pd.read_excel(
    'final-testData-no-label-Obama-tweets.xlsx',
    sheet_name='Obama'
)

# Write output file
with open('obama_predictions.txt', 'w') as f:
    f.write('(setf x *(\n')
    for i, row in dg.iterrows():
        pred = classify(str(row['<e>Obama</e> has to maintain his professionalism throughout this entire campaign...very strong individual!']))
        f.write(f'({i} {pred})\n')
    f.write(') )')

print('Predictions written to obama_predictions.txt')