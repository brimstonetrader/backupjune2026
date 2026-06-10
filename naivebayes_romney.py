import pandas as pd
import math
df = pd.read_excel(
    'training-Obama-Romney-tweets.xlsx',
    sheet_name='Romney'
)

# Filter valid rows, then split
df = df[df['Unnamed: 4'].isin([1, 0, -1]) & df['Anootated tweet'].notna()]
train, test_df = df.iloc[:-250], df.iloc[-250:]


train_pos = train[train['Unnamed: 4'] == 1]
train_neu = train[train['Unnamed: 4'] == 0]
train_neg = train[train['Unnamed: 4'] == -1]
max_count = max(len(train_pos), len(train_neu), len(train_neg))

import numpy as np
train = pd.concat([
    train_pos.sample(max_count, replace=True, random_state=42),
    train_neu.sample(max_count, replace=True, random_state=42),
    train_neg.sample(max_count, replace=True, random_state=42),
])

dpos = {}
dneu = {}
dneg = {}
for _, row in train.iterrows():
    tweet = row['Anootated tweet']
    label = row['Unnamed: 4']
    for word in str(tweet).split():
        if label == 1:  dpos[word] = dpos.get(word, 0) + 1
        if label == 0:  dneu[word] = dneu.get(word, 0) + 1
        if label == -1: dneg[word] = dneg.get(word, 0) + 1

vocab = set(dpos) | set(dneu) | set(dneg)
possum, neusum, negsum = sum(dpos.values()), sum(dneu.values()), sum(dneg.values())
common = set()
for w in dpos.keys(): 
    if dpos[w] > 6 and dneu.get(w,0) > 6 and dneg.get(w,0) > 6: 
        common.add(w)


def classify(text):
    p_pos = p_neu = p_neg = 0
    total = len(train)
    for word in text.split():
        # if word not in common:
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
'final-testData-no-label-Romney-tweets.xlsx',
    sheet_name='Romney'
)

with open('romney_predictions.txt', 'w') as f:
    f.write('(setf x *(\n')
    for i, row in dg.iterrows():
        pred = classify(str(row['<e>Romney</e> got 3 less minutes and had to debate Candy Crowley,  he still out performed both of them.']))
        f.write(f'({i} {pred})\n')
    f.write(') )')
print('Predictions written to romney_predictions.txt')
