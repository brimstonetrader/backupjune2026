from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import re
# pip install vaderSentiment

df = pd.read_excel(
    'training-Obama-Romney-tweets.xlsx',
    sheet_name='Romney'
)

# Filter valid rows, then split
df = df[df['Unnamed: 4'].isin([1, 0, -1]) & df['Anootated tweet'].notna()]

analyzer = SentimentIntensityAnalyzer()

def classify(text):
    # strip the <e> tags first since vader doesn't know them
    text = re.sub(r'<[^>]+>', '', str(text))
    scores = analyzer.polarity_scores(text)
    # scores = {'neg': 0.x, 'neu': 0.x, 'pos': 0.x, 'compound': -1 to +1}
    
    compound = scores['compound']
    if compound >= 0.05:   return 1   # positive
    elif compound <= -0.05: return -1  # negative
    else:                   return 0   # neutral


rate = [0, 0]
for _, row in df.iterrows():
    if classify(str(row['Anootated tweet'])) == row['Unnamed: 4']: rate[0] += 1
    rate[1] += 1
print('Accuracy: ', rate[0] / rate[1])

preds = [classify(str(r['Anootated tweet'])) for _, r in df.iterrows()]
actuals = list(df['Unnamed: 4'])

for label, name in [(1, 'Positive'), (-1, 'Negative')]:
    tp = sum(p == label == a for p, a in zip(preds, actuals))
    fp = sum(p == label != a for p, a in zip(preds, actuals))
    fn = sum(p != label == a for p, a in zip(preds, actuals))
    p, r = tp/(tp+fp or 1), tp/(tp+fn or 1)
    print(f"{name}: P={p:.2f} R={r:.2f} F1={2*p*r/(p+r or 1):.2f}")

