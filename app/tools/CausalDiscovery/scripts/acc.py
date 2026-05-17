import pandas as pd

with open('../data/raw/24471163/N15/N15/artificialResults_0/gt.txt', 'r') as f:
    gt = []
    for i, line in enumerate(f):
        if i > 2:
            gt.append(list(map(lambda x: int(x), line.strip().split())))

df = pd.read_csv('../data/output/24471163/N15/final_graph.csv').iloc[:, 1:]
print(df.head())
tp, fp = 0, 0
for g in gt:
    if df.iloc[g[0], g[1]] == -1 and df.iloc[g[1], g[0]] == 1:
        tp += 1
    else:
        fp += 1

print(tp, fp)