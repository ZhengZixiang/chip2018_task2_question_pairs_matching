# -*- coding: utf-8 -*-
import os
import pandas as pd

filenames = os.listdir('./dataset/val_result/')
all = pd.read_csv('./dataset/test.csv')
all['label'] = 0
for filename in filenames:
    df = pd.read_csv('./dataset/val_result/' + filename)
    all['label'] += df['label']
all['label'] = all['label'] // 6
all.to_csv('./result/merge.csv', index=False)
