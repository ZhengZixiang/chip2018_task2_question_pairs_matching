# -*- coding: utf-8 -*-
import keras
import pandas as pd

from data_helper import *
from keras.models import load_model, model_from_json

MAXSEQUENCE_LENGTH = 30

test_q1, test_q2 = load_test_set(MAXSEQUENCE_LENGTH, word_level=True)
test_features = pd.read_csv('dataset/test_features.csv')

pick_columns = ['len_diff', 'edit_distance',
                'adjusted_common_word_ratio', 'adjusted_common_char_ratio',
                'pword_dside_rate', 'pword_oside_rate',
                'pchar_dside_rate', 'pchar_oside_rate']
test_features = test_features[pick_columns]

model = model_from_json(open('model/network_architecture.json').read())
model.load_weights('./model/word_level_0.hdf5')
preds = model.predict_classes([test_q1, test_q2, test_features], batch_size=1024, verbose=1)
preds = preds.ravel() + 0.5
preds[preds['y_pred'] < -.5]