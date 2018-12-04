# -*- coding: utf-8 -*-
import time
import keras.callbacks as kcallbacks
import getopt
import sys
import warnings

from data_helper import *
from keras.layers import CuDNNLSTM, Dense, Input, Dropout, LSTM, Activation, BatchNormalization, concatenate, Subtract, Multiply, Bidirectional
from keras.layers.embeddings import Embedding
from keras.models import Model

np.random.seed(10000)
warnings.filterwarnings('ignore')

MAX_SEQUENCE_LENGTH = 10
EMBEDDING_DIM = 300
BATCH_SIZE = 32
NUM_CELL = 75
LSTM_DROPOUT = 0.5
DENSE_DROPOUT = 0.3
MODEL_ARCHITECTURE_PATH = './model/network_architecture.json'
# MODEL_WEIGHTS_PATH = './model/best.hdf5'
MODEL_WEIGHTS_PATH = './model/word_level_0.hdf5'
WORD_LEVEL = True


class SiameseLSTM(object):

    def __init__(self, mode):
        self.word2index, self.index2word, self.embed_matrix = load_embed_matrix(WORD_LEVEL)
        self.features = load_features(mode)
        print('embed_matrix: ', self.embed_matrix.shape)
        question1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
        question2 = Input(shape=(MAX_SEQUENCE_LENGTH,))
        embed_layer = Embedding(self.embed_matrix.shape[0], EMBEDDING_DIM, weights=[self.embed_matrix],
                                input_length=MAX_SEQUENCE_LENGTH, trainable=False)
        q1_embed = embed_layer(question1)
        q2_embed = embed_layer(question2)

        shared_lstm_1 = Bidirectional(CuDNNLSTM(NUM_CELL, return_sequences=True))
        shared_lstm_2 = Bidirectional(CuDNNLSTM(NUM_CELL))

        q1 = shared_lstm_1(q1_embed)
        q1 = Dropout(LSTM_DROPOUT)(q1)
        q1 = BatchNormalization()(q1)
        q1 = shared_lstm_2(q1)

        q2 = shared_lstm_1(q2_embed)
        q2 = Dropout(LSTM_DROPOUT)(q2)
        q2 = BatchNormalization()(q2)
        q2 = shared_lstm_2(q2)

        d = Subtract()([q1, q2])
        distance = Multiply()([d, d])
        angle = Multiply()([q1, q2])

        #　train_features =
        features_input = Input(shape=(self.features.shape[1],))
        features_dense = BatchNormalization()(features_input)
        features_dense = Dense(64, activation='relu')(features_dense)

        merged = concatenate([distance, angle, features_dense])
        merged = Dropout(DENSE_DROPOUT)(merged)
        merged = BatchNormalization()(merged)

        merged = Dense(256, activation='relu')(merged)
        merged = Dropout(DENSE_DROPOUT)(merged)
        merged = BatchNormalization()(merged)

        merged = Dense(64, activation='relu')(merged)
        merged = Dropout(DENSE_DROPOUT)(merged)
        merged = BatchNormalization()(merged)

        is_duplicate = Dense(1, activation='sigmoid')(merged)
        self.model = Model(inputs=[question1, question2, features_input], outputs=is_duplicate)

    def cross_val(self):
        train_q1, train_q2, train_label = load_dataset(MAX_SEQUENCE_LENGTH, 'train', self.word2index, WORD_LEVEL)
        best_val_score = {}
        split_index = {}
        for i in range(10):
            split_index[i] = np.arange(i * 2000, (i+1) * 2000)

        for model_count in range(10):
            if model_count != 0:
                continue
            print('MODEL: ', model_count)

            # split data into train/val set
            idx_val = split_index[model_count]
            idx_train = []
            for i in range(10):
                if i != model_count:
                    idx_train.extend(list(split_index[i]))
         
            q1_train = train_q1[idx_train]
            q2_train = train_q2[idx_train]
            y_train = train_label[idx_train]
            f_train = self.features.iloc[idx_train]

            q1_val = train_q1[idx_val]
            q2_val = train_q2[idx_val]
            y_val = train_label[idx_val]
            f_val = self.features.iloc[idx_val]

            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.model.summary()

            # define save model
            best_weights_filepath = './model/word_level_' + str(model_count) + '.hdf5'
            early_stopping = kcallbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
            save_best_model = kcallbacks.ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1,
                                                         save_best_only=True, mode='auto')

            hist = self.model.fit([q1_train, q2_train, f_train],
                                  y_train,
                                  validation_data=([q1_val, q2_val, f_val], y_val),
                                  epochs=30,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  callbacks=[early_stopping, save_best_model],
                                  verbose=1)

            self.model.load_weights(best_weights_filepath)
            print(model_count, 'validation_loss: ', min(hist.history['val_loss']))
            best_val_score[model_count] = min(hist.history['val_loss'])

            # predict on the val set
            preds = self.model.predict([q1_val, q2_val, f_val], batch_size=1024, verbose=1)
            val_preds = pd.DataFrame({'y_pre': preds.ravel()})
            val_preds['val_index'] = idx_val
            save_path = './dataset/val_result/val_' + str(model_count) + '.csv'
            val_preds.to_csv(save_path, index=0)
            print(model_count, 'val preds saved.')

            # predict on the test set
            # preds = self.model.predict([test_q1, test_q2, test_features], batch_size=1024, verbose=1)
            # test_preds = pd.DataFrame({'y_pre': preds.ravel()})
            # save_path = 'dataset/val_result/test_' + str(model_count) + '.csv'
            # test_preds.to_csv(save_path, index=0)
            # print(model_count, 'test preds saved.')

        f = open('./record.txt', 'a')
        f.write(str(model_count) + str(best_val_score))
        f.write('\n')
        f.close()

    def test(self):
        test_q1, test_q2 = load_dataset(MAX_SEQUENCE_LENGTH, 'test', self.word2index, WORD_LEVEL)
        for i in range(10):
            self.model.load_weights('./model/word_level_'+str(i)+'.hdf5')
            preds = self.model.predict([test_q1, test_q2, self.features], batch_size=1024, verbose=1)
            preds = preds.ravel() + 0.5
            preds = preds.astype(int)
            df = pd.read_csv('./dataset/test.csv')
            df['label'] = preds
            df.to_csv('./dataset/result_'+str(i)+'.csv', index=False)

    def train(self):
        train_q1, train_q2, train_label = load_dataset(MAX_SEQUENCE_LENGTH, 'train', self.word2index, WORD_LEVEL)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()
        self.model.fit([train_q1, train_q2, self.features],
                       train_label,
                       validation_split=0,
                       epochs=30,
                       batch_size=BATCH_SIZE,
                       shuffle=True,
                       verbose=1)
        self.model.save_weights('./dataset/word_level.hdf5')


if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], 'm:', ['mode='])
    mode = 'train'
    for op, value in opts:
        if op in ('-m', '--mode'):
            mode = value

    if mode == 'test':
        model = SiameseLSTM(mode)
        model.test()
    elif mode == 'val':
        model = SiameseLSTM('train')
        model.cross_val()
    elif mode == 'train':
        model = SiameseLSTM(mode)
        start = time.time()
        model.train()
        end = time.time()
        print('Training time {0:.3f}分钟'.format((end - start) / 60))
