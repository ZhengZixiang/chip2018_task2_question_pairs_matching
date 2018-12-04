# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

EMBEDDING_DIM = 300
WORD_EMBEDDING_PATH = './dataset/word_embedding.txt'
CHAR_EMBEDDING_PATH = './dataset/char_embedding.txt'
QUESTION_ID_PATH = './dataset/question_id.csv'


def sentences2indices(x, word2index, max_seq_len):
    '''
    Converts an array of sentences(string) into an array of indices corresponding to words in the sentences.
    The output shape should be shuch that it can be given to 'Embedding()'

    :param x: array of sentences(string), of shape(m, 1)
    :param word2index: a dictionary containing the each word mapped to its index
    :param max_len: maximum number of words in a sentence. You can assume every sentence in x is no longer than this.

    :return: x_indices: array of indices corresponding to words in the sentences from x, of shape(m, max_len)
    '''
    m = x.shape[0]  # Number of training examples
    # Inintial x_indices as a numpy matrix of zeors and the correct shape
    x_indices = np.zeros((m, max_seq_len))
    for i in range(m):
        # Split the sentences into a list of words
        words_list = x[i].split(' ')
        # oop over the words of words_list
        for j, word in enumerate(words_list):
            if j >= max_seq_len:
                break
            # Set the (i, j)th entry of x_indices to the index of the correct word
            if word2index.get(word) is not None:
                x_indices[i, j] = word2index[word]
    return x_indices


def load_dataset(max_seq_len, mode, word2index, word_level=True):
    # Load data and preprocessing
    if mode == 'test':
        path = os.path.join('dataset', 'test.csv')
    else:
        path = os.path.join('dataset', 'train.csv')
    question = pd.read_csv(QUESTION_ID_PATH)
    dataset = pd.read_csv(path)

    # Transfer qid in train.csv to concrete question string in question_id.csv
    dataset = pd.merge(dataset, question, left_on=['qid1'], right_on=['qid'], how='left')
    dataset = pd.merge(dataset, question, left_on=['qid2'], right_on=['qid'], how='left')

    if word_level:
        dataset = dataset[['label', 'wid_x', 'wid_y']]
    else:
        dataset = dataset[['label', 'cid_x', 'cid_y']]
    dataset.columns = ['label', 'q1', 'q2']

    q1_indices = sentences2indices(dataset.q1.values, word2index, max_seq_len).astype(np.int32)
    q2_indices = sentences2indices(dataset.q2.values, word2index, max_seq_len).astype(np.int32)
    print(mode + '_q1: ', q1_indices.shape)
    print(mode + '_q2: ', q2_indices.shape)

    if mode == 'test':
        return q1_indices, q2_indices
    else:
        label = dataset.label.values
        return q1_indices, q2_indices, label


def load_features(mode):
    # 读取手工特征
    features = pd.read_csv('./dataset/' + mode + '_features.csv')
    pick_columns = ['len_diff', 'edit_distance',
                    'adjusted_common_word_ratio', 'adjusted_common_char_ratio',
                    'pword_dside_rate', 'pword_oside_rate',
                    'pchar_dside_rate', 'pchar_oside_rate']

    features = features[pick_columns]
    print(features.info())
    return features


def load_embed_matrix(word_level):
    if word_level:
        embed_path = WORD_EMBEDDING_PATH
    else:
        embed_path = CHAR_EMBEDDING_PATH

    # Load word2vec
    word2vec = pd.read_csv(embed_path, sep='\t', header=None, index_col=0)

    # Create index dictionary
    word = word2vec.index.values
    word2index = dict([(word[i], i+1) for i in range(len(word))])
    index2word = dict([(i + 1, word[i]) for i in range(len(word))])

    # Generate embedding matrix
    vocab_len = len(word2index) + 1

    # Initialize the embedding matrix as numpy arrays of zeros
    embed_matrix = np.zeros((vocab_len, EMBEDDING_DIM))

    # Set each row 'index' of the embedding matrix to be word vector representation of the 'index'th word of the vocabulary
    for word, index in word2index.items():
        embed_matrix[index, :] = word2vec.loc[word].values
    return word2index, index2word, embed_matrix,
