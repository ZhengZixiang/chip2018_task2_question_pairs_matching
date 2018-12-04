# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def select_columns(data, question, word_level, need_label):
    data = pd.merge(data, question, left_on=['qid1'], right_on=['qid'], how='left')
    data = pd.merge(data, question, left_on=['qid2'], right_on=['qid'], how='left')
    if word_level:
        if need_label:
            data = data[['label', 'wid_x', 'wid_y']]
        else:
            data = data[['wid_x', 'wid_y']]
    else:
        if need_label:
            data = data[['label', 'cid_x', 'cid_y']]
        else:
            data = data[['cid_x', 'cid_y']]

    if need_label:
        data.columns = ['label', 'q1', 'q2']
    else:
        data.columns = ['q1', 'q2']
    return data


def get_len_diff(data):
    '''
    Get the difference of length and normalize by the longest one of question pairs
    '''
    q1_len = data.q1.apply(lambda x: len(x.split(' '))).values
    q2_len = data.q2.apply(lambda x: len(x.split(' '))).values
    len_diff = np.abs(q1_len - q2_len) / np.max([q1_len, q2_len], axis=0)

    return len_diff


def get_num_common_words(question, data):
    '''
    get the common words in q1 and q2
    '''
    q1_word_set = data.q1.apply(lambda x: x.split(' ')).apply(set).values
    q2_word_set = data.q2.apply(lambda x: x.split(' ')).apply(set).values

    result = [len(q1_word_set[i] & q2_word_set[i]) for i in range(len(q1_word_set))]
    result = pd.DataFrame(result, index=data.index)
    result.columns = ['num_common_words']
    return result


def get_common_words_ratio(data):
    q1_word_set = data.q1.apply(lambda x: x.split(' ')).apply(set).values
    q2_word_set = data.q2.apply(lambda x: x.split(' ')).apply(set).values
    q1_word_len = data.q1.apply(lambda x: len(x.split(' '))).values
    q2_word_len = data.q2.apply(lambda x: len(x.split(' '))).values

    result = [len(q1_word_set[i] & q2_word_set[i])/max(q1_word_len[i], q2_word_len[i]) for i in range(len(q1_word_set))]
    result = pd.DataFrame(result, index=data.index)
    result.columns = ['common_word_ratio']
    return result


def get_tfidf_vector(data, vectorizer):
    # vectorizer = TfidfVectorizer().fit(question.wid.values)
    q1_tfidf = vectorizer.transform(data.q1.values)
    q2_tfidf = vectorizer.transform(data.q2.values)

    return vectorizer.vocabulary_, q1_tfidf, q2_tfidf


def adjust_common_word_ratio_by_tfidf(data, word2index, q1_tfidf, q2_tfidf):
    adjusted_common_words_ratio = []
    for i in range(q1_tfidf.shape[0]):
        q1_words = {}
        q2_words = {}
        for word in data.loc[i, 'q1'].lower().split():
            q1_words[word] = q1_words.get(word, 0) + 1
        for word in data.loc[i, 'q2'].lower().split():
            q2_words[word] = q2_words.get(word, 0) + 1

        sum_shared_word_in_q1 = sum([q1_words[w] * q1_tfidf[i, word2index[w]] for w in q1_words if w in q2_words])
        sum_shared_word_in_q2 = sum([q2_words[w] * q2_tfidf[i, word2index[w]] for w in q2_words if w in q1_words])
        sum_total = sum([q1_words[w] * q1_tfidf[i, word2index[w]] for w in q1_words]) + sum([q2_words[w] * q2_tfidf[i, word2index[w]] for w in q2_words])
        if 1e-6 > sum_total:
            adjusted_common_words_ratio.append(0.)
        else:
            adjusted_common_words_ratio.append(1.0 * (sum_shared_word_in_q1 + sum_shared_word_in_q2) / sum_total)

    return adjusted_common_words_ratio



def generate_powerful_word(data):
    """
    计算数据中词语的影响力
    0. 出现语句对数量
    1. 出现语句对比例
    2. 正确语句对比例
    3. 单侧语句对比例
    4. 单侧语句对正确比例
    5. 双侧语句对比例
    6. 双侧语句对正确比例
    """
    words_power = {}
    for i in data.index:
        label = int(data.loc[i, 'label'])
        q1_words = data.loc[i, 'q1'].lower().split()
        q2_words = data.loc[i, 'q2'].lower().split()
        all_words = set(q1_words + q2_words)
        q1_words = set(q1_words)
        q2_words = set(q2_words)
        for word in all_words:
            if word not in words_power:
                words_power[word] = [0. for i in range(7)]
            words_power[word][0] += 1.
            words_power[word][1] += 1.

            if (word in q1_words and word not in q2_words) or (word not in q1_words and word in q2_words):
                # 计算单侧语句数量
                words_power[word][3] += 1.
                if 0 == label:
                    words_power[word][2] += 1.
                    words_power[word][4] += 1.

            if word in q1_words and word in q2_words:
                # 计算双侧语句数量
                words_power[word][5] += 1.
                if 1 == label:
                    # 计算正确语句对比例
                    words_power[word][2] += 1.
                    words_power[word][6] += 1.

    for word in words_power:
        # 计算出现语句对比例
        words_power[word][1] /= data.shape[0]
        # 计算正确语句对比例
        words_power[word][2] /= data.shape[0]
        # 计算单侧语句对正确比例
        if words_power[word][3] > 1e-6:
            words_power[word][4] /= words_power[word][3]
        # 计算单侧语句对比例
        words_power[word][3] /= words_power[word][0]
        # 计算双侧语句对正确比例
        if words_power[word][5] > 1e-6:
            words_power[word][6] /= words_power[word][5]
        # 计算双侧语句对比例
        words_power[word][5] /= words_power[word][0]

    sorted_words_power = sorted(words_power.items(), key=lambda d: d[1][0], reverse=True)

    return sorted_words_power


def powerful_words_dside_tag(pword, data, threshold_num, threshold_rate):
    """
    若问题两侧存在有预测力的powerful words，则设置标签为1，否则为0
    """

    # 筛选powerful words（有预测力的）
    pword_dside = []
    pword = filter(lambda x: x[1][0] * x[1][5] >= threshold_num, pword)  # 保证统计可靠性
    pword_sort = sorted(pword, key=lambda d: d[1][6], reverse=True)
    pword_dside.extend(map(lambda x: x[0], filter(lambda x: x[1][6] >= threshold_rate, pword_sort)))  # 保证抽取到真正powerful的word

    pword_dside_tags = []
    for i in data.index:
        tags = []
        q1_words = set(data.loc[i, 'q1'].lower().split())
        q2_words = set(data.loc[i, 'q2'].lower().split())
        for word in pword_dside:
            if word in q1_words and word in q2_words:
                tags.append(1.0)
            else:
                tags.append(0.0)
        pword_dside_tags.append(tags)
    return pword_dside, pword_dside_tags


def powerful_words_oside_tag(pword, data, threshold_num, threshold_rate):
    pword_oside = []
    pword = filter(lambda x: x[1][0] * x[1][3] >= threshold_num, pword)
    pword_oside.extend(map(lambda x: x[0], filter(lambda x: x[1][4] >= threshold_rate, pword)))

    pword_oside_tags = []
    for i in data.index:
        tags = []
        q1_words = set(data.loc[i, 'q1'].lower().split())
        q2_words = set(data.loc[i, 'q2'].lower().split())
        for word in pword_oside:
            if word in q1_words and word not in q2_words:
                tags.append(1.0)
            elif word not in q1_words and word in q2_words:
                tags.append(1.0)
            else:
                tags.append(0.0)

        pword_oside_tags.append(tags)
    return pword_oside, pword_oside_tags


def powerful_words_dside_rate(sorted_words_power, pword_dside, data):
    num_least = 300
    words_power = dict(sorted_words_power)
    pword_dside_rate = []
    for i in data.index:
        rate = 1.0
        q1_words = set(data.loc[i, 'q1'].lower().split())
        q2_words = set(data.loc[i, 'q2'].lower().split())
        share_words = list(q1_words.intersection(q2_words))
        for word in share_words:
            if word in pword_dside:
                rate *= (1.0 - words_power[word][6])
        pword_dside_rate.append(1-rate)

    return pword_dside_rate


def powerful_words_oside_rate(sorted_words_power, pword_oside, data):
    num_least = 300

    words_power = dict(sorted_words_power)
    pword_oside_rate = []
    for i in data.index:
        rate = 1.0
        q1_words = set(data.loc[i, 'q1'].lower().split())
        q2_words = set(data.loc[i, 'q2'].lower().split())
        q1_diff = list(set(q1_words).difference(set(q2_words)))
        q2_diff = list(set(q2_words).difference(set(q1_words)))
        all_diff = set(q1_diff + q2_diff)
        for word in all_diff:
            if word in pword_oside:
                rate *= (1.0 - words_power[word][4])
        pword_oside_rate.append(1-rate)

    return pword_oside_rate


def edit_distance(q1, q2):
    str1 = q1.split(' ')
    str2 = q2.split(' ')
    matrix = [[i+j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i-1] == str2[j-1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+d)

        if i >1 and j > 1 and str1[i-1] == str2[j-2] and str1[i-2] == str2[j-1]:
            d = 0
            matrix[i][j] = min(matrix[i][j], matrix[i-2][j-2]+d)


    return matrix[len(str1)][len(str2)]


def get_edit_distance(data):
    q1_len = data['q1'].apply(lambda x: len(x.split(' '))).values
    q2_len = data['q2'].apply(lambda x: len(x.split(' '))).values

    # normalize the edit_distance by the max(len(q1), len(q2))
    dist = [edit_distance(data.loc[i, 'q1'], data.loc[i, 'q2']) / np.max([q1_len, q2_len], axis=0)[i] for i in data.index]

    return dist


def generate_features_csv(kind):
    question = pd.read_csv('./dataset/question_id.csv')
    raw = pd.read_csv('./dataset/' + kind + '.csv')
    unlabeled_data = select_columns(raw, question, True, False)
    labeled_data = select_columns(pd.read_csv('./dataset/train.csv'), question, True, True)

    len_diff = get_len_diff(unlabeled_data)

    vocab, q1_tfidf, q2_tfidf = get_tfidf_vector(unlabeled_data, TfidfVectorizer().fit(question.wid.values))
    adjusted_common_word_ratio = adjust_common_word_ratio_by_tfidf(unlabeled_data, vocab, q1_tfidf, q2_tfidf)

    edit = get_edit_distance(unlabeled_data)

    sorted_words_power = generate_powerful_word(labeled_data)
    pword_dside, pword_dside_tags = powerful_words_dside_tag(sorted_words_power, unlabeled_data, 1, 0.7)
    pword_dside_rate = powerful_words_dside_rate(sorted_words_power, pword_dside, unlabeled_data)
    pword_oside, pword_oside_tags = powerful_words_oside_tag(sorted_words_power, unlabeled_data, 1, 0.7)
    pword_oside_rate = powerful_words_oside_rate(sorted_words_power, pword_oside, unlabeled_data)

    unlabeled_data = select_columns(raw, question, False, False)
    labeled_data = select_columns(pd.read_csv('./dataset/train.csv'), question, False, True)

    vocab, q1_tfidf, q2_tfidf = get_tfidf_vector(unlabeled_data, TfidfVectorizer().fit(question.cid.values))
    adjusted_common_char_ratio = adjust_common_word_ratio_by_tfidf(unlabeled_data, vocab, q1_tfidf, q2_tfidf)

    sorted_chars_power = generate_powerful_word(labeled_data)
    pchar_dside, pchar_dside_tags = powerful_words_dside_tag(sorted_chars_power, unlabeled_data, 1, 0.7)
    pchar_dside_rate = powerful_words_dside_rate(sorted_chars_power, pchar_dside, unlabeled_data)
    pchar_oside, pchar_oside_tags = powerful_words_oside_tag(sorted_chars_power, unlabeled_data, 1, 0.7)
    pchar_oside_rate = powerful_words_oside_rate(sorted_chars_power, pchar_oside, unlabeled_data)

    df = pd.DataFrame({'len_diff': len_diff, 'edit_distance': edit,
                       'adjusted_common_word_ratio': adjusted_common_word_ratio, 'adjusted_common_char_ratio': adjusted_common_char_ratio,
                       'pword_dside_rate': pword_dside_rate, 'pword_oside_rate': pword_oside_rate,
                       'pchar_dside_rate': pchar_dside_rate, 'pchar_oside_rate': pchar_oside_rate})
    df.to_csv('./dataset/' + kind + '_features.csv', index=False)


if __name__ == '__main__':
    generate_features_csv('train')
    generate_features_csv('test')
