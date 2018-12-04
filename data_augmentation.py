# -*- coding: utf-8 -*-
import numpy as np

def all_pair_dijkstra(train_graph, connected_component, max_distance):
    m = len(connected_component)
    cc = list(connected_component)
    distance = {}

    matrix = np.zeros((m, m))

    for i in range(m):
        dist = dijk

def generate_graph(train):
    """
    把输入数据转化为以字典表示的无向图
    :param train:
    :return:
    """