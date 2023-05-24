import pandas as pd
import math
import random
import numpy as np
from collections import deque
import collections

def get_adjacency_list(data):
    adjacency_list = dict({})
    edges_r = dict({})
    time = []
    count_edges = 0
    is_loop = False
    for row in data.itertuples():
        #Список смежности для все ребер за все время
        count_add = 0
        if (row[1] in adjacency_list):
            if (row[2] not in adjacency_list[row[1]]):
                adjacency_list[row[1]].append(row[2])
                count_add += 1
        else:
            adjacency_list[row[1]] = [row[2]]
            count_add += 1
        if (row[2] in adjacency_list):
            if (row[1] not in adjacency_list[row[2]]):
                adjacency_list[row[2]].append(row[1])
                count_add += 1
        else:
            adjacency_list[row[2]] = [row[1]]
            count_add += 1

        if count_add >= 1:
          if count_add == 1:
              is_loop = True
          count_edges += 1

        #Заносим все ребра
        if (row[1] < row[2]):
            fir = row[1]
            sec = row[2]
        else:
            fir = row[2]
            sec = row[1]
        if (fir, sec) in edges_r:
            edges_r[(fir, sec)].append(row[4])
        else:
            edges_r[(fir, sec)] = [row[4]]

        time.append(row[4])
    return [adjacency_list, min(time), max(time), edges_r, count_edges, is_loop]


def get_adjacency_list_until_s(data, s):
    adjacency_list_until_s = dict({})
    for row in data.itertuples():
        #Список смежности до момента s
        if row[4] < s:
            if (row[1] in adjacency_list_until_s):
                if (row[2] not in adjacency_list_until_s[row[1]]):
                    adjacency_list_until_s[row[1]].append(row[2])
            else:
                adjacency_list_until_s[row[1]] = [row[2]]
            if (row[2] in adjacency_list_until_s):
                if (row[1] not in adjacency_list_until_s[row[2]]):
                    adjacency_list_until_s[row[2]].append(row[1])
            else:
                adjacency_list_until_s[row[2]] = [row[1]]

    return adjacency_list_until_s


def floyd_warshall(adjacency_list):
    d = [[[] for _ in range(len(adjacency_list))] for __ in range(len(adjacency_list))]
    for node_1 in adjacency_list:
        for node_2 in adjacency_list:
            d[list(adjacency_list).index(node_1)][list(adjacency_list).index(node_2)] = [math.inf, node_1, node_2]

    for node in adjacency_list:
        for neighbour in adjacency_list[node]:
            d[list(adjacency_list).index(node)][list(adjacency_list).index(neighbour)] = [1, node, neighbour]

    for i in range(len(d)):
        d_i = [d[j][:] for j in range(len(d))]
        for u in range(len(d)):
            for v in range(len(d)):
                d[u][v][0] = min(d_i[u][v][0], d_i[u][i][0] + d_i[i][v][0])

    for i in range(len(d)):
        d[i][i][0] = 0
    return d

def get_info_dataset(data, q):
    adjacency_list, tmin, tmax, edges_r, count_edges, is_loop = get_adjacency_list(data)
    ver_count = len(adjacency_list)
    print(ver_count)

    s = (tmax + tmin) * q
    print(tmin, tmax, s)

    adjacency_list_until_s = get_adjacency_list_until_s(data, s)
    d = [[[math.inf, node, neighbour] for neighbour in adjacency_list_until_s[node]] for node in adjacency_list_until_s]
    pairs_dist_2 = set()
    d = floyd_warshall(adjacency_list_until_s)

    for i in range(len(d)):
        for j in range(len(d)):
            if d[i][j][0] == 2 and ((d[i][j][1], d[i][j][2]) not in pairs_dist_2) and (
                    (d[i][j][2], d[i][j][1]) not in pairs_dist_2):
                pairs_dist_2.add((d[i][j][1], d[i][j][2]))

    # Все ребра которых небыло до s
    edges_p = []
    edges_n = []
    for i in range(1, ver_count):
        for j in range(i + 1, ver_count):
            if (j in adjacency_list[i]):
                if (min(edges_r[(i, j)]) >= s):
                    edges_p.append((i, j))
            else:
                edges_n.append((i, j))

    # Ограничить расстоянием
    edges_p = list(set(edges_p) & pairs_dist_2)
    edges_n = list(set(edges_n) & pairs_dist_2)

    return adjacency_list, adjacency_list_until_s, edges_r, edges_p, edges_n, tmin, tmax, count_edges, is_loop