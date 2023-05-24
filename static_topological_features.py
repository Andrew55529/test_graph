import pandas as pd
import math
import random
import numpy as np
from collections import deque
import collections
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import RocCurveDisplay


# dataset_path = 'datasets/' + 'test' + '.txt'
# dataset_path = 'datasets/' + 'out.radoslaw_email_email' + '.txt'
# dataset_path = 'datasets/' + 'out.prosper-loans' + '.txt'
# data = pd.read_csv(dataset_path, sep='\s+', names=['id_from', 'id_to', 'weight', 'time'], header=None)

def common_neighbours(u, v, adjacency_list):
    return len(set(adjacency_list[u]) & set(adjacency_list[v]))


def adamic_adar(u, v, adjacency_list):
    AA = 0
    if u == v:
        return AA
    else:
        Z = set(adjacency_list[u]) & set(adjacency_list[v])
        for z in Z:
            AA += 1 / math.log(len(adjacency_list[z]))
        return AA


def jaccard_coefficient(u, v, adjacency_list):
    return len(set(adjacency_list[u]) & set(adjacency_list[v])) / len(set(adjacency_list[u]) | set(adjacency_list[v]))


def preferential_attachment(u, v, adjacency_list):
    return len(adjacency_list[u]) * len(adjacency_list[v])


def get_static_topological_features(adjacency_list_until_s, edges_p, edges_n):
    # Заменить ребра на
    for i in range(len(edges_p)):
        CN = common_neighbours(edges_p[i][0], edges_p[i][1], adjacency_list_until_s)
        AA = adamic_adar(edges_p[i][0], edges_p[i][1], adjacency_list_until_s)
        JC = jaccard_coefficient(edges_p[i][0], edges_p[i][1], adjacency_list_until_s)
        PA = preferential_attachment(edges_p[i][0], edges_p[i][1], adjacency_list_until_s)
        edges_p[i] = [CN, AA, JC, PA]

    for i in range(len(edges_n)):
        CN = common_neighbours(edges_n[i][0], edges_n[i][1], adjacency_list_until_s)
        AA = adamic_adar(edges_n[i][0], edges_n[i][1], adjacency_list_until_s)
        JC = jaccard_coefficient(edges_n[i][0], edges_n[i][1], adjacency_list_until_s)
        PA = preferential_attachment(edges_n[i][0], edges_n[i][1], adjacency_list_until_s)
        edges_n[i] = [CN, AA, JC, PA]

    X_new = edges_n + edges_p;
    Y_new = [0] * len(edges_n) + [1] * len(edges_p);
    # Делим 25% тестовых
    model = LogisticRegression()
    X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y_new, test_size=0.25, random_state=0)
    # Обучение
    model.fit(X_train, Y_train)
    accuracy = accuracy_score(Y_test, model.predict(X_test))
    auc_roc = roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1])
    print(classification_report(Y_test, model.predict(X_test)))
    svc_disp = RocCurveDisplay.from_estimator(model, X_test, Y_test)
    plt.show()