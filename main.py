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

from all_info_dataset import get_info_dataset
from graph_properties import get_graph_properties
from static_topological_features import get_static_topological_features

# dataset_path = 'datasets/' + 'test' + '.txt'
dataset_path = 'datasets/' + 'out.radoslaw_email_email' + '.txt'
# dataset_path = 'datasets/' + 'out.prosper-loans' + '.txt'
data = pd.read_csv(dataset_path, sep='\s+', names=['id_from', 'id_to', 'weight', 'time'], header=None)

q = 0.5
adjacency_list, adjacency_list_until_s, edges_r, edges_p, edges_n, tmin, tmax, count_edges, is_loop = get_info_dataset(data, q)
get_graph_properties(adjacency_list, count_edges, is_loop)
get_static_topological_features(adjacency_list_until_s, edges_p, edges_n)
