import numpy as np
from collections import defaultdict

import os


def load_cora(content = 'cora.content', cites = 'cora.cites'):
    if content == 'cora.content' and cites == 'cora.cites':
        content = os.path.dirname(os.path.abspath(__file__))+'/'+content
        cites = os.path.dirname(os.path.abspath(__file__))+'/'+cites
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open(content) as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = [float(i) for i in info[1:-1]]
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open(cites) as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists