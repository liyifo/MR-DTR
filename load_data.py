
import torch
import dill
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
import itertools
from collections import defaultdict
import dgl
import numpy as np
import  torch.nn.functional as F
from sklearn.preprocessing import normalize



def load_mimic3(device, args):
    data_path = "./data/MIMIC-III/records_final.pkl"
    voc_path = "./data//MIMIC-III/voc_final.pkl"

    ddi_adj_path = "./data/MIMIC-III/ddi_A_final.pkl"
    ddi_mask_path = "./data/MIMIC-III/ddi_mask_H.pkl"
    ehr_adj_path = './data/MIMIC-III/ehr_adj_final.pkl'
    #molecule_path = "./data/MIMIC-III/atc3toSMILES.pkl"

    with open(ddi_adj_path, 'rb') as Fin:
        ddi_adj = dill.load(Fin)
    with open(data_path, 'rb') as Fin:
        data = dill.load(Fin)
    # with open(molecule_path, 'rb') as Fin:
    #     molecule = dill.load(Fin)
    with open(voc_path, 'rb') as Fin:
        voc = dill.load(Fin)
    with open(ehr_adj_path, 'rb') as Fin:
        ehr_adj = dill.load(Fin)

    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
    voc_size = (
        len(diag_voc.idx2word),
        len(pro_voc.idx2word),
        len(med_voc.idx2word)
    )

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point : split_point + eval_len]
    data_eval = data[split_point + eval_len :]

    print(f"Diag num:{len(diag_voc.idx2word)}")
    print(f"Proc num:{len(pro_voc.idx2word)}")
    print(f"Med num:{len(med_voc.idx2word)}")
    
    
    return data_train, data_eval, data_test, voc_size, ddi_adj, ehr_adj


def dnntsp_process(med, item_embedding_matrix, device):

    def get_nodes(baskets):
        # convert tensor to int
        # baskets = [basket.tolist() for basket in baskets]
        items = torch.tensor(list(set(itertools.chain.from_iterable(baskets))))
        return items
    def convert_to_gpu(data):
        if device != -1 and torch.cuda.is_available():
            data = data.to(device)
        return data
    def get_edges_weight(baskets):
        edges_weight_dict = defaultdict(float)
        for basket in baskets:
            # basket = basket.tolist()
            for i in range(len(basket)):
                for j in range(i + 1, len(basket)):
                    edges_weight_dict[(basket[i], basket[j])] += 1.0
                    edges_weight_dict[(basket[j], basket[i])] += 1.0
        return edges_weight_dict

    user_data = med
    nodes = get_nodes(baskets=user_data[:-1])


    nodes_feature = item_embedding_matrix(convert_to_gpu(nodes).long())
    # print(nodes.shape)
    project_nodes = torch.tensor(list(range(nodes.shape[0])))
    # construct fully connected graph, containing N nodes, unweighted
    # (0, 0), (0, 1), ..., (0, N-1), (1, 0), (1, 1), ..., (1, N-1), ...
    # src -> [0, 0, 0, ... N-1, N-1, N-1, ...],  dst -> [0, 1, ..., N-1, ..., 0, 1, ..., N-1]

    src = torch.stack([project_nodes for _ in range(project_nodes.shape[0])], dim=1).flatten().tolist()
    dst = torch.stack([project_nodes for _ in range(project_nodes.shape[0])], dim=0).flatten().tolist()
    g = dgl.graph((src, dst), num_nodes=project_nodes.shape[0])


    edges_weight_dict = get_edges_weight(user_data[:-1])
    # add self-loop
    for node in nodes.tolist():
        if edges_weight_dict[(node, node)] == 0.0:
            edges_weight_dict[(node, node)] = 1.0
    # normalize weight
    
    max_weight = max(edges_weight_dict.values())
    for i, j in edges_weight_dict.items():
        edges_weight_dict[i] = j / max_weight
    # get edge weight for each timestamp, shape (T, N*N)
    # print(edges_weight_dict)

    edges_weight = []
    for basket in user_data[:-1]:
        # basket = basket.tolist()
        # list containing N * N weights of elements
        edge_weight = []
        for node_1 in nodes.tolist():
            for node_2 in nodes.tolist():
                if (node_1 in basket and node_2 in basket) or (node_1 == node_2):
                    # each node has a self connection
                    edge_weight.append(edges_weight_dict[(node_1, node_2)])
                else:
                    edge_weight.append(0.0)
        edges_weight.append(torch.Tensor(edge_weight))
    # tensor -> shape (T, N*N)
    edges_weight = torch.stack(edges_weight).to(device)

    return g, nodes_feature, edges_weight, nodes, user_data

def collate_set_across_user(batch_data, item_total, device):
    def get_truth_data(truth_data):
        truth_list = []
        for basket in truth_data:
            one_hot_items = F.one_hot(torch.tensor(basket), num_classes=item_total)
            one_hot_basket, _ = torch.max(one_hot_items, dim=0)
            truth_list.append(one_hot_basket)
        truth = torch.stack(truth_list)
        return truth

    def convert_to_gpu(data):
        if device != -1 and torch.cuda.is_available():
            data = data.to(device)
        return data
    
    def convert_all_data_to_gpu(*data):
        res = []
        for item in data:
            item = convert_to_gpu(item)
            res.append(item)
        return tuple(res)

    ret = list()
    for idx, item in enumerate(zip(*batch_data)):
        # assert type(item) == tuple

        if isinstance(item[0], dgl.DGLGraph):
            ret.append(dgl.batch(item))
        elif isinstance(item[0], torch.Tensor):

            if idx == 2:
                # pad edges_weight sequence in time dimension batch, (T, N*N)
                # (T_max, N*N)
                max_length = max([data.shape[0] for data in item])
                edges_weight, lengths = list(), list()
                for data in item:
                    if max_length != data.shape[0]:
                        edges_weight.append(torch.cat((data, torch.stack(
                            [torch.eye(int(data.shape[1] ** 0.5)).flatten() for _ in range(max_length - data.shape[0])],
                            dim=0)), dim=0))
                    else:
                        edges_weight.append(data)
                    lengths.append(data.shape[0])
                # (T_max, N_1*N_1 + N_2*N_2 + ... + N_b*N_b)
                ret.append(torch.cat(edges_weight, dim=1))
                # (batch, )
                ret.append(torch.tensor(lengths))
            else:
                # nodes_feature -> (N_1 + N_2, .. + N_b, item_embedding) or nodes -> (N_1 + N_2, .. + N_b, )
                ret.append(torch.cat(item, dim=0))
        # user_data
        elif isinstance(item[0], list):
            data_list = item
        else:
            raise ValueError(f'batch must contain tensors or graphs; found {type(item[0])}')

    truth_data = get_truth_data([dt[-1] for dt in data_list])
    ret.append(truth_data)

    # tensor (batch, items_total), for frequency calculation
    users_frequency = np.zeros([len(batch_data), item_total])
    for idx, baskets in enumerate(data_list):
        for basket in baskets:
            for item in basket:
                users_frequency[idx, item] = users_frequency[idx, item] + 1
    users_frequency = normalize(users_frequency, axis=1, norm='max')
    ret.append(torch.Tensor(users_frequency))


    # (g, nodes_feature, edges_weight, lengths, nodes, truth_data, individual_frequency)
    g, nodes_feature, edges_weight, lengths, nodes, truth_data, individual_frequency = ret
    return convert_all_data_to_gpu(g, nodes_feature, edges_weight, lengths, nodes, truth_data, individual_frequency)


def read_data_inf(dataset):
    if dataset == 'mimic3':
        voc_path = "./data/voc_final.pkl"
        ddi_adj_path = "./data/ddi_A_final.pkl"
        data_path = "./data/records_final.pkl"
        ehr_adj_path = './data/ehr_adj_final.pkl'

        with open(ddi_adj_path, 'rb') as Fin:
            ddi_adj = dill.load(Fin)
        with open(data_path, 'rb') as Fin:
            data = dill.load(Fin)
        with open(voc_path, 'rb') as Fin:
            voc = dill.load(Fin)
        with open(ehr_adj_path, 'rb') as Fin:
            ehr_adj = dill.load(Fin)

        diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
        voc_size = (
            len(diag_voc.idx2word),
            len(pro_voc.idx2word),
            len(med_voc.idx2word)
        )
    
        return voc_size, ddi_adj, ehr_adj, len(data)
    else:
        voc_path = "./data/voc_final_4.pkl"
        ddi_adj_path = "./data/ddi_A_final_4.pkl"
        data_path = "./data/records_final_4.pkl"
        ehr_adj_path = './data/ehr_adj_final_4.pkl'

        with open(ddi_adj_path, 'rb') as Fin:
            ddi_adj = dill.load(Fin)
        with open(data_path, 'rb') as Fin:
            data = dill.load(Fin)
        with open(voc_path, 'rb') as Fin:
            voc = dill.load(Fin)
        with open(ehr_adj_path, 'rb') as Fin:
            ehr_adj = dill.load(Fin)

        diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
        voc_size = (
            len(diag_voc.idx2word),
            len(pro_voc.idx2word),
            len(med_voc.idx2word)
        )
    
        return voc_size, ddi_adj, ehr_adj, len(data)
