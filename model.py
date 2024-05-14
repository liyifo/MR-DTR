
import torch
import torch.nn as nn
from typing import List
import dgl
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
import dgl.function as fn
from load_Graph_data import *


class PeriodicTimeEncoder(nn.Module):
    def __init__(self, embedding_dimension: int):
        super(PeriodicTimeEncoder, self).__init__()

        self.embedding_dimension = embedding_dimension
        self.scale_factor = (1 / (embedding_dimension // 2)) ** 0.5

        self.w = nn.Parameter(torch.randn(1, embedding_dimension // 2))
        self.b = nn.Parameter(torch.randn(1, embedding_dimension // 2))

    def forward(self, input_relative_time: torch.Tensor):
        """

        :param input_relative_time: shape (batch_size, temporal_feature_dimension) or (batch_size, max_neighbors_num, temporal_feature_dimension)
               input_time_dim = 1 since the feature denotes relative time (scalar)
        :return:
            time_encoding, shape (batch_size, embedding_dimension) or (batch_size, max_neighbors_num, embedding_dimension)
        """
        # print(input_relative_time.shape)
        # cos_encoding, shape (batch_size, embedding_dimension // 2) or (batch_size, max_neighbors_num, embedding_dimension // 2)
        cos_encoding = torch.cos(torch.matmul(input_relative_time, self.w) + self.b)
        # sin_encoding, shape (batch_size, embedding_dimension // 2) or (batch_size, max_neighbors_num, embedding_dimension // 2)
        sin_encoding = torch.sin(torch.matmul(input_relative_time, self.w) + self.b)

        # time_encoding, shape (batch_size, embedding_dimension) or (batch_size, max_neighbors_num, embedding_dimension)
        time_encoding = self.scale_factor * torch.cat([cos_encoding, sin_encoding], dim=-1)

        return time_encoding


class weighted_graph_conv(nn.Module):
    """
        Apply graph convolution over an input signal.
    """
    def __init__(self, in_features: int, out_features: int):
        super(weighted_graph_conv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, graph, node_features, edge_weights):
        graph = graph.local_var()
        # multi W first to project the features, with bias
        # (N, F) / (N, T, F)
        graph.ndata['n'] = node_features
        # edge_weights, shape (T, N^2)
        # one way: use dgl.function is faster and less requirement of GPU memory
        graph.edata['e'] = edge_weights.t().unsqueeze(dim=-1)  # (E, T, 1)
        graph.update_all(fn.u_mul_e('n', 'e', 'msg'), fn.sum('msg', 'h'))

        # another way: use user defined function, needs more GPU memory
        # graph.edata['e'] = edge_weights.t()
        # graph.update_all(self.gcn_message, self.gcn_reduce)

        node_features = graph.ndata.pop('h')
        output = self.linear(node_features)
        return output

    @staticmethod
    def gcn_message(edges):
        if edges.src['n'].dim() == 2:
            # (E, T, 1) (E, 1, F),  matmul ->  matmul (E, T, F)
            return {'msg': torch.matmul(edges.data['e'].unsqueeze(dim=-1), edges.src['n'].unsqueeze(dim=1))}

        elif edges.src['n'].dim() == 3:
            # (E, T, 1) (E, T, F),  mul -> (E, T, F)
            return {'msg': torch.mul(edges.data['e'].unsqueeze(dim=-1), edges.src['n'])}

        else:
            raise ValueError(f"wrong shape for edges.src['n'], the length of shape is {edges.src['n'].dim()}")

    @staticmethod
    def gcn_reduce(nodes):
        # propagate, the first dimension is nodes num in a batch
        # h, tensor, shape, (N, neighbors, T, F) -> (N, T, F)
        return {'h': torch.sum(nodes.mailbox['msg'], 1)}


class weighted_GCN(nn.Module):
    def __init__(self, in_features: int, hidden_sizes: List[int], out_features: int):
        super(weighted_GCN, self).__init__()
        gcns, relus, bns = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        # layers for hidden_size
        input_size = in_features
        for hidden_size in hidden_sizes:
            gcns.append(weighted_graph_conv(input_size, hidden_size))
            relus.append(nn.ReLU())
            bns.append(nn.BatchNorm1d(hidden_size))
            input_size = hidden_size

        # output layer
        gcns.append(weighted_graph_conv(hidden_sizes[-1], out_features))
        relus.append(nn.ReLU())
        bns.append(nn.BatchNorm1d(out_features))
        self.gcns, self.relus, self.bns = gcns, relus, bns

    def forward(self, graph: dgl.DGLGraph, node_features: torch.Tensor, edges_weight: torch.Tensor):
        """
        :param graph: a graph
        :param node_features: shape (n_1+n_2+..., n_features)
               edges_weight: shape (T, n_1^2+n_2^2+...)
        :return:
        """
        h = node_features
        for gcn, relu, bn in zip(self.gcns, self.relus, self.bns):
            # (n_1+n_2+..., T, features)
            h = gcn(graph, h, edges_weight)
            if h.shape[0] != 1:
                h = bn(h.transpose(1, -1)).transpose(1, -1)
            h = relu(h)
        return h


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, nodes_embedding_projection, patient_feature, drug_memory):

        stacked_inputs = torch.stack([nodes_embedding_projection, patient_feature, drug_memory], dim=1) # shape: (med_num, 3, embedding_dim)
        

        queries = self.query(stacked_inputs) # shape: (med_num, 3, embedding_dim)
        keys = self.key(stacked_inputs) # shape: (med_num, 3, embedding_dim)
        values = self.value(stacked_inputs) # shape: (med_num, 3, embedding_dim)
        

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.embed_dim ** 0.5) # shape: (med_num, 3, 3)
        attention_weights = self.softmax(attention_scores) # shape: (med_num, 3, 3)
        

        weighted_values = torch.matmul(attention_weights, values) # shape: (med_num, 3, embedding_dim)
        

        output = torch.sum(weighted_values, dim=1) # shape: (med_num, embedding_dim)
        
        return output

class stacked_weighted_GCN_blocks(nn.ModuleList):
    def __init__(self, *args, **kwargs):
        super(stacked_weighted_GCN_blocks, self).__init__(*args, **kwargs)

    def forward(self, *input):
        g, nodes_feature, edge_weights = input
        h = nodes_feature
        for module in self:
            h = module(g, h, edge_weights)
        return h

class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(torch.nn.Module):
    def __init__(self, voc_size, emb_dim, ehr_adj, ddi_adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        ehr_adj = self.normalize(ehr_adj + np.eye(ehr_adj.shape[0]))
        ddi_adj = self.normalize(ddi_adj + np.eye(ddi_adj.shape[0]))

        self.ehr_adj = torch.FloatTensor(ehr_adj).to(device)
        self.ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)
        self.gcn3 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        ehr_node_embedding = self.gcn1(self.x, self.ehr_adj)
        ddi_node_embedding = self.gcn1(self.x, self.ddi_adj)

        ehr_node_embedding = F.relu(ehr_node_embedding)
        ddi_node_embedding = F.relu(ddi_node_embedding)
        ehr_node_embedding = self.dropout(ehr_node_embedding)
        ddi_node_embedding = self.dropout(ddi_node_embedding)

        ehr_node_embedding = self.gcn2(ehr_node_embedding, self.ehr_adj)
        ddi_node_embedding = self.gcn3(ddi_node_embedding, self.ddi_adj)
        return ehr_node_embedding, ddi_node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx



class TimeRec_GCN(nn.Module):

    def __init__(self, voc_size: list, num_users: int, embedding_dimension: int,
                 embedding_dropout: float, temporal_attention_dropout: float, temporal_information_importance: float, 
                 ehr_adj, ddi_adj, 
                 device:int, hop_num: int=3, temporal_feature_dimension: int=1):
        """
        :param num_items: int, number of items
        :param num_users: int, number of users
        :param hop_num: int, , number of hops
        :param embedding_dimension: int, dimension of embedding
        :param temporal_feature_dimension: int, the input dimension of temporal feature
        :param embedding_dropout: float, embedding dropout rate
        :param temporal_attention_dropout: float, temporal attention dropout rate
        :param temporal_information_importance: float, importance of temporal information
        """
        super(TimeRec_GCN, self).__init__()

        self.voc_size = voc_size
        self.num_users = num_users
        self.hop_num = hop_num
        self.embedding_dimension = embedding_dimension
        self.temporal_feature_dimension = temporal_feature_dimension
        self.embedding_dropout = embedding_dropout
        self.temporal_attention_dropout = temporal_attention_dropout
        self.temporal_information_importance = temporal_information_importance

        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        # self.items_embedding = nn.Embedding(num_items, embedding_dimension)
        # change

        # self.diagnosis_encoder = torch.nn.GRU(embedding_dimension, embedding_dimension, batch_first=True)
        # self.procedure_encoder = torch.nn.GRU(embedding_dimension, embedding_dimension, batch_first=True)
        self.gru_fcn = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dimension*2, embedding_dimension),
            # torch.nn.GroupNorm(num_groups=8, num_channels=embedding_dimension)
        )

        self.med_gcn =  GCN(voc_size=voc_size[2], emb_dim=embedding_dimension, ehr_adj=ehr_adj, ddi_adj=ddi_adj, device=device)


        self.diagnosis_embedding = nn.Sequential(
            nn.Embedding(voc_size[0], embedding_dimension),
            nn.Dropout(embedding_dropout)
        )
        self.procedure_embedding = nn.Sequential(
            nn.Embedding(voc_size[1], embedding_dimension),
            nn.Dropout(embedding_dropout)
        )
        self.medication_embedding = nn.Sequential(
            nn.Embedding(voc_size[2], embedding_dimension),
            nn.Dropout(embedding_dropout)
        )

        self.users_embedding = nn.Embedding(num_users, embedding_dimension)

        self.leaky_relu_func = nn.LeakyReLU(negative_slope=0.2)

        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.temporal_attention_dropout = nn.Dropout(temporal_attention_dropout)

        self.nhead = 2
        self.diagnosis_transformer_encoder = nn.TransformerEncoderLayer(embedding_dimension, self.nhead, batch_first=True, dropout=0.2)
        self.procedure_transformer_encoder = nn.TransformerEncoderLayer(embedding_dimension, self.nhead, batch_first=True, dropout=0.2)
        # self.diag_self_attend = SelfAttend(embedding_dimension)
        # self.proc_self_attend = SelfAttend(embedding_dimension)


        # self.stacked_gcn = stacked_weighted_GCN_blocks([weighted_GCN(embedding_dimension,
        #                                                                     [embedding_dimension],
        #                                                                     embedding_dimension)])


        # the TMS dataset only has periodic temporal information, and thus does not need the semantic temporal information encoder

        self.periodic_time_encoder = PeriodicTimeEncoder(embedding_dimension=embedding_dimension)

        self.fc_projection = nn.Linear(hop_num * embedding_dimension, embedding_dimension)
        self.self_attention = SelfAttention(embedding_dimension)
        self.final_fcn = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dimension, 1)
        )
        self.device = device

        self.inter = torch.nn.Parameter(torch.FloatTensor(1))
        # self.inter2 = torch.nn.Parameter(torch.FloatTensor(1))
        self.inter3 = torch.nn.Parameter(torch.FloatTensor(1))
        self.inter4 = torch.nn.Parameter(torch.FloatTensor(1))
        self.init_weights()

    def forward(self, hops_nodes_indices: list, hops_nodes_temporal_features: list, central_nodes_temporal_feature, diagnosis_list, procedure_list, time_list):
        """

        :param hops_nodes_indices: list, shape (1 + hop_num, batch_size, max_neighbors_num)
        hop: 0 -> self, 1 -> 1 hop, 2 -> 2 hop ..., odd number -> item, even number -> user
        :param hops_nodes_temporal_features: list, shape, (1 + hop_num, batch_size, max_neighbors_num, temporal_feature_dimension)
        :param hops_nodes_length: list, shape (1 + hop_num, batch_size)
        :param central_nodes_temporal_feature: Tensor, shape (batch_size, temporal_feature_dimension)
        :return:
            set_prediction, shape (batch_size, num_items),
        """


        # shape (num_items, embedding_dimension)
        query_embeddings = self.medication_embedding(
            torch.LongTensor([i for i in range(self.voc_size[2])]).to(self.device))

        # list, shape (1 + hop_num, batch_size, embedding_dimension)
        nodes_hops_embedding = []
        # Transformer
        ## tensor and mask (visit_num, max_diag_num)

        diagnosis_feature = [torch.sum(self.diagnosis_embedding(torch.LongTensor(i).to(self.device)), keepdim=True, dim=0) for i in diagnosis_list] # (item_num,emb_dim) -> (emb_dim) -> (visit_num, emb_dim)
        procedure_feature = [torch.sum(self.procedure_embedding(torch.LongTensor(i).to(self.device)), keepdim=True, dim=0) for i in procedure_list]
        diagnosis_feature = torch.cat(diagnosis_feature, dim=0).unsqueeze(0)
        procedure_feature = torch.cat(procedure_feature, dim=0).unsqueeze(0) # (1,visit_num, emd_dim)

        

        d_mask_matrix = torch.triu(torch.ones(len(diagnosis_list), len(diagnosis_list)), diagonal=1).repeat(self.nhead,1,1).to(self.device) # (visit_num, max_diag_num)
 
        diagnosis_feature = self.diagnosis_transformer_encoder(diagnosis_feature, src_mask=d_mask_matrix) # (1,visit_num, emd_dim)
        procedure_feature = self.procedure_transformer_encoder(procedure_feature, src_mask=d_mask_matrix)
        diagnosis_feature = diagnosis_feature[:,-1,:]
        procedure_feature = procedure_feature[:,-1,:] # （1， emb_dim）
        



        patient_feature = torch.cat([diagnosis_feature, procedure_feature], dim=-1) # (1, embedding_dimension*2)
        patient_feature = self.gru_fcn(patient_feature) # (1, embedding_dimension)
        patient_feature = patient_feature * query_embeddings # （131， embedding_dimension）
        # patient_prediction = patient_feature @ query_embeddings.transpose(0, 1) # (1, med_num)
        ehr_embedding, ddi_embedding = self.med_gcn() # (med_num, embedding_dimension)
        drug_memory = ehr_embedding - ddi_embedding * self.inter

    


        # (batch_size) -> (batch_size, embedding_dimension)
        central_nodes_time_embedding = self.periodic_time_encoder(torch.Tensor([[central_nodes_temporal_feature]]).to(self.device))

        for hop_index in range(len(hops_nodes_indices)):
            # hop_nodes_indices -> tensor (batch_size, max_neighbors_num)
            # hop_nodes_temporal_features -> Tensor (batch_size, max_neighbors_num, temporal_feature_dimension)

            hop_nodes_indices, hop_nodes_temporal_features = hops_nodes_indices[hop_index], hops_nodes_temporal_features[hop_index]


            if hop_index % 2 == 0:
                # skip central node itself feature
                if hop_index == 0:
                    continue
                else:
                    # shape (batch_size, max_neighbors_num, embedding_dimension)
                    hop_nodes_embedding = self.users_embedding(torch.LongTensor([hop_nodes_indices]).to(self.device))
            else:
                # shape (batch_size, max_neighbors_num, embedding_dimension)
                hop_diagnosis_nodes_embedding = self.diagnosis_embedding(torch.LongTensor([hop_nodes_indices[0]]).to(self.device))
                hop_procedure_nodes_embedding = self.procedure_embedding(torch.LongTensor([hop_nodes_indices[1]]).to(self.device))
                hop_medication_nodes_embedding = self.medication_embedding(torch.LongTensor([hop_nodes_indices[2]]).to(self.device))
                hop_nodes_embedding = torch.cat([hop_diagnosis_nodes_embedding, hop_procedure_nodes_embedding, hop_medication_nodes_embedding], dim=1)


            hop_nodes_embedding = self.embedding_dropout(hop_nodes_embedding)
            
            # shape (batch_size, num_items, max_neighbors_num),  (num_items, embedding_dimension) einsum (batch_size, max_neighbors_num, embedding_dimension)
            # shape (1, num_items, neighbors_num)
            attention = torch.einsum('if,bnf->bin', query_embeddings, hop_nodes_embedding)
            # mask based on hops_nodes_length, shape (batch_size, num_items, max_neighbors_num)


            hop_nodes_time_embedding = self.periodic_time_encoder(torch.Tensor([hop_nodes_temporal_features]).unsqueeze(dim=-1).to(self.device))

            # shape (batch_size, num_items, max_neighbors_num),  (batch_size, num_items, embedding_dimension) einsum (batch_size, max_neighbors_num, embedding_dimension)
            temporal_attention = torch.einsum('bif,bnf->bin', torch.stack([central_nodes_time_embedding for _ in range(self.voc_size[2])], dim=1), hop_nodes_time_embedding)
            temporal_attention = self.temporal_attention_dropout(temporal_attention)
            # print(attention.shape, temporal_attention.shape, hop_nodes_time_embedding.shape, central_nodes_time_embedding.shape)

            #attention = attention + self.temporal_information_importance * temporal_attention

            attention = self.leaky_relu_func(attention)

            # shape (batch_size, med_num, max_neighbors_num)
            attention_scores = F.softmax(attention, dim=-1)

            # shape (batch_size, med_num, embedding_dimension),  (batch_size, med_num, max_neighbors_num) bmm (batch_size, max_neighbors_num, embedding_dimension)
            hop_embedding = torch.bmm(attention_scores, hop_nodes_embedding)


            nodes_hops_embedding.append(hop_embedding)

        # (batch_size, med_num, hop_num, embedding_dimension)
        nodes_hops_embedding = self.embedding_dropout(torch.stack(nodes_hops_embedding, dim=2))
        # make final prediction with concatenation operation
        # nodes_embedding_projection, shape (batch_size, med_num, embedding_dimension)
        nodes_embedding_projection = self.fc_projection(nodes_hops_embedding.flatten(start_dim=2))

        # nodes_embedding_projection = torch.cat([nodes_embedding_projection, patient_feature.unsqueeze(0), drug_memory.unsqueeze(0)], dim=2)

        #nodes_embedding_projection = nodes_embedding_projection.squeeze(0)
        #nodes_embedding_projection = nodes_embedding_projection@(nodes_embedding_projection.t()@patient_feature) + patient_feature + drug_memory@(drug_memory.t()@patient_feature)
        


        nodes_embedding_projection = self.inter4*nodes_embedding_projection.squeeze(0) + patient_feature + self.inter3*drug_memory

        #set_prediction = self.self_attention(nodes_embedding_projection, patient_feature, drug_memory)

        
        # set_prediction, shape (1, med_num),   (1, med_num, embedding_dimension) * (med_num, embedding_dimension)
        # set_prediction = (nodes_embedding_projection * query_embeddings).sum(dim=-1)
        

        # set_prediction = set_prediction + patient_prediction
        #set_prediction = set_prediction 


        
        set_prediction = self.final_fcn(nodes_embedding_projection).t()
        #set_prediction = self.final_fcn(nodes_embedding_projection).squeeze(2)
        

        neg_pred_prob = torch.sigmoid(set_prediction)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return set_prediction, batch_neg
    
    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1

        self.diagnosis_embedding[0].weight.data.uniform_(-initrange, initrange)
        self.procedure_embedding[0].weight.data.uniform_(-initrange, initrange)
        self.medication_embedding[0].weight.data.uniform_(-initrange, initrange)
        self.users_embedding.weight.data.uniform_(-initrange, initrange)
