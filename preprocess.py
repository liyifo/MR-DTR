
import dill
from collections import defaultdict
import random
import pickle

def read_mimic3():

    data_path = "./data/records_final.pkl"
    voc_path = "./data/voc_final.pkl"

    ddi_adj_path = "./data/ddi_A_final.pkl"
    ddi_mask_path = "./data/ddi_mask_H.pkl"
    ehr_adj_path = './data/ehr_adj_final.pkl'
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

    print(f"Diag num:{len(diag_voc.idx2word)}")
    print(f"Proc num:{len(pro_voc.idx2word)}")
    print(f"Med num:{len(med_voc.idx2word)}")
    
    graph = build_graph(data)
    

    split_point = int(len(data) * 2 / 3)
    eval_len = int(len(data[split_point:]) / 2)

    generate_data(graph, 0, split_point, 'train', data)
    generate_data(graph, split_point, split_point + eval_len, 'eval', data)
    generate_data(graph, split_point + eval_len, len(data), 'test', data)


def read_mimic4():

    data_path = "./data/records_final_4.pkl"
    voc_path = "./data/voc_final_4.pkl"

    ddi_adj_path = "./data/ddi_A_final_4.pkl"
    ddi_mask_path = "./data/ddi_mask_H_4.pkl"
    ehr_adj_path = './data/ehr_adj_final_4.pkl'
    #molecule_path = "./data/MIMIC-III/atc3toSMILES.pkl"


    with open(data_path, 'rb') as Fin:
        data = dill.load(Fin)
    # with open(molecule_path, 'rb') as Fin:
    #     molecule = dill.load(Fin)
    with open(voc_path, 'rb') as Fin:
        voc = dill.load(Fin)


    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
    voc_size = (
        len(diag_voc.idx2word),
        len(pro_voc.idx2word),
        len(med_voc.idx2word)
    )

    print(f"Diag num:{len(diag_voc.idx2word)}")
    print(f"Proc num:{len(pro_voc.idx2word)}")
    print(f"Med num:{len(med_voc.idx2word)}")
    
    graph = build_graph(data)
    

    split_point = int(len(data) * 2 / 3)
    eval_len = int(len(data[split_point:]) / 2)

    generate_data(graph, 0, split_point, 'train', data)
    generate_data(graph, split_point, split_point + eval_len, 'eval', data)
    generate_data(graph, split_point + eval_len, len(data), 'test', data)

def build_graph(data_list):
    '''
    init: 
    {'patient': {
        patient_id: {
            'diagnosis': {diagnosis_id: []}, 
            'procedure': {}, 
            'medication': {}
            }, 
        },
    'diagnosis': {diagnosis_id: {patient_id: []}, 
    'procedure': {}, 
    'medication': {}, 
    'temporal_feature': {},
    'label': {}}
    '''
    transformed_data = {'patient': {}, 'diagnosis': {}, 'procedure': {}, 'medication': {}, 'temporal_feature': {}, 'label': {}}

    for patient_id, visits in enumerate(data_list):
        transformed_data['patient'][patient_id] = {'diagnosis': {}, 'procedure': {}, 'medication': {}}

        for visit in visits[:-1]:
            diagnosis_list, procedure_list, medication_list, timestamp = visit


            for diagnosis_id in diagnosis_list:
                if diagnosis_id not in transformed_data['diagnosis']:
                    transformed_data['diagnosis'][diagnosis_id] = {}
                if patient_id not in transformed_data['diagnosis'][diagnosis_id]:
                    transformed_data['diagnosis'][diagnosis_id][patient_id] = []
                if diagnosis_id not in transformed_data['patient'][patient_id]['diagnosis']:
                    transformed_data['patient'][patient_id]['diagnosis'][diagnosis_id] = []
                transformed_data['diagnosis'][diagnosis_id][patient_id].append(timestamp)
                transformed_data['patient'][patient_id]['diagnosis'][diagnosis_id].append(timestamp)


            for procedure_id in procedure_list:
                if procedure_id not in transformed_data['procedure']:
                    transformed_data['procedure'][procedure_id] = {}
                if patient_id not in transformed_data['procedure'][procedure_id]:
                    transformed_data['procedure'][procedure_id][patient_id] = []
                if procedure_id not in transformed_data['patient'][patient_id]['procedure']:
                    transformed_data['patient'][patient_id]['procedure'][procedure_id] = []
                transformed_data['procedure'][procedure_id][patient_id].append(timestamp)
                transformed_data['patient'][patient_id]['procedure'][procedure_id].append(timestamp)


            for medication_id in medication_list:
                if medication_id not in transformed_data['medication']:
                    transformed_data['medication'][medication_id] = {}
                if patient_id not in transformed_data['medication'][medication_id]:
                    transformed_data['medication'][medication_id][patient_id] = []
                if medication_id not in transformed_data['patient'][patient_id]['medication']:
                    transformed_data['patient'][patient_id]['medication'][medication_id] = []
                transformed_data['medication'][medication_id][patient_id].append(timestamp)
                transformed_data['patient'][patient_id]['medication'][medication_id].append(timestamp)


            


        transformed_data['temporal_feature'][patient_id] = visits[-1][3]

        transformed_data['label'][patient_id] = visits[-1][2]

    return transformed_data

def generate_data(data_dict, left, right, name, origin_data):
    def patient_process(item_dict, type, sample_neighbors_num=1000):
        nonlocal tmp_last_node_indices
        nonlocal node_indices
        nonlocal node_temporal_features
        for item_idx in last_node_indices[type]:

            #print(item_idx, type, patient_id)
            if 0 < sample_neighbors_num < len(item_dict[item_idx].keys()):
                select_user_idx = random.sample(list(item_dict[item_idx].keys()), sample_neighbors_num)
            else:
                select_user_idx = list(item_dict[item_idx].keys())
            select_user_idx = [x for x in select_user_idx if x not in patient_set]
            tmp_last_node_indices += select_user_idx
            

            for user_idx in select_user_idx:
                temporal_features_list = item_dict[item_idx][user_idx]

                node_indices += [int(user_idx)] * len(temporal_features_list)
                node_temporal_features += temporal_features_list
    
    def item_process(user_dict, type, sample_neighbors_num=1000):
        tmp_node_indices = []
        nonlocal node_indices
        nonlocal node_temporal_features
        nonlocal tmp_last
        for user_idx in last_node_indices:
            # add to tmp_last_node_indices for next-hop neighbors retrieval with sampling (except for the first hop, which uses all the neighbors)
            if hop != 1 and 0 < sample_neighbors_num < len(user_dict[user_idx][type].keys()):
                select_item_idx = random.sample(list(user_dict[user_idx][type].keys()), sample_neighbors_num)
            else:
                select_item_idx = list(user_dict[user_idx][type].keys())


            for item_idx in select_item_idx:
                temporal_features_list = user_dict[user_idx][type][item_idx]

                tmp_node_indices += [int(item_idx)] * len(temporal_features_list)
                node_temporal_features += temporal_features_list
        node_indices.append(tmp_node_indices)
        tmp_last.append(list(set(tmp_node_indices)))


    # final_data = []
    #print('yes', data_dict['medication'][128][9])
    #print('yes', data_dict['patient'][9]['medication'][128])
    for patient_id in range(left, right):
        print(f'{patient_id-left}/{right-left}')
        hops_information = []
        hops_nodes_indices = []
        hops_nodes_temporal_features = []

        central_node_temporal_feature = data_dict['temporal_feature'][patient_id]
        central_node_label = data_dict['label'][patient_id]


        last_node_indices = []
        patient_set = set()

        # hop_num = 3
        for hop in range(4):

            if hop == 0:
                node_indices = [int(patient_id)]
                node_temporal_features = data_dict['temporal_feature'][patient_id]
                last_node_indices = [patient_id]
            # hop >= 1
            else:
                node_indices = []
                node_temporal_features = []
                tmp_last_node_indices = []

                # get neighboring users for each node in last_node_indices
                if hop % 2 == 0:
                    patient_process(data_dict['diagnosis'], 0)
                    patient_process(data_dict['procedure'], 1)
                    patient_process(data_dict['medication'], 2)
                    patient_set.update(tmp_last_node_indices)
                    last_node_indices = list(set(tmp_last_node_indices))
                # get neighboring items for each node in last_node_indices
                else: 
                    tmp_last = []
                    item_process(data_dict['patient'], 'diagnosis')
                    item_process(data_dict['patient'], 'procedure')
                    item_process(data_dict['patient'], 'medication')
                    last_node_indices = tmp_last

                # remove duplicated neighboring nodes
                

            # hops_information.append([node_indices, node_temporal_features])
            hops_nodes_indices.append(node_indices)
            hops_nodes_temporal_features.append(node_temporal_features)
        with open(f'./dataset/data_{name}_{right-left}.pkl', "ab") as file:
            pickle.dump([hops_nodes_indices, hops_nodes_temporal_features, central_node_temporal_feature, central_node_label, [i[0] for i in origin_data[patient_id]], [i[1] for i in origin_data[patient_id]], [i[3] for i in origin_data[patient_id]]], file)
            '''
            (hop_nums, max_neighbors_num)
            (hop_nums, max_neighbors_num)
            (1)
            (ans_num)

            '''








if __name__ == '__main__':
    #read_mimic3()
    read_mimic4()
