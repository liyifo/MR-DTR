
import pickle
import torch
import numpy as np
from util import *
from metrics import multi_label_metric, ddi_rate_score
from torch.optim import Adam
from collections import defaultdict
import time
import dill
import torch.nn.functional as F
import os
import math
from util import *


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def read_all_test_data():
    data = []
    file = open('./dataset/data_eval_907.pkl', 'rb')
    for i in range(907):
        input = pickle.load(file)
        data.append(input)
        llprint("\r read step: {} / {}".format(i, 907))
    
    file.close()
    return data


def Test_mimic3(model, model_path, device, voc_size):
    with open(model_path, 'rb') as Fin:
        model.load_state_dict(torch.load(Fin, map_location=device))
    model = model.to(device).eval()
    data_test = read_all_test_data()
    print('--------------------Begin Testing--------------------')
    ddi_list, ja_list, prauc_list, f1_list, med_list = [], [], [], [], []
    tic, result, sample_size = time.time(), [], round(len(data_test) * 0.8)
    for _ in range(10):
        test_sample = np.random.choice(len(data_test), sample_size, replace=True)
        test_sample = [data_test[i] for i in test_sample]
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = \
            test_Eval(model, test_sample, voc_size)
        result.append([ddi_rate, ja, avg_f1, prauc, avg_med])
    result = np.array(result)
    mean, std = result.mean(axis=0), result.std(axis=0)
    metric_list = ['ddi_rate', 'ja', 'avg_f1', 'prauc', 'med']
    outstring = ''.join([
        "{}:\t{:.4f} $\\pm$ {:.4f} & \n".format(metric_list[idx], m, s)
        for idx, (m, s) in enumerate(zip(mean, std))
    ])
    print(outstring)
    print('average test time: {}'.format((time.time() - tic) / 10))
    print('parameters', get_n_params(model))

def test_Eval(model, test_sample, voc_size):
    model.eval()  

    smm_record = [] 
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)] 
    med_cnt, visit_cnt = 0, 0 

    step=0
    for input in test_sample: 
        step = step+1
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], [] 

        target_output, _ = model(*input[:3], input[4], input[5], input[6])


        y_gt_tmp = np.zeros(voc_size[2])
        y_gt_tmp[input[3]] = 1
        y_gt.append(y_gt_tmp)


        target_output = torch.sigmoid(target_output).detach().cpu().numpy()[0]
        y_pred_prob.append(target_output)
        


        y_pred_tmp = target_output.copy()
        y_pred_tmp[y_pred_tmp >= 0.5] = 1
        y_pred_tmp[y_pred_tmp < 0.5] = 0
        y_pred.append(y_pred_tmp)


        y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
        y_pred_label.append(sorted(y_pred_label_tmp))
        visit_cnt += 1
        med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)

        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint("\reval step: {} / {}".format(step, 907))


    ddi_rate = ddi_rate_score(smm_record, path="./data/ddi_A_final.pkl")



    llprint(
        "\nDDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
            ddi_rate*1.0,
            np.mean(ja)*1.0,
            np.mean(prauc)*1.0,
            np.mean(avg_p)*1.0,
            np.mean(avg_r)*1.0,
            np.mean(avg_f1)*1.0,
            med_cnt*1.0 / visit_cnt,
        )
    )

    return (
        ddi_rate,
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        med_cnt / visit_cnt,
    )


def Eval_mimic3(model, voc_size):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0 
    file = open('./dataset/data_test_907.pkl', 'rb')

    
    for step in range(907): 
        input = pickle.load(file)
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], [] 

        target_output, _ = model(*input[:3], input[4], input[5], input[6])


        y_gt_tmp = np.zeros(voc_size[2])
        y_gt_tmp[input[3]] = 1
        y_gt.append(y_gt_tmp)


        target_output = torch.sigmoid(target_output).detach().cpu().numpy()[0]
        y_pred_prob.append(target_output)
        


        y_pred_tmp = target_output.copy()
        y_pred_tmp[y_pred_tmp >= 0.5] = 1
        y_pred_tmp[y_pred_tmp < 0.5] = 0
        y_pred.append(y_pred_tmp)


        y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
        y_pred_label.append(sorted(y_pred_label_tmp))
        visit_cnt += 1  
        med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)


        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint("\reval step: {} / {}".format(step, 907))
    file.close()


    ddi_rate = ddi_rate_score(smm_record, path="./data/ddi_A_final.pkl")



    llprint(
        "\nDDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
            ddi_rate*1.0,
            np.mean(ja)*1.0,
            np.mean(prauc)*1.0,
            np.mean(avg_p)*1.0,
            np.mean(avg_r)*1.0,
            np.mean(avg_f1)*1.0,
            med_cnt*1.0 / visit_cnt,
        )
    )

    return (
        ddi_rate,
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        med_cnt / visit_cnt,
    )


    


def Train_mimic3(model, device, voc_size, args):
    model.to(device=device)
    print('parameters', count_parameters(model))
    optimizer = Adam(list(model.parameters()), lr=args.lr)

    # start iterations
    history = defaultdict(list)
    best_epoch, best_ja = 0, 0
    EPOCH = 100


    for epoch in range(EPOCH):
        tic = time.time()
        print(f'----------------Epoch {epoch}------------------')
        skip_num = 0
        model.train()
        loss_list = []
        file = open('./dataset/data_train_3628.pkl', 'rb')
        
        for step in range(3628):
            input = pickle.load(file)
            

            loss_bce_target = torch.zeros((1, voc_size[2])).to(device)
            loss_bce_target[:, input[3]] = 1


            loss_multi_target = -torch.ones((1, voc_size[2])).long()
            for id, item in enumerate(input[3]):
                loss_multi_target[0][id] = item
            loss_multi_target = loss_multi_target.to(device)


            result, loss_ddi = model(*input[:3], input[4], input[5], input[6])
            #print('result', result)


            loss_bce = F.binary_cross_entropy_with_logits(result, loss_bce_target)
            loss_multi = F.multilabel_margin_loss(F.sigmoid(result), loss_multi_target)




            result = F.sigmoid(result).detach().cpu().numpy()[0] # sigmoid
            result[result >= 0.5] = 1
            result[result < 0.5] = 0
            y_label = np.where(result == 1)[0]
            current_ddi_rate = ddi_rate_score(
                [[y_label]], path="./data/ddi_A_final.pkl"
            )
            #loss = 0.95 * loss_bce + 0.05 * loss_multi
            #loss = 0.60 * loss_bce + 0.05 * loss_multi + 0.35 *loss_ddi


            if current_ddi_rate <= args.target_ddi:
                loss = 0.95 * loss_bce + 0.05 * loss_multi
            else:
                # beta = min(0, 1 + (args.target_ddi - current_ddi_rate) / args.kp)
                # loss = (
                #     beta * (0.95 * loss_bce + 0.05 * loss_multi)
                #     + (1 - beta) * loss_ddi
                # )
                beta = args.kp * (1 - (current_ddi_rate /args. target_ddi))
                beta = min(math.exp(beta), 1)
                loss = beta * (0.95 * loss_bce + 0.05 * loss_multi) \
                    + (1 - beta) * loss_ddi

            # print(f'loss:{epoch}-{loss}')
            loss_list.append(loss.detach().cpu().item())


            optimizer.zero_grad()
            loss.backward() # retain_graph=True
            optimizer.step()
            

            llprint("\rtraining step: {} / {}".format(step, 3628))

        print()
        tic2 = time.time()
        print('loss:', sum(loss_list) / len(loss_list))
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = Eval_mimic3(
            model, voc_size
        )
        print(
            "training time: {}, validate time: {}".format(
                tic2 - tic, time.time() - tic2
            )
        )
        print(f'skip num:{skip_num}')

        history["ja"].append(ja)
        history["ddi_rate"].append(ddi_rate)
        history["avg_p"].append(avg_p)
        history["avg_r"].append(avg_r)
        history["avg_f1"].append(avg_f1)
        history["prauc"].append(prauc)
        history["med"].append(avg_med)


        if epoch >= 5:
            print(
                "ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}".format(
                    np.mean(history["ddi_rate"][-5:]),
                    np.mean(history["med"][-5:]),
                    np.mean(history["ja"][-5:]),
                    np.mean(history["avg_f1"][-5:]),
                    np.mean(history["prauc"][-5:]),
                )
            )


        torch.save(model.state_dict(), os.path.join("saved", "Epoch_{}_TARGET_{:.2}_JA_{:.4}_DDI_{:.4}.model".format(epoch, args.target_ddi*1.0, ja*1.0, ddi_rate*1.0)))

        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja = ja

        print(f"best_epoch: {best_epoch}/{best_ja}")
        file.close()

    dill.dump(history,open(os.path.join("saved", "history.pkl""wb")))
