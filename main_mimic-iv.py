
import torch
import argparse
from load_data import *
from train_4 import *
from model import *
from ETGNN import *

# Parameters setting
parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", default=False, help="test mode")
parser.add_argument("--resume_path", type=str, default='./saved/Epoch_11_TARGET_0.05_JA_0.4957_DDI_0.06534.model', help="resume path")
parser.add_argument("--device", type=int, default=1, help="gpu id to run on, negative for cpu")

parser.add_argument("--temporal_information_importance", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
parser.add_argument('--dp', default=0.2, type=float, help='dropout ratio')
parser.add_argument('--emb_dim', type=int, default=64, help='embedding dimension size')
parser.add_argument("--target_ddi", type=float, default=0.05, help="target ddi")
parser.add_argument("--kp", type=float, default=0.05, help="coefficient of P signal")



args = parser.parse_args()
    


torch.manual_seed(1203)
np.random.seed(2048)
def get_model_name(args):
    model_name = [
        f'dim_{args.emb_dim}',  f'lr_{args.lr}', f'coef_{args.kp}',
        f'dp_{args.dp}', f'ddi_{args.target_ddi}'
    ]
    if args.embedding:
        model_name.append('embedding')
    return '-'.join(model_name)

# run framework
def main():
    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')


    # load data
    # data_train, data_eval, data_test, voc_size, ddi_adj, ehr_adj = load_mimic3(device, args)

    # load model
    voc_size, ddi_adj, ehr_adj, patient_num  = read_data_inf('mimic4')
    #voc_size, ddi_adj, ehr_adj, patient_num  = read_data_inf('mimic4')
    # print('patient_num', patient_num)
    model = TimeRec_GCN(voc_size, patient_num, args.emb_dim, args.dp, args.dp, args.temporal_information_importance, ehr_adj, ddi_adj, args.device).to(device)
    print('voc_size:', voc_size)
    if args.test:
        Test_mimic3(model, args.resume_path, device, voc_size)
    else:
        Train_mimic3(model, device, voc_size, args)




if __name__ == '__main__':
    main()

