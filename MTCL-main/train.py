import os
import torch
import numpy as np
from tqdm import tqdm
from core.dataset import MMDataLoader,MMSAATBDataLoader
from core.scheduler import get_scheduler
from core.utils import AverageMeter, save_model, setup_seed, save_args, save_excel, eliminated_confilict,Loss_Branch
from models.MTCL import build_model
from core.metric import MetricsTop
import wandb
import shutil
import argparse
import pandas as pd

os.environ["WANDB_MODE"] = "offline"
parser = argparse.ArgumentParser()

parser.add_argument("--datasetName", type=str, default="sims", required=False)
parser.add_argument("--dataPath", type=str, default="./datasets/CHSIMSV2.pkl", required=False)
parser.add_argument("--seq_lens", type=tuple, default=[50, 925, 232], required=False) #[50, 375, 500] [50, 50, 50] [39, 400, 55] [50, 925, 232]
parser.add_argument("--feature_dims", type=tuple, default=[768, 25, 177], required=False) #[768, 5, 20] [768, 74, 35] [768, 33, 709] [768, 25, 177]
parser.add_argument("--train_mode", type=str, default='regression', required=False) 
parser.add_argument("--use_bert", type=bool, default=True, required=False)

parser.add_argument("--num_workers", type=int, default=8, required=False)
parser.add_argument("--CUDA_VISIBLE_DEVICES", type=str, default='1', required=False)
parser.add_argument("--seed", type=int, default=123, required=False)
parser.add_argument("--batch_size", type=int, default=64, required=False)
parser.add_argument("--n_epochs", type=int, default=30, required=False)
parser.add_argument("--early_stop", type=int, default=10, required=False)
parser.add_argument("--lr", type=float, default=4.5e-05, required=False)
parser.add_argument("--weight_decay", type=float, default=1e-4, required=False)
parser.add_argument("--dropout", type=float, default=0, required=False)

parser.add_argument("--project_name", type=str, default='BiLSTM_l', required=False)
parser.add_argument("--models_code", type=str, default='MTCL', required=False)
parser.add_argument("--LSTM_layers", type=int, default=2, required=False)
parser.add_argument("--eliminate_rate", type=float, default=0.44, required=False) 


args = parser.parse_args()

def main(seed, save_path):
    save_path = os.path.join(save_path, str(seed))
    
    if seed is not None:
        setup_seed(seed)
    print("seed: {}".format(seed))

    print(args)
    model = build_model(args).to(device)
    dataLoader = MMDataLoader(args)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler_warmup = get_scheduler(optimizer, args)
    loss_fn = torch.nn.MSELoss()
    loss_fn_none = torch.nn.MSELoss(reduction='none')
    metrics = MetricsTop().getMetics(args.datasetName)

    # train
    best_mae, best_epoch, best_acc, early_stop_num = float('inf'), 0,0,0
    for epoch in range(1, args.n_epochs+1):
        train(model, dataLoader['train'], optimizer, loss_fn, loss_fn_none, epoch, metrics)
        result_eval = evaluate(model, dataLoader['valid'], optimizer, loss_fn, loss_fn_none, epoch, metrics, mode="valid")
        result_test = evaluate(model, dataLoader['test'], optimizer, loss_fn, loss_fn_none, epoch, metrics, mode="test")
        if result_eval["MAE"] < best_mae:
            best_mae = result_eval["MAE"]
            best_epoch = epoch
            best_eval = result_eval
            best_result = result_test
            early_stop_num=0
            save_model(save_path, epoch, model, optimizer) 
        early_stop_num+=1
        if(early_stop_num>args.early_stop):
            break
        scheduler_warmup.step()
    
    # print/save result
    print("best_epoch:",best_epoch)
    print("best_epoch_val:",best_eval)
    print("best_epoch_test:",best_result)

    content = 'best_epoch:' + f'{best_epoch}\n' + 'best_epoch_valid:\n' + f'{best_eval}\n' + 'best_epoch_test:\n' +  f'{best_result}\n'
    with open(os.path.join(save_path,'result.txt'),"w") as f:
        f.write(str(content))
    
    return best_result, best_epoch

def train(model, train_loader, optimizer, loss_fn, loss_fn_none, epoch, metrics):
    train_pbar = tqdm(enumerate(train_loader))
    losses_m, losses_t, losses_a, losses_v, losses = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    y_pred, y_true = [[],[],[],[]], [[],[],[],[]]
    pred, true, train_results = [[],[],[],[]], [[],[],[],[]], [[],[],[],[]]
    model.train()
    for cur_iter, data in train_pbar:

        img, audio, text, vlens, alens = data['vision'].to(device), data['audio'].to(device), data['text'].to(device), data['vision_lengths'], data['audio_lengths']
        label_m, label_t, label_a, label_v = data['labels']['M'].to(device), data['labels']['T'].to(device), data['labels']['A'].to(device), data['labels']['V'].to(device)
        label_m, label_t, label_a, label_v = label_m.view(-1, 1), label_t.view(-1, 1), label_a.view(-1, 1), label_v.view(-1, 1)
        batchsize = img.shape[0]

        output_M, output_T_f, output_T_c, output_A_f, output_A_c, output_V_f, output_V_c, Djs_T, Djs_A, Djs_V = model(text, audio, img, vlens, alens)

        loss_m = loss_fn(output_M, label_m)
        loss_t = Loss_Branch(output_T_f,output_T_c, label_t, label_m, Djs_T, args.eliminate_rate)
        loss_a = Loss_Branch(output_A_f,output_A_c, label_a, label_m, Djs_A, args.eliminate_rate)
        loss_v = Loss_Branch(output_V_f,output_V_c, label_v, label_m, Djs_V, args.eliminate_rate)

        y_pred[0].append(output_M.cpu())
        y_true[0].append(label_m.cpu())

        loss = loss_m + loss_t + loss_a + loss_v

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses_m.update(loss_m.item(), batchsize)
        losses_t.update(loss_t.item(), batchsize)
        losses_a.update(loss_a.item(), batchsize)
        losses_v.update(loss_v.item(), batchsize)
        losses.update(loss.item(), batchsize)
        train_pbar.set_description('train')
        train_pbar.set_postfix({'epoch': '{}'.format(epoch),
                                'loss_m': '{:.5f}'.format(losses_m.value_avg),
                                'loss_t': '{:.5f}'.format(losses_t.value_avg),
                                'loss_a': '{:.5f}'.format(losses_a.value_avg),
                                'loss_v': '{:.5f}'.format(losses_v.value_avg),
                                'loss': '{:.5f}'.format(losses.value_avg),
                                'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])})

    pred[0], true[0] = torch.cat(y_pred[0]), torch.cat(y_true[0])
    train_results[0] = metrics(pred[0], true[0])
    print('train: ', train_results[0])
  
def evaluate(model, loader, optimizer, loss_fn, loss_fn_none, epoch, metrics, mode):
    test_pbar = tqdm(enumerate(loader))
    losses_m, losses_t, losses_a, losses_v, losses = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    y_pred, y_true = [[],[],[],[]], [[],[],[],[]]
    pred, true, test_results = [[],[],[],[]], [[],[],[],[]], [[],[],[],[]]
    model.eval()
    with torch.no_grad():
        for cur_iter, data in test_pbar:
            img, audio, text, vlens, alens = data['vision'].to(device), data['audio'].to(device), data['text'].to(device), data['vision_lengths'], data['audio_lengths']
            label_m, label_t, label_a, label_v = data['labels']['M'].to(device), data['labels']['T'].to(device), data['labels']['A'].to(device), data['labels']['V'].to(device)
            label_m, label_t, label_a, label_v = label_m.view(-1, 1), label_t.view(-1, 1), label_a.view(-1, 1), label_v.view(-1, 1)
            batchsize = img.shape[0]

            output_M, output_T_f, output_T_c, output_A_f, output_A_c, output_V_f, output_V_c, Djs_T, Djs_A, Djs_V = model(text, audio, img, vlens, alens)

            loss_m = loss_fn(output_M, label_m)
            loss_t = Loss_Branch(output_T_f,output_T_c, label_t, label_m, Djs_T, args.eliminate_rate)
            loss_a = Loss_Branch(output_A_f,output_A_c, label_a, label_m, Djs_A, args.eliminate_rate)
            loss_v = Loss_Branch(output_V_f,output_V_c, label_v, label_m, Djs_V, args.eliminate_rate)

            y_pred[0].append(output_M.cpu())
            y_true[0].append(label_m.cpu())

            loss = loss_m + loss_t + loss_a + loss_v

            losses_m.update(loss_m.item(), batchsize)
            losses_t.update(loss_t.item(), batchsize)
            losses_a.update(loss_a.item(), batchsize)
            losses_v.update(loss_v.item(), batchsize)
            losses.update(loss.item(), batchsize)
            test_pbar.set_description(mode)
            test_pbar.set_postfix({'epoch': '{}'.format(epoch),
                                    'loss_m': '{:.5f}'.format(losses_m.value_avg),
                                    'loss_t': '{:.5f}'.format(losses_t.value_avg),
                                    'loss_a': '{:.5f}'.format(losses_a.value_avg),
                                    'loss_v': '{:.5f}'.format(losses_v.value_avg),
                                    'loss': '{:.5f}'.format(losses.value_avg),
                                    'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])})

        pred[0], true[0] = torch.cat(y_pred[0]), torch.cat(y_true[0])
        test_results[0] = metrics(pred[0], true[0])

        print(f'{mode}: ', test_results[0])
        return test_results[0]

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    print("device: {}:{}".format(device, args.CUDA_VISIBLE_DEVICES))

    save_path = os.path.join('result', args.models_code, args.project_name, str(args.seed)+f'lr:{args.lr}')
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
        best_result, best_epoch = main(args.seed, save_path)
    else:
        print("repeat!!!!!!!!!!!!")
    
    