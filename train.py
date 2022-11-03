import os
import argparse
import itertools
import apex
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
from model import DSEA
from data import DBP15K
from loss import L1_Loss
from utils import add_inverse_rels, get_train_batch, get_hits1,get_hits_stable


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--data", default="data/DBP15K")
    parser.add_argument("--lang", default="zh_en")
    parser.add_argument("--rate", type=float, default=0.3)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--r_hidden", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=3)
    parser.add_argument("--epoch", type=int, default=160)
    parser.add_argument("--neg_epoch", type=int, default=5)
    parser.add_argument("--test_epoch", type=int, default=5)
    parser.add_argument("--stable_test", action="store_true", default=False)
    args = parser.parse_args()
    return args

def init_data(args, device):
    data = DBP15K(args.data, args.lang, rate=args.rate)[0] 
    data.x1 = data.x1.to(device) 
    data.x2 = data.x2.to(device)
    data.x_name1 = data.x_name1.to(device)
    data.x_name2 = data.x_name2.to(device)  
    data.onehot1 = data.onehot1.to(device)
    data.onehot2 = data.onehot2.to(device)
    data.edge_index_all1, data.rel_all1 = add_inverse_rels(data.edge_index1, data.rel1)
    data.edge_index_all2, data.rel_all2 = add_inverse_rels(data.edge_index2, data.rel2)
    return data

def get_emb(model, data,data_batch):
    model.eval()
    with torch.no_grad():
        x1,x_name1,onehot1,x2,x_name2,onehot2 = model(data.x1, data.edge_index1, data.rel1, data.edge_index_all1, data.rel_all1,data.x_name1,data.onehot1,
                                                      data.x2, data.edge_index2, data.rel2, data.edge_index_all2, data.rel_all2,data.x_name2,data.onehot2,data_batch)

    return x1, x2,x_name1,x_name2,onehot1,onehot2   
    
def train(model, criterion, optimizer, data,data_batch,train_batch):
    model.train()
    x1,x_name1,onehot1,x2,x_name2,onehot2 = model(data.x1, data.edge_index1, data.rel1, data.edge_index_all1, data.rel_all1,data.x_name1,data.onehot1,
                                                      data.x2, data.edge_index2, data.rel2, data.edge_index_all2, data.rel_all2,data.x_name2,data.onehot2,data_batch)
    optimizer.zero_grad()
    loss = criterion(x1, x2, x_name1, x_name2,onehot1,onehot2,data_batch, train_batch)
    with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    optimizer.step()
    return loss
      
def test(model, data,ite1,ite2,batchsize,stable=False,):
    with torch.no_grad():
        for ite in range(ite1+ite2):
            x1, x2, x_name1,x_name2,onehot1,onehot2= get_emb(model, data,data.pair_set[ite*batchsize:(ite+1)*batchsize])
        print('-'*16+'Train_set'+'-'*16)
        get_hits1(x1, x2, x_name1,x_name2,onehot1,onehot2,data.degree1,data.degree2,data.train_set)
        print('-'*16+'Test_set'+'-'*17)
        get_hits1(x1, x2, x_name1,x_name2,onehot1,onehot2,data.degree1,data.degree2,data.test_set)
        if stable:
            get_hits_stable(x1, x2, x_name1,x_name2,onehot1,onehot2,data.test_set)
        print()
    return x1, x2, x_name1,x_name2,onehot1,onehot2
    
def main(args):
    gc.collect()
    torch.cuda.empty_cache() 
    torch.autograd.set_detect_anomaly(True) 
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    data = init_data(args, device).to(device)
    batchsize = 8
    num_ite1 = len(data.train_set)//batchsize
    num_ite2= len(data.test_set)//batchsize
    model =DSEA(data.x1.size(1), args.r_hidden).to(device)
    optimizer = torch.optim.Adam(itertools.chain(model.parameters()))
    model, optimizer = apex.amp.initialize(model, optimizer)
    criterion = L1_Loss(args.gamma)
    test(model, data,num_ite1,num_ite2,batchsize,args.stable_test,)
    for epoch in range(args.epoch):
        torch.cuda.empty_cache()
        if (epoch+1)%args.neg_epoch == 0:
            x1, x2, x_name1, x_name2,onehot1,onehot2= get_emb(model, data,data.train_set[0:batchsize])
            loss_1=0
            for ite in range(num_ite1):
                train_batch = get_train_batch(x1, x2, x_name1, x_name2,onehot1,onehot2, data.train_set[ite*batchsize:(ite+1)*batchsize], args.k)
                loss=train(model, criterion, optimizer, data,data.train_set[ite*batchsize:(ite+1)*batchsize], train_batch)
                loss_1 = float(loss)+loss_1
            loss_1=loss_1/num_ite1
            print('Epoch:', epoch+1, '/', args.epoch, '\tLoss: %.3f'%loss_1, '\r', end='')
        if (epoch+1)%args.test_epoch == 0:
            print()
            test(model, data,num_ite1,num_ite2,batchsize,args.stable_test,)
    
if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    args = parse_args()
    main(args)
