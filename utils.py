import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
import numpy as np

def add_inverse_rels(edge_index, rel):
    edge_index_all = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    rel_all = torch.cat([rel, rel+rel.max()+1])
    return edge_index_all, rel_all

def get_train_batch(x1, x2, x_name1,x_name2,onehot1,onehot2,train_set, k=6):
    
    e1_neg1 = torch.cdist(x1[train_set[:, 0]], x1, p=1).topk(k+1, largest=False)[1].t()[1:]
    e1_neg2 = torch.cdist(x1[train_set[:, 0]], x2, p=1).topk(k+1, largest=False)[1].t()[1:]
    e2_neg1 = torch.cdist(x2[train_set[:, 1]], x2, p=1).topk(k+1, largest=False)[1].t()[1:]
    e2_neg2 = torch.cdist(x2[train_set[:, 1]], x1, p=1).topk(k+1, largest=False)[1].t()[1:]
    
    e1_neg1_name = torch.cdist(x_name1[train_set[:, 0]], x_name1, p=1).topk(k+1, largest=False)[1].t()[1:]
    e1_neg2_name = torch.cdist(x_name1[train_set[:, 0]], x_name2, p=1).topk(k+1, largest=False)[1].t()[1:]
    e2_neg1_name = torch.cdist(x_name2[train_set[:, 1]], x_name2, p=1).topk(k+1, largest=False)[1].t()[1:]
    e2_neg2_name = torch.cdist(x_name2[train_set[:, 1]], x_name1, p=1).topk(k+1, largest=False)[1].t()[1:]
    
    e1_neg1_onehot = torch.cdist(onehot1[train_set[:, 0]], onehot1, p=1).topk(k+1, largest=False)[1].t()[1:]
    e1_neg2_onehot = torch.cdist(onehot1[train_set[:, 0]], onehot2, p=1).topk(k+1, largest=False)[1].t()[1:]
    e2_neg1_onehot = torch.cdist(onehot2[train_set[:, 1]], onehot2, p=1).topk(k+1, largest=False)[1].t()[1:]
    e2_neg2_onehot = torch.cdist(onehot2[train_set[:, 1]], onehot1, p=1).topk(k+1, largest=False)[1].t()[1:]
    
    train_batch = torch.stack([e1_neg1, e1_neg2, e2_neg1, e2_neg2,e1_neg1_name,e1_neg2_name,e2_neg1_name,
                               e2_neg2_name,e1_neg1_onehot,e1_neg2_onehot,e2_neg1_onehot,e2_neg2_onehot], dim=0)
    return train_batch

def softmax(x, axis):
    x = x - torch.max(x,axis)[0].unsqueeze(1)
    y = torch.exp(x)
    return y / torch.sum(y,axis).unsqueeze(1)

def concat(a,b,c):
    d=torch.cat((a.unsqueeze(0),b.unsqueeze(0),c.unsqueeze(0)),0)
    return d

def get_hits1(x1, x2, x_name1,x_name2,onehot1,onehot2,degree1,degree2, pair, dist='L1', Hn_nums=(1, 10)):
    degree1=degree1[pair[:, 0]]
    degree2=degree2[pair[:, 1]]
    x_name1=torch.as_tensor(x_name1,dtype=torch.float32)
    x_name2=torch.as_tensor(x_name2,dtype=torch.float32)
    onehot1=torch.as_tensor(onehot1,dtype=torch.float32)
    onehot2=torch.as_tensor(onehot2,dtype=torch.float32)
    # degree1_l=torch.le(degree1,6)
    # degree1_m=torch.gt(degree1,6)&torch.lt(degree1,12)
    # degree1_h=torch.ge(degree1,12)
    pair_num = pair.size(0) 
    S = torch.cdist(x1[pair[:, 0]], x2[pair[:, 1]], p=1)
    for k in Hn_nums:
        pred_topk= S.topk(k, largest=False)[1]   
        Hk = (pred_topk == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item()/pair_num
        # HK1=(pred_topk == torch.arange(pair_num, device=S.device).view(-1, 1))[degree1_l]
        # HK1=HK1.sum().item()/HK1.size(0)*100
        # HK2=(pred_topk == torch.arange(pair_num, device=S.device).view(-1, 1))[degree1_m]
        # HK2=HK2.sum().item()/HK2.size(0)*100
        # HK3=(pred_topk == torch.arange(pair_num, device=S.device).view(-1, 1))[degree1_h]
        # HK3=HK3.sum().item()/HK3.size(0)*100
        print('Hits@%d: %.2f%%    ' % (k, Hk*100),end='')
    rank = torch.where(S.sort()[1] == torch.arange(pair_num, device=S.device).view(-1, 1))[1].float()
    MRR = (1/(rank+1)).mean().item()
    print('MRR: %.3f' % MRR)
  
def get_hits_stable(x1, x2, x_name1,x_name2,onehot1,onehot2,pair):
    x_name1=torch.as_tensor(x_name1,dtype=torch.float32)
    x_name2=torch.as_tensor(x_name2,dtype=torch.float32)
    onehot1=torch.as_tensor(onehot1,dtype=torch.float32)
    onehot2=torch.as_tensor(onehot2,dtype=torch.float32)
    pair_num = pair.size(0)
    S = F.normalize(torch.cdist(x1[pair[:, 0]], x2[pair[:, 1]], p=1),dim=0) 
    index = (S.softmax(1)+S.softmax(0)).flatten().argsort(descending=True) 
    index_e1 = index//pair_num 
    index_e2 = index%pair_num 
    aligned_e1 = torch.zeros(pair_num, dtype=torch.bool)
    aligned_e2 = torch.zeros(pair_num, dtype=torch.bool)
    true_aligned = 0
    for _ in range(pair_num*100):
        if aligned_e1[index_e1[_]] or aligned_e2[index_e2[_]]:
            continue
        if index_e1[_] == index_e2[_]:
            true_aligned += 1
        aligned_e1[index_e1[_]] = True
        aligned_e2[index_e2[_]] = True
    print('Both:\tHits@Stable: %.2f%%   ' % (true_aligned/pair_num*100))
