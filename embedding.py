import torch
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,SAGEConv
from torch_geometric.datasets import Planetoid
from GNN import GCN
import json
import torch.nn as nn

def normalize(mx):                                          
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def dis(x, y):
    return torch.mean(F.relu(3+torch.sum(torch.abs(x-y), dim=-1)/10))

def  load_data(dataset):
    path1='data/DBP15K/'+dataset
    edges1=[]
    inf1=open(path1)
    G1 = nx.Graph()
    for line in inf1:
        strs = line.strip().split('\t')
        edges1.append([int(strs[0]),int(strs[2])])
    G1.add_edges_from(edges1)
    dict1 = {int(element):i for i,element in enumerate(sorted(G1.nodes()))}    
    print(len(dict1))
    edges2=[]
    inf1=open(path1)
    for line in inf1:
        strs = line.strip().split('\t')
        edges2.append([dict1[int(strs[0])], dict1[int(strs[2])]])
        edges2.append([dict1[int(strs[2])], dict1[int(strs[0])]])
    edges2 = torch.tensor(edges2, dtype=torch.int64).T
    features=np.ones([100000,300])
    features = normalize(features)                                            
    features = torch.tensor(np.array(features), dtype=torch.float32)
    print(edges2.shape)
    return features, edges2

features1, edge1= load_data('dbp_wd/triples_1')  
features2, edge2= load_data('dbp_wd/triples_2')          
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(300, 300, 300) 
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)                                            
model.train()
for epoch in range(10):   
    optimizer.zero_grad()
    out1 = model(features1, edge1)                            
    out2 = model(features2, edge2) 
    loss = dis(out1,out2)       
    loss.backward()
    optimizer.step()
    print(f"epoch:{epoch+1}, loss:{loss.item()}")     
model.eval()
out1 = model(features1, edge1)                              
out2 = model(features2, edge2)
out=torch.cat((out1,out2),0)
out = out.tolist()                           
path='data/DBP15K/dbp_wd/db_vectorList.json'
with open(path,'w') as file_obj:
     json.dump(out,file_obj)

