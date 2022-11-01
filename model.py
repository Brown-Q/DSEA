import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spmm
from torch_geometric.utils import softmax, degree
from torch.nn import init
  
class GCN(nn.Module):
    def __init__(self, hidden):
        super(GCN, self).__init__()

    def forward(self, x, edge_index):
        edge_index_j, edge_index_i = edge_index
        deg = degree(edge_index_i, x.size(0), dtype=x.dtype)
        # print(deg)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[edge_index_j]*deg_inv_sqrt[edge_index_i]
        x = F.relu(spmm(edge_index[[1, 0]], norm, x.size(0), x.size(0), x))
        return x

class Highway(nn.Module):
    def __init__(self, x_hidden):
        super(Highway, self).__init__()
        self.lin = nn.Linear(x_hidden, x_hidden, bias=False)

    def forward(self, x1, x2):
        gate = F.leaky_relu(self.lin(x1)+1e-8)
        x = F.leaky_relu(torch.mul(gate, x2)+torch.mul(1-gate, x1))
        return x

class GAT_E(nn.Module):
    def __init__(self, e_hidden, r_hidden):
        super(GAT_E, self).__init__()
        self.a_h1 = nn.Linear(r_hidden, 1, bias=False)
        self.a_h2 = nn.Linear(r_hidden, 1, bias=False)
        self.a_t1 = nn.Linear(r_hidden, 1, bias=False)
        self.a_t2 = nn.Linear(r_hidden, 1, bias=False)
        self.w_h = nn.Linear(e_hidden, r_hidden, bias=False)
        self.w_t = nn.Linear(e_hidden, r_hidden, bias=False)
        
    def forward(self, x_e, edge_index, rel):
        edge_index_h, edge_index_t = edge_index
        x_r_h = self.w_h(x_e)
        x_r_t = self.w_t(x_e)        
        e1 = self.a_h1(x_r_h).squeeze()[edge_index_h]+self.a_h2(x_r_t).squeeze()[edge_index_t]
        e2 = self.a_t1(x_r_h).squeeze()[edge_index_h]+self.a_t2(x_r_t).squeeze()[edge_index_t]       
        alpha = softmax(F.leaky_relu(e1).float(), rel)
        x_r_h = spmm(torch.cat([rel.view(1, -1), edge_index_h.view(1, -1)], dim=0), alpha, rel.max()+1, x_e.size(0), x_r_h)        
        alpha = softmax(F.leaky_relu(e2).float(), rel)
        x_r_t = spmm(torch.cat([rel.view(1, -1), edge_index_t.view(1, -1)], dim=0), alpha, rel.max()+1, x_e.size(0), x_r_t)
        x_r = x_r_h+x_r_t
        return x_r
 
class GAT_R(nn.Module):
    def __init__(self, e_hidden, r_hidden):
        super(GAT_R, self).__init__()
        self.a_h = nn.Linear(e_hidden, 1, bias=False)
        self.a_t = nn.Linear(e_hidden, 1, bias=False)
        self.a_r = nn.Linear(r_hidden, 1, bias=False)
        
    def forward(self, x_e, x_r, edge_index, rel):
        edge_index_h, edge_index_t = edge_index
        e_h = self.a_h(x_e).squeeze()[edge_index_h]
        e_t = self.a_t(x_e).squeeze()[edge_index_t]
        e_r = self.a_r(x_r).squeeze()[rel]
        alpha = softmax(F.leaky_relu(e_h+e_r).float(), rel)
        x = spmm(torch.cat([rel.view(1, -1),edge_index_t.view(1, -1)], dim=0), alpha, x_e.size(0), x_e.size(0), x_e)
        return x

class GAT(nn.Module):
    def __init__(self, hidden):
        super(GAT, self).__init__()
        self.a_i = nn.Linear(hidden, 1, bias=False)
        self.a_j = nn.Linear(hidden, 1, bias=False)
        self.a_r = nn.Linear(hidden, 1, bias=False)
               
    def forward(self, x, edge_index):
        edge_index_j, edge_index_i = edge_index
        e_i = self.a_i(x).squeeze()[edge_index_i]
        e_j = self.a_j(x).squeeze()[edge_index_j]
        e = e_i+e_j
        alpha = softmax(F.leaky_relu(e).float(), edge_index_i)
        x = F.leaky_relu(spmm(edge_index[[1, 0]], alpha, x.size(0), x.size(0), x))
        x=x[:,:300]
        return x
    
class AttentionNetwork(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mk = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 1, bias=True),
            nn.Sigmoid(),
        )

    def concat(self,a,b,c):
        d=torch.cat((a.unsqueeze(0),b.unsqueeze(0),c.unsqueeze(0)),0)
        return d              
    
    def forward(self, x1, x_name1,onehot1,x2,x_name2,onehot2, data_batch):
        x1=torch.as_tensor(x1,dtype=torch.float32)
        x2=torch.as_tensor(x2,dtype=torch.float32)
        x_name1=torch.as_tensor(x_name1,dtype=torch.float32)
        x_name2=torch.as_tensor(x_name2,dtype=torch.float32)
        onehot1=torch.as_tensor(onehot1,dtype=torch.float32)
        onehot2=torch.as_tensor(onehot2,dtype=torch.float32)
        x1_train, x2_train = x1[data_batch[:, 0]], x2[data_batch[:, 1]]
        x1_train_name, x2_train_name = x_name1[data_batch[:, 0]], x_name2[data_batch[:, 1]]
        x1_train_onehot, x2_train_onehot = onehot1[data_batch[:, 0]], onehot2[data_batch[:, 1]]
        X1=self.concat(x1_train,x1_train_name,x1_train_onehot)
        X2=self.concat(x2_train,x2_train_name,x2_train_onehot)
        X1_1=self.concat(x2,x_name2,onehot2)
        X2_1=self.concat(x1,x_name1,onehot1)
        X1_2=self.concat(x2.t(),x_name2.t(),onehot2.t())
        X2_2=self.concat(x1.t(),x_name1.t(),onehot1.t())
        Attention1=nn.Softmax(dim=-1)(torch.matmul(X1,X1_2))
        Attention2=nn.Softmax(dim=-1)(torch.matmul(X2,X2_2))
        Wei_kg1=torch.matmul(Attention1,X1_1)
        Wei_kg2=torch.matmul(Attention2,X2_1)
        Re_kg1=Wei_kg1+X1
        Re_kg2=Wei_kg2+X2
        Re_kg1.transpose_(1,0)
        Re_kg2.transpose_(1,0)
        kg1=nn.Softmax(dim=-1)(torch.cat([self.mk(Re_kg1[:,0,:]),self.mk(Re_kg1[:,1,:]),self.mk(Re_kg1[:,2,:])],dim=-1))
        kg2=nn.Softmax(dim=-1)(torch.cat([self.mk(Re_kg2[:,0,:]),self.mk(Re_kg2[:,1,:]),self.mk(Re_kg2[:,2,:])],dim=-1))
        x_name1[data_batch[:, 0]]=x_name1[data_batch[:, 0]]*kg1[:,1].unsqueeze(1)
        onehot1[data_batch[:, 0]]=onehot1[data_batch[:, 0]]*kg1[:,2].unsqueeze(1)
        x_name2[data_batch[:, 1]]=x_name2[data_batch[:, 1]]*kg2[:,1].unsqueeze(1)
        onehot2[data_batch[:, 1]]=onehot2[data_batch[:, 1]]*kg2[:,2].unsqueeze(1)
        return x1, x_name1,onehot1,x2,x_name2,onehot2            
   
    
class DSEA(nn.Module):
    def __init__(self, e_hidden=300, r_hidden=100,embedding_dim=300,dim=900,hidden_size=150,num_layers=1,input_size=100):
        super(DSEA, self).__init__()
        self.layer=2
        self.gcn1 = GCN(e_hidden)
        self.highway1 = Highway(e_hidden)
        self.gcn2 = GCN(e_hidden)
        self.highway2 = Highway(e_hidden)
        self.gat_e = GAT_E(e_hidden, r_hidden)
        self.gat_r = GAT_R(e_hidden, r_hidden)
        self.gat = GAT(self.layer*e_hidden)
        self.lstm_layer=nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size,num_layers=num_layers,
                        bias=True,batch_first=False,dropout=0.5,bidirectional=True)
        self.lstm_layer2=nn.LSTM(input_size=300,hidden_size=300,num_layers=num_layers,
                        bias=True,batch_first=False,dropout=0.5,bidirectional=False)
        self.attentionnetwork=AttentionNetwork(dim)

    def forward(self, x_e1, edge_index1, rel1, edge_index_all1, rel_all1,x_name1,onehot1,
                x_e2, edge_index2, rel2, edge_index_all2, rel_all2,x_name2,onehot2,data_batch):
        x_e1 = self.highway1(x_e1, self.gcn1(x_e1, edge_index_all1))
        x_e1 = self.highway2(x_e1, self.gcn2(x_e1, edge_index_all1))
        x_r = self.gat_e(x_e1, edge_index1, rel1)
        x_e1 = torch.cat([x_e1, self.gat_r(x_e1, x_r, edge_index1, rel1)], dim=1)
        x_e1 = torch.cat([x_e1, self.gat(x_e1, edge_index_all1)], dim=1)
        lstm_input=torch.reshape(x_name1.clone() ,(-1,3,300))
        x_e1=lstm_input.shape[1]**self.layer*x_e1
        lstm_input.transpose_(1,0)
        output1,(h_n,c_n)=self.lstm_layer(lstm_input)
        output1=torch.reshape(output1.clone().transpose_(1,0),(-1,900))
        lstm_input2=torch.reshape(onehot1.clone() ,(-1,3,300))
        lstm_input2.transpose_(1,0)
        output2,(h_n,c_n)=self.lstm_layer2(lstm_input2)
        output2=torch.reshape(output2.clone().transpose_(1,0),(-1,900))
        x_e2 = self.highway1(x_e2, self.gcn1(x_e2, edge_index_all2))
        x_e2 = self.highway2(x_e2, self.gcn2(x_e2, edge_index_all2))
        x_r = self.gat_e(x_e2, edge_index2, rel2)
        x_e2 = torch.cat([x_e2, self.gat_r(x_e2, x_r, edge_index2, rel2)], dim=1)
        x_e2 = torch.cat([x_e2, self.gat(x_e2, edge_index_all2)], dim=1)
        lstm_input=torch.reshape(x_name2.clone() ,(-1,3,300))
        x_e2=lstm_input.shape[1]**self.layer*x_e2
        lstm_input.transpose_(1,0)
        output3,(h_n,c_n)=self.lstm_layer(lstm_input)
        output3=torch.reshape(output3.clone().transpose_(1,0),(-1,900))
        lstm_input2=torch.reshape(onehot2.clone() ,(-1,3,300))
        lstm_input2.transpose_(1,0)
        output4,(h_n,c_n)=self.lstm_layer2(lstm_input2)
        output4=torch.reshape(output4.clone().transpose_(1,0),(-1,900))
        x1, x_name1,onehot1,x2,x_name2,onehot2=self.attentionnetwork(x_e1,output1,output2,x_e2,output3,output4,data_batch)
        x_name1=nn.Softmax(dim=-1)(x_name1)
        x_name2=nn.Softmax(dim=-1)(x_name2)
        onehot1=nn.Softmax(dim=-1)(onehot1)
        onehot2=nn.Softmax(dim=-1)(onehot2)
        return x1, x_name1, onehot1, x2, x_name2,onehot2
