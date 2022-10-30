import torch
import torch.nn as nn
import torch.nn.functional as F


class L1_Loss(nn.Module):
    def __init__(self, gamma=3):
        super(L1_Loss, self).__init__()
        self.gamma = gamma
        self.belt=1.2
        self.lamn=10
        
    def dis(self, x, y):
        return torch.sum(torch.abs(x-y), dim=-1)/self.lamn
    
    
    def forward(self, x1, x2,x_name1,x_name2,onehot1,onehot2, train_set, train_batch):
        x_name1=torch.as_tensor(x_name1,dtype=torch.float32)
        x_name2=torch.as_tensor(x_name2,dtype=torch.float32)
        x1_train, x2_train = x1[train_set[:, 0]], x2[train_set[:, 1]]
        x1_train_name, x2_train_name = x_name1[train_set[:, 0]], x_name2[train_set[:, 1]]
        x1_train_onehot, x2_train_onehot = onehot1[train_set[:, 0]], onehot2[train_set[:, 1]]
        
        x1_neg1 = x1[train_batch[0].view(-1)].reshape(-1, train_set.size(0), x1.size(1)).unsqueeze(1)
        x1_neg2 = x2[train_batch[1].view(-1)].reshape(-1, train_set.size(0), x2.size(1)).unsqueeze(1)
        x2_neg1 = x2[train_batch[2].view(-1)].reshape(-1, train_set.size(0), x2.size(1)).unsqueeze(1)
        x2_neg2 = x1[train_batch[3].view(-1)].reshape(-1, train_set.size(0), x1.size(1)).unsqueeze(1)
        dis_x1_x2 = self.dis(x1_train.unsqueeze(1), x2_train.unsqueeze(1))
        
        x1_neg1_name = x_name1[train_batch[4].view(-1)].reshape(-1, train_set.size(0), x_name1.size(1)).unsqueeze(1)
        x1_neg2_name = x_name2[train_batch[5].view(-1)].reshape(-1, train_set.size(0), x_name2.size(1)).unsqueeze(1)
        x2_neg1_name = x_name2[train_batch[6].view(-1)].reshape(-1, train_set.size(0), x_name2.size(1)).unsqueeze(1)
        x2_neg2_name = x_name1[train_batch[7].view(-1)].reshape(-1, train_set.size(0), x_name1.size(1)).unsqueeze(1)
        dis_x1_x2_name = self.dis(x1_train_name.unsqueeze(1), x2_train_name.unsqueeze(1))
        
        x1_neg1_onehot = onehot1[train_batch[8].view(-1)].reshape(-1, train_set.size(0), onehot1.size(1)).unsqueeze(1)
        x1_neg2_onehot = onehot2[train_batch[9].view(-1)].reshape(-1, train_set.size(0), onehot2.size(1)).unsqueeze(1)
        x2_neg1_onehot = onehot2[train_batch[10].view(-1)].reshape(-1, train_set.size(0), onehot2.size(1)).unsqueeze(1)
        x2_neg2_onehot = onehot1[train_batch[11].view(-1)].reshape(-1, train_set.size(0), onehot1.size(1)).unsqueeze(1)
        dis_x1_x2_onehot = self.dis(x1_train_onehot.unsqueeze(1), x2_train_onehot.unsqueeze(1))
       
        loss11 = torch.mean(F.relu(self.gamma+dis_x1_x2-self.dis(x1_train, x1_neg1))
                            +F.relu(self.belt+dis_x1_x2_name-self.dis(x1_train_name, x1_neg1_name))
                            +F.relu(self.belt+dis_x1_x2_onehot-self.dis(x1_train_onehot, x1_neg1_onehot))
                            )
        loss12 = torch.mean(F.relu(self.gamma+dis_x1_x2-self.dis(x1_train, x1_neg2))
                            +F.relu(self.belt+dis_x1_x2_name-self.dis(x1_train_name, x1_neg2_name))
                            +F.relu(self.belt+dis_x1_x2_onehot-self.dis(x1_train_onehot, x1_neg2_onehot))
                            )
        loss21 = torch.mean(F.relu(self.gamma+dis_x1_x2-self.dis(x2_train, x2_neg1))
                            +F.relu(self.belt+dis_x1_x2_name-self.dis(x2_train_name, x2_neg1_name))
                            +F.relu(self.belt+dis_x1_x2_onehot-self.dis(x2_train_onehot, x2_neg1_onehot))
                            )
        loss22 = torch.mean(F.relu(self.gamma+dis_x1_x2-self.dis(x2_train, x2_neg2))
                            +F.relu(self.belt+dis_x1_x2_name-self.dis(x2_train_name, x2_neg2_name))
                            +F.relu(self.belt+dis_x1_x2_onehot-self.dis(x2_train_onehot, x2_neg2_onehot))
                            )

        loss = (loss11+loss12+loss21+loss22)/4
        return loss
