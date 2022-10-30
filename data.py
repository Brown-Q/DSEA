import os
import json
import torch
from torch_geometric.io import read_txt_array
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import sort_edge_index
import numpy as np
import torch.nn as nn

class DBP15K(InMemoryDataset):
    def __init__(self, root, pair, KG_num=1, rate=0.3, rate2=0.4,seed=1):
        self.pair = pair
        self.KG_num = KG_num
        self.rate = rate
        self.rate2= rate2
        self.seed = seed
        torch.manual_seed(seed)
        super(DBP15K, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['zh_en', 'fr_en', 'ja_en']

    @property
    def processed_file_names(self):
        return '%s_%d_%.1f_%.1f_%d.pt' % (self.pair, self.KG_num, self.rate, self.rate2,self.seed)

    def process(self):
        x1_path = os.path.join(self.root, self.pair, 'ent_ids_1')
        x2_path = os.path.join(self.root, self.pair, 'ent_ids_2')
        x_name1_path=os.path.join(self.root, self.pair, 'name_vec1.txt')
        x_name2_path=os.path.join(self.root, self.pair, 'name_vec2.txt')
        g1_path = os.path.join(self.root, self.pair, 'triples_1')
        g2_path = os.path.join(self.root, self.pair, 'triples_2')
        emb_path = os.path.join(self.root, self.pair, self.pair[:2]+'_vectorList.json')
        onehot_path1=os.path.join(self.root, self.pair, 'onehot_1.json')
        onehot_path2=os.path.join(self.root, self.pair, 'onehot_2.json')
        degree_path1=os.path.join(self.root, self.pair, 'degree_1.json')
        degree_path2=os.path.join(self.root, self.pair, 'degree_2.json')
        x1, edge_index1, rel1, assoc1,onehot1,degree1 = self.process_graph(g1_path, x1_path, emb_path,onehot_path1,degree_path1)
        x2, edge_index2, rel2, assoc2,onehot2,degree2= self.process_graph(g2_path, x2_path, emb_path,onehot_path2,degree_path2)
        x_name1=self.loadNe(x_name1_path)
        x_name2=self.loadNe(x_name2_path)
        pair_path = os.path.join(self.root, self.pair, 'ref_ent_ids')
        pair_set = self.process_pair(pair_path, assoc1, assoc2)
        pair_set = pair_set[:, torch.randperm(pair_set.size(1))]
        train_set = pair_set[:, :int(self.rate*pair_set.size(1))]
        test_set = pair_set[:, int(self.rate2*pair_set.size(1)):]

        if self.KG_num == 1:
            data = Data(x1=x1, edge_index1=edge_index1, rel1=rel1,x_name1=x_name1,assoc1=assoc1,onehot1=onehot1,onehot2=onehot2,
                        x2=x2, edge_index2=edge_index2, rel2=rel2,x_name2=x_name2,degree1=degree1,degree2=degree2,
                        train_set=train_set.t(), test_set=test_set.t(),pair_set=pair_set.t())
        else:
            x = torch.cat([x1, x2], dim=0)
            edge_index = torch.cat([edge_index1, edge_index2+x1.size(0)], dim=1)
            rel = torch.cat([rel1, rel2+rel1.max()+1], dim=0)
            data = Data(x=x, edge_index=edge_index, rel=rel,train_set=train_set.t(), test_set=test_set.t())
        torch.save(self.collate([data]), self.processed_paths[0])

    def loadNe(self,path):
        f1 = open(path)
        vectors = []
        for i, line in enumerate(f1):
            vect = line.rstrip()
            vect = np.fromstring(vect, sep=' ')
            vectors.append(vect)
        embeddings = np.vstack(vectors)
        embeddings=torch.tensor(embeddings)
        embeddings=torch.as_tensor(embeddings, dtype=torch.float32)
        return embeddings

    def process_graph(self, triple_path, ent_path, emb_path,onehot_path,degree_path):
        g = read_txt_array(triple_path, sep='\t', dtype=torch.long)
        subj, rel, obj = g.t()
        assoc = torch.full((rel.max().item()+1,), -1, dtype=torch.long)
        assoc[rel.unique()] = torch.arange(rel.unique().size(0))
        rel = assoc[rel] 
        idx = []
        with open(ent_path, 'r') as f:
            for line in f:
                info = line.strip().split('\t')
                idx.append(int(info[0]))
        idx = torch.tensor(idx)
        with open(emb_path, 'r', encoding='utf-8') as f:
            embedding_list = torch.tensor(json.load(f))
        print(embedding_list.shape)
        x = embedding_list[idx]
        with open(onehot_path, 'r', encoding='utf-8') as f:
            onehot = torch.tensor(json.load(f))
        with open(degree_path, 'r', encoding='utf-8') as f:
            degree = torch.tensor(json.load(f))
        assoc = torch.full((idx.max().item()+1, ), -1, dtype=torch.long)
        assoc[idx] = torch.arange(idx.size(0))
        subj, obj = assoc[subj], assoc[obj]
        edge_index = torch.stack([subj, obj], dim=0)
        edge_index, rel = sort_edge_index(edge_index, rel)
        return x, edge_index, rel, assoc,onehot,degree
    def process_pair(self, path, assoc1, assoc2):
        e1, e2 = read_txt_array(path, sep='\t', dtype=torch.long).t()
        return torch.stack([assoc1[e1], assoc2[e2]], dim=0)
