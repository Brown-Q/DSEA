import torch
from torch import nn
from transformers import BertModel,BertConfig,BertTokenizer
import numpy as np
import re
import pickle
import jieba
import torch.nn as nn

pretrained_path = 'bert-multi/'
config = BertConfig.from_pretrained(pretrained_path)
tokenizer = BertTokenizer.from_pretrained(pretrained_path)
model = BertModel.from_pretrained(pretrained_path,config=config)

def gen_mean(vals, p):
    p = float(p)
    return np.power(
        np.mean(
            np.power(
                np.array(vals, dtype=complex),
                p),
            axis=0),
        1 / p
    )
operations = dict([('mean', (lambda word_embeddings: [np.mean(word_embeddings, axis=0)], lambda embeddings_size: embeddings_size)),])

def get_sentence_embedding(embeddings, chosen_operations, con):
    word_embeddings = embeddings[0,:,:].detach().numpy()
    for o in chosen_operations:
        concat_embs = operations[o][0](word_embeddings)
    sentence_embedding = np.concatenate([concat_embs[0],[0]*132])
    return sentence_embedding, con

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

id2embed1 = dict()
id2embed2 = dict()
id2name = dict()
path='./data/DBP15K/dbp_wd/'
inf = open(path + 'ent_ids_1')
con = 0
for i1, line in enumerate(inf):
    strs = line.strip().split('\t')
    id2name[int(strs[0])] = strs[1]
    wordline = strs[1].split('/')[-1].lower().replace('_',' ')
    words = re.findall(r'\b\w+\b', wordline)
    words_new=''
    for i in words:
        for j in i:
            words_new=words_new+j
    batch = tokenizer.encode_plus(words_new)  
    input_ids = torch.tensor([batch['input_ids']])
    token_type_ids = torch.tensor([batch['token_type_ids']])
    attention_mask = torch.tensor([batch['attention_mask']])
    embedding = model(input_ids,token_type_ids=token_type_ids)
    embed, con = get_sentence_embedding(embedding[0], ['mean'], con)
    id2embed1[strs[0]] = embed

con1 = 0
inf = open(path + 'ent_ids_2')
for i2, line in enumerate(inf):
    strs = line.strip().split('\t')
    id2name[int(strs[0])] = strs[1]
    wordline = strs[1].split('/')[-1].lower().replace('_', ' ')
    words = re.findall(r'\b\w+\b', wordline)
    words_new=''
    for i in words:
        for j in i:
            words_new=words_new+j
    batch = tokenizer.encode_plus(words_new) 
    input_ids = torch.tensor([batch['input_ids']])
    token_type_ids = torch.tensor([batch['token_type_ids']])
    attention_mask = torch.tensor([batch['attention_mask']])
    embedding = model(input_ids,token_type_ids=token_type_ids)
    embed, con1 = get_sentence_embedding(embedding[0], ['mean'], con1)
    id2embed2[strs[0]] = embed

outf = open(path + 'name_vec1.txt', 'w')
outf.truncate(0)
for id in id2embed1:
    embed=id2embed1[id]
    dis_str = ''
    for i in embed:
        dis_str = dis_str+ str(i) + ' '
    dis_str = dis_str[:-1]
    outf.write(dis_str + '\n')
outf.close()

outf = open(path + 'name_vec2.txt', 'w')
outf.truncate(0)
for id in id2embed2:
    embed=id2embed2[id]
    dis_str = ''
    for i in embed:
        dis_str = dis_str+ str(i) + ' '
    dis_str = dis_str[:-1]
    outf.write(dis_str + '\n')
outf.close()


