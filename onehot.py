# import tensorflow as tf
import numpy as np
import json
import torch
import networkx as nx
import pandas as pd
import csv
import torch.nn as nn
from torch.nn import functional

def k_shell(graph):
    importance_dict = {}
    level = 1
    while len(graph.degree()):
        importance_dict[level] = []
        while True:
            level_node_list = []
            for key,value in graph.degree():
                if value <= level:
                    level_node_list.append(key)
            graph.remove_nodes_from(level_node_list)
            importance_dict[level].extend(level_node_list)
            if not len(graph.degree()):
                return importance_dict
            if min(graph.degree(), key=lambda x: x[1])[1] > level:
                break
        level = min(graph.degree(), key=lambda x: x[1])[1]

def Hindex(indexList):
    indexSet = sorted(list(set(indexList)), reverse=True)
    for index in indexSet:
        clist = [i for i in indexList if i >= index]
        if index <= len(clist):
            break
    return index

def LHindex(graph):
    node = {} 
    H_index = {} 
    LH_index = {}  
    for i in graph.nodes():
        node[i] = graph.degree(i)
    for i in graph.nodes():
        test = [node[j] for j in graph[i]]
        H_index[i] = Hindex(test)
    for i in graph.nodes():
        LH_index[i] = sum(H_index[j] for j in graph[i])
    maximum=max(LH_index.values())
    minimum=min(LH_index.values())
    for key,value in LH_index.items():
        LH_index[key]=(float(value)-float(minimum))/(float(maximum)-float(minimum))
    return LH_index

def get_embedding(id2fre):
    importance=dict()
    for i in id2fre:
        importance[i[0]]=i[1]
    print('total ent fre dic length: ' + str(len(importance)))
    value=dict()
    list1=list(importance.values())
    j=0
    for i in list1:
        if i not in value:
            value[i]=j
            j=j+1
    Degree=[]
    for i in list1:
        degree=value[i]
        Degree.append(degree)
    oneHot=functional.one_hot(torch.tensor(Degree), num_classes=3000)[:,0:300]
    return  oneHot,list1

def get_im1oneHot(path):
    inf2 = open(path)
    id2fre = dict()
    G = nx.Graph()
    edge = []
    for line in inf2:
        strs = line.strip().split('\t')
        edge.append([strs[0],strs[2]])
    G.add_edges_from(edge)
    result = k_shell(G)
    for key in result:
        for value in result[key]:
            id2fre[value] = key
    id2fre=sorted(id2fre.items(), key=lambda e: int(e[0]), reverse=False)  
    oneHot,list1=get_embedding(id2fre)
    return  oneHot

def get_im2oneHot(path):
    inf2 = open(path)
    id2fre = dict()
    G = nx.Graph()
    edge = []
    for line in inf2:
        strs = line.strip().split('\t')
        edge.append([strs[0],strs[2]])
    G.add_edges_from(edge)
    lhindex = LHindex(G)
    for i in lhindex:
        id2fre[i]=int(1000*lhindex[i])
    id2fre=sorted(id2fre.items(), key=lambda e: int(e[0]), reverse=False)  
    oneHot,list1=get_embedding(id2fre)
    return  oneHot

def get_im3oneHot(path):
    inf2 = open(path)
    id2fre = dict()
    G = nx.Graph()
    edge = []
    for line in inf2:
        strs = line.strip().split('\t')
        edge.append([strs[0],strs[2]])
    G.add_edges_from(edge)
    for i in G:
        id2fre[i]=G.degree(i)
    id2fre=sorted(id2fre.items(), key=lambda e: int(e[0]), reverse=False)  
    oneHot,list1=get_embedding(id2fre)
    return  oneHot,list1

path1='data/DBP15K/dbp_wd/triples_1'
path2='data/DBP15K/dbp_wd/triples_2'
path3='data/DBP15K/dbp_wd/onehot_1.json'
path4='data/DBP15K/dbp_wd/onehot_2.json'
path5='data/DBP15K/dbp_wd/degree_1.json'
path6='data/DBP15K/dbp_wd/degree_2.json'
onehot1 = get_im1oneHot(path1)
onehot2 = get_im2oneHot(path1)
onehot3,list1 = get_im3oneHot(path1)
onehot=torch.cat((onehot1,onehot2,onehot3), 1)
onehot=torch.tensor(onehot,dtype=float)  
onehot = onehot.tolist()
with open(path3,'w') as file_obj:
     json.dump(onehot,file_obj)
with open(path5,'w') as file_obj:
     json.dump(list1,file_obj)     
onehot1 = get_im1oneHot(path2)
onehot2 = get_im2oneHot(path2)
onehot3,list2 = get_im3oneHot(path2)
onehot=torch.cat((onehot1,onehot2,onehot3), 1)
onehot=torch.tensor(onehot,dtype=float) 
onehot = onehot.tolist()
with open(path4,'w') as file_obj:
     json.dump(onehot,file_obj)
with open(path6,'w') as file_obj:
     json.dump(list2,file_obj) 
    
