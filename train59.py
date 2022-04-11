# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 22:36:29 2019

@author: User
"""
import copy
import pandas as pd
from data import SampleGenerator
from Procedure59 import ProcedureEngine
import codecs
import os
from torch_geometric.data import Data
import torch

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

config = {'alias': 'RNNGCN+Pair+RNNGCNNotype+shareweight',
          'model_dir': 'checkpoints/{}_Epoch{}_valmse{:.4f}_testmse{:.4f}_ndcg{:.4f}.model',
          'num_epoch': 300,
          'batch_size': 30,
          'n_shops': 7329,  # 商店数目 ？
          'n_distances': 6,  # 商店距离数
          'n_prices': 5,  # 价格档次
          'optimizer': 'adam',
          'adam_lr': 1e-5,
          'l2_regularization': 0,
          'type_hidden_size': 16,
          'distance_hidden_size': 8,
          'price_hidden_size': 5,
          'region_hidden_size': 16,
          'shop_mlp_layers': [32, 64, 16],
          'combine_mlp_layers': [24, 32, 64, 128, 128, 64, 16, 8, 1],
          # 消融
          # 'combine_mlp_layers':[16,32,64,128,128,64,16,8,1],
          'score_mlp_layers': [32, 64, 8],
          'GNNStep': 3,
          'GCNLayers': [22, 32, 16],
          'pretrained': False,  # 是否使用预训练好的模型
          'pretrained_path': '',
          'distances': ['0-50', '50-100', '100-200',
                        '200-300', '300-400', '400-500', '500-1000']
          }

## mlp进入维度 == 各特征拼接后维度

data_set = 'b_fengtai'
# pre = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+os.path.sep+"..")
# pre = os.path.abspath(os.path.dirname(os.path.abspath(pre))+os.path.sep+"..")

pre = '/home/featurize'

# 加载数据集
# 总数据
total_dir = pre+'/data/GraphShopData/'+data_set+'.csv'
total_set = pd.read_csv(total_dir)
# 训练数据
train_dir = pre+'/data/GraphShopData/'+data_set+'_train.csv'
train_set = pd.read_csv(train_dir)
# 验证数据
val_dir = pre+'/data/GraphShopData/'+data_set+'_val.csv'
val_set = pd.read_csv(val_dir)
# 测试数据
test_dir = pre+'/data/GraphShopData/'+data_set+'_test.csv'
test_set = pd.read_csv(test_dir)


# 邻居表
neighbor_dir = pre+'/data/GraphShopData/'+data_set+'_neighbors.txt'
f = open(neighbor_dir, 'r')
neighbors = f.read()
# 将txt转换为dic
neighbors = eval(neighbors)
f.close()

# 距离表
distances = config['distances']

# 构造连边1
edge_indexs = []
for dis in distances:
    edge1 = pre+'/data/GraphShopData/'+data_set+'_'+dis+' edge1 sameType.txt'
    edge2 = pre+'/data/GraphShopData/'+data_set+'_'+dis+' edge2 sameType.txt'
    f1 = open(edge1, 'r')
    f2 = open(edge2, 'r')
    edge_set = [eval(f1.read()), eval(f2.read())]
    f1.close()
    f2.close()
    edge_indexs.append(torch.tensor(edge_set, dtype=torch.long))

# 构造连边2
edge_indexs_no_same = []
for dis in distances:
    edge1_nosame = pre+'/data/GraphShopData/' + \
        data_set+'_'+dis+' edge1 NOsameType.txt'
    edge2_nosame = pre+'/data/GraphShopData/' + \
        data_set+'_'+dis+' edge2 NOsameType.txt'
    f1 = open(edge1_nosame, 'r')
    f2 = open(edge2_nosame, 'r')
    edge_set_nosame = [eval(f1.read()), eval(f2.read())]
    f1.close()
    f2.close()
    edge_indexs_no_same.append(torch.tensor(edge_set_nosame, dtype=torch.long))

# 类型字典构造
dir_pre = os.path.abspath(os.path.dirname(
    os.path.abspath(__file__))+os.path.sep+".")
fr = open(dir_pre+"/type2index.txt", 'r+', encoding='utf-8')
type2index = eval(fr.read())  # 读取的str转换为字典
fr.close()

# 类型列表
types = list(type2index.keys())
config['n_types'] = len(types)

# 在列表中的筛选出来
total_set, train_set, val_set, test_set = total_set[total_set['subType'].isin(types)], train_set[train_set['subType'].isin(
    types)], val_set[val_set['subType'].isin(types)], test_set[test_set['subType'].isin(types)]

# 地区字典构造
regions = set(total_set['region_number'])
config['n_regions'] = len(regions)
region2Index = {}
for index, region in enumerate(regions):
    region2Index[region] = index  # 给各个地区一个编号

# 样本生成器
sample_generator = SampleGenerator(config, total_set, neighbors, train_set,
                                   val_set, test_set,  type2index, region2Index, edge_indexs, edge_indexs_no_same)
# 训练引擎
engine = ProcedureEngine(config)


for epoch in range(config['num_epoch']):
    print('_'*80)
    print('Epoch {} starts !'.format(epoch))
    # 训练
    engine.train_an_epoch(sample_generator, epoch)
    # 测试
    val_mse, test_mse, ndcg = engine.evaluate(sample_generator)
    # 存储
    engine.save(config['alias'], epoch, val_mse, test_mse, ndcg)
    print("_"*80)
# email2Me()
