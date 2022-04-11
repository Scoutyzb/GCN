# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 23:13:18 2019

@author: User
"""
import torch
from utils import use_optimizer, save_checkpoint
from metrics import Metrics
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
import os
import time


class Engine(object):
    def __init__(self, config):
        self.config = config
        self._metron = Metrics(1)
        self.opt = use_optimizer(self.model, config)
        self.crit = torch.nn.MSELoss()
        torch.autograd.set_detect_anomaly(True)

        if torch.cuda.is_available():
            self.crit = self.crit.cuda()

    def train_single_batch(self, types, groups, targets, ratings, prices, lengthes, neighbortypes,
                           neighbordistances, neighborratings, neighborcomments,
                           neighborprices, neighborgroups, graph_features, edge_indexs, edge_indexs_no_same, indexs):

        if torch.cuda.is_available():
            types = types.cuda()
            groups = groups.cuda()
            targets = targets.cuda()
            ratings = ratings.cuda()
            prices = prices.cuda()
            neighbortypes = neighbortypes.cuda()
            neighbordistances = neighbordistances.cuda()
            neighborratings = neighborratings.cuda()
            neighborcomments = neighborcomments.cuda()
            neighborprices = neighborprices.cuda()
            neighborgroups = neighborgroups.cuda()

        self.opt.zero_grad()
        scores = self.model(types, groups, ratings, prices, lengthes, neighbortypes,
                            neighbordistances, neighborratings, neighborcomments,
                            neighborprices, neighborgroups, graph_features, edge_indexs, edge_indexs_no_same, indexs)  # (batch_size)
        loss = self.crit(scores, targets)

        loss.backward(retain_graph=True)

        self.opt.step()
        return loss

    def train_an_epoch(self, sample_generator, epoch_id):
        self.model.train()
        total_loss = 0

        # 获取一定的训练集
        batches, indexs = sample_generator.generate_train_batch(
            self.config['batch_size'])

        # 图特征
        edge_indexs = sample_generator.edge_indexs
        edge_indexs_no_same = sample_generator.edge_indexs_no_same
        graph_features = sample_generator.graph_features

        # batches 列表套列表
        for batch_id, (batch, index) in enumerate(zip(batches, indexs)):
            # batch 是一组batch shopid
            if len(batch) == 1:
                continue
            (types, groups, targets, ratings, prices, lengthes, neighbortypes,
                neighbordistances, neighborratings, neighborcomments,
                neighborprices, neighborgroups) = sample_generator.get_train_batch(batch)
            loss = self.train_single_batch(types, groups, targets, ratings, prices, lengthes, neighbortypes,
                                           neighbordistances, neighborratings, neighborcomments,
                                           neighborprices, neighborgroups, graph_features, edge_indexs, edge_indexs_no_same, index)

            print('[Training Epoch{}] Batch {}, loss {}'.format(epoch_id,
                  batch_id, loss.item()))
            total_loss += loss.item()
        # loss.backward()

    def evaluate(self, sample_generator):
        loss_fn = torch.nn.MSELoss()

        # 图特征
        edge_indexs = sample_generator.edge_indexs
        edge_indexs_no_same = sample_generator.edge_indexs_no_same
        graph_features = sample_generator.graph_features

        # 生成val_data 列表
        val_data, val_indexs = sample_generator.generate_val_batch()

        # 获取val_data 特征
        (types, groups, targets, ratings, prices, lengthes, neighbortypes,
         neighbordistances, neighborratings, neighborcomments,
         neighborprices, neighborgroups) = sample_generator.get_train_batch(val_data)

        # 转换cuda
        if torch.cuda.is_available():
            types = types.cuda()
            groups = groups.cuda()
            targets = targets.cuda()
            ratings = ratings.cuda()
            prices = prices.cuda()
            neighbortypes = neighbortypes.cuda()
            neighbordistances = neighbordistances.cuda()
            neighborratings = neighborratings.cuda()
            neighborcomments = neighborcomments.cuda()
            neighborprices = neighborprices.cuda()
            neighborgroups = neighborgroups.cuda()

        # TODO delete
        batch_id = 10000

        # 验证
        val_scores = self.model(types, groups, ratings, prices, lengthes, neighbortypes,
                                neighbordistances, neighborratings, neighborcomments,
                                neighborprices, neighborgroups, graph_features, edge_indexs, edge_indexs_no_same,  val_indexs)
        val_mse = loss_fn(val_scores, targets)
        ########################################################################

        # 生成test_data id列表
        test_data, test_indexs = sample_generator.generate_test_batch()

        # 获取test_data 特征
        (types, groups, targets, ratings, prices, lengthes, neighbortypes,
         neighbordistances, neighborratings, neighborcomments,
         neighborprices, neighborgroups, regions) = sample_generator.get_test_batch(test_data)

        if torch.cuda.is_available():
            types = types.cuda()
            groups = groups.cuda()
            targets = targets.cuda()
            ratings = ratings.cuda()
            prices = prices.cuda()
            neighbortypes = neighbortypes.cuda()
            neighbordistances = neighbordistances.cuda()
            neighborratings = neighborratings.cuda()
            neighborcomments = neighborcomments.cuda()
            neighborprices = neighborprices.cuda()
            neighborgroups = neighborgroups.cuda()

        # TODO delete
        batch_id = 100000
        print(batch_id)

        # 验证
        test_scores = self.model(types, groups, ratings, prices, lengthes, neighbortypes,
                                 neighbordistances, neighborratings, neighborcomments,
                                 neighborprices, neighborgroups, graph_features, edge_indexs, edge_indexs_no_same,  test_indexs)
        test_mse = loss_fn(test_scores, targets)
        ################################################################

        # 计算NDCG
        result = pd.DataFrame({'regions': regions.tolist(), 'targets': targets.tolist(),
                               'scores': test_scores.tolist()})
        regionList = list(result['regions'].value_counts().index)
        ndcg = []
        for iregion in regionList:
            if len(result[result['regions'] == iregion]['targets'].values) > 1:
                ndcg.append(ndcg_score(result[result['regions'] == iregion]['targets'].values.reshape((1, -1)),
                                       result[result['regions'] == iregion]['scores'].values.reshape((1, -1))))
        ndcg = np.mean(ndcg)
        return val_mse, test_mse, ndcg

    def save(self, alias, epoch_id, val_mse, test_mse, ndcg):
        model_dir = self.config['model_dir'].format(
            alias, epoch_id, val_mse, test_mse, ndcg)
        save_checkpoint(self.model, model_dir)
