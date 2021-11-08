# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 23:13:18 2019

@author: User
"""
import torch
from utils import use_optimizer,save_checkpoint
from metrics import Metrics
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score 
import os
import time

class Engine(object):
    def __init__(self,config):
        self.config=config
        self._metron=Metrics(1)
        self.opt=use_optimizer(self.model,config)
        self.crit=torch.nn.MSELoss()
        torch.autograd.set_detect_anomaly(True)
        
        if torch.cuda.is_available():
            self.crit=self.crit.cuda()
        
    def train_single_batch(self,types,groups,targets,ratings,prices,lengthes,neighbortypes,
                neighbordistances,neighborratings,neighborcomments,
                neighborprices,neighborgroups,tpg,edge_index,batch_id):
        
        if torch.cuda.is_available():
            types=types.cuda()
            groups=groups.cuda()
            targets=targets.cuda()
            ratings=ratings.cuda()
            prices=prices.cuda()
            neighbortypes=neighbortypes.cuda()
            neighbordistances=neighbordistances.cuda()
            neighborratings=neighborratings.cuda()
            neighborcomments=neighborcomments.cuda()
            neighborprices=neighborprices.cuda()
            neighborgroups=neighborgroups.cuda()
        
        self.opt.zero_grad()
        # predir = os.path.dirname(os.path.abspath(__file__))+os.path.sep+"."
        # torch.onnx.export(self.model,(types,regions,ratings,prices,neighbortypes,
        #         neighbordistances,neighborratings,neighborcomments,
        #         neighborprices,neighborgroups),predir+'/output.onnx',input_names=['类型','地区','评分','价格','邻居种类','邻居距离'
        #         ,'邻居得分','邻居评论数','邻居价格','邻居组数'],opset_version=11)
        scores=self.model(types,groups,ratings,prices,lengthes,neighbortypes,
                neighbordistances,neighborratings,neighborcomments,
                neighborprices,neighborgroups,tpg,edge_index,batch_id) #(batch_size)
        loss=self.crit(scores,targets)
        timebackend = time.time()

        loss.backward(retain_graph=True)

        print("后向传播：")
        print(time.time()-timebackend)
        self.opt.step()
        # loss=loss.item()
        return loss

    
    def train_an_epoch(self,sample_generator,epoch_id,tpg,edge_index):
        self.model.train()
        total_loss=0
        
        batches=sample_generator.generate_train_batch(self.config['batch_size'])
        #batches
        for batch_id, batch in enumerate(batches):
            #batch 是一组batch shopid
            if len(batch)==1:
                continue
            (types,groups,targets,ratings,prices,lengthes,neighbortypes,
                neighbordistances,neighborratings,neighborcomments,
                neighborprices,neighborgroups)=sample_generator.get_train_batch(batch)
            loss=self.train_single_batch(types,groups,targets,ratings,prices,lengthes,neighbortypes,
                neighbordistances,neighborratings,neighborcomments,
                neighborprices,neighborgroups,tpg,edge_index,batch_id)

            print('[Training Epoch{}] Batch {}, loss {}'.format(epoch_id,
                  batch_id,loss.item()))
            total_loss+=loss.item()
        # loss.backward()
            
    def evaluate(self,sample_generator,epoch_id,tpg,edge_index):
        loss_fn=torch.nn.MSELoss()
        
        
        val_data=sample_generator.generate_val_batch(self.config['test_size'])
        print(val_data[0])
        (types,groups,targets,ratings,prices,lengthes,neighbortypes,
                neighbordistances,neighborratings,neighborcomments,
                neighborprices,neighborgroups)=sample_generator.get_train_batch(val_data)
        
        if torch.cuda.is_available():
            types=types.cuda()
            groups=groups.cuda()
            targets=targets.cuda()
            ratings=ratings.cuda()
            prices=prices.cuda()
            neighbortypes=neighbortypes.cuda()
            neighbordistances=neighbordistances.cuda()
            neighborratings=neighborratings.cuda()
            neighborcomments=neighborcomments.cuda()
            neighborprices=neighborprices.cuda()
            neighborgroups=neighborgroups.cuda()
        
        batch_id = 10000
        val_scores=self.model(types,groups,ratings,prices,lengthes,neighbortypes,
                neighbordistances,neighborratings,neighborcomments,
                neighborprices,neighborgroups,tpg,edge_index,batch_id)
        
        val_mse=loss_fn(val_scores,targets)
        ########################################################################
        
        evaluate_data=sample_generator.generate_test_batch(self.config['test_size'])
        (types,groups,targets,ratings,prices,lengthes,neighbortypes,
                neighbordistances,neighborratings,neighborcomments,
                neighborprices,neighborgroups,regions)=sample_generator.get_test_batch(evaluate_data)
        
        if torch.cuda.is_available():
            types=types.cuda()
            groups=groups.cuda()
            targets=targets.cuda()
            ratings=ratings.cuda()
            prices=prices.cuda()
            neighbortypes=neighbortypes.cuda()
            neighbordistances=neighbordistances.cuda()
            neighborratings=neighborratings.cuda()
            neighborcomments=neighborcomments.cuda()
            neighborprices=neighborprices.cuda()
            neighborgroups=neighborgroups.cuda()
        batch_id = 100000
        print(batch_id)
        test_scores=self.model(types,groups,ratings,prices,lengthes,neighbortypes,
                neighbordistances,neighborratings,neighborcomments,
                neighborprices,neighborgroups,tpg,edge_index,batch_id)
        
        test_mse=loss_fn(test_scores,targets)
        
        result=pd.DataFrame({'regions':regions.tolist(),'targets':targets.tolist(),
                             'scores':test_scores.tolist()})
        regionList=list(result['regions'].value_counts().index)
        ndcg=[]
        for iregion in regionList:
            if len(result[result['regions']==iregion]['targets'].values) > 1:
                ndcg.append(ndcg_score(result[result['regions']==iregion]['targets'].values.reshape((1,-1)),
                                   result[result['regions']==iregion]['scores'].values.reshape((1,-1))))
        ndcg=np.mean(ndcg)
        return val_mse,test_mse,ndcg
    
    def save(self,alias,epoch_id,val_mse,test_mse,ndcg):
        model_dir=self.config['model_dir'].format(alias,epoch_id,val_mse,test_mse,ndcg)
        save_checkpoint(self.model,model_dir)