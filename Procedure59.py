# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 21:36:11 2019

@author: User
"""

import torch
from engine import Engine
from ShopModels import Pair_Shop_MLP,GCN2Layers, GCN_Shop_MLP,GCN1Layers
from CombineMLP import Combine_MLP,CombineGNN_MLP
from scoreMLP import Pair_score_MLP, RNN_score_MLP, GNN_score_MLP
import numpy as np


class Procedure(torch.nn.Module):
    def __init__(self,config):
        super(Procedure,self).__init__()
        self.config=config
        
        # type-based 类型的嵌入向量
        self.Pair_embedding_types=torch.nn.Embedding(config['n_types']+1,
                                                config['type_hidden_size'],
                                                padding_idx=config['n_types'])
        
        # 距离的嵌入向量
        self.embedding_Pair_distances=torch.nn.Embedding(config['n_distances']+1,
                                                    config['distance_hidden_size'],
                                                    padding_idx=config['n_distances'])
        
        # 价格的嵌入向量
        self.embedding_Pair_prices=torch.nn.Embedding(config['n_prices']+1,
                                                 config['price_hidden_size'],
                                                 padding_idx=config['n_prices'])
        # 
        self.Pair_shop_mlp=Pair_Shop_MLP(config)
        self.Pair_score_mlp=Pair_score_MLP(config)
        
        



        
        self.GCN_Shop_MLP = GCN_Shop_MLP(config)
        self.GCN1Layers = GCN1Layers(config)
        self.LSTM = torch.nn.LSTM(input_size=16,
                hidden_size=16,
                num_layers=3,
                )
        self.RNN_score_MLP = RNN_score_MLP(config)


        self.GCN2_Shop_MLP = GCN_Shop_MLP(config)
        self.GCN2Layers= GCN2Layers(config) 
        self.GNN_score_MLP = GNN_score_MLP(config)


        self.GCN_Shop_MLP_no_sameType = GCN_Shop_MLP(config)
        self.GCN1Layers_no_sameType = GCN1Layers(config)
        self.LSTM_no_same_Type = torch.nn.LSTM(input_size=16,
                hidden_size=16,
                num_layers=3,
                )
        self.GRN_no_same_Type = RNN_score_MLP(config)

        self.combine_mlp=Combine_MLP(config)
        
        
        #cuda部分
        if torch.cuda.is_available():
            self.Pair_embedding_types=self.Pair_embedding_types.cuda()
            self.embedding_Pair_distances=self.embedding_Pair_distances.cuda()
            self.embedding_Pair_prices=self.embedding_Pair_prices.cuda()
            self.Pair_shop_mlp=self.Pair_shop_mlp.cuda()
            self.Pair_score_mlp=self.Pair_score_mlp.cuda()
                   

            self.GCN_Shop_MLP = self.GCN_Shop_MLP.cuda()
            self.GCN1Layers = self.GCN1Layers.cuda()
            self.LSTM = self.LSTM.cuda()
            self.RNN_score_MLP = self.RNN_score_MLP.cuda()
            # self.RNN_shop_mlp=self.RNN_Shop_mlp.cuda()
            # self.RNN_score_mlp=self.RNN_score_mlp.cuda()
            # self.distanceLSTM=self.distanceLSTM.cuda()
            self.GCN2_Shop_MLP = self.GCN2_Shop_MLP.cuda()

            self.GCN2Layers = self.GCN2Layers.cuda()
            self.GNN_score_MLP = self.GNN_score_MLP.cuda()
            self.combine_mlp=self.combine_mlp.cuda()
        
            self.GCN_Shop_MLP_no_sameType = self.GCN_Shop_MLP_no_sameType.cuda()
            self.GCN1Layers_no_sameType = self.GCN1Layers_no_sameType.cuda()
            self.LSTM_no_same_Type =  self.LSTM_no_same_Type.cuda()
            self.GRN_no_same_Type = self.GRN_no_same_Type.cuda()
        
    def forward(self,types,groups,ratings,prices,lengthes,neighbortypes,
                neighbordistances,neighborratings,neighborcomments,
                neighborprices,neighborgroups,graph_features,edge_indexs,edge_index_no_same,indexs): 
       
       
        #Pair based aggregation
        Pair_type_embedding=self.Pair_embedding_types(types)
        Pair_neighbortypes_embedding=self.Pair_embedding_types(neighbortypes)   #(batch_size, n_shops, shop_hidden_size)
        
        Pair_neighbordistances_embedding=self.embedding_Pair_distances(neighbordistances)
        Pair_neighborprices_embedding=self.embedding_Pair_prices(neighborprices)
        
        pair_shops_hidden=self.Pair_shop_mlp(Pair_neighbortypes_embedding,Pair_neighbordistances_embedding,neighborratings,
                                  neighborcomments,Pair_neighborprices_embedding,neighborgroups) #(10,15,16)
        
        
        # recent partition
        aggregator=[]
        for i,j in enumerate(lengthes):
            if j ==0:
                t = torch.FloatTensor(np.zeros((1,pair_shops_hidden.shape[-1]))) #如果没有邻居，就是全0向量
                if torch.cuda.is_available():
                    t = t.cuda()
                aggregator.append(t)
            else:
                aggregator.append(torch.mean(pair_shops_hidden[i:i+1,:j,:],1,False)) #有邻居,就是取平均向量，所有邻居的平均！
        
        pair_context=torch.cat(aggregator)   #(batch_size,hidden_size) (10,16)
        Pair_result=self.Pair_score_mlp(pair_context,Pair_type_embedding)


        

        
        # batch_size = len(Pair_type_embedding)  
        # j = batch_size*batch_id
        # if batch_id == 10000:
        #     j=2491
        #     jstart=2491
        # if batch_id == 100000:
        #     j=3321
        #     jstart=3321
        # for i,_ in enumerate(ratings):
        #     j+=1

        #竞争模块

        RNN_hiddens = []
        for graph_feature,edge_index in zip(graph_features,edge_indexs):
            (all_types,all_prices,all_groups)= graph_feature
            x = self.GCN_Shop_MLP(all_types.cuda(),all_prices.cuda(),all_groups.cuda())
            h = self.GCN1Layers(x,edge_index.cuda())
            h = h.view(1,h.shape[0],h.shape[1])
            RNN_hiddens.append(h)
        
        RNN_hidden = torch.cat(RNN_hiddens).cuda()
        output, (hn, cn) = self.LSTM(RNN_hidden)

        
        hn_s = []
        for index in indexs:
          hn_index = hn[-1][index].view(1,-1)
          hn_s.append(hn_index)
        hn_cat = torch.cat(hn_s,0)
       
        RNN_result = self.RNN_score_MLP(hn_cat) 


        #集聚模块

        RNN_hidden_no_same = []
        for graph_feature,edge_index_no_same in zip(graph_features,edge_index_no_same):
            (all_types,all_prices,all_groups)=graph_feature
            x = self.GCN_Shop_MLP_no_sameType(all_types.cuda(),all_prices.cuda(),all_groups.cuda())
            h = self.GCN1Layers_no_sameType(x,edge_index.cuda())
            h = h.view(1,h.shape[0],h.shape[1])
            RNN_hidden_no_same.append(h)
        
        RNN_hidden_no_same = torch.cat(RNN_hidden_no_same).cuda()
        output, (hn, cn) = self.LSTM(RNN_hidden_no_same)


        hn_s = []
        for index in indexs:
          hn_index = hn[-1][index].view(1,-1)
          hn_s.append(hn_index)
        hn_cat = torch.cat(hn_s,0)

        RNN_result_nosame = self.GRN_no_same_Type(hn_cat)






        #消融
        # final_scores=self.combine_mlp(RNN_result,GNN_result)
        # final_scores=self.combine_mlp(Pair_result,RNN_result,GNN_result)
        # final_scores=self.combine_mlp(Pair_result,GNN_result)
        # final_scores = self.combine_mlp(Pair_result,RNN_result)
        final_scores = self.combine_mlp(Pair_result,RNN_result,RNN_result_nosame)



        

        #预测
        scores=final_scores.squeeze(1)
        return scores


class ProcedureEngine(Engine):
    def __init__(self,config):
        #Main Model
        self.model=Procedure(config) 
        super(ProcedureEngine,self).__init__(config)
        if config['pretrained']==True:
          PATH = config['pretrained_path']
          self.model.load_state_dict(torch.load(PATH))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # self.model.eval()
        #use pretrained
        # model_dict=self.model.state_dict()
        # print(torch.cuda.is_available())
        # if torch.cuda.is_available():
        #     pair=torch.load('checkpoints/GNN60_Epoch3_valmse4.2323_testmse4.0734_ndcg0.7604.model')
        #     cnn=torch.load('checkpoints/GNN61_Epoch12_valmse4.8727_testmse4.8142_ndcg0.7532.model')
        #     rnn=torch.load('checkpoints/GNN62_Epoch2_valmse4.5752_testmse4.4591_ndcg0.7548.model')
        # else:
        #     pair=torch.load('checkpoints/GNN60_Epoch3_valmse4.2323_testmse4.0734_ndcg0.7604.model',map_location=torch.device('cpu'))
        #     cnn=torch.load('checkpoints/GNN61_Epoch12_valmse4.8727_testmse4.8142_ndcg0.7532.model',map_location=torch.device('cpu'))
        #     rnn=torch.load('checkpoints/GNN62_Epoch2_valmse4.5752_testmse4.4591_ndcg0.7548.model',map_location=torch.device('cpu'))
        # pretrained_pair={k:v for k,v in pair.items() if k in model_dict}
        # model_dict.update(pretrained_pair)
        
        # pretrained_cnn={k:v for k,v in cnn.items() if k in model_dict}
        # model_dict.update(pretrained_cnn)
        
        # pretrained_rnn={k:v for k,v in rnn.items() if k in model_dict}
        # model_dict.update(pretrained_rnn)
        
        # self.model.load_state_dict(model_dict)
        
# =============================================================================
#         for i,p in enumerate(self.model.parameters()):
#             if i < 49 or i> 56 :
#                 p.requires_grad = False
# =============================================================================
        
        
        
        
        
        
        