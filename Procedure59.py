# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 21:36:11 2019

@author: User
"""

import torch
from engine import Engine
from ShopModels import Pair_Shop_MLP,CNN_Shop_MLP,GCN2Layers, GCN_Shop_MLP
from CombineMLP import Combine_MLP,CombineGNN_MLP
#from ScoreMLP import Score_MLP
from scoreMLP import Pair_score_MLP, CNN_score_MLP, GNN_score_MLP
import sys
import numpy as np
import time


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
        
        
        #CNN
        self.CNN_embedding_types=torch.nn.Embedding(config['n_types']+1,
                                                config['type_hidden_size'],
                                                padding_idx=config['n_types'])
        self.embedding_CNN_distances=torch.nn.Embedding(config['n_distances']+1,
                                                    config['distance_hidden_size'],
                                                    padding_idx=config['n_distances'])
        self.embedding_CNN_prices=torch.nn.Embedding(config['n_prices']+1,
                                                 config['price_hidden_size'],
                                                 padding_idx=config['n_prices'])
        self.CNN_shop_mlp=CNN_Shop_MLP(config)
        self.CNN_score_mlp=CNN_score_MLP(config)
        
        self.typeConv=torch.nn.Sequential(
                torch.nn.Conv2d(1,1,3,1,1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(1,1,3,1,1),
                torch.nn.ReLU()
                )
        
        
        
        #RNN 类型的嵌入层  #当输入为type+1的时候，代表0 0 
        self.GNN_embedding_types=torch.nn.Embedding(config['n_types']+1,
                                                config['type_hidden_size'],
                                                padding_idx=config['n_types'])
        
        self.embedding_GNN_prices=torch.nn.Embedding(config['n_prices']+1,
                                                 config['price_hidden_size'],
                                                 padding_idx=config['n_prices'])
        
        self.embedding_GNN_distances=torch.nn.Embedding(config['n_distances']+1,
                                                    config['distance_hidden_size'],
                                                    padding_idx=config['n_distances'])
        # self.RNN_Shop_mlp=RNN_Shop_MLP(config)
        # self.RNN_score_mlp=RNN_score_MLP(config)
        
        # self.distanceLSTM=torch.nn.LSTM(
        #         input_size=16,
        #         hidden_size=16,
        #         num_layers=3,
        #         batch_first=True
        #         )
        self.GNN_Shop_Mlp = GCN_Shop_MLP(config)
        self.GCN2Layers= GCN2Layers(config) 
        self.GNN_score_MLP = GNN_score_MLP(config)
        self.combine_mlp=CombineGNN_MLP(config)
        
        
        #cuda部分
        if torch.cuda.is_available():
            self.Pair_embedding_types=self.Pair_embedding_types.cuda()
            self.embedding_Pair_distances=self.embedding_Pair_distances.cuda()
            self.embedding_Pair_prices=self.embedding_Pair_prices.cuda()
            self.Pair_shop_mlp=self.Pair_shop_mlp.cuda()
            self.Pair_score_mlp=self.Pair_score_mlp.cuda()
                   
            self.CNN_embedding_types=self.CNN_embedding_types.cuda()
            self.embedding_CNN_distances=self.embedding_CNN_distances.cuda()
            self.embedding_CNN_prices=self.embedding_CNN_prices.cuda()
            self.CNN_shop_mlp=self.CNN_shop_mlp.cuda()
            self.CNN_score_mlp=self.CNN_score_mlp.cuda()
            self.typeConv=self.typeConv.cuda()
            
            # self.RNN_embedding_types=self.RNN_embedding_types.cuda()
            self.GNN_embedding_types = self.GNN_embedding_types.cuda()
            # self.embedding_RNN_prices=self.embedding_RNN_prices.cuda()
            self.embedding_GNN_prices=self.embedding_GNN_prices.cuda()

            self.embedding_GNN_distances=self.embedding_GNN_distances.cuda()
            # self.RNN_shop_mlp=self.RNN_Shop_mlp.cuda()
            # self.RNN_score_mlp=self.RNN_score_mlp.cuda()
            # self.distanceLSTM=self.distanceLSTM.cuda()
            self.GCN2Layers = self.GCN2Layers.cuda()
            self.GNN_score_MLP = self.GNN_score_MLP.cuda()
            self.combine_mlp=self.combine_mlp.cuda()
            self.GNN_Shop_Mlp = self.GNN_Shop_Mlp.cuda()
        
        
        
    def forward(self,types,groups,ratings,prices,lengthes,neighbortypes,
                neighbordistances,neighborratings,neighborcomments,
                neighborprices,neighborgroups,tpg,edge_index,batch_id): 
        #Pair based aggregation
        pairtime1 = time.time()

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
                aggregator.append(torch.mean(pair_shops_hidden[i:i+1,:j,:],1,False)) #有邻居,就是取平均向量，所有邻居的平均？！
        
        pair_context=torch.cat(aggregator)   #(batch_size,hidden_size) (10,16)
        
        Pair_result=self.Pair_score_mlp(pair_context,Pair_type_embedding)

        pairduration = time.time()-pairtime1
        print("Pair Duration Time is %f"%(pairduration))

        # #CNN partition
        # cnntime1 = time.time()

        # CNN_type_embedding=self.CNN_embedding_types(types)
        # CNN_neighbortypes_embedding=self.CNN_embedding_types(neighbortypes)   #(batch_size, n_shops, shop_hidden_size)
        
        # CNN_neighbordistances_embedding=self.embedding_CNN_distances(neighbordistances)
        # CNN_neighborprices_embedding=self.embedding_CNN_prices(neighborprices)
        
        # CNN_shops_hidden=self.CNN_shop_mlp(CNN_neighbortypes_embedding,CNN_neighbordistances_embedding,neighborratings,
        #                           neighborcomments,CNN_neighborprices_embedding,neighborgroups) #(10,15,16)
        
        # type_based_matrix=self.g_type_based_matrix(CNN_shops_hidden,neighbortypes) #(10,126,16)
        # if torch.cuda.is_available():
        #     type_based_matrix=type_based_matrix.cuda()
        # type_matrix=type_based_matrix.view(type_based_matrix.shape[0],1,type_based_matrix.shape[1],
        #                                    type_based_matrix.shape[2])
        # type_context_1=self.typeConv(type_matrix).squeeze()
        # type_context_2=torch.empty(type_context_1.shape[0],type_context_1.shape[-1])
        # if torch.cuda.is_available(): #(10,126,16)
        #     type_context_2=type_context_2.cuda() #(10,16)
        # for i,t in enumerate(types):
        #     type_context_2[i]=type_context_1[i][t.item()]  
        # #type_context=type_context.view(type_context.size(0),-1) #15位 #(10,15)
        # #type_context_3=self.GNN_CNN_MLP(type_context_2)
        # CNN_result=self.CNN_score_mlp(type_context_2,CNN_type_embedding)
        
        # cnnduration = time.time()-cnntime1
        # print("CNN Time Duartion is %f"%(cnnduration))

        gnntime1 = time.time()
        #RNN partition #batch.size * 1
        (all_types,all_prices,all_groups)=tpg
        x = self.GNN_Shop_Mlp(all_types.cuda(),all_prices.cuda(),all_groups.cuda())
        x = x.cuda()
        edge_index = edge_index.cuda()
        
        
        
        batch_size = len(Pair_type_embedding)
        
        
        j = batch_size*batch_id
        if batch_id == 10000:
            j=2491
            jstart=2491
        if batch_id == 100000:
            j=3321
            jstart=3321

        for i,_ in enumerate(ratings):
            j+=1
        
        #
        # if 0 == ((torch.zeros(data.x.shape[0],data.x.shape[1]) != data.x).sum()):
        #     pass
        # else:
        #     pass
        # if torch.cuda.is_available():
        #     data=data.cuda()  
        # print(data)
        GNN_hidden = self.GCN2Layers(x,edge_index)
        if batch_id==10000 or batch_id==100000:
            GNN_result = self.GNN_score_MLP(GNN_hidden[jstart:j,:])
        else:
            GNN_result = self.GNN_score_MLP(GNN_hidden[batch_id*batch_size:(batch_id+1)*batch_size,:])
        # if torch.cuda.is_available():
            # GNN_hidden=GNN_hidden.cuda()  
        # distance_based_matrix=self.g_distance_based_matrix(rnn_neighbors_hidden,neighbordistances) #(10,6,16)
        #distance_context1=distance_based_matrix.reshape(distance_based_matrix.shape[0],-1)
        #RNN input (batch_size,neibors_len,embeddingsize)
        # distance_based_context,(h_n,h_c)=self.distanceLSTM(distance_based_matrix)
        # distance_context2=distance_based_context[:,-1,:] #(10,16)
        # RNN_result=self.RNN_score_mlp(distance_context2,RNN_type_embedding)
        
        
        gnnduration = time.time()-gnntime1
        print("GNN Time Duartion is %f"%(gnnduration))

        final_scores=self.combine_mlp(Pair_result,GNN_result)
        


        scores=final_scores.squeeze(1)
        # print(type(scores))
        return scores

    def g_type_based_matrix(self,shops_hidden,neighbortypes):
        '''
            shops_hidden: tensor (batch_size,n_neighbors,hidden_size)
            neighbortypes: tensor (batch_size,n_neighbors)
        '''
        neighbors_types=neighbortypes.cpu()
        batch_size,n_types=shops_hidden.shape[0],self.config['n_types']
        matrix=torch.tensor(np.zeros((batch_size,n_types,16))).float()

        
        for i in range(batch_size):
            for j in range(n_types):
                index=np.where(neighbors_types[i]==j)
                vector1=shops_hidden[i][index]
                if len(vector1)!=0:
                    matrix[i][j]=torch.sum(vector1).float()
                    
        return matrix # tensor (batch_size,n_types,32)    

    def g_distance_based_matrix(self,rnn_neighbors_hidden,neighbordistances):
        # TODO
        n_distances= 6
        neighbordistances=neighbordistances.cpu().data.numpy()
        batch_size=neighbordistances.shape[0]

        #batsize 时间步数 , hidden_size
        matrix=torch.tensor(np.zeros((batch_size,n_distances,16))).float()


        vector=rnn_neighbors_hidden
        try: 
            #第一个训练数据的第一个lenght？
            lastVector=vector[0][0]
        except:
            lastVector=torch.zeros(16)
        for i in range(batch_size):
            # n_distances 距离的种类数？
            # n_distances = len(neighbordistances[i])
            for k,j in zip(range(n_distances),reversed(range(n_distances))):
                #j由大到小 也就是由远及近 j代表的是真实距离 k代表的是index
                index=np.where(neighbordistances[i]==j)
                
                vector1=vector[i][index]
                #似乎有问题
                if len(vector1)!=0:
                    lastVector=torch.mean(vector1).float()# cvj/alpha
                    matrix[i][k]=lastVector
                else:
                    matrix[i][k]=lastVector
        if torch.cuda.is_available():            
            return matrix.cuda() # tensor (batch_size,time_step,Rnninput_size)  
        else:
            return matrix
class ProcedureEngine(Engine):
    def __init__(self,config):
        self.model=Procedure(config)
        super(ProcedureEngine,self).__init__(config)
        
        # PATH = 'checkpoints/GNNNoCNN_Epoch6_valmse3.5584_testmse3.5024_ndcg0.9117.model'

        # self.model.load_state_dict(torch.load(PATH))
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
        
        
        
        
        
        
        