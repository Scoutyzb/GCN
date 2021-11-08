# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 22:38:59 2019

@author: User
"""

import torch
import random
import pandas as pd
import numpy as np
from tkinter import _flatten
from random import choice


def g_type_based_matrix(n_types,neighbortypes,neighbortypes_embedding,neighbordistances_embedding,neighborratings,
                                  neighborcomments,neighborprices_embedding,neighborgroups):
    '''
        n_types: scalar
        neighbortypes: tensor (batch_size,n_shops) 
        neighbortypes_embedding: tensor (batch_size,n_shops,type_hidden_size)
    '''
    neighbortypes=neighbortypes.data.numpy()
    
    batch_size=neighbortypes_embedding.shape[0]
    matrix=torch.tensor(np.zeros((batch_size,n_types,32))).float()
    
    # vector: tensor (batch_size,n_neighbors,type_hidden_size)
    neighborratings=neighborratings.view(neighborratings.shape[0],neighborratings.shape[1],1)
    neighborcomments=neighborcomments.view(neighborcomments.shape[0],neighborcomments.shape[1],1)
    neighborgroups=neighborgroups.view(neighborgroups.shape[0],neighborgroups.shape[1],1)
    vector=torch.cat([neighbortypes_embedding,neighbordistances_embedding,
                      neighborratings,neighborcomments,neighborprices_embedding,neighborgroups],dim=-1) 
    
    for i in range(batch_size):
        for j in range(n_types):
            index=np.where(neighbortypes[i]==j)
            vector1=vector[i][index]
            if len(vector1)!=0:
                matrix[i][j]=torch.sum(vector1).float()
                
    return matrix # tensor (batch_size,n_types,32)    

def g_distance_based_matrix(neighbordistances,neighbortypes_embedding,
                        neighborprices_embedding,neighborratings,neighborcomments,neighborgroups):
    '''
        object: rnn input (batch_size,time_step,Rnninput_size)
        neighbordistances: tensor (batch_size,n_shops) 大于5说明不存在
        neighbortypes_embedding: tensor (batch_size,n_shops,type_hidden_size)
    '''
    n_distances=6 
    
    neighbordistances=neighbordistances.data.numpy()
    
    batch_size=neighbortypes_embedding.shape[0]
    matrix=torch.tensor(np.zeros((batch_size,n_distances,24))).float()
    
    # vector: tensor (batch_size,n_neighbors,hidden_size)
    neighborratings=neighborratings.view(neighborratings.shape[0],neighborratings.shape[1],1)
    neighborcomments=neighborcomments.view(neighborcomments.shape[0],neighborcomments.shape[1],1)
    neighborgroups=neighborgroups.view(neighborgroups.shape[0],neighborgroups.shape[1],1)
    vector=torch.cat([neighbortypes_embedding,neighborprices_embedding,
                      neighborratings,neighborcomments,neighborgroups],dim=-1) 
    lastVector=vector[0][0]
    for i in range(batch_size):
        for j in range(n_distances):
            index=np.where(neighbordistances[i]==j)
            vector1=vector[i][index]
            if len(vector1)!=0:
                lastVector=torch.mean(vector1).float()
                matrix[i][j]=lastVector
            else:
                matrix[i][j]=lastVector
                
    return matrix # tensor (batch_size,time_step,Rnninput_size)        
        
def gen_neg(ori_shops,Neighbors,type2Index):
    result={}
    for shop in ori_shops['shopID']:
        n50=Neighbors[str(shop)]['50m']
        n100=Neighbors[str(shop)]['100m']
        n150=Neighbors[str(shop)]['150m']
        n200=Neighbors[str(shop)]['200m']
        n250=Neighbors[str(shop)]['250m']
        n300=Neighbors[str(shop)]['300m']
        bneighbors=n50+n100+n150+n200+n250+n300
        
        bneighbortypes=[ori_shops[ori_shops['shopID']==int(x)]['type'].values.tolist() for x in bneighbors]
        unappear_types=set(type2Index.keys())-set(_flatten(bneighbortypes))
        n_unappear=len(list(unappear_types))
        
        negative_types=[random.sample(list(unappear_types),int(n_unappear/2))][0]
        
        result[shop]=negative_types
    return result  
            

class SampleGenerator(object):
    def __init__(self,config,content,Neighbors,train_set,val_set,test_set,shopID2NegativeTypes,type2Index,region2Index):
        self.config=config
        self.type2Index=type2Index
        self.region2Index=region2Index
        self.content=content # all shops
        self.content.index=self.content['shopID']
        self.Neighbors=Neighbors
        self.train_set=train_set
        self.val_set=val_set
        self.test_set=test_set 
        self.content['region_number']=content['region_number'].apply(lambda x: 
            self.region2Index[x])
        self.train_set['region_number']=train_set['region_number'].apply(lambda x: 
            self.region2Index[x])
        self.val_set['region_number']=val_set['region_number'].apply(lambda x:
            self.region2Index[x])    
        self.test_set['region_number']=test_set['region_number'].apply(lambda x: 
            self.region2Index[x])
        
        self.shops=content['shopID'] #series
        self.ratings=content['rating'] #series
        self.comments=content['comments'] #series
        self.price=self.classify_price(content['price']) #series
        self.group=content['group'] #series
        self.type=self.classify_type(content['subType']) #series
        self.region=content['region_number'] #series
        self.coordinate=content['coordinate'] #series
        
        self.length=len(self.shops) 
        self.shopID2NegativeTypes=shopID2NegativeTypes

        
    def classify_price(self,prices):
        result=[]
        for price in prices:
            if np.isnan(price):
                result.append(0)
            elif price<20:
                result.append(1)
            elif price<50:
                result.append(2)
            elif price<100:
                result.append(3)
            else:
                result.append(4)
        assert len(result)==len(prices)
        result=pd.Series(result,index=self.content['shopID'])
        return result
    
    def classify_type(self,types):
        result=types.apply(lambda x: self.type2Index[x])
        return result

    def get_shopids(self,type):
        assert type in ['train','test','all']
        if type == 'train':
            return self.train_set['shopID'].tolist() 
        if type == 'test':
            return self.test_set['shopID'].tolist()
        if type == 'all':
            return self.content['shopID'].tolist()

    def generate_train_batch(self,batch_size):
# =============================================================================
#         regions=list(set(self.region))
#         shops=[]
#         for region in regions:
#             shops_in_region=self.train_set[self.train_set['region_number']==region]['shopID'].tolist()
#             if shops_in_region!=[]:
#                 shops.append(choice(shops_in_region))
# =============================================================================
        
        shops=self.train_set['shopID'].tolist()# train shopID列表
        length=len(shops) 
        from math import ceil
        n_batch=ceil(length/batch_size)
        batches=[]
        for i in range(n_batch):
            batches.append(shops[i*batch_size:(i+1)*batch_size])
        return batches    #[[...],...,[123992379, 107966842, 77358959, 66264666, 123671868]]
        
 
    
# =============================================================================
#     def generate_test_batch(self,test_size):
#         regions=list(set(self.region))
#         shops=[]
#         for region in regions:
#             shops_in_region=self.test_set[self.test_set['region_number']==region]['shopID'].tolist()
#             if shops_in_region!=[]:
#                 shops.append(choice(shops_in_region))
#         
#         return shops
# =============================================================================
        
    def generate_test_batch(self,test_size):
        shops=self.test_set['shopID'].tolist()  
        return shops
    
    def generate_val_batch(self,val_size):
        shops=self.val_set['shopID'].tolist()  
        return shops        
    
    def get_data_x(self,i):
            (types,groups,targets,ratings,prices,lengthes,neighbortypes,
                neighbordistances,neighborratings,neighborcomments,
                neighborprices,neighborgroups,regions) = self.get_test_batch(i)
            return (types,prices,groups)
    def get_test_batch(self,i):
        targets,types,groups,ratings,prices=[],[],[],[],[]
        (neighbors,neighbortypes,neighbordistances,neighborratings,neighborcomments,neighborprices,neighborgroups,
         lengthes)=([],[],[],[],[],[],[],[])
        regions = []
        max_n=0
        for shop in i:
            # t_neighbors 邻居列表 t_distances 对应的距离列表
            (t_neighbors,t_neighbortypes,t_distances,t_ratings,t_comments,t_price,
             t_group,t_region)=self.shop_groups(shop)
            if max_n<len(t_neighbors):
                #这个batch中邻居最大者的邻居数量
                max_n=len(t_neighbors)
            
            targets.append(self.comments[shop])
            types.append(self.type[shop])

            groups.append(self.group[shop])
            ratings.append(self.ratings[shop])
            prices.append(self.price[shop])
            regions.append(self.region[shop])
            
            neighbors.append(t_neighbors)
            neighbortypes.append(t_neighbortypes)
            neighbordistances.append(t_distances)
            neighborratings.append(t_ratings)
            neighborcomments.append(t_comments)
            neighborprices.append(t_price)
            neighborgroups.append(t_group)
            #邻居数量
            lengthes.append(len(t_neighbors))
               
                  
        for i,_ in enumerate(neighbors): 
            n_pad=max_n-len(neighbors[i])#需要补足的邻居数
            # neighbors[i]=neighbors[i]+[self.config['n_shops']]*n_pad#补足操作，PAD
            neighbortypes[i]=neighbortypes[i]+[self.config['n_types']]*n_pad
            neighbordistances[i]=neighbordistances[i]+[self.config['n_distances']]*n_pad
            neighborratings[i]=neighborratings[i]+[0]*n_pad#ratting为0所以没有
            neighborcomments[i]=neighborcomments[i]+[0]*n_pad
            neighborprices[i]=neighborprices[i]+[self.config['n_prices']]*n_pad
            neighborgroups[i]=neighborgroups[i]+[0]*n_pad
        
        
        targets=torch.FloatTensor(targets)   #(batch_size*n)
        types=torch.tensor(_flatten(types))
        groups=torch.tensor(_flatten(groups))
        regions=torch.tensor(_flatten(regions))

        ratings=torch.FloatTensor(_flatten(ratings))
        prices=torch.FloatTensor(_flatten(prices))
        #neighbors=torch.tensor(neighbors)
        neighbortypes=torch.LongTensor(neighbortypes)
        neighbordistances=torch.LongTensor(neighbordistances)
        neighborratings=torch.FloatTensor(neighborratings)
        neighborcomments=torch.FloatTensor(neighborcomments)
        neighborprices=torch.LongTensor(neighborprices)
        neighborgroups=torch.FloatTensor(neighborgroups)

        return (types,groups,targets,ratings,prices,lengthes,neighbortypes,
                neighbordistances,neighborratings,neighborcomments,
                neighborprices,neighborgroups,regions)
    
    def get_train_batch(self,i):
        return self.get_test_batch(i)[0:-1]

        
    def shop_groups(self,shop):
        #某个商店id
        bneighbors,bdistances=[],[]
        for index,dist in enumerate(['50m','100m','150m','200m','250m','300m']):
            #字典套字典 
            n=self.Neighbors[str(shop)][dist]
            #n是一个id列表,里面是int id
            n=[int(x) for x in n]
            #去重
            n=set(n)
            if dist=='50m':
                try:
                    #移除自己
                    n.remove(int(shop))
                except:
                    print("processing")
            #取交集,去除非训练集数据
            n=n&set(self.train_set['shopID'].tolist())
            n=list(n)
            # 邻居列表
            bneighbors=bneighbors+n
            # 0 0 1 2
            bdistances=bdistances+[index]*len(n)
            
            
        indexes=bneighbors
        bneighbortypes=self.type[indexes].tolist()
        bratings=self.ratings[indexes].tolist()
        bcomments=self.comments[indexes].tolist()
        bprice=self.price[indexes].tolist()
        bgroup=self.group[indexes].tolist()
        bregion=self.region[indexes].tolist()
        
        return (bneighbors,bneighbortypes,bdistances,bratings,bcomments,bprice,bgroup,bregion)
    
    
        
        
