# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:33:01 2019

@author: User
"""

import torch

def use_optimizer(model,config):
    if config['optimizer']=='adam':
        optimizer=torch.optim.Adam(model.parameters(),lr=config['adam_lr'],weight_decay=config['l2_regularization'])        
    return optimizer    

def save_checkpoint(model,model_dir):
    torch.save(model.state_dict(),model_dir)
    
    print("You have saved %s"%(model_dir))

#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
import smtplib
from email.mime.text import MIMEText
from email.header import Header


def email2Me(): 
    # 第三方 SMTP 服务
    mail_host="smtp.qq.com"  #设置服务器
    mail_user="775734337@qq.com"    #用户名
    mail_pass="qufyzjwmhzbmbege"   #口令 
    
    
    sender = '775734337@qq.com'
    receivers = ['scoutyzb@yeah.net']  # 接收邮件，可设置为你的QQ邮箱或者其他邮箱
    
    message = MIMEText('训练已经结束，请交还实例', 'plain', 'utf-8')
    message['From'] = Header("菜鸟教程", 'utf-8')
    message['To'] =  Header("测试", 'utf-8')
    
    subject = '训练结束通知信息'
    message['Subject'] = Header(subject, 'utf-8')
    
    
    try:
        smtpObj = smtplib.SMTP() 
        smtpObj.connect(mail_host, 587)    # 25 为 SMTP 端口号
        smtpObj.login(mail_user,mail_pass)  
        smtpObj.sendmail(sender, receivers, message.as_string())
        print("邮件发送成功")
    except smtplib.SMTPException as e:
        print(e)
        print("Error: 无法发送邮件")

