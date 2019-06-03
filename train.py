# -*- coding: utf-8 -*-

import tensorflow as tf
import config as cfg
from utils import read_one_picture
from utils import U_net
from utils import batch_namequeue,get_list
import numpy as np
import random

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

x=tf.placeholder(dtype=tf.float32,shape=[None,cfg.w_inputs,cfg.h_inputs,cfg.input_channels],name='x')
y=tf.placeholder(dtype=tf.float32,shape=[None,cfg.w_inputs,cfg.h_inputs,1],name='y')
keep_prob=tf.placeholder(tf.float32,name='keep_prob')

model=U_net(y,name='train')
model.u_net_model(x,cfg.w_inputs,cfg.h_inputs,cfg.batch_size,cfg.input_channels,cfg.n_classs,keep_prob)
model.loss(y)

b=tf.constant(1,tf.float32)
y_pred=tf.multiply(b,model.y_pred,name='logits') #乘上b，主要是为后面测试用

train_op=tf.train.AdamOptimizer(cfg.learn_rate).minimize(model.cost)
saver=tf.train.Saver(tf.global_variables())
sess=tf.Session(config=config)

ckpt=tf.train.get_checkpoint_state(cfg.model_path)
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess,ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

merged=tf.summary.merge_all()

train_writer=tf.summary.FileWriter(cfg.train_summary_path,sess.graph)
test_writer=tf.summary.FileWriter(cfg.test_summary_path)
#save=tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=2)


data_list=get_list(cfg.train_list_path)
for epoch in range(cfg.init_epoch+1,cfg.train_iter+1):
    if epoch%2 !=0:
        random.shuffle(data_list) #每迭代一次，就打乱数据一次
        #训练
        for data in batch_namequeue(data_list,cfg.batch_size):
            label=[];image=[]
            for i in data:
                image_path,label_path=i.strip('\n').split(',')  #之前生成data_list时在每行后面加上了换行符\n
                label.append(read_one_picture(label_path,gray=True,resize_val=255.0))
                image.append(read_one_picture(image_path,gray=True,dtype=np.float32))            
            _,lo,ac=sess.run([train_op,model.cost,model.accuracy],feed_dict={x:image,y:label,keep_prob:0.5})          
        
        #记录数据
        if epoch%cfg.summary_n==0:
            train_summary=sess.run(merged,feed_dict={x:image,y:label,keep_prob:1})
            train_writer.add_summary(train_summary,epoch)
            print('{}--loss:{}--accuracy:{}'.format(epoch,lo,ac))
            
        if epoch==1:
            saver.save(sess,cfg.model_path)
        else:
            saver.save(sess,cfg.model_path,write_meta_graph=False) 
            
    else:   
        label=[];image=[]
        test_data_list=get_list(cfg.test_list_path)
        o=0
        for i in test_data_list:
            if o< cfg.batch_size:
                image_path,label_path=i.strip('\n').split(',')  
                label.append(read_one_picture(label_path,gray=True,resize_val=255.0))
                image.append(read_one_picture(image_path,gray=True,dtype=np.float32))
                o+=1
        _,test_c=sess.run([y_pred,model.cost],feed_dict={x:image,y:label,keep_prob:1}) 
        print(test_c)
        test_summary=sess.run(merged,feed_dict={x:image,y:label,keep_prob:1})
        test_writer.add_summary(test_summary,epoch-1)
        
train_writer.flush()
test_writer.flush()
train_writer.close()
test_writer.close()
sess.close()