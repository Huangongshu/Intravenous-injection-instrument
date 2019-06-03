# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:58:41 2019

@author: huan
"""
import tensorflow as tf
import numpy as np
from utils import read_picture,u_net_model

config=tf.ConfigProto()
config.gpu_options.allow_growth=True

w_inputs=240
h_inputs=200
name='u_net'
input_channels=1
output_channels=1
n_classs=1

path='C:/Users/huan/Desktop/u_net_train_pic/test1/'
test=read_picture(path,gray=True,dtype=np.float32) 
batch_size=len(test)
x=tf.placeholder(dtype=tf.float32,shape=[None,w_inputs,h_inputs,input_channels],name='x')
keep_prob=tf.placeholder(tf.float32,name='keep_prob')

y_pred=u_net_model(x,w_inputs,h_inputs,batch_size,input_channels,n_classs,keep_prob,name='train')
sess=tf.Session(config=config)
saver=tf.train.Saver()
ckpt =tf.train.get_checkpoint_state('D:/python/model/u_net_finger/model/')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess,ckpt.model_checkpoint_path)

merged=tf.summary.merge_all()
writer=tf.summary.FileWriter('D:/python/model/u_net_finger/test_result/',sess.graph,filename_suffix=name)
y_p,summary=sess.run([y_pred,merged],feed_dict={x:test,keep_prob:1})
writer.add_summary(summary)
writer.flush()
writer.close()
sess.close()