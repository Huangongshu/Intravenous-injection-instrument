# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 20:32:46 2018

@author: huan
"""
import tensorflow as tf
import numpy as np
from utils import read_picture
from skimage import io

config=tf.ConfigProto()
config.gpu_options.allow_growth=True

path='C:/Users/huan/Desktop/u_net_train_pic/test/'
model_path='C:/Users/huan/Desktop/shop/model/U-net/.meta'
parameter_path='C:/Users/huan/Desktop/shop/model/U-net/'

test=read_picture(path,gray=True,dtype=np.float32) 

with tf.Session(config=config) as sess:
    save=tf.train.import_meta_graph('D:/python/model/u_net/model/.meta')
    save.restore(sess,tf.train.latest_checkpoint('D:/python/model/u_net/model/'))
    graph=tf.get_default_graph()
    x=graph.get_tensor_by_name("x:0")
    y_p=graph.get_tensor_by_name("logits:0")
    keep_prob=graph.get_tensor_by_name('keep_prob:0')
    prediction=sess.run(y_p,feed_dict={x:test,keep_prob:1})
    for i in range(len(prediction)):        
        im=prediction[i,:,:,:]
        im=np.squeeze(im)
        io.imshow(im)
        io.imsave('D:/python/model/u_net/test_result_pic/'+str(i)+'.png',im)
