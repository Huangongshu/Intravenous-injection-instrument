# -*- coding: utf-8 -*-
list_path='D:/python/model/u_net/data_list.txt'
batch_size=20
n_classs=1

init_epoch=1
num_epoch=20000
train_iter=init_epoch+num_epoch

summary_n=5000
save_n=5000
learn_rate=0.01

input_channels=1
w_inputs=240
h_inputs=200

model_path='D:/python/model/u_net/model/model/'
summary_path='D:/python/model/u_net/summary/'