# -*- coding: utf-8 -*-

train_list_path='D:/python/model/u_net/train_data_list.txt'
test_list_path='D:/python/model/u_net/test_data_list.txt'

batch_size=20
n_classs=1

init_epoch=0
num_epoch=20000
train_iter=init_epoch+num_epoch

summary_n=1000
save_n=1000
learn_rate=0.01

input_channels=1
w_inputs=240
h_inputs=200

model_path='D:/python/model/u_net/model/'
train_summary_path='D:/python/model/u_net/summary/train_summary/'
test_summary_path='D:/python/model/u_net/summary/test_summary/'