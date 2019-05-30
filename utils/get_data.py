# -*- coding: utf-8 -*-
"""
Created on Sun Dec 9 16:50:30 2018

@author: huan
"""
import tensorflow as tf;from glob import glob
import numpy as np;import os
from skimage import io,color,transform,img_as_float 
import cv2 
#from PIL import Image

config=tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
#config=tf.ConfigProto(intra_op_parallelism_threads=0)
#config.gpu_options.per_process_gpu_memory_fraction=0.9

def read_picture(path,dtype=None,gray=False,resize=False,shape=None,resize_val=None):
    cate=[path+x for x in os.listdir(path) if os.path.isfile(path+x)]
    data=[]
    for im in cate:
        im=io.imread(im,as_gray=gray)
#        im = img_as_float(im)  #变为浮点型[0-1]。
#        im= (im - im.min()) * (1 / (im.max() - im.min()))  #比例缩放的归一化
        if resize==True:
            im=transform.resize(im,shape)
        if resize_val !=None:
            i=im.copy()
            i[im>0]=1.0
            im=i
        im=np.asarray(im,dtype=dtype)
        im=np.atleast_3d(im)
        data.append(im)
    return data

def get_picture_data_or_label(path,h,w):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    images=[];labels=[]
    with tf.Session(config=config) as sess:
        with tf.name_scope('get_data_label_layer'):
            for index,element in enumerate(cate):
                for i in glob(element+'/*.jpg'):
                    im=tf.gfile.GFile(i,'rb').read()
                    im=tf.image.decode_jpeg(im)
                    im=tf.image.rgb_to_grayscale(im)
                    img=tf.image.resize_images(im,[h,w],method=0)
                    img=np.asarray(img.eval(),np.float32)
                    images.append(img)
                    labels.append(index)
        return images

def get_picture_data_tif(path,w,h,dtype=None,big_number=True):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    images=[]
    with tf.Session(config=config) as sess:
        with tf.name_scope('get_data_layer'):
            for index,element in enumerate(cate):
                for i in glob(element+'/*.tif'):
                    im=io.imread(i)
                    im=np.asarray(im)
                    im=np.atleast_3d(im)
                    im1=tf.image.resize_images(im,[w,h],method=0)
                    if big_number==True:    
                        im1=np.asarray(im1.eval(),dtype)                        
                    else:
                        im1=np.asarray((im1/255).eval(),dtype)
                    images.append(im1)
    return images

def get_picture_data_png(path,w,h,dtype=None,gray=False,big_number=True):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    images=[]
    with tf.Session(config=config) as sess:
        with tf.name_scope('get_data_layer'):
            for index,element in enumerate(cate):
                for i in glob(element+'/*.png'):
                    im=io.imread(i)
                    im=np.asarray(im)
                    im=np.atleast_3d(im)
                    im=tf.image.resize_images(im,[w,h],method=0)
                    if gray==True:
#                        im=tf.image.rgb_to_grayscale(im)
                        im=color.rgb2gray(im)
                    if big_number==True:    
                        im=np.asarray(im.eval(),dtype)                        
                    else:
                        im=np.asarray((im/255).eval(),dtype)
                    images.append(im)
    return images

def get_picture_data_no_resize(path,dtype=None):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    images=[]
    with tf.Session(config=config) as sess:
        with tf.name_scope('get_data_layer'):
            for index,element in enumerate(cate):
                for i in glob(element+'/*.tif'):
                    im=io.imread(i)
                    im=np.asarray(im,dtype)
                    images.append(im)
    return images

def get_label_data_png(path):
    cate=[path+x for x in os.listdir(path) if os.path.isfile(path+x)]
    labels=[]
    for i in cate:
        im=cv2.imread(i)
        im=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
        labels.append(im)
    return labels

def read_one_picture(path,dtype=None,gray=False,resize=False,shape=None,resize_val=None):
    im=io.imread(path,as_gray=gray)
    if resize==True:
        im=transform.resize(im,shape)
    if resize_val !=None:
        i=im.copy()
        i[im>0]=1.0
        im=i
    im=np.asarray(im,dtype=dtype)
    im=np.atleast_3d(im)
    return im

def main():
    path1='D:/data_set/data_set/train/'
    train=get_picture_data_no_resize(path1,512,512,dtype=np.int32)
    path='D:/data_set/segment_picture/label/'
    path1='D:/data_set/segment_picture/train/train/'
    label=read_picture(path,512,512,gray=True)
    train=read_picture(path1,512,512)

if __name__=='__main__':
    main()

#path2='D:/data_set/segment_picture/label/label/'
#label=read_picture(path2,gray=True,dtype=np.float32)
