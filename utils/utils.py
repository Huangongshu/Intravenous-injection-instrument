import tensorflow as tf
import numpy as np
import PIL as Image

def pixel_wise_cross_entropy(y_pred,y):#效果没有dice好
    flat_logit=tf.reshape(y_pred,[-1])
    flat_label=tf.reshape(y,[-1])
    loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logit,labels=flat_label))
    return loss

def dice_coe(x,y,smooth=1e-6):
    inse=tf.reduce_sum(x*y,axis=(1,2,3))
    l=tf.reduce_sum(x*x,axis=(1,2,3))
    r=tf.reduce_sum(y*y,axis=(1,2,3))
    dice=(2.0*inse+smooth)/(l+r+smooth)
    dice=tf.reduce_mean(dice)
    return 1-dice

def IOU(x,y,smooth=1e-10):
    inse=tf.reduce_sum(x*y,axis=(1,2,3))
    l=tf.reduce_sum(x*x,axis=(1,2,3))
    r=tf.reduce_sum(y*y,axis=(1,2,3))
    j=(2.0*inse+smooth)/(l+r+smooth)
    I=tf.reduce_mean(j)
    return I

def batch_normalization(input,output_n,axis=[0],epsilon=0.001):
    '''
    output_n:the parameter is the number of  output channels
             of feature map
    '''
    mean,var=tf.nn.moments(input,axis)
    shift=tf.Variable(tf.zeros([output_n]),name='shift')
    scale=tf.Variable(tf.ones([output_n]),name='scale')
    input=tf.nn.batch_normalization(input,mean,var,shift,scale,epsilon)
    return input

def batch_data(x,y,batch_size):
    assert len(x)==len(y)
    for start in range(0,len(x)-batch_size+1,batch_size):
        a=slice(start,start+batch_size)
        yield x[a],y[a] 
        
def weighted_loss(logits, labels, num_classes, head=None):
    """ median-frequency re-weighting """
    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, num_classes))
        epsilon = tf.constant(value=1e-10)#是为了防止0出现，log0为无穷，肯定不可以让他出现
        logits = logits + epsilon#对预测值加上epsilon来防止0出现
        # consturct one-hot label array
        label_flat = tf.reshape(labels, (-1, 1))
        # should be [batch ,num_classes]
#        label_flat=tf.cast(tf.multiply(label_flat,255.0),tf.int32)
        labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))
        softmax = tf.nn.softmax(logits)
        if head.all==None:
            cross_entropy = -tf.reduce_sum(labels * tf.log(softmax + epsilon))
        else:
            cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon),head))          
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
#        tf.add_to_collection('losses', cross_entropy_mean)
#        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return cross_entropy_mean,labels        
        
def cal_loss(logits,labels,num_classes,weight=[]):
    loss_weight=np.array(weight) #考虑到大目标占的像素多，其交叉熵也较大，因此面积大的目标
    #设置权重小，而小物体的权重大一些
    return weighted_loss(logits,labels,num_classes,loss_weight)   

def writeImage(image, filename,label_colours,n_class,real_v):
    r = image.copy()
    g = image.copy()
    b = image.copy()
    for l in range(0,n_class):
        r[image==real_v[l]] = label_colours[l,0]
        g[image==real_v[l]] = label_colours[l,1]
        b[image==real_v[l]] = label_colours[l,2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:,:,0] = r/1.0
    rgb[:,:,1] = g/1.0
    rgb[:,:,2] = b/1.0
    im = Image.fromarray(np.uint8(rgb))
    im.save(filename)

def batch_namequeue(data,batch_size):
    for start in range(0,len(data)-batch_size+1,batch_size):
        s=slice(start,start+batch_size)
        yield data[s]
        
def get_list(path):
    f=open(path,'r')
    c=[]
    k=f.readlines()
    for i in range(len(k)):
        c.append(k[i])
    f.close()
    return c
    
#v=get_list('D:/python/model/u_net/data_list.txt')
#import random
#random.shuffle(v)