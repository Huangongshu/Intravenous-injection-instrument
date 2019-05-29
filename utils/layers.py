# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 11:38:26 2018

@author: huan
"""
import tensorflow as tf
from utils import batch_normalization

def u_net_model(image,w_inputs,h_inputs,batch_size,input_channels=3,n_classs=1,keep_prob=0.5,base_channel=64,name=None):
    
    with tf.variable_scope('u-net'): 
        
        image=batch_normalization(image,input_channels,axis=[0,1,2])
        tf.summary.image(name+'/image',image,batch_size) 
        
        #layer1
        with tf.variable_scope('conv1_1'):
            conv1_1_weight=tf.get_variable('conv1_1_weigt',[3,3,input_channels,base_channel],
                                         initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv1_1_bias=tf.get_variable('conv1_1_bias',[base_channel], 
                                       initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv1_1_result=tf.nn.conv2d(image,conv1_1_weight,[1,1,1,1],padding='SAME',name='conv1_1')
            conv1_1_relu=tf.nn.relu(tf.add(conv1_1_result,conv1_1_bias),name='conv1_1_relu')
#            conv1_1_relu=tf.nn.dropout(conv1_1_relu,keep_prob)
            
        with tf.variable_scope('conv1_2'):            
            conv1_2_weight=tf.get_variable('conv1_2_weigt',[3,3,base_channel,base_channel],
                                         initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv1_2_bias=tf.get_variable('conv1_2_bias',[base_channel],
                                       initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv1_2_result=tf.nn.conv2d(conv1_1_relu,conv1_2_weight,[1,1,1,1],padding='SAME',name='conv1_2')
            conv1_2_relu=tf.nn.relu(tf.add(conv1_2_result,conv1_2_bias),name='conv1_2_relu')#h_w_inputs
#            conv1_2_relu=tf.nn.dropout(conv1_2_relu,keep_prob)
#            conv1_2_relu=batch_normalization(conv1_2_relu,base_channel,axis=[0,1,2])    
            
        with tf.variable_scope('pool1'):
            pool1=tf.nn.max_pool(conv1_2_relu,[1,2,2,1],[1,2,2,1],padding='SAME')#h_w_inputs/2
            pool1=tf.nn.dropout(pool1,keep_prob)
            
        #layer2
        with tf.variable_scope('conv2_1'):
            conv2_1_weight=tf.get_variable('conv2_1_weigt',[3,3,base_channel,base_channel*2],
                                         initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv2_1_bias=tf.get_variable('conv2_1_bias',[base_channel*2], 
                                       initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv2_1_result=tf.nn.conv2d(pool1,conv2_1_weight,[1,1,1,1],padding='SAME',name='conv2_1')
            conv2_1_relu=tf.nn.relu(tf.add(conv2_1_result,conv2_1_bias),name='conv2_1_relu')
#            conv2_1_relu=tf.nn.dropout(conv2_1_relu,keep_prob)
        
        with tf.variable_scope('conv2_2'):
            conv2_2_weight=tf.get_variable('conv2_2_weigt',[3,3,base_channel*2,base_channel*2],
                                         initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv2_2_bias=tf.get_variable('conv2_2_bias',[base_channel*2],
                                       initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv2_2_result=tf.nn.conv2d(conv2_1_relu,conv2_2_weight,[1,1,1,1],padding='SAME',name='conv2_2')
            conv2_2_relu=tf.nn.relu(tf.add(conv2_2_result,conv2_2_bias),name='conv2_2_relu')#h_w_inputs/2              
#            conv2_2_relu=tf.nn.dropout(conv2_2_re lu,keep_prob) 
            conv2_2_relu=batch_normalization(conv2_2_relu,base_channel*2,axis=[0,1,2])       
            
        with tf.variable_scope('pool2'):
            pool2=tf.nn.max_pool(conv2_2_relu,[1,2,2,1],[1,2,2,1],padding='SAME')#h_w_inputs/4
            pool2=tf.nn.dropout(pool2,keep_prob)    
            
        #layer3
        with tf.variable_scope('conv3_1'):
            conv3_1_weight=tf.get_variable('conv3_1_weigt',[3,3,base_channel*2,base_channel*4],
                                         initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv3_1_bias=tf.get_variable('conv3_1_bias',[base_channel*4], 
                                       initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv3_1_result=tf.nn.conv2d(pool2,conv3_1_weight,[1,1,1,1],padding='SAME',name='conv3_1')
            conv3_1_relu=tf.nn.relu(tf.add(conv3_1_result,conv3_1_bias),name='conv3_1_relu')
#            conv3_1_relu=tf.nn.dropout(conv3_1_relu,keep_prob)
       
        with tf.variable_scope('conv3_2'):
            conv3_2_weight=tf.get_variable('conv3_2_weigt',[3,3,base_channel*4,base_channel*4],
                                         initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv3_2_bias=tf.get_variable('conv3_2_bias',[base_channel*4],
                                       initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv3_2_result=tf.nn.conv2d(conv3_1_relu,conv3_2_weight,[1,1,1,1],padding='SAME',name='conv3_2')
            conv3_2_relu=tf.nn.relu(tf.add(conv3_2_result,conv3_2_bias),name='conv3_2_relu')#h_w_inputs/4  
#            conv3_2_relu=batch_normalization(conv3_2_relu,base_channel*4,axis=[0,1,2])    
            
        with tf.variable_scope('pool3'):
            pool3=tf.nn.max_pool(conv3_2_relu,[1,2,2,1],[1,2,2,1],padding='SAME')#h_w_inputs/8
            pool3=tf.nn.dropout(pool3,keep_prob)    

        #layer4    
        with tf.variable_scope('conv4_1'):
            conv4_1_weight=tf.get_variable('conv4_1_weight',[3,3,base_channel*4,base_channel*8],
                                         initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv4_1_bias=tf.get_variable('conv4_1_bias',[base_channel*8], 
                                       initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv4_1_result=tf.nn.conv2d(pool3,conv4_1_weight,[1,1,1,1],padding='SAME',name='conv4_1_result')
            conv4_1_relu=tf.nn.relu(tf.add(conv4_1_result,conv4_1_bias),name='conv4_1_relu')
#            conv4_1_relu=tf.nn.dropout(conv4_1_relu,keep_prob)
            
        with tf.variable_scope('conv4_2'):            
            conv4_2_weight=tf.get_variable('conv4_2_weight',[3,3,base_channel*8,base_channel*8],
                                         initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv4_2_bias=tf.get_variable('conv4_2_bias',[base_channel*8],
                                       initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv4_2_result=tf.nn.conv2d(conv4_1_relu,conv4_2_weight,[1,1,1,1],padding='SAME',name='conv4_2_result')
            conv4_2_relu=tf.nn.relu(tf.add(conv4_2_result,conv4_2_bias),name='conv4_2_relu')     
#            conv4_2_relu=tf.nn.dropout(conv4_2_relu,keep_prob)
            conv4_2_relu=batch_normalization(conv4_2_relu,base_channel*8,axis=[0,1,2])  
                 
        with tf.variable_scope('pool4'):
            pool4=tf.nn.max_pool(conv4_2_relu,[1,2,2,1],[1,2,2,1],padding='SAME')#h_w_inputs/16
            pool4=tf.nn.dropout(pool4,keep_prob)         
            
        #layer5        
        with tf.variable_scope('conv5_1'):
            conv5_1_weight=tf.get_variable('conv5_1_weigt',[3,3,base_channel*8,base_channel*16],
                                         initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv5_1_bias=tf.get_variable('conv5_1_bias',[base_channel*16],  
                                       initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv5_1_result=tf.nn.conv2d(pool4,conv5_1_weight,[1,1,1,1],padding='SAME',name='conv5_1')
            conv5_1_relu=tf.nn.relu(tf.add(conv5_1_result,conv5_1_bias),name='conv5_1_relu')
#            conv5_1_relu=tf.nn.dropout(conv5_1_relu,keep_prob)
        
        with tf.variable_scope('conv5_2'):
            conv5_2_weight=tf.get_variable('conv5_2_weigt',[3,3,base_channel*16,base_channel*16],
                                         initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv5_2_bias=tf.get_variable('conv5_2_bias',[base_channel*16],
                                       initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv5_2_result=tf.nn.conv2d(conv5_1_relu,conv5_2_weight,[1,1,1,1],padding='SAME',name='conv5_2')
            conv5_2_relu=tf.nn.relu(tf.add(conv5_2_result,conv5_2_bias),name='conv5_2_relu')         
#            conv5_2_relu=tf.nn.dropout(conv5_2_relu,keep_prob)
#            conv5_2_relu=batch_normalization(conv5_2_relu,base_channel*16,axis=[0,1,2])       
            
        with tf.variable_scope('unsample1'):            
            unsam1_weight=tf.get_variable('unsam1_weight',[2,2,base_channel*8,base_channel*16],
                                           initializer=tf.truncated_normal_initializer(stddev=1.0))
            unsam1_bias=tf.get_variable('unsam1_bias',[base_channel*8],
                                         initializer=tf.truncated_normal_initializer(stddev=1.0))
            unsam1_result=tf.nn.conv2d_transpose(conv5_2_relu,unsam1_weight,
                                                 [batch_size,w_inputs//8,h_inputs//8,base_channel*8],[1,2,2,1])
            unsam1_relu=tf.nn.relu(tf.add(unsam1_result,unsam1_bias))  #h_w_inputs/4
           #之前出过错误，因为只改一层的上采样的输出的batchsize导致一层没改，使得无法批量导入图片数据
            unsam1_relu=tf.nn.dropout(unsam1_relu,keep_prob)
        
            merged_layer1=tf.concat([unsam1_relu,conv4_2_relu],axis=-1)

        #layer6
        with tf.variable_scope('conv6_1'):
            conv6_1_weigt=tf.get_variable('conv6_1_weigt',[3,3,base_channel*16,base_channel*8],
                                         initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv6_1_bias=tf.get_variable('conv6_1_bias',[base_channel*8],
                                       initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv6_1_result=tf.nn.conv2d(merged_layer1,conv6_1_weigt,[1,1,1,1],padding='SAME',name='conv6_1_result')
            conv6_1_relu=tf.nn.relu(tf.add(conv6_1_result,conv6_1_bias),name='conv6_1_relu')
#            conv6_1_relu=tf.nn.dropout(conv6_1_relu,keep_prob)

        with tf.variable_scope('conv6_2'):
            conv6_2_weigt=tf.get_variable('conv6_2_weigt',[3,3,base_channel*8,base_channel*8],
                                         initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv6_2_bias=tf.get_variable('conv6_2_bias',[base_channel*8],
                                       initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv6_2_result=tf.nn.conv2d(conv6_1_relu,conv6_2_weigt,[1,1,1,1],padding='SAME',name='conv6_2_result')
            conv6_2_relu=tf.nn.relu(tf.add(conv6_2_result,conv6_2_bias),name='conv6_2_relu')                           
#            conv6_2_relu=tf.nn.dropout(conv6_2_relu,keep_prob) 
            conv6_2_relu=batch_normalization(conv6_2_relu,base_channel*8,axis=[0,1,2])   
                
        with tf.variable_scope('unsample2'):
            #由于上采样中扩大对应的是池化层的缩小，所以其设置应该和池化层相同，而不是前面的卷积层相同
            unsam2_weight=tf.get_variable('unsam2_weight',[2,2,base_channel*4,base_channel*8],
                                           initializer=tf.truncated_normal_initializer(stddev=1.0))
            #和池化层一样，核大小为2
            unsam2_bias=tf.get_variable('unsam2_bias',[base_channel*4],
                                         initializer=tf.truncated_normal_initializer(stddev=1.0))
            unsam2_result=tf.nn.conv2d_transpose(conv6_2_relu,unsam2_weight,
                                                 [batch_size,w_inputs//4,h_inputs//4,base_channel*4],[1,2,2,1])
            unsam2_relu=tf.nn.relu(tf.add(unsam2_result,unsam2_bias))  #h_w_inputs/2
            unsam2_relu=tf.nn.dropout(unsam2_relu,keep_prob)            
            
            merged_layer2=tf.concat([unsam2_relu,conv3_2_relu],axis=-1)

            #layer7
        with tf.variable_scope('conv7_1'):
            conv7_1_weight=tf.get_variable('conv7_1_weight',[3,3,base_channel*8,base_channel*4],
                                      initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv7_1_bias=tf.get_variable('conv7_1_bias',[base_channel*4],
                                    initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv7_1_result=tf.nn.conv2d(merged_layer2,conv7_1_weight,[1,1,1,1],padding='SAME',name='conv7_1_result')
            conv7_1_relu=tf.nn.relu(tf.add(conv7_1_result,conv7_1_bias),name='conv7_1_relu')
#            conv7_1_relu=tf.nn.dropout(conv7_1_relu,keep_prob) 
            
        with tf.variable_scope('conv7_2'):
            conv7_2_weight=tf.get_variable('conv7_2_weight',[3,3,base_channel*4,base_channel*4],
                                      initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv7_2_bias=tf.get_variable('conv7_2_bias',[base_channel*4],
                                    initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv7_2_result=tf.nn.conv2d(conv7_1_relu,conv7_2_weight,[1,1,1,1],padding='SAME',name='conv7_2_result')
            conv7_2_relu=tf.nn.relu(tf.add(conv7_2_result,conv7_2_bias),name='conv7_2_relu')                      
#            conv7_2_relu=tf.nn.dropout(conv7_2_relu,keep_prob) 
#            conv7_2_relu=batch_normalization(conv7_2_relu,base_channel*4,axis=[0,1,2])       
            
        with tf.variable_scope('unsample3'):            
            unsam3_weight=tf.get_variable('unsam3_weight',[2,2,base_channel*2,base_channel*4],
                                          initializer=tf.truncated_normal_initializer(stddev=1.0))
            unsam3_bias=tf.get_variable('unsam3_bias',[base_channel*2],
                                        initializer=tf.truncated_normal_initializer(stddev=1.0))
            unsam3_result=tf.nn.conv2d_transpose(conv7_2_relu,unsam3_weight,
                                                 [batch_size,w_inputs//2,h_inputs//2,base_channel*2],[1,2,2,1])
            unsam3_relu=tf.nn.relu(tf.add(unsam3_result,unsam3_bias),name='unsam3_relu')
            unsam3_relu=tf.nn.dropout(unsam3_relu,keep_prob)            
            
            merged_layer3=tf.concat([unsam3_relu,conv2_2_relu],axis=-1) 
            
            #layer8
        with tf.variable_scope('conv8_1'):
            conv8_1_weight=tf.get_variable('conv8_1_weight',[3,3,base_channel*4,base_channel*2],
                                      initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv8_1_bias=tf.get_variable('conv8_1_bias',[base_channel*2],
                                    initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv8_1_result=tf.nn.conv2d(merged_layer3,conv8_1_weight,[1,1,1,1],padding='SAME',name='conv8_1_result')
            conv8_1_relu=tf.nn.relu(tf.add(conv8_1_result,conv8_1_bias),name='conv8_1_relu')
#            conv8_1_relu=tf.nn.dropout(conv8_1_relu,keep_prob)    
            
        with tf.variable_scope('conv8_2'):
            conv8_2_weight=tf.get_variable('conv8_2_weight',[3,3,base_channel*2,base_channel*2],
                                      initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv8_2_bias=tf.get_variable('conv8_2_bias',[base_channel*2],
                                    initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv8_2_result=tf.nn.conv2d(conv8_1_relu,conv8_2_weight,[1,1,1,1],padding='SAME',name='conv8_2_result')
            conv8_2_relu=tf.nn.relu(tf.add(conv8_2_result,conv8_2_bias),name='conv8_2_relu')            
#            conv8_2_relu=tf.nn.dropout(conv8_2_relu,keep_prob)  
            conv8_2_relu=batch_normalization(conv8_2_relu,base_channel*2,axis=[0,1,2])       
            
        with tf.variable_scope('unsample4'):            
            unsam4_weight=tf.get_variable('unsam4_weight',[2,2,base_channel,base_channel*2],
                                          initializer=tf.truncated_normal_initializer(stddev=1.0))
            unsam4_bias=tf.get_variable('unsam4_bias',[base_channel],
                                        initializer=tf.truncated_normal_initializer(stddev=1.0))
            unsam4_result=tf.nn.conv2d_transpose(conv8_2_relu,unsam4_weight,
                                                 [batch_size,w_inputs,h_inputs,base_channel],[1,2,2,1])
            unsam4_relu=tf.nn.relu(tf.add(unsam4_result,unsam4_bias),name='unsam4_relu')
            unsam4_relu=tf.nn.dropout(unsam4_relu,keep_prob)            
            
            merged_layer4=tf.concat([unsam4_relu,conv1_2_relu],axis=-1) 

            #layer9
        with tf.variable_scope('conv9_1'):
            conv9_1_weight=tf.get_variable('conv9_1_weight',[3,3,base_channel*2,base_channel],
                                      initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv9_1_bias=tf.get_variable('conv9_1_bias',[base_channel],
                                    initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv9_1_result=tf.nn.conv2d(merged_layer4,conv9_1_weight,[1,1,1,1],padding='SAME',name='conv9_1_result')
            conv9_1_relu=tf.nn.relu(tf.add(conv9_1_result,conv9_1_bias),name='conv9_1_relu')
#            conv9_1_relu=tf.nn.dropout(conv9_1_relu,keep_prob) 
            
        with tf.variable_scope('conv9_2'):
            conv9_2_weight=tf.get_variable('conv9_2_weight',[3,3,base_channel,base_channel],
                                      initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv9_2_bias=tf.get_variable('conv9_2_bias',[base_channel],
                                    initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv9_2_result=tf.nn.conv2d(conv9_1_relu,conv9_2_weight,[1,1,1,1],padding='SAME',name='conv9_2_result')
            conv9_2_relu=tf.nn.relu(tf.add(conv9_2_result,conv9_2_bias),name='conv9_2_relu') 
#            conv9_2_relu=tf.nn.dropout(conv9_2_relu,keep_prob)  
            conv9_2_relu=batch_normalization(conv9_2_relu,base_channel,axis=[0,1,2])  
#            print(conv9_2_relu.shape)
            
        with tf.variable_scope('conv9_3'):
            conv9_3_weight=tf.get_variable('conv9_3_weight',[3,3,base_channel,n_classs],
                                      initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv9_3_bias=tf.get_variable('conv9_3_bias',[n_classs],
                                    initializer=tf.truncated_normal_initializer(stddev=1.0))
            conv9_3_result=tf.nn.conv2d(conv9_2_relu,conv9_3_weight,[1,1,1,1],padding='SAME',name='conv9_3_result')
            conv9_3_add=tf.add(conv9_3_result,conv9_3_bias,name='conv9_3_add')
            conv9_3_sigmoid=tf.nn.sigmoid(conv9_3_add,name='conv9_3_sigmoid')  #不能用relu只能用sigmoid函数           
            #由于层数太多，不加BN，loss下降不了
            #观察效果
            image2=tf.reshape(conv9_3_sigmoid,[-1,w_inputs,h_inputs,n_classs])
            tf.summary.image(name+'/image1',image2,batch_size)     
            
    return conv9_3_sigmoid        