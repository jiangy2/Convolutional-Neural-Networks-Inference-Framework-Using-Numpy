# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 20:55:01 2019

@author: admin
"""
import torch
import torch.nn.functional as F
import time

def maxout(x,max_out):

    out_shape = (1,int(x.shape[1] / max_out))
    y = torch.zeros(out_shape)
    y,indice = torch.max(x.view(-1, 2), 1)
    return y.view(1,-1)

img = torch.randn((1, 3, 220, 220),dtype = torch.float32)
#conv1
conv1_start = time.clock()
conv1_filter = torch.rand((64,3,7,7),dtype = torch.float32)
conv1_oput = F.conv2d(img, conv1_filter, stride=2, padding=3)

relu1_start = time.clock()
conv1_oput_relu = F.relu(conv1_oput)

pool1_start = time.clock()
conv1_oput_pool = F.max_pool2d(conv1_oput_relu, kernel_size=3, stride=2, padding=1)

#conv2a
conv2a_start=time.clock()
conv2a_filter = torch.rand((64,64,1,1),dtype = torch.float32)
conv2a_oput = F.conv2d(conv1_oput_pool, conv2a_filter, stride=1, padding=0)

relu2a_start = time.clock()
conv2a_oput_relu = F.relu(conv2a_oput)

#conv2
conv2_start = time.clock()
conv2_filter = torch.rand((192,64,3,3),dtype = torch.float32)
conv2_oput = F.conv2d(conv2a_oput_relu, conv2_filter, stride=1, padding=1)

relu2_start = time.clock()
conv2_oput_relu = F.relu(conv2_oput)

pool2_start = time.clock()
conv2_oput_pool = F.max_pool2d(conv2_oput_relu, kernel_size=3, stride=2, padding=1)


#conv3a
conv3a_start = time.clock()
conv3a_filter = torch.rand((192,192,1,1),dtype = torch.float32)
conv3a_oput = F.conv2d(conv2_oput_pool, conv3a_filter, stride=1, padding=0)

relu3a_start = time.clock()
conv3a_oput_relu = F.relu(conv3a_oput)

#conv3
conv3_start = time.clock()
conv3_filter = torch.rand((384,192,3,3),dtype = torch.float32)
conv3_oput = F.conv2d(conv3a_oput_relu, conv3_filter, stride=1, padding=1)

#relu3
relu3_start = time.clock()
conv3_oput_relu = F.relu(conv3_oput)

#pool3
pool3_start = time.clock()
conv3_oput_pool = F.max_pool2d(conv3_oput_relu, kernel_size=3, stride=2, padding=1)

#conv4a
conv4a_start = time.clock()
conv4a_filter = torch.rand((384,384,1,1),dtype = torch.float32)
conv4a_oput = F.conv2d(conv3_oput_pool, conv4a_filter, stride=1, padding=0)
#relu4
relu4a_start = time.clock()
conv4a_oput_relu = F.relu(conv4a_oput)

#conv4
conv4_start = time.clock()
conv4_filter = torch.rand((256,384,3,3),dtype = torch.float32)
conv4_oput = F.conv2d(conv4a_oput_relu, conv4_filter, stride=1, padding=1)

relu4_start = time.clock()
conv4_oput_relu = F.relu(conv4_oput)

#conv5a
conv5a_start = time.clock()
conv5a_filter = torch.rand((256,256,1,1),dtype = torch.float32)
conv5a_oput = F.conv2d(conv4_oput_relu, conv5a_filter, stride=1, padding=0)

relu5a_start = time.clock()
conv5a_oput_relu = F.relu(conv5a_oput)

#conv5
conv5_start = time.clock()
conv5_filter = torch.rand((256,256,3,3),dtype = torch.float32)
conv5_oput = F.conv2d(conv5a_oput_relu, conv5_filter, stride=1, padding=1)

relu5_start = time.clock()
conv5_oput_relu = F.relu(conv5_oput)

#conv6a
conv6a_start = time.clock()
conv6a_filter = torch.rand((256,256,1,1),dtype = torch.float32)
conv6a_oput = F.conv2d(conv5_oput_relu, conv6a_filter, stride=1, padding=0)

relu6a_start = time.clock()
conv6a_oput_relu = F.relu(conv6a_oput)

#conv6
conv6_start = time.clock()
conv6_filter = torch.rand((256,256,3,3),dtype = torch.float32)
conv6_oput = F.conv2d(conv6a_oput_relu, conv6_filter, stride=1, padding=1)

relu6_start = time.clock()
conv6_oput_relu = F.relu(conv6_oput)

pool4_start = time.clock()
conv6_oput_pool = F.max_pool2d(conv6_oput_relu, kernel_size=3, stride=2, padding=1)

conv_end = time.clock()

concat = conv6_oput_pool.view(-1, 7*7*256)


maxout_cof=2

time_fc1_start = time.clock()

fc1_filter = torch.rand((maxout_cof*1*32*128, 7*7*256),dtype = torch.float32)
fc1_oput = F.linear(concat, fc1_filter)
fc1_oput_maxout = maxout(fc1_oput, maxout_cof)

time_fc2_start = time.clock()

fc2_filter = torch.rand((maxout_cof*1*32*128, 1*32*128),dtype = torch.float32)
fc2_oput = F.linear(fc1_oput_maxout, fc2_filter)
fc2_oput_maxout = maxout(fc2_oput, maxout_cof)

time_fc3_start = time.clock()

fc3_filter = torch.rand((1*1*128, 1*32*128),dtype = torch.float32)
fc3_oput = F.linear(fc2_oput_maxout, fc3_filter)

time_fc_end = time.clock()

print(relu1_start-conv1_start)
print(pool1_start-relu1_start)
print(conv2a_start-pool1_start)
print(relu2a_start-conv2a_start)
print(conv2_start-relu2a_start)
print(relu2_start-conv2_start)
print(pool2_start-relu2_start)
print(conv3a_start-pool2_start)
print(relu3a_start-conv3a_start)
print(conv3_start-relu3a_start)
print(relu3_start-conv3_start)
print(pool3_start-relu3_start)
print(conv4a_start-pool3_start)
print(relu4a_start-conv4a_start)
print(conv4_start-relu4a_start)
print(relu4_start-conv4_start)
print(conv5a_start-relu4_start)
print(relu5a_start-conv5a_start)
print(conv5_start-relu5a_start)
print(relu5_start-conv5_start)
print(conv6a_start-relu5_start)
print(relu6a_start-conv6a_start)
print(conv6_start-relu6a_start)
print(relu6_start-conv6_start)
print(pool4_start-relu6_start)
print(conv_end-pool4_start)
#print('all convs',conv_end-conv1_start)
print(time_fc2_start-time_fc1_start)
print(time_fc3_start-time_fc2_start)
print(time_fc_end-time_fc3_start)
#print('all fc', time_fc_end-time_fc1_start)
 
#print('conv1', relu1_start-conv1_start)
#print('relu1', pool1_start-relu1_start)
#print('pool1', conv2a_start-pool1_start)
#print('conv2a', relu2a_start-conv2a_start)
#print('relu2a', conv2_start-relu2a_start)
#print('conv2', relu2_start-conv2_start)
#print('relu2', pool2_start-relu2_start)
#print('pool2', conv3a_start-pool2_start)
#print('conv3a',relu3a_start-conv3a_start)
#print('relu3a',conv3_start-relu3a_start)
#print('conv3',relu3_start-conv3_start)
#print('relu3',pool3_start-relu3_start)
#print('pool3',conv4a_start-pool3_start)
#print('conv4a',relu4a_start-conv4a_start)
#print('relu4a',conv4_start-relu4a_start)
#print('conv4',relu4_start-conv4_start)
#print('relu4',conv5a_start-relu4_start)
#print('conv5a',relu5a_start-conv5a_start)
#print('relu5a',conv5_start-relu5a_start)
#print('conv5',relu5_start-conv5_start)
#print('relu5',conv6a_start-relu5_start)
#print('conv6a',relu6a_start-conv6a_start)
#print('relu6a',conv6_start-relu6a_start)
#print('conv6',relu6_start-conv6_start)
#print('relu6',pool4_start-relu6_start)
#print('pool4',conv_end-pool4_start)
##print('all convs',conv_end-conv1_start)
#print('fc1', time_fc2_start-time_fc1_start)
#print('fc2', time_fc3_start-time_fc2_start)
#print('fc3', time_fc_end-time_fc3_start)
##print('all fc', time_fc_end-time_fc1_start)

#print('conv1',conv1_oput.shape)
#print('conv1_pool',conv1_oput_pool.shape)
#
#print('conv2a',conv2a_oput.shape)
#print('conv2',conv2_oput.shape)
#print('conv2_pool',conv2_oput_pool.shape)
#
#print('conv3a',conv3a_oput.shape)
#print('conv3',conv3_oput.shape)
#print('conv3_pool',conv3_oput_pool.shape)
#
#print('conv4a',conv4a_oput.shape)
#print('conv4',conv4_oput.shape)
#print('conv5a',conv5a_oput.shape)
#print('conv5',conv5_oput.shape)
#print('conv6a',conv6a_oput.shape)
#print('conv6',conv6_oput.shape)
#print('conv6_pool',conv6_oput_pool.shape)