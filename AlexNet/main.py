# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 00:44:34 2019

@author: admin
"""

import numpy as np
import time
import cnn_func_utconv_tfpool as npcnn
#import cnn_func_utconv_torchpool as npcnn
import facenet_fc_navie as npfc


img = np.random.random((1,227,227,3)).astype('float32')

time_conv1_start=time.clock()
test_filter_1 = np.random.random((11,11,3,96)).astype('float32')
l1_feature_map = npcnn.conv(img, test_filter_1,pad=0,stride=(4,4))
time_relu1_start=time.clock()
l1_feature_map_relu = npcnn.relu(l1_feature_map)
time_pool1_start=time.clock()
l1_feature_map_pool = npcnn.pooling(l1_feature_map_relu,size=3,stride=2,padding=0)


time_conv2_start=time.clock()
test_filter_2 = np.random.random((5,5,96,256)).astype('float32')
l2_feature_map = npcnn.conv(l1_feature_map_pool, test_filter_2, pad='SAME', stride=(1,1))
time_relu2_start=time.clock()
l2_feature_map_relu = npcnn.relu(l2_feature_map)
time_pool2_start=time.clock()
l2_feature_map_pool = npcnn.pooling(l2_feature_map_relu,size=3,stride=2,padding=0)

time_conv3_start=time.clock()
test_filter_3 = np.random.random((3,3,256,384)).astype('float32')
l3_feature_map = npcnn.conv(l2_feature_map_pool, test_filter_3,pad='SAME',stride=(1,1))
time_relu3_start=time.clock()
l3_feature_map_relu = npcnn.relu(l3_feature_map)

time_conv4_start=time.clock()
test_filter_4 = np.random.random((3,3,384,384)).astype('float32')
l4_feature_map = npcnn.conv(l3_feature_map_relu, test_filter_4,pad='SAME',stride=(1,1))
time_relu4_start=time.clock()
l4_feature_map_relu = npcnn.relu(l4_feature_map)

time_conv5_start=time.clock()
test_filter_5 = np.random.random((3,3,384,256)).astype('float32')
l5_feature_map = npcnn.conv(l4_feature_map_relu, test_filter_5,pad='SAME',stride=(1,1))
time_relu5_start=time.clock()
l5_feature_map_relu = npcnn.relu(l5_feature_map)
time_pool3_start=time.clock()
l5_feature_map_pool = npcnn.pooling(l5_feature_map_relu,size=3,stride=2,padding=0)
time_conv_end=time.clock()


Net_dim = [6*6*256, 4096, 4096, 1000]
a0 = np.random.random_sample(size=(1, Net_dim[0])).astype(np.float32)

time_fc1_start = time.clock()
z1 = npfc.forward_step_alex(a0, Net_dim[1],16)
time_fc1_relu_start = time.clock()
a1 = npcnn.relu(z1)

time_fc2_start = time.clock()
z2 = npfc.forward_step_alex(a1, Net_dim[2],16)
time_fc2_relu_start = time.clock()
a2 = npcnn.relu(z2)

time_fc3_start = time.clock()
z3 = npfc.forward_step_alex(a2, Net_dim[3],16)
time_fc_end = time.clock()


#print('conv1',time_relu1_start-time_conv1_start)
#print('relu1',time_pool1_start-time_relu1_start)
#print('pool1',time_conv2_start-time_pool1_start)
#print('conv2',time_relu2_start-time_conv2_start)
#print('relu2',time_pool2_start-time_relu2_start)
#print('pool2',time_conv3_start-time_pool2_start)
#print('conv3',time_relu3_start-time_conv3_start)
#print('relu3',time_conv4_start-time_relu3_start)
#print('conv4',time_relu4_start-time_conv4_start)
#print('relu4',time_conv5_start-time_relu4_start)
#print('conv5',time_relu5_start-time_conv5_start)
#print('relu5',time_pool3_start-time_relu5_start)
#print('pool3',time_conv_end-time_pool3_start)
#print('fc1 linear',time_fc1_relu_start-time_fc1_start)
#print('fc1 relu',time_fc2_start-time_fc1_relu_start)
#print('fc2 linear',time_fc2_relu_start-time_fc2_start)
#print('fc2 relu',time_fc3_start-time_fc2_relu_start)
#print('fc3 linear',time_fc_end-time_fc3_start)

print(time_relu1_start-time_conv1_start)
print(time_pool1_start-time_relu1_start)
print(time_conv2_start-time_pool1_start)
print(time_relu2_start-time_conv2_start)
print(time_pool2_start-time_relu2_start)
print(time_conv3_start-time_pool2_start)
print(time_relu3_start-time_conv3_start)
print(time_conv4_start-time_relu3_start)
print(time_relu4_start-time_conv4_start)
print(time_conv5_start-time_relu4_start)
print(time_relu5_start-time_conv5_start)
print(time_pool3_start-time_relu5_start)
print(time_conv_end-time_pool3_start)
print(time_fc1_relu_start-time_fc1_start)
print(time_fc2_start-time_fc1_relu_start)
print(time_fc2_relu_start-time_fc2_start)
print(time_fc3_start-time_fc2_relu_start)
print(time_fc_end-time_fc3_start)