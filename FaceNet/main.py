# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 00:02:57 2019

@author: admin
"""
import numpy as np
import time
import cnn_func_mod_tfpool as npcnn
import facenet_fc_navie as npfc

#np.random.seed(0)

img = np.random.random((1,220,220,3)).astype('float32')

#conv1
time_conv1_start=time.clock()
test_filter_1 = np.random.random((7,7,3,64)).astype('float32')
l1_feature_map = npcnn.conv(img, test_filter_1,pad='SAME',stride=(2,2))
time_relu1_start=time.clock()
l1_feature_map_relu = npcnn.relu(l1_feature_map)
time_pool1_start=time.clock()
l1_feature_map_pool = npcnn.pooling(l1_feature_map_relu,size=3,stride=2,padding='SAME')

#conv2a
time_conv2a_start=time.clock()
test_filter_2a = np.random.random((1,1,64,64)).astype('float32')
l2a_feature_map = npcnn.conv(l1_feature_map_pool, test_filter_2a, pad='SAME', stride=(1,1))
time_relu2a_start=time.clock()
l2a_feature_map_relu = npcnn.relu(l2a_feature_map)

#conv2
time_conv2_start=time.clock()
test_filter_2 = np.random.random((3,3,64,192)).astype('float32')
l2_feature_map = npcnn.conv(l2a_feature_map_relu, test_filter_2, pad='SAME', stride=(1,1))
time_relu2_start=time.clock()
l2_feature_map_relu = npcnn.relu(l2_feature_map)
time_pool2_start=time.clock()
l2_feature_map_pool = npcnn.pooling(l2_feature_map_relu,size=3,stride=2,padding='SAME')

#conv3a
time_conv3a_start=time.clock()
test_filter_3a = np.random.random((1,1,192,192)).astype('float32')
l3a_feature_map = npcnn.conv(l2_feature_map_pool, test_filter_3a, pad='SAME',stride=(1,1))
time_relu3a_start=time.clock()
l3a_feature_map_relu = npcnn.relu(l3a_feature_map)

#conv3
time_conv3_start=time.clock()
test_filter_3 = np.random.random((3,3,192,384)).astype('float32')
l3_feature_map = npcnn.conv(l3a_feature_map_relu, test_filter_3,pad='SAME',stride=(1,1))
time_relu3_start=time.clock()
l3_feature_map_relu = npcnn.relu(l3_feature_map)
time_pool3_start=time.clock()
l3_feature_map_pool = npcnn.pooling(l3_feature_map_relu,size=3,stride=2,padding='SAME')

#conv4a
time_conv4a_start=time.clock()
test_filter_4a = np.random.random((1,1,384,384)).astype('float32')
l4a_feature_map = npcnn.conv(l3_feature_map_pool, test_filter_4a,pad='SAME',stride=(1,1))
time_relu4a_start=time.clock()
l4a_feature_map_relu = npcnn.relu(l4a_feature_map)

#conv4
time_conv4_start=time.clock()
test_filter_4 = np.random.random((3,3,384,256)).astype('float32')
l4_feature_map = npcnn.conv(l4a_feature_map_relu, test_filter_4,pad='SAME',stride=(1,1))
time_relu4_start=time.clock()
l4_feature_map_relu = npcnn.relu(l4_feature_map)


#conv5a
time_conv5a_start=time.clock()
test_filter_5a = np.random.random((1,1,256,256)).astype('float32')
l5a_feature_map = npcnn.conv(l4_feature_map_relu, test_filter_5a,pad='SAME',stride=(1,1))
time_relu5a_start=time.clock()
l5a_feature_map_relu = npcnn.relu(l5a_feature_map)

#conv5
time_conv5_start=time.clock()
test_filter_5 = np.random.random((3,3,256,256)).astype('float32')
l5_feature_map = npcnn.conv(l5a_feature_map_relu, test_filter_5,pad='SAME',stride=(1,1))
time_relu5_start=time.clock()
l5_feature_map_relu = npcnn.relu(l5_feature_map)


#conv6a
time_conv6a_start=time.clock()
test_filter_6a = np.random.random((1,1,256,256)).astype('float32')
l6a_feature_map = npcnn.conv(l5_feature_map_relu, test_filter_6a,pad='SAME',stride=(1,1))
time_relu6a_start=time.clock()
l6a_feature_map_relu = npcnn.relu(l6a_feature_map)

#conv6
time_conv6_start=time.clock()
test_filter_6 = np.random.random((3,3,256,256)).astype('float32')
l6_feature_map = npcnn.conv(l6a_feature_map_relu, test_filter_6,pad='SAME',stride=(1,1))
time_relu6_start=time.clock()
l6_feature_map_relu = npcnn.relu(l6_feature_map)
time_pool4_start=time.clock()
l6_feature_map_pool = npcnn.pooling(l6_feature_map_relu,size=3,stride=2,padding='SAME')
time_conv_end=time.clock()

Net_dim = [7*7*256, 1*32*128, 1*32*128, 1*1*128]#original
maxout_layer = [1,2]
k=[2,2]
    
#X = np.random.randn(1, Net_dim[0])
a0 = l6_feature_map_pool.reshape(-1, Net_dim[0])

#a0 = np.random.random_sample(size=(1, Net_dim[0])).astype(np.float32)#12544
time_fc1_start = time.clock()
z1 = npfc.forward_step(a0, Net_dim[1],64)
a1 = npfc.maxout(z1,2)
time_fc2_start = time.clock()
z2 = npfc.forward_step(a1,Net_dim[2],64)
a2 = npfc.maxout(z2,2)
time_fc3_start = time.clock()
z3 = npfc.forward(a2,Net_dim[-1])

time_fc_end = time.clock()


print(time_relu1_start-time_conv1_start)
print(time_pool1_start-time_relu1_start)
print(time_conv2a_start-time_pool1_start)
print(time_relu2a_start-time_conv2a_start)
print(time_conv2_start-time_relu2a_start)
print(time_relu2_start-time_conv2_start)
print(time_pool2_start-time_relu2_start)
print(time_conv3a_start-time_pool2_start)
print(time_relu3a_start-time_conv3a_start)
print(time_conv3_start-time_relu3a_start)
print(time_relu3_start-time_conv3_start)
print(time_pool3_start-time_relu3_start)
print(time_conv4a_start-time_pool3_start)
print(time_relu4a_start-time_conv4a_start)
print(time_conv4_start-time_relu4a_start)
print(time_relu4_start-time_conv4_start)
print(time_conv5a_start-time_relu4_start)
print(time_relu5a_start-time_conv5a_start)
print(time_conv5_start-time_relu5a_start)
print(time_relu5_start-time_conv5_start)
print(time_conv6a_start-time_relu5_start)
print(time_relu6a_start-time_conv6a_start)
print(time_conv6_start-time_relu6a_start)
print(time_relu6_start-time_conv6_start)
print(time_pool4_start-time_relu6_start)
print(time_conv_end-time_pool4_start)
print(time_fc2_start-time_fc1_start)
print(time_fc3_start-time_fc2_start)
print(time_fc_end-time_fc3_start)


#print('conv1',time_relu1_start-time_conv1_start)
#print('relu1',time_pool1_start-time_relu1_start)
#print('pool1',time_conv2a_start-time_pool1_start)
#print('conv2a',time_relu2a_start-time_conv2a_start)
#print('relu2a',time_conv2_start-time_relu2a_start)
#print('conv2',time_relu2_start-time_conv2_start)
#print('relu2',time_pool2_start-time_relu2_start)
#print('pool2',time_conv3a_start-time_pool2_start)
#print('conv3a',time_relu3a_start-time_conv3a_start)
#print('relu3a',time_conv3_start-time_relu3a_start)
#print('conv3',time_relu3_start-time_conv3_start)
#print('relu3',time_pool3_start-time_relu3_start)
#print('pool3',time_conv4a_start-time_pool3_start)
#print('conv4a',time_relu4a_start-time_conv4a_start)
#print('relu4a',time_conv4_start-time_relu4a_start)
#print('conv4',time_relu4_start-time_conv4_start)
#print('relu4',time_conv5a_start-time_relu4_start)
#print('conv5a',time_relu5a_start-time_conv5a_start)
#print('relu5a',time_conv5_start-time_relu5a_start)
#print('conv5',time_relu5_start-time_conv5_start)
#print('relu5',time_conv6a_start-time_relu5_start)
#print('conv6a',time_relu6a_start-time_conv6a_start)
#print('relu6a',time_conv6_start-time_relu6a_start)
#print('conv6',time_relu6_start-time_conv6_start)
#print('relu6',time_pool4_start-time_relu6_start)
#print('pool4',time_conv_end-time_pool4_start)
#print('fc1',time_fc2_start-time_fc1_start)
#print('fc2',time_fc3_start-time_fc2_start)
#print('fc3',time_fc_end-time_fc3_start)
#output size check

#print('time cost of conv1',time_conv1_end-time_conv1_start,'s')
#print('time cost of conv2a',time_conv2a_end-time_conv1_end,'s')
#print('time cost of conv2',time_conv2_end-time_conv2a_end,'s')
#print('time cost of conv3a',time_conv3a_end-time_conv2_end,'s')
#print('time cost of conv3',time_conv3_end-time_conv3a_end,'s')
#print('time cost of conv4a',time_conv4a_end-time_conv3_end,'s')
#print('time cost of conv4',time_conv4_end-time_conv4a_end,'s')
#print('time cost of conv5a',time_conv5a_end-time_conv4_end,'s')
#print('time cost of conv5',time_conv5_end-time_conv5a_end,'s')
#print('time cost of conv6a',time_conv6a_end-time_conv5_end,'s')
#print('time cost of conv6',time_conv6_end-time_conv6a_end,'s')
#print('time cost of conv layers',time_conv6_end-time_conv1_start,'s')
#print('time cost of fc1',time_fc2_start-time_fc1_start,'s')
#print('time cost of fc2',time_fc3_start-time_fc2_start,'s')
#print('time cost of fc3',time_fc_end-time_fc3_start,'s')
#print('time cost of full connected layers',time_fc_end-time_fc1_start,'s')


#print(time_conv1_end-time_conv1_start)
#print(time_conv2a_end-time_conv1_end)
#print(time_conv2_end-time_conv2a_end)
#print(time_conv3a_end-time_conv2_end)
#print(time_conv3_end-time_conv3a_end)
#print(time_conv4a_end-time_conv3_end)
#print(time_conv4_end-time_conv4a_end)
#print(time_conv5a_end-time_conv4_end)
#print(time_conv5_end-time_conv5a_end)
#print(time_conv6a_end-time_conv5_end)
#print(time_conv6_end-time_conv6a_end)
#print(time_conv6_end-time_conv1_start)
#print(time_fc2_start-time_fc1_start)
#print(time_fc3_start-time_fc2_start)
#print(time_fc_end-time_fc3_start)
#print(time_fc_end-time_fc1_start)




#print('covn1',l1_feature_map.shape)
#print('pool1',l1_feature_map_pool.shape)
#
#print('covn2a',l2a_feature_map.shape)
#print('covn2',l2_feature_map.shape)
#print('pool2',l2_feature_map_pool.shape)
#
#print('covn3a',l3a_feature_map.shape)
#print('covn3',l3_feature_map.shape)
#print('pool3',l3_feature_map_pool.shape)
#
#print('covn4a',l4a_feature_map.shape)
#print('covn4',l4_feature_map.shape)
#
#print('covn5a',l5a_feature_map.shape)
#print('covn5',l5_feature_map.shape)
#
#print('covn6a',l6a_feature_map.shape)
#print('covn6',l6_feature_map.shape)
#print('pool4',l6_feature_map_pool.shape)
#
#print('concat',a0.shape)

