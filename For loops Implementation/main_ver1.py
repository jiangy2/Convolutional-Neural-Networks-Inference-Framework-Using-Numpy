# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:43:09 2019

@author: admin
"""

import numpy as np
import cnn_func_ver1 as npcnn
import facenet_fc_navie as npfc
import time


#import skimage.data
#from skimage.transform import resize
#img = skimage.data.chelsea()
#img = resize(img, (220, 220))

#time_start=time.time()
# Reading the image
#img = skimage.io.imread("test.jpg")
#img = skimage.data.checkerboard()

#img = skimage.data.chelsea()
#img = np.random.random((220,220,3))
#img = skimage.data.camera()


img = np.random.random((220,220,3))

time_conv1_start=time.time()
#test_filter_1 = np.random.random((64,7,7,3))
test_filter_1_shape = (64,7,7,3)
l1_feature_map = npcnn.conv(img,test_filter_1_shape,stride=2,padding=3)
l1_feature_map_relu = npcnn.relu(l1_feature_map)
l1_feature_map_pool = npcnn.pooling(l1_feature_map_relu,size=3,stride=2,padding=1)
time_conv1_end=time.time()
print('time cost of conv1',time_conv1_end-time_conv1_start,'s')

#test_filter_2a = np.random.random((64,1,1,64))
test_filter_2a_shape = (64,1,1,64)
l2a_feature_map = npcnn.conv(l1_feature_map_pool, test_filter_2a_shape,stride=1,padding=0)
l2a_feature_map_relu = npcnn.relu(l2a_feature_map)

#test_filter_2 = np.random.random((192,3,3,64))
test_filter_2_shape = (192,3,3,64)
l2_feature_map = npcnn.conv(l2a_feature_map_relu, test_filter_2_shape,stride=1,padding=1)
l2_feature_map_relu = npcnn.relu(l2_feature_map)
l2_feature_map_pool = npcnn.pooling(l2_feature_map_relu,size=3,stride=2,padding=1)

time_conv2a_2_end=time.time()
print('time cost of conv2a and conv2',time_conv2a_2_end-time_conv1_end,'s')


#test_filter_3a = np.random.random((192,1,1,192))
test_filter_3a_shape = (192,1,1,192)
l3a_feature_map = npcnn.conv(l2_feature_map_pool, test_filter_3a_shape,stride=1,padding=0)
l3a_feature_map_relu = npcnn.relu(l3a_feature_map)

#test_filter_3 = np.random.random((384,3,3,192))
test_filter_3_shape = (384,3,3,192)
l3_feature_map = npcnn.conv(l3a_feature_map_relu, test_filter_3_shape,stride=1,padding=1)
l3_feature_map_relu = npcnn.relu(l3_feature_map)
l3_feature_map_pool = npcnn.pooling(l3_feature_map_relu,size=3,stride=2,padding=1)

time_conv3_end=time.time()
print('time cost of conv3',time_conv3_end-time_conv2a_2_end,'s')

#test_filter_4a = np.random.random((384,1,1,384))
test_filter_4a_shape = (384,1,1,384)
l4a_feature_map = npcnn.conv(l3_feature_map_pool, test_filter_4a_shape,stride=1,padding=0)
l4a_feature_map_relu = npcnn.relu(l4a_feature_map)

#test_filter_4 = np.random.random((256,3,3,384))
test_filter_4_shape = (256,3,3,384)
l4_feature_map = npcnn.conv(l4a_feature_map_relu, test_filter_4_shape,stride=1,padding=1)
l4_feature_map_relu = npcnn.relu(l4_feature_map)

time_conv4a_4_end=time.time()
print('time cost of conv4a and conv4',time_conv4a_4_end-time_conv3_end,'s')

#test_filter_5a = np.random.random((256,1,1,256))
test_filter_5a_shape = (256,1,1,256)
l5a_feature_map = npcnn.conv(l4_feature_map_relu, test_filter_5a_shape,stride=1,padding=0)
l5a_feature_map_relu = npcnn.relu(l5a_feature_map)

#test_filter_5 = np.random.random((256,3,3,256))
test_filter_5_shape = (256,3,3,256)
l5_feature_map = npcnn.conv(l5a_feature_map_relu, test_filter_5_shape,stride=1,padding=1)
l5_feature_map_relu = npcnn.relu(l5_feature_map)

time_conv5a_5_end=time.time()
print('time cost of conv5a and conv5',time_conv5a_5_end-time_conv4a_4_end,'s')

#test_filter_6a = np.random.random((256,1,1,256))
test_filter_6a_shape = (256,1,1,256)
l6a_feature_map = npcnn.conv(l5_feature_map_relu, test_filter_6a_shape,stride=1,padding=0)
l6a_feature_map_relu = npcnn.relu(l6a_feature_map)

#test_filter_6 = np.random.random((256,3,3,256))
test_filter_6_shape = (256,3,3,256)
l6_feature_map = npcnn.conv(l6a_feature_map_relu, test_filter_6_shape,stride=1,padding=1)
l6_feature_map_relu = npcnn.relu(l6_feature_map)
l6_feature_map_pool = npcnn.pooling(l6_feature_map_relu,size=3,stride=2,padding=1)

time_conv6a_6_end=time.time()
print('time cost of conv6a and conv6',time_conv6a_6_end-time_conv5a_5_end,'s')

Net_dim = [7*7*256, 1*32*128, 1*32*128, 1*1*128]#original
maxout_layer = [1,2]
k=[2,2]
    
#X = np.random.randn(1, Net_dim[0])
a0 = l6_feature_map_pool.reshape(-1, Net_dim[0])

Net_dim = [7*7*256, 1*32*128, 1*32*128, 1*1*128]
#a0 = np.random.random_sample(size=(1, Net_dim[0])).astype(np.float32)#12544
time_fc_start = time.time()
z1 = npfc.forward_step(a0, Net_dim[1],64)
a1 = npfc.maxout(z1,2)
z2 = npfc.forward_step(a1,Net_dim[2],64)
a2 = npfc.maxout(z2,2)
z3 = npfc.forward(a2,Net_dim[-1])
#z3 = npfc.forward(npfc.maxout(npfc.forward_step(npfc.maxout(npfc.forward_step(a0, Net_dim[1],64),2),Net_dim[2],64),2),Net_dim[-1])
time_fc_end = time.time()
print('time cost of full connected layers',time_fc_end-time_fc_start,'s')