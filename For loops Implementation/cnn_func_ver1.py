# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:40:12 2019

@author: admin
"""

import numpy as np
import sys

def conv_(img, conv_filter, stride):
    filter_size = conv_filter.shape[1]
    #result = np.zeros((img.shape))
    result = np.zeros((np.uint16((img.shape[0]-filter_size)/stride+1),
                            np.uint16((img.shape[1]-filter_size)/stride+1)))
    #Looping through the image to apply the convolution operation.
    r1 = 0
    for r in np.uint16(np.arange(filter_size/2.0, 
                          img.shape[0]-filter_size/2.0+1, stride)):
        c1 = 0
        for c in np.uint16(np.arange(filter_size/2.0, 
                                           img.shape[1]-filter_size/2.0+1, stride)):
            """
            Getting the current region to get multiplied with the filter.
            How to loop through the image and get the region based on 
            the image and filer sizes is the most tricky part of convolution.
            """
#            curr_region = img[r-np.uint16(np.floor(filter_size/2.0)):r+np.uint16(np.ceil(filter_size/2.0)), 
#                              c-np.uint16(np.floor(filter_size/2.0)):c+np.uint16(np.ceil(filter_size/2.0)),:]
            #Element-wise multipliplication between the current region and the filter.
#            curr_result = curr_region * conv_filter
#            conv_sum = np.sum(curr_result) #Summing the result of multiplication.
#            result[r1, c1] = conv_sum #Saving the summation in the convolution layer feature map.
            result[r1, c1] = np.sum(img[r-np.uint16(np.floor(filter_size/2.0)):r+np.uint16(np.ceil(filter_size/2.0)), 
                              c-np.uint16(np.floor(filter_size/2.0)):c+np.uint16(np.ceil(filter_size/2.0)),:]*conv_filter)
            
            c1 = c1 +1
        r1 = r1 +1
    
    #Clipping the outliers of the result matrix.
    #final_result = result[np.uint16(filter_size/2.0):result.shape[0]-np.uint16(filter_size/2.0), 
    #                      np.uint16(filter_size/2.0):result.shape[1]-np.uint16(filter_size/2.0)]
    #return final_result
    return result
def conv(img,conv_filter_shape,stride=1,padding=0):
    
    conv_filter = np.random.random(conv_filter_shape)

    if len(img.shape) != len(conv_filter.shape) - 1: # Check whether number of dimensions is the same
        print("Error: Number of dimensions in conv filter and image do not match.")  
        exit()
    if len(img.shape) > 2 or len(conv_filter.shape) > 3: # Check if number of image channels matches the filter depth.
        if img.shape[-1] != conv_filter.shape[-1]:
            print("Error: Number of channels in both image and filter must match.")
            sys.exit()
    if conv_filter.shape[1] != conv_filter.shape[2]: # Check if filter dimensions are equal.
        print('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')
        sys.exit()
    if conv_filter.shape[1]%2==0: # Check if filter diemnsions are odd.
        print('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.')
        sys.exit()
    img = np.pad(img,((padding,padding),(padding,padding),(0,0)),'constant')
    # An empty feature map to hold the output of convolving the filter(s) with the image.
    feature_maps = np.zeros((int((img.shape[0]-conv_filter.shape[1])/stride +1), 
                                int((img.shape[1]-conv_filter.shape[1])/stride+1), 
                                conv_filter.shape[0]))
#    feature_maps = np.zeros((np.uint16((img.shape[0]-conv_filter.shape[1])/stride+1),
#                            np.uint16((img.shape[1]-conv_filter.shape[1])/stride+1),
#                            conv_filter.shape[0]))
    # Convolving the image by the filter(s).
    for filter_num in range(conv_filter.shape[0]):
        #print("Filter ", filter_num + 1)
        #curr_filter = conv_filter[filter_num, :] # getting a filter from the bank.
        """ 
        Checking if there are mutliple channels for the single filter.
        If so, then each channel will convolve the image.
        The result of all convolutions are summed to return a single feature map.
        """
        if len(conv_filter[filter_num, :].shape) > 2:
#            conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0], stride) # Array holding the sum of all feature maps.
#            for ch_num in range(1, curr_filter.shape[-1]): # Convolving each channel with the image and summing the results.
#                conv_map = conv_map + conv_(img[:, :, ch_num], 
#                                  curr_filter[:, :, ch_num], stride)
            conv_map = conv_(img, conv_filter[filter_num, :], stride)
        else: # There is just a single channel in the filter.
            conv_map = conv_(img, conv_filter[filter_num, :], stride)
        feature_maps[:, :, filter_num] = conv_map # Holding feature map with the current filter.
    return feature_maps # Returning all feature maps.
    

def pooling(feature_map, size=3, stride=2, padding=0):
    #Preparing the output of the pooling operation. 
    feature_map = np.pad(feature_map,((padding,padding),(padding,padding),(0,0)),'constant')
    pool_out = np.zeros((np.uint16((feature_map.shape[0]-size)/stride+1),
                            np.uint16((feature_map.shape[1]-size)/stride+1),
                            feature_map.shape[-1]))

    r2 = 0
    for r in np.arange(0,feature_map.shape[0]-size+1, stride):
        c2 = 0
        for c in np.arange(0, feature_map.shape[1]-size+1, stride):
            pool_out[r2, c2, :] = np.max(feature_map[r:r+size,  c:c+size, :],axis=(0,1))
            c2 = c2 + 1
        r2 = r2 +1
    return pool_out

def relu(feature_map):

    relu_out = np.maximum(feature_map,0)

    return relu_out