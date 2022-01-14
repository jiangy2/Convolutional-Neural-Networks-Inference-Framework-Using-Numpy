# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 17:20:24 2019

@author: admin
"""

from __future__ import division

import numpy as np


def array_offset(x):
    """Get offset of array data from base data in bytes."""
    if x.base is None:
        return 0

    base_start = x.base.__array_interface__['data'][0]
    start = x.__array_interface__['data'][0]
    return start - base_start


def calc_pad(pad, in_siz, out_siz, stride, ksize):
    """Calculate padding width.
    Args:
        pad: padding method, "SAME", "VALID", or manually speicified.
        ksize: kernel size [I, J].
    Returns:
        pad_: Actual padding width.
    """
    if pad == 'SAME':
        return (out_siz - 1) * stride + ksize - in_siz
    elif pad == 'VALID':
        return 0
    else:
        return pad


def calc_size(h, kh, pad, sh):
    """Calculate output image size on one dimension.
    Args:
        h: input image size.
        kh: kernel size.
        pad: padding strategy.
        sh: stride.
    Returns:
        s: output size.
    """

    if pad == 'VALID':
        return np.ceil((h - kh + 1) / sh)
    elif pad == 'SAME':
        return np.ceil(h / sh)
    else:
        return int(np.ceil((h - kh + pad + 1) / sh))



def extract_sliding_windows(x, ksize, pad, stride, floor_first=True):
    """Converts a tensor to sliding windows.
    Args:
        x: [N, H, W, C]
        k: [KH, KW]
        pad: [PH, PW]
        stride: [SH, SW]
    Returns:
        y: [N, (H-KH+PH+1)/SH, (W-KW+PW+1)/SW, KH * KW, C]
    """
    n = x.shape[0]
    h = x.shape[1]
    w = x.shape[2]
    c = x.shape[3]
    kh = ksize[0]
    kw = ksize[1]
    sh = stride[0]
    sw = stride[1]

    h2 = int(calc_size(h, kh, pad, sh))
    w2 = int(calc_size(w, kw, pad, sw))
    ph = int(calc_pad(pad, h, h2, sh, kh))
    pw = int(calc_pad(pad, w, w2, sw, kw))

    ph0 = int(np.floor(ph / 2))
    ph1 = int(np.ceil(ph / 2))
    pw0 = int(np.floor(pw / 2))
    pw1 = int(np.ceil(pw / 2))

    if floor_first:
        pph = (ph0, ph1)
        ppw = (pw0, pw1)
    else:
        pph = (ph1, ph0)
        ppw = (pw1, pw0)
    x = np.pad(
        x, ((0, 0), pph, ppw, (0, 0)),
        mode='constant',
        constant_values=(0.0, ))

    y = np.zeros([n, h2, w2, kh, kw, c])
    for ii in range(h2):
        for jj in range(w2):
            xx = ii * sh
            yy = jj * sw
            y[:, ii, jj, :, :, :] = x[:, xx:xx + kh, yy:yy + kw, :]

    return y


def conv(x, w, pad='SAME', stride=(1, 1)):
    """2D convolution (technically speaking, correlation).
    Args:
        x: [N, H, W, C]
        w: [I, J, C, K]
        pad: [PH, PW]
        stride: [SH, SW]
    Returns:
        y: [N, H', W', K]
    """
    ksize = w.shape[:2]
    x = extract_sliding_windows(x, ksize, pad, stride)
    ws = w.shape
    w = w.reshape([ws[0] * ws[1] * ws[2], ws[3]])
    xs = x.shape
    x = x.reshape([xs[0] * xs[1] * xs[2], -1])
    temp = x.dot(w[:,0]).reshape(-1,1)
    for i in range(1,ws[3]):
        temp1 = x.dot(w[:,i]).reshape(-1,1)
        temp = np.hstack((temp,temp1))
    y = temp.copy()
    y = y.reshape([xs[0], xs[1], xs[2], -1])
    return y


def relu(feature_map):

    relu_out = np.maximum(feature_map,0)

    return relu_out


def pooling(feature_map, size=3, stride=2, padding='SAME'):
    ksize = (size,size)
    feature_map1= extract_sliding_windows(feature_map, ksize, padding, (stride,stride))
    fs = feature_map1.shape
    pool_out = np.zeros((fs[0], fs[1], fs[2], fs[-1])).astype(np.float32)
#    print('pool',pool_out.shape)
    f = feature_map1.reshape([fs[0] * fs[1] * fs[2], -1, fs[-1] ])
    pool_out = np.max(f,1).reshape(*pool_out.shape)
    
    

    return pool_out