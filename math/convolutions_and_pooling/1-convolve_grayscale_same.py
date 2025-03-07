#!/usr/bin/env python3
"""Same Convolution on an image"""


import numpy as np


def convolve_grayscale_same(images, kernel):
    """Function that performs a same convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2
    padded_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    convolved_images = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            convolved_images[:, i, j] = np.sum(padded_images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))

    return convolved_images    
    """kh, kw = kernel.shape
    m, hm, wm = images.shape
    ph = int(kh / 2)
    pw = int(kw / 2)
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    convoluted = np.zeros((m, hm, wm))
    for h in range(hm):
        for w in range(wm):
            square = padded[:, h: h + kh, w: w + kw]
            insert = np.sum(square * kernel, axis=1).sum(axis=1)
            convoluted[:, h, w] = insert
    return convoluted"""
