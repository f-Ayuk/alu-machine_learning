#!/usr/bin/env python3
"""Normalization"""


import numpy as np


def normalize(X, m, s):
    """Normalize
    m = mean of all features of X
    s = standard deviation of all features of X"""
    return (X - m) / s
