#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 17:14:41 2021

@author: vittorio
"""


import tensorflow as tf
import numpy as np

st = tf.constant([[1,2,3,4]])
cavallo = tf.constant([[2,4,6,8],[5,6,7,8]])
genny = tf.transpose(cavallo)

juju = tf.tensordot(tf.tensordot(st,cavallo[0,:],1),st,1)

carlo = np.array([[1,2,3],[4,5,6],[7,8,9]])

