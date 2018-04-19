#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DNN Layers:
    SoftmaxLayer

"""
import tensorflow as tf
from ..activations import get_activation


class SoftmaxDense(object):

    def __init__(self, input_data, input_dim, output_dim, weights=None,
                 biases=None, activation=None, name='Dense'):
        assert len(input_data.get_shape()) == 2, \
            "全连接层的输入必须要flatten, 即shape=[batch_size, input_dim]"
        self._input_data = input_data
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._activation = get_activation(activation)
        self._name = name

        with tf.name_scope(self._name):
            # initialize weights
            if weights is None:
                w_bound = tf.sqrt(6. / (input_dim + output_dim))
                weights = tf.Variable(
                    tf.random_uniform(
                        minval=-w_bound, maxval=w_bound, dtype='float32',
                        shape=[input_dim, output_dim]),
                    name='weights'
                )
            self._weights = weights
            tf.summary.histogram('weights', self._weights)
            # initialize biases
            if biases is None:
                biases = tf.Variable(
                    tf.constant(0.1, shape=[self._output_dim]),
                    name='biases')
            self._biases = biases
            tf.summary.histogram('biases', biases)

        self.call()

    def call(self):
        # output
        linear_output = tf.matmul(self._input_data, self._weights) + \
                            self._biases
        self._output = (
            linear_output if self._activation is None
            else self._activation(linear_output)
        )

    def loss(self, y):
        y = tf.cast(y, tf.int32)
        cross_entroy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.output, labels=y, name='xentroy')
        loss = tf.reduce_mean(cross_entroy, name='xentroy_mean')
        return loss

    def get_pre_y(self):
        # TODO 待修改
        # pre_y = tf.reshape(tf.round(tf.sigmoid(self._output)), [-1])
        pre_y = tf.arg_max(input=self._output, dimension=1)
        return pre_y

    @property
    def input_data(self):
        return self._input_data

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def name(self):
        return self._name

    @property
    def weights(self):
        return self._weights

    @property
    def biases(self):
        return self._biases

    @property
    def output(self):
        return self._output
