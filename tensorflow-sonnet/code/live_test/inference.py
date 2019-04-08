from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import i3d


def att_model(image_holder_rgb, image_holder_flow, dropout_holder, is_train_holder, num_classes):
    # Inference Module
    with tf.variable_scope('RGB'):
        net_rgb = i3d.InceptionI3d(num_classes=400, spatial_squeeze=True, final_endpoint='Logits')
        logits_rgb, _ = net_rgb(inputs=image_holder_rgb, is_training=is_train_holder)
        pred_op_rgb, _ = net_rgb(inputs=image_holder_rgb, is_training=is_train_holder)

    with tf.variable_scope('Flow'):
        net_flow = i3d.InceptionI3d(num_classes=400, spatial_squeeze=True, final_endpoint='Logits')
        logits_flow, _ = net_flow(inputs=image_holder_flow, is_training=is_train_holder)

    with tf.variable_scope('TwoStream'):
        epsilon = 1e-3
        # weights_rgb (batch_size, 1)

        bn_mean_rgb, bn_var_rgb = tf.nn.moments(logits_rgb, [1], keep_dims=True)
        logits_rgb = tf.div(tf.subtract(logits_rgb, bn_mean_rgb), (tf.sqrt(bn_var_rgb) + epsilon))
        if is_train_holder:
            weights_rgb_1 = tf.layers.dense(logits_rgb, 1, activation=tf.nn.leaky_relu,
                                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                            name='weights_rgb')
        elif not is_train_holder:
            weights_rgb_1 = tf.layers.dense(logits_rgb, 1, activation=None,
                                            kernel_initializer=None,
                                            name='weights_rgb')
        weights_rgb = tf.exp(weights_rgb_1) / tf.reduce_sum(tf.exp(weights_rgb_1))
        attention_rgb = tf.multiply(logits_rgb, weights_rgb, name='attention_rgb')

        bn_mean_flow, bn_var_flow = tf.nn.moments(logits_flow, [1], keep_dims=True)
        logits_flow = tf.div(tf.subtract(logits_flow, bn_mean_flow), (tf.sqrt(bn_var_flow) + epsilon))
        if is_train_holder:
            weights_flow = tf.layers.dense(logits_flow, 1, activation=tf.nn.leaky_relu,
                                           kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                           name='weights_flow')
        elif not is_train_holder:
            weights_flow = tf.layers.dense(logits_flow, 1, activation=None,
                                           kernel_initializer=None,
                                           name='weights_flow')
        weights_flow = tf.exp(weights_flow) / tf.reduce_sum(tf.exp(weights_flow))
        attention_flow = tf.multiply(logits_flow, weights_flow, name='attention_flow')

        logits = tf.concat([attention_rgb, attention_flow], axis=-1, name='concat')
        logits_dropout = tf.nn.dropout(logits, dropout_holder, name='dropout')
        fc_out = tf.layers.dense(logits_dropout, num_classes, use_bias=True, name='fc')

    if is_train_holder:
        return fc_out
    elif not is_train_holder:
        return fc_out, weights_rgb_1, logits_rgb


def att_model_3stream(image_holder_rgb, image_holder_flow, image_holder_warped_flow, dropout_holder, is_train_holder, num_classes):
    # Inference Module
    with tf.variable_scope('RGB'):
        net_rgb = i3d.InceptionI3d(num_classes=400, spatial_squeeze=True, final_endpoint='Logits')
        # logits (batch_size, 400)
        logits_rgb, _ = net_rgb(inputs=image_holder_rgb, is_training=is_train_holder, dropout_keep_prob=dropout_holder)

    with tf.variable_scope('Flow'):
        net_flow = i3d.InceptionI3d(num_classes=400, spatial_squeeze=True, final_endpoint='Logits')
        logits_flow, _ = net_flow(inputs=image_holder_flow, is_training=is_train_holder, dropout_keep_prob=dropout_holder)

    with tf.variable_scope('WarpedFlow'):
        net_warped_flow = i3d.InceptionI3d(num_classes=400, spatial_squeeze=True, final_endpoint='Logits')
        logits_warped_flow, _ = net_warped_flow(inputs=image_holder_warped_flow, is_training=is_train_holder,
                                  dropout_keep_prob=dropout_holder)
    with tf.variable_scope('TwoStream'):
        epsilon = 1e-3
        # weights_rgb (batch_size, 1)
        bn_mean_rgb, bn_var_rgb = tf.nn.moments(logits_rgb, [1], keep_dims=True)
        logits_rgb = tf.div(tf.subtract(logits_rgb, bn_mean_rgb), (tf.sqrt(bn_var_rgb) + epsilon))

        weights_rgb_1 = tf.layers.dense(logits_rgb, 1, activation=None,
                                      kernel_initializer=None,
                                      name='weights_rgb')
        weights_rgb = tf.exp(weights_rgb_1) / tf.reduce_sum(tf.exp(weights_rgb_1))
        attention_rgb = tf.multiply(logits_rgb, weights_rgb, name='attention_rgb')

        bn_mean_flow, bn_var_flow = tf.nn.moments(logits_flow, [1], keep_dims=True)
        logits_flow = tf.div(tf.subtract(logits_flow, bn_mean_flow), (tf.sqrt(bn_var_flow) + epsilon))

        weights_flow = tf.layers.dense(logits_flow, 1, activation=None,
                                      kernel_initializer=None,
                                      name='weights_flow')
        weights_flow = tf.exp(weights_flow) / tf.reduce_sum(tf.exp(weights_flow))
        attention_flow = tf.multiply(logits_flow, weights_flow, name='attention_flow')

        bn_mean_warped_flow, bn_var_warped_flow = tf.nn.moments(logits_warped_flow, [1], keep_dims=True)
        logits_warped_flow = tf.div(tf.subtract(logits_warped_flow, bn_mean_warped_flow), (tf.sqrt(bn_var_warped_flow) + epsilon))

        weights_warped_flow = tf.layers.dense(logits_warped_flow, 1, activation=None,
                                       kernel_initializer=None,
                                       name='weights_warped_flow')
        weights_warped_flow = tf.exp(weights_warped_flow) / tf.reduce_sum(tf.exp(weights_warped_flow))
        attention_warped_flow = tf.multiply(logits_warped_flow, weights_warped_flow, name='attention_warped_flow')

        logits = tf.concat([attention_rgb, attention_flow, attention_warped_flow], axis=-1, name='concat')
        logits_dropout = tf.nn.dropout(logits, dropout_holder, name='dropout')
        fc_out = tf.layers.dense(logits_dropout, num_classes, use_bias=True, name='fc')
        if is_train_holder:
            return fc_out
        elif not is_train_holder:
            return fc_out, weights_rgb_1, logits_rgb


def att_model_ek(clip_holder_rgb, clip_holder_flow, num_class, is_train_holder, dropout_holder):
    # Inference Module
    with tf.variable_scope('RGB'):
        net_rgb = i3d.InceptionI3d(num_classes=400, spatial_squeeze=True, final_endpoint='Logits')
        # logits (batch_size, 400)
        logits_rgb, _ = net_rgb(inputs=clip_holder_rgb, is_training=is_train_holder, dropout_keep_prob=dropout_holder)

    with tf.variable_scope('Flow'):
        net_flow = i3d.InceptionI3d(num_classes=400, spatial_squeeze=True, final_endpoint='Logits')
        logits_flow, _ = net_flow(inputs=clip_holder_flow, is_training=is_train_holder,
                                  dropout_keep_prob=dropout_holder)

    with tf.variable_scope('TwoStream'):
        epsilon = 1e-3
        # weights_rgb (batch_size, 1)
        bn_mean_rgb, bn_var_rgb = tf.nn.moments(logits_rgb, [1], keep_dims=True)
        logits_rgb = tf.div(tf.subtract(logits_rgb, bn_mean_rgb), (tf.sqrt(bn_var_rgb) + epsilon))

        weights_rgb = tf.layers.dense(logits_rgb, 1, activation=None,
                                      kernel_initializer=None,
                                      name='weights_rgb')
        weights_rgb = tf.exp(weights_rgb) / tf.reduce_sum(tf.exp(weights_rgb))
        attention_rgb = tf.multiply(logits_rgb, weights_rgb, name='attention_rgb')

        bn_mean_flow, bn_var_flow = tf.nn.moments(logits_flow, [1], keep_dims=True)
        logits_flow = tf.div(tf.subtract(logits_flow, bn_mean_flow), (tf.sqrt(bn_var_flow) + epsilon))

        weights_flow = tf.layers.dense(logits_flow, 1, activation=None,
                                       kernel_initializer=None,
                                       name='weights_flow')
        weights_flow = tf.exp(weights_flow) / tf.reduce_sum(tf.exp(weights_flow))
        attention_flow = tf.multiply(logits_flow, weights_flow, name='attention_flow')

        logits = tf.concat([attention_rgb, attention_flow], axis=-1, name='concat')
        logits_dropout = tf.nn.dropout(logits, dropout_holder, name='dropout_n')
        # (batch_size, num_classes)
        v_fc_out = tf.layers.dense(logits_dropout, num_class['verb'], use_bias=True, name='fc')

        n_logits_dropout = tf.nn.dropout(logits_rgb, dropout_holder, name='dropout_v')
        # (batch_size, num_classes)
        n_fc_out = tf.layers.dense(n_logits_dropout, num_class['noun'], use_bias=True)
        
        return v_fc_out, n_fc_out
