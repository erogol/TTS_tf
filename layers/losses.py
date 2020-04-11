import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import math_ops

def loss_l1(y_hat, y, lens):
    #mask = tf.sequence_mask(lens, maxlen=y.shape[1], dtype=tf.float32)
    #mask = tf.cast(mask, tf.float32)
    loss = tf.reduce_mean(math_ops.abs(y_hat - y))
    #mask = tf.tile(tf.expand_dims(mask, axis=-1), [1, 1, loss.shape[2]])
    #loss_masked = loss * mask
    #return tf.reduce_sum(loss_masked) / tf.reduce_sum(mask)
    return loss

def loss_l2(y_hat, y, lens):
    # mask = tf.sequence_mask(lens, dtype=tf.float32)
    # loss = tf.keras.losses.mean_squared_error(y, y_hat)
    loss = tf.reduce_mean(math_ops.squared_difference(y_hat, y))
    #mask = tf.tile(tf.expand_dims(mask, axis=-1), [1, 1, loss.shape[2]])
    #loss_masked = loss * mask
    #return tf.reduce_sum(loss_masked) / tf.reduce_sum(mask)
    return loss

def stopnet_loss(y_hat, y):
    return tf.reduce_mean(            
        keras.losses.binary_crossentropy(y,
                                         y_hat,
                                         from_logits=True))
