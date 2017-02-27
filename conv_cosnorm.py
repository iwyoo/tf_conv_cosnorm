import tensorflow as tf

"""
conv2d_cosnorm
Usage :
  x : An input Tensor with shape [batch, height, width, channel]
  w : A convolutional kernel Tensor with 
           shape [height, width, in, out]. (channel == in)
  strides : strides of convolutional layer.
  padding : padding of convolutional layer.
  bias : small values like 0.0001.
"""
def conv2d_cosnorm(x, w, strides, padding, bias=0.0001):
  x_shape = tf.shape(x)
  x_b = tf.fill(
    tf.pack([x_shape[0], x_shape[1], x_shape[2], 1]), bias)
  x = tf.concat(3, [x_b, x])

  w_shape = tf.shape(w)
  w_b = tf.fill(
    tf.pack([w_shape[0], w_shape[1], 1, w_shape[3]]), bias)
  w = tf.concat(2, [w_b, w])

  x2 = tf.square(x)
  w1 = tf.ones_like(w)
  x2_len = tf.nn.conv2d(x2, w1, strides, padding)
  x_len = tf.sqrt(x2_len)

  x1 = tf.ones_like(x)
  w2 = tf.square(w)
  w2_len = tf.nn.conv2d(x1, w2, strides, padding)
  w_len = tf.sqrt(w2_len)

  y = tf.nn.conv2d(x, w, strides, padding)
  return y / (x_len * w_len)

def conv3d_cosnorm(x, w, strides, padding, bias=None): # CHECK
  # Future
  pass
