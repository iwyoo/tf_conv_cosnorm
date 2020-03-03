# tf_conv_cosnorm
Tensorflow implementation of cosine normalization [1] for convolutional layer.
There is no implementation detail of convolution version in the original paper [1].
Therefore, the performance of this implementation is not guaranteed.

This fork adds a keras layer wrapper for iwyoo's TF implementation. 

## Usage (Tensorflow)
```
# Example of using tf.nn.conv2d
conv = tf.nn.conv2d(x, w, strides, padding)
relu = tf.nn.relu(conv + bias)

# conv2d_cosnorm
conv = conv2d_cosnorm(x, w, strides, padding)
relu = tf.nn.relu(conv) # No bias needed
```

## Usage (Keras)
```
# Keras Conv2d
from keras.layers import Conv2d
model.add(Conv2d(64, (3, 3), strides=(1, 1)))

# Norm_Conv2d
from sim_layer import Norm_Conv2d
model.add(Norm_Conv2d(64, (3, 3), strides=(1, 1)))
```

## Test
A modified version of test code [2] is used to dubug and test of this implementation.

## References
- [1] Chunjie, Luo, and Yang Qiang. "Cosine Normalization: Using Cosine Similarity Instead of Dot Product in Neural Networks." (https://arxiv.org/abs/1702.05870)
- [2] https://github.com/aymericdamien/TensorFlow-Examples
- [3] https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py
- [4] https://keras.io/examples/cifar10_cnn/