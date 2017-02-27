# tf_conv_cosnorm
Tensorflow implementation of cosine normalization [1] for convolutional layer.
There is no implementation detail of convolution version in the original paper [1].
Therefore, the performance of this implementation is not guaranteed.

## Usage
```
# Example of using tf.nn.conv2d
conv = tf.nn.conv2d(x, w, strides, padding)
relu = tf.nn.relu(conv + bias)

# conv2d_cosnorm
conv = conv2d_cosnorm(x, w, strides, padding)
relu = tf.nn.relu(conv) # No bias needed
```

## Test
A modified version of test code [2] is used for dubug and test of this implementation.

## References
- [1] Chunjie, Luo, and Yang Qiang. "Cosine Normalization: Using Cosine Similarity Instead of Dot Product in Neural Networks." (https://arxiv.org/abs/1702.05870)
- [2] https://github.com/aymericdamien/TensorFlow-Examples
