import numpy as np
import tensorflow as tf
from keras.utils import np_utils
import cPickle


# Extract data from batch file
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

file_test = 'cifar-10-batches-py/test_batch'
test_batch = unpickle(file_test)
X_test = np.array(test_batch['data'], dtype=np.float32)
y_test = np_utils.to_categorical(test_batch['labels'])

# Network Parameters
num_input = X_test.shape[1]  # CIFAR-10 data input (img shape: 32*32*3)
num_classes = y_test.shape[1]  # CIFAR-10 total classes (0-9 digits)

# Training Parameters
beta1 = 0.9
beta2 = 0.999
alpha = 0.15


tf.reset_default_graph()

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input], name='data')
Y = tf.placeholder(tf.float32, [None, num_classes], name='labels')
keep_prob = tf.placeholder(tf.float32, name='dropout')  # dropout (keep probability)
learning_rate = tf.placeholder(tf.float32, name='learning_rate')


def gaussian_noise_layer(x, std):
    noise = np.random.normal(0.0, std, (x.shape[0], x.shape[1]))
    return x + noise


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def leaky_relu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def conv2d(x, W, b, strides=1, padding='SAME'):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return leaky_relu(x, 0.1)


def maxpool2d(x, k=2, dropout=0.5):
    # MaxPool2D wrapper
    return tf.nn.dropout(tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                                        padding='SAME'), dropout)


# Create model
def conv_net(x, weights, biases, dropout):
    # CIFAR-10 data input is a 1-D vector of 3072 features (32*32 pixels * 3 Channels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 32, 32, 3])

    # Convolution Layer 1
    conv1a = conv2d(x, weights['wc1a'], biases['bc1a'])
    conv1b = conv2d(conv1a, weights['wc1b'], biases['bc1b'])
    conv1c = conv2d(conv1b, weights['wc1c'], biases['bc1c'])
    # Max Pooling (down-sampling)
    pool1 = maxpool2d(conv1c, k=2, dropout=dropout)

    # Convolution Layer 2
    conv2a = conv2d(pool1, weights['wc2a'], biases['bc2a'])
    conv2b = conv2d(conv2a, weights['wc2b'], biases['bc2b'])
    conv2c = conv2d(conv2b, weights['wc2c'], biases['bc2c'])
    # Max Pooling (down-sampling)
    pool2 = maxpool2d(conv2c, k=2, dropout=dropout)

    # Convolution Layer 3
    conv3a = conv2d(pool2, weights['wc3a'], biases['bc3a'], padding='VALID')
    conv3b = conv2d(conv3a, weights['wc3b'], biases['bc3b'])
    conv3c = conv2d(conv3b, weights['wc3c'], biases['bc3c'])
    # Global Average Pooling (down-sampling)
    pool3 = tf.layers.average_pooling2d(conv3c, pool_size=[6, 6], strides=[1, 1])

    # Output, class prediction
    # Reshape pool3 output to fit fully connected layer input
    out = tf.reshape(pool3, [-1, weights['out'].get_shape().as_list()[0]])
    out = tf.add(tf.matmul(out, weights['out']), biases['out'])
    return out


# Store layers weight & bias
weights = {
    # 3x3 conv, 3 inputs, 128 outputs
    'wc1a': tf.get_variable("wc1a", shape=[3, 3, 3, 128]),
    # 3x3 conv, 128 inputs, 128 outputs
    'wc1b': tf.get_variable("wc1b", shape=[3, 3, 128, 128]),
    # 3x3 conv, 128 inputs, 128 outputs
    'wc1c': tf.get_variable("wc1c", shape=[3, 3, 128, 128]),
    # 3x3 conv, 128 inputs, 256 outputs
    'wc2a': tf.get_variable("wc2a", shape=[3, 3, 128, 256]),
    # 3x3 conv, 128 inputs, 256 outputs
    'wc2b': tf.get_variable("wc2b", shape=[3, 3, 256, 256]),
    # 3x3 conv, 128 inputs, 256 outputs
    'wc2c': tf.get_variable("wc2c", shape=[3, 3, 256, 256]),
    # 3x3 conv, 256 inputs, 512 outputs
    'wc3a': tf.get_variable("wc3a", shape=[3, 3, 256, 512]),
    # 1x1 conv, 512 inputs, 256 outputs
    'wc3b': tf.get_variable("wc3b", shape=[1, 1, 512, 256]),
    # 1x1 conv, 256 inputs, 128 outputs
    'wc3c': tf.get_variable("wc3c", shape=[1, 1, 256, 128]),
    # 1x1x128 inputs, 10 outputs (class prediction)
    'out': tf.get_variable("w_out", shape=[1 * 1 * 128, num_classes])
}

biases = {
    'bc1a': tf.get_variable("bc1a", shape=[128]),
    'bc1b': tf.get_variable("bc1b", shape=[128]),
    'bc1c': tf.get_variable("bc1c", shape=[128]),
    'bc2a': tf.get_variable("bc2a", shape=[256]),
    'bc2b': tf.get_variable("bc2b", shape=[256]),
    'bc2c': tf.get_variable("bc2c", shape=[256]),
    'bc3a': tf.get_variable("bc3a", shape=[512]),
    'bc3b': tf.get_variable("bc3b", shape=[256]),
    'bc3c': tf.get_variable("bc3c", shape=[128]),
    'out': tf.get_variable("b_out", shape=[num_classes])
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                 labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   beta1=beta1,
                                   beta2=beta2)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

model_path = 'tf_model_data/cnn_cifar10/cnn_cifar10.ckpt'

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Add ops to restore all the variables.
saver = tf.train.Saver()

# Start training
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, model_path)
    print("Model restored.")
    # Run accuracy op
    acc = sess.run(accuracy, feed_dict={X: X_test[:256],
                                        Y: y_test[:256],
                                        keep_prob: 1.0})
    print("Testing Accuracy: " + "{:.4f}".format(acc))
