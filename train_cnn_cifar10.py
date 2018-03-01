import numpy as np
import tensorflow as tf
from keras.utils import np_utils
import math
import cPickle
import time

start = time.time()


# Extract data from batch file
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

# 'data_batch_1' contains 10,000 CIFAR-10 images and their labels
file1 = 'cifar-10-batches-py/data_batch_1'
data_batch_1 = unpickle(file1)
X_train = np.array(data_batch_1['data'], dtype=np.float32)[:4000]
y_train = np_utils.to_categorical(data_batch_1['labels'])[:4000]

# Training Parameters
lr_max = 0.003
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_epsilon = 1e-8
alpha = 0.15
beta = 0.01
rampup_length = 80
rampdown_length = 50

# Split data and labels into mini-batches
batch_size = 100
total_batch = len(X_train) // batch_size
X_batches = np.array_split(X_train, total_batch)
Y_batches = np.array_split(y_train, total_batch)

# Training & displaying steps
num_steps = 300
display_step = 15

# Network Parameters
num_input = X_train.shape[1]  # CIFAR-10 data input (img shape: 32*32*3)
num_classes = y_train.shape[1]  # CIFAR-10 total classes (0-9 digits)


tf.reset_default_graph()

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input], name='data')
Y = tf.placeholder(tf.float32, [None, num_classes], name='labels')
keep_prob = tf.placeholder(tf.float32, name='dropout')  # dropout (keep probability)
learning_rate = tf.placeholder(tf.float32, name='learning_rate')


# Create some wrappers for simplicity
def gaussian_noise_layer(x, std):
    noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=std, dtype=tf.float32)
    return x + noise


def rampup(epoch):
    if epoch < rampup_length:
        p = max(0.0, float(epoch)) / float(rampup_length)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0


def rampdown(epoch):
    if epoch >= (num_steps - rampdown_length):
        ep = (epoch - (num_steps - rampdown_length)) * 0.5
        return math.exp(-(ep * ep) / rampdown_length)
    else:
        return 1.0


def leaky_relu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def batch_norm(x, epsilon=1e-3):
    shape = x.shape
    last = len(shape) - 1
    axis = list(range(last))
    mean, var = tf.nn.moments(x, axis)
    scale = tf.Variable(tf.ones([shape[last]]))
    offset = tf.Variable(tf.zeros([shape[last]]))
    return tf.nn.batch_normalization(x, mean, var, offset, scale, epsilon)


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

    # Add gaussian noise layer
    x = gaussian_noise_layer(x, alpha)

    # Convolution Layer 1
    conv1a = batch_norm(conv2d(x, weights['wc1a'], biases['bc1a']))
    conv1b = batch_norm(conv2d(conv1a, weights['wc1b'], biases['bc1b']))
    conv1c = batch_norm(conv2d(conv1b, weights['wc1c'], biases['bc1c']))
    # Max Pooling (down-sampling)
    pool1 = maxpool2d(conv1c, k=2, dropout=dropout)

    # Convolution Layer 2
    conv2a = batch_norm(conv2d(pool1, weights['wc2a'], biases['bc2a']))
    conv2b = batch_norm(conv2d(conv2a, weights['wc2b'], biases['bc2b']))
    conv2c = batch_norm(conv2d(conv2b, weights['wc2c'], biases['bc2c']))
    # Max Pooling (down-sampling)
    pool2 = maxpool2d(conv2c, k=2, dropout=dropout)

    # Convolution Layer 3
    conv3a = batch_norm(conv2d(pool2, weights['wc3a'], biases['bc3a'], padding='VALID'))
    conv3b = batch_norm(conv2d(conv3a, weights['wc3b'], biases['bc3b']))
    conv3c = batch_norm(conv2d(conv3b, weights['wc3c'], biases['bc3c']))
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
    'wc1a': tf.get_variable(name='wc1a', shape=[3, 3, 3, 128]),
    # 3x3 conv, 128 inputs, 128 outputs
    'wc1b': tf.get_variable(name='wc1b', shape=[3, 3, 128, 128]),
    # 3x3 conv, 128 inputs, 128 outputs
    'wc1c': tf.get_variable(name='wc1c', shape=[3, 3, 128, 128]),
    # 3x3 conv, 128 inputs, 256 outputs
    'wc2a': tf.get_variable(name='wc2a', shape=[3, 3, 128, 256]),
    # 3x3 conv, 128 inputs, 256 outputs
    'wc2b': tf.get_variable(name='wc2b', shape=[3, 3, 256, 256]),
    # 3x3 conv, 128 inputs, 256 outputs
    'wc2c': tf.get_variable(name='wc2c', shape=[3, 3, 256, 256]),
    # 3x3 conv, 256 inputs, 512 outputs
    'wc3a': tf.get_variable(name='wc3a', shape=[3, 3, 256, 512]),
    # 1x1 conv, 512 inputs, 256 outputs
    'wc3b': tf.get_variable(name='wc3b', shape=[1, 1, 512, 256]),
    # 1x1 conv, 256 inputs, 128 outputs
    'wc3c': tf.get_variable(name='wc3c', shape=[1, 1, 256, 128]),
    # 1x1x128 inputs, 10 outputs (class prediction)
    'out': tf.get_variable(name='w_out', shape=[1 * 1 * 128, num_classes])
}

biases = {
    'bc1a': tf.get_variable(name='bc1a', shape=[128]),
    'bc1b': tf.get_variable(name='bc1b', shape=[128]),
    'bc1c': tf.get_variable(name='bc1c', shape=[128]),
    'bc2a': tf.get_variable(name='bc2a', shape=[256]),
    'bc2b': tf.get_variable(name='bc2b', shape=[256]),
    'bc2c': tf.get_variable(name='bc2c', shape=[256]),
    'bc3a': tf.get_variable(name='bc3a', shape=[512]),
    'bc3b': tf.get_variable(name='bc3b', shape=[256]),
    'bc3c': tf.get_variable(name='bc3c', shape=[128]),
    'out': tf.get_variable(name='b_out', shape=[num_classes])
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)


# Compute l2 regularization
def l2_regularization(W, B):
    result = 0.
    for w in W:
        result += tf.nn.l2_loss(W[w])
    for b in B:
        result += tf.nn.l2_loss(B[b])
    return result

# Define loss and optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=Y))
loss_op = tf.reduce_mean(cross_entropy + beta * l2_regularization(weights, biases))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   beta1=adam_beta1,
                                   beta2=adam_beta2,
                                   epsilon=adam_epsilon)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy_op')

model_path = 'tf_model_data/cnn_cifar10/cnn_cifar10.ckpt'

# Add ops to save all the variables.
saver = tf.train.Saver()

# Start training
with tf.Session() as sess:

    # Restore variables from disk.
    saver.restore(sess, model_path)

    # Initialize learning_rate
    lr = 0.

    for step in range(1, num_steps + 1):

        # Reset loss and accuracy for current training step
        avg_loss = 0.
        avg_acc = 0.

        # Ramp up/down learning rate
        rampup_value = rampup(step)
        rampdown_value = rampdown(step)
        lr = rampup_value * rampdown_value * lr_max

        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Run optimization op (backprop), loss op and accuracy op
            _, loss, acc = sess.run([train_op, loss_op, accuracy], feed_dict={X: batch_x,
                                                                              Y: batch_y,
                                                                              keep_prob: 0.5,
                                                                              learning_rate: lr})
            # Compute average loss and accuracy
            avg_loss += loss / total_batch
            avg_acc += acc / total_batch

        # Display loss and accuracy per step
        if step % display_step == 0 or step == 1:
            print("Step " + str(step) +
                  ", Loss=" + "{:.4f}".format(avg_loss) +
                  ", Accuracy=" + "{:.4f}".format(avg_acc))

    # Stop timer
    end = time.time()
    print('\nOptimization Finished!'
          '\nTotal training time: {:.2f} {}'.format((end - start) / 60, 'minutes'))

    # Save model data to model_path
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)

    # Calculate loss and accuracy for 256 CIFAR-10 test images
    file_test = 'cifar-10-batches-py/test_batch'
    test_batch = unpickle(file_test)
    X_test = np.array(test_batch['data'], dtype=np.float32)
    y_test = np_utils.to_categorical(test_batch['labels'])
    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: X_test[:256],
                                                         Y: y_test[:256],
                                                         keep_prob: 1.0})
    print("Testing Loss: " + "{:.4f}".format(loss)
          + ", Testing Accuracy: " + "{:.4f}".format(acc))
