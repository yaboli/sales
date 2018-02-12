import numpy as np
import tensorflow as tf
from keras.utils import np_utils
import cPickle
from pathlib import Path
import shutil

# Delete old data and event summaries
dir1 = 'tf_model_data/cnn_cifar10/'
path1 = Path(dir1)
dir2 = 'tensorboard/cnn_cifar10/'
path2 = Path(dir2)
if path1.is_dir():
    shutil.rmtree(dir1)
if path2.is_dir():
    shutil.rmtree(dir2)


# Extract data from batch file
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


# 'data_batch_1' contains 10,000 CIFAR-10 images and their labels
file1 = 'cifar-10-batches-py/data_batch_1'
data_batch_1 = unpickle(file1)
X_train = np.array(data_batch_1['data'], dtype=np.float32)
y_train = np_utils.to_categorical(data_batch_1['labels'])
data_size = X_train.shape[0]

# Training Parameters
lr_max = 0.003
beta1 = 0.9
beta2 = 0.999
alpha = 0.15
beta = 0.01

# Split data and labels into mini-batches
batch_size = 100
total_batch = len(X_train) // batch_size
X_batches = np.array_split(X_train, total_batch)
Y_batches = np.array_split(y_train, total_batch)

# Training & displaying steps
num_steps = 300
display_step = 1

# Network Parameters
num_input = X_train.shape[1]  # CIFAR-10 data input (img shape: 32*32*3)
num_classes = y_train.shape[1]  # CIFAR-10 total classes (0-9 digits)

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input], name='data')
Y = tf.placeholder(tf.float32, [None, num_classes], name='labels')
keep_prob = tf.placeholder(tf.float32, name='dropout')  # dropout (keep probability)
learning_rate = tf.placeholder(tf.float32, name='learning_rate')


# Create some wrappers for simplicity
def gaussian_noise_layer(x, std):
    noise = np.random.normal(0.0, std, (x.shape[0], x.shape[1]))
    return x + noise


def gaussian(amplitude, scale, t, scope):
    return amplitude * np.exp(-scale * (1 - t / scope) ** 2)


def leaky_relu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def batch_norm(x, epsilon=0.001):
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


# Initialize stddev of filter weights (Ref. "He et al. 2015")
def init_stddev(filter_size, num_filters):
    return (2 / (filter_size ** 2 * num_filters)) ** 0.5

# Store layers weight & bias
weights = {
    # 3x3 conv, 3 inputs, 128 outputs
    'wc1a': tf.Variable(tf.random_normal([3, 3, 3, 128], mean=0.0, stddev=init_stddev(3, 128)), name='wc1a'),
    # 3x3 conv, 128 inputs, 128 outputs
    'wc1b': tf.Variable(tf.random_normal([3, 3, 128, 128], mean=0.0, stddev=init_stddev(3, 128)), name='wc1b'),
    # 3x3 conv, 128 inputs, 128 outputs
    'wc1c': tf.Variable(tf.random_normal([3, 3, 128, 128], mean=0.0, stddev=init_stddev(3, 128)), name='wc1c'),
    # 3x3 conv, 128 inputs, 256 outputs
    'wc2a': tf.Variable(tf.random_normal([3, 3, 128, 256], mean=0.0, stddev=init_stddev(3, 256)), name='wc2a'),
    # 3x3 conv, 128 inputs, 256 outputs
    'wc2b': tf.Variable(tf.random_normal([3, 3, 256, 256], mean=0.0, stddev=init_stddev(3, 256)), name='wc2b'),
    # 3x3 conv, 128 inputs, 256 outputs
    'wc2c': tf.Variable(tf.random_normal([3, 3, 256, 256], mean=0.0, stddev=init_stddev(3, 256)), name='wc2c'),
    # 3x3 conv, 256 inputs, 512 outputs
    'wc3a': tf.Variable(tf.random_normal([3, 3, 256, 512], mean=0.0, stddev=init_stddev(3, 512)), name='wc3a'),
    # 1x1 conv, 512 inputs, 256 outputs
    'wc3b': tf.Variable(tf.random_normal([1, 1, 512, 256], mean=0.0, stddev=init_stddev(1, 256)), name='wc3b'),
    # 1x1 conv, 256 inputs, 128 outputs
    'wc3c': tf.Variable(tf.random_normal([1, 1, 256, 128], mean=0.0, stddev=init_stddev(1, 128)), name='wc3c'),
    # 1x1x128 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1 * 1 * 128, num_classes], mean=0.0, stddev=0.01), name='w_out')
}

biases = {
    'bc1a': tf.Variable(tf.random_normal([128]), name='bc1a'),
    'bc1b': tf.Variable(tf.random_normal([128]), name='bc1b'),
    'bc1c': tf.Variable(tf.random_normal([128]), name='bc1c'),
    'bc2a': tf.Variable(tf.random_normal([256]), name='bc2a'),
    'bc2b': tf.Variable(tf.random_normal([256]), name='bc2b'),
    'bc2c': tf.Variable(tf.random_normal([256]), name='bc2c'),
    'bc3a': tf.Variable(tf.random_normal([512]), name='bc3a'),
    'bc3b': tf.Variable(tf.random_normal([256]), name='bc3b'),
    'bc3c': tf.Variable(tf.random_normal([128]), name='bc3c'),
    'out': tf.Variable(tf.random_normal([num_classes]), name='b_out')
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
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                 labels=Y))
loss_op = tf.reduce_mean(loss_op + beta * l2_regularization(weights, biases))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   beta1=beta1,
                                   beta2=beta2)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy_op')

model_path = 'tf_model_data/cnn_cifar10/cnn_cifar10.ckpt'

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Add ops to save all the variables.
saver = tf.train.Saver()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Initialize learning_rate
    lr = 0.

    for step in range(1, num_steps + 1):

        # Reset loss and accuracy for current training step
        avg_loss = 0.
        avg_acc = 0.

        # Ramp up or down learning rate
        if step <= 80:
            lr = gaussian(lr_max, 5, step, 80)
        elif step > 250:
            lr = gaussian(lr_max, 12.5, step - 200, 50)
        else:
            lr = lr_max

        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Add Gaussian noise (stddev=0.15) to data
            batch_x = gaussian_noise_layer(batch_x, alpha)
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

    print("Optimization Finished!")

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
