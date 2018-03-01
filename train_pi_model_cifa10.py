import numpy as np
import tensorflow as tf
from keras.utils import np_utils
import cPickle
import random
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
w_max = 100
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
weight_t = tf.placeholder(tf.float32, name='weight_t')


# Create some wrappers for simplicity
def gaussian_noise_layer(x, std):
    noise = np.random.normal(0.0, std, (x.shape[0], x.shape[1]))
    return x + noise


def gaussian(amplitude, scale, t, scope):
    return amplitude * np.exp(-scale * (1 - t / scope) ** 2)


def random_translate(images):
    # randomly generate an operation flag
    flag = random.randrange(0, 3)
    if flag == 0:
        return tf.map_fn(lambda img: tf.image.random_flip_left_right(img), images)
    elif flag == 1:
        return tf.map_fn(lambda img: tf.image.random_flip_up_down(img), images)
    else:
        return tf.map_fn(lambda img: tf.image.rot90(img), images)


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

    # # Apply random translation to input tensor
    # x = random_translate(x)

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

# Construct two parallel and identical computation graphs
z = conv_net(X, weights, biases, keep_prob)
z_ = conv_net(X, weights, biases, keep_prob)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z,
                                                                 labels=Y))
loss_op = tf.reduce_mean(loss_op + weight_t * tf.losses.mean_squared_error(predictions=z_,
                                                                           labels=z))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   beta1=beta1,
                                   beta2=beta2)
train_op = optimizer.minimize(loss_op)

# Evaluate model
prediction = tf.nn.softmax(z)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

model_path = 'tf_model_data/pi_model_cifar10/pi_model_cifar10.ckpt'

# Add ops to save all the variables.
saver = tf.train.Saver()

# Start training
with tf.Session() as sess:

    # Restore variables from disk.
    saver.restore(sess, model_path)

    # Initialize learning_rate and unsupervised weight
    lr = 0.
    w_t = 0.

    for step in range(1, num_steps + 1):

        # Reset loss and accuracy for current training step
        avg_loss = 0.
        avg_acc = 0.

        # Ramp up/down learning rate and unsupervised weight
        if step <= 80:
            lr = gaussian(lr_max, 5, step, 80)
            w_t = gaussian(w_max, 5, step, 80)
        elif step > 250:
            lr = gaussian(lr_max, 12.5, step - 200, 50)
            w_t = w_max
        else:
            lr = lr_max
            w_t = w_max

        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Add Gaussian noise (stddev=0.15) to data
            batch_x = gaussian_noise_layer(batch_x, alpha)
            # Run optimization op (backprop), loss op and accuracy op
            _, loss, acc = sess.run([train_op, loss_op, accuracy], feed_dict={X: batch_x,
                                                                              Y: batch_y,
                                                                              keep_prob: 0.5,
                                                                              learning_rate: lr,
                                                                              weight_t: w_t})
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
                                                         keep_prob: 1.0,
                                                         weight_t: w_t})
    print("Testing Loss: " + "{:.4f}".format(loss)
          + ", Testing Accuracy: " + "{:.4f}".format(acc))
