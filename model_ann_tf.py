import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from preprocessing import preprocess
from pathlib import Path
import shutil
import time

start = time.time()

# Delete old data and event summaries
dir1 = 'tf_model_data/ann/'
path1 = Path(dir1)
dir2 = 'tensorboard/ann/'
path2 = Path(dir2)
if path1.is_dir():
    shutil.rmtree(dir1)
if path2.is_dir():
    shutil.rmtree(dir2)

# Load in data
X_train, X_cv, _, y_train, y_cv, _ = preprocess()
data_size = len(X_train)

# One-hot encode labels
y_train = np_utils.to_categorical(y_train)
y_cv = np_utils.to_categorical(y_cv)

# Hyper Parameters
learning_rate = 0.0001
beta1 = 0.9
beta2 = 0.999
keep_prob = 0.5
beta = 0.01
epochs = 100
display_step = 1

# Split data and labels into mini batches
batch_size = 100
total_batch = len(X_train) // batch_size
X_batches = np.array_split(X_train, total_batch)
Y_batches = np.array_split(y_train, total_batch)

# Network Parameters
n_hidden_1 = 512  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
num_input = X_train.shape[1]  # number of features
num_classes = y_train.shape[1]  # number of classes

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'w1': tf.Variable(tf.random_normal([num_input, n_hidden_1], mean=0.0, stddev=0.05), name='W_1'),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], mean=0.0, stddev=0.05), name='W_2'),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]), name='W_OUT')
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='B_1'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='B_2'),
    'out': tf.Variable(tf.random_normal([num_classes]), name='B_OUT')
}

# Add histogram summaries for weights
tf.summary.histogram('h_w1_sum', weights['w1'])
tf.summary.histogram('h_w2_sum', weights['w2'])
tf.summary.histogram('h_wout_sum', weights['out'])

# Add histogram summaries for biases
tf.summary.histogram('h_b1_summ', biases['b1'])
tf.summary.histogram('h_b2_summ', biases['b2'])
tf.summary.histogram('h_bout_summ', biases['out'])


def batch_norm(x, epsilon=0.001):
    shape = x.shape
    last = len(shape) - 1
    axis = list(range(last))
    mean, var = tf.nn.moments(x, axis)
    scale = tf.Variable(tf.ones([shape[last]]))
    offset = tf.Variable(tf.zeros([shape[last]]))
    return tf.nn.batch_normalization(x, mean, var, offset, scale, epsilon)


# Create model
def neural_net_model(x):
    # Hidden fully connected layer with 512 neurons, Relu activation, batch normalization and dropout
    z1 = tf.nn.relu(tf.add(tf.matmul(x, weights['w1']), biases['b1']))
    bn1 = batch_norm(z1)
    layer_1 = tf.nn.dropout(bn1, keep_prob=keep_prob)
    # Hidden fully connected layer with 256 neurons, Relu activation, batch normalization and dropout
    z2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['w2']), biases['b2']))
    bn2 = batch_norm(z2)
    layer_2 = tf.nn.dropout(bn2, keep_prob=keep_prob)
    # Output fully connected layer with 1 neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net_model(X)
prediction = tf.nn.softmax(logits)


def l2_regularization(W, B):
    result = 0.
    for w in W:
        result += tf.nn.l2_loss(W[w])
    for b in B:
        result += tf.nn.l2_loss(B[b])
    return result

with tf.name_scope("loss"):
    # Define loss and regularization
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                     labels=Y))
    loss_op = tf.reduce_mean(loss_op + beta * l2_regularization(weights, biases))
    # Add scalar summary for cost tensor
    tf.summary.scalar("loss", loss_op)

# Define optimization op
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   beta1=beta1,
                                   beta2=beta2)
train_op = optimizer.minimize(loss_op)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Add scalar summary for accuracy tensor
    tf.summary.scalar("accuracy", accuracy)

# Merge all the summaries and write them out to corresponding directory
merged = tf.summary.merge_all()

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

model_path = 'tf_model_data/ann/model_ann.ckpt'

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # op to write logs to Tensorboard
    train_writer = tf.summary.FileWriter("tensorboard/ann/train")
    train_writer.add_graph(sess.graph)

    # Training cycle
    for epoch in range(1, epochs + 1):

        # Reset loss and accuracy for current training step
        avg_loss = 0.
        avg_acc = 0.

        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Run optimization op (backprop), loss op and accuracy op
            _, loss, acc = sess.run([train_op, loss_op, accuracy], feed_dict={X: batch_x,
                                                                              Y: batch_y})
            # Compute average loss and accuracy
            avg_loss += loss / total_batch
            avg_acc += acc / total_batch

        # Display logs per epoch step
        if epoch % display_step == 0 or epoch == 1:
            print("Epoch " + str(epoch) +
                  ", Loss=" + "{:.4f}".format(avg_loss) +
                  ", Accuracy=" + "{:.4f}".format(avg_acc))

        # Write summary after each training epoch
        train_summary = sess.run(merged, feed_dict={X: X_train,
                                                    Y: y_train})
        train_writer.add_summary(train_summary, epoch)

    # Close writers
    train_writer.close()

    # Stop timer
    end = time.time()
    print('\nOptimization Finished!'
          '\nTotal training time: {:.2f} {}'.format((end - start) / 60, 'minutes'))

    # Calculate loss and accuracy for CV set
    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: X_cv,
                                                         Y: y_cv})
    print("Validation Loss: " + "{:.4f}".format(loss)
          + ", Validation Accuracy: " + "{:.4f}".format(acc))

    # Save model data
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)
