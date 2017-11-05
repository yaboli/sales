import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from preprocess_training_set import preprocess
from sklearn import metrics
import itertools
from sklearn.preprocessing import label_binarize

# load in data
_, X_cv, X_test, _, y_cv, y_test = preprocess()

# number of classes
num_classes = 2

# One-hot encode labels
y_cv_enc = label_binarize(y_cv, range(num_classes + 1))[:, :-1]
y_test_enc = label_binarize(y_test, range(num_classes + 1))[:, :-1]

# Parameters
model_path = './model_data/model_ann_tf.ckpt'

# Network Parameters
n_hidden_1 = 512  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
num_input = 33  # number of features

tf.reset_default_graph()
# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.get_variable("W_1", shape=[num_input, n_hidden_1]),
    'h2': tf.get_variable("W_2", shape=[n_hidden_1, n_hidden_2]),
    'out': tf.get_variable("W_OUT", shape=[n_hidden_2, num_classes])
}
biases = {
    'b1': tf.get_variable("B_1", shape=[n_hidden_1]),
    'b2':  tf.get_variable("B_2", shape=[n_hidden_2]),
    'out':  tf.get_variable("B_OUT", shape=[num_classes])
}


def neural_net_model(x):
    # Hidden fully connected layer with 512 neurons, Relu activation
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # Hidden fully connected layer with 256 neurons, Relu activation
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    # Output fully connected layer with 1 neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# Construct model
prediction = neural_net_model(X)

# Define loss op
cost = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y)))

# Define accuracy op
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Add ops to restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, model_path)
    # Evaluate costs and accuracies
    print("\n------------- Model report -------------")
    print("Cost (validation):", cost.eval({X: X_cv, Y: y_cv_enc}))
    print("Accuracy (validation):", accuracy.eval({X: X_cv, Y: y_cv_enc}))
    print("Cost (test):", cost.eval({X: X_test, Y: y_test_enc}))
    print("Accuracy (test):", accuracy.eval({X: X_test, Y: y_test_enc}))
    pred_test = tf.argmax(prediction, 1).eval(feed_dict={X: X_test})


# ------------------ Compute and plot Confusion Matrix ------------------
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("\nNormalized confusion matrix")
    else:
        print('\nConfusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, pred_test)
np.set_printoptions(precision=4)

# Plot non-normalized confusion matrix
plt.figure()
class_names = ['0', '1']
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.savefig('confusion_mtrx_plot.png')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.savefig('confusion_mtrx_plot_normalized.png')
