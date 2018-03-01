import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from preprocessing import preprocess
from sklearn import metrics
import itertools
from sklearn.preprocessing import label_binarize
from itertools import cycle
from scipy import interp

# load in data
_, _, X_test, _, _, y_test = preprocess()

# number of classes
num_classes = 2

# One-hot encode labels
y_test_enc = label_binarize(y_test, range(num_classes + 1))[:, :-1]

# Parameters
model_path = './model_data/model_ann_tf.ckpt'

# Network Parameters
n_hidden_1 = 512  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
num_input = X_test.shape[1]  # number of features
epsilon = 0.001
keep_prob = 1

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


def batch_norm(x, epsilon=0.001):
    shape = x.shape
    last = len(shape) - 1
    axis = list(range(last))
    mean, var = tf.nn.moments(x, axis)
    scale = tf.Variable(tf.ones([shape[last]]))
    offset = tf.Variable(tf.zeros([shape[last]]))
    return tf.nn.batch_normalization(x, mean, var, offset, scale, epsilon)


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


# Construct cascaded
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
    print("Cost:", cost.eval({X: X_test, Y: y_test_enc}))
    print("Accuracy:", accuracy.eval({X: X_test, Y: y_test_enc}))
    pred_test = tf.argmax(prediction, 1).eval(feed_dict={X: X_test})
    pred_prob_test = sess.run(tf.nn.softmax(prediction), feed_dict={X: X_test})


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
        print('\nConfusion matrix')

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
np.set_printoptions(precision=6)

# Plot non-normalized confusion matrix
plt.figure()
class_names = ['0', '1']
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
plt.savefig('cm_plot.png')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.savefig('cm_plot_norm.png')

# ------------------ ROC ------------------
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_test_enc[:, i], pred_prob_test[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test_enc.ravel(), pred_prob_test.ravel())
roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= num_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

# colors = cycle(['aqua', 'darkorange'])
# for i, color in zip(range(num_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(i+1, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristics of all classes')
plt.legend(loc="lower right")
plt.savefig('roc.png')
