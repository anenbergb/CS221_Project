
"""
Script to train a basic action classification system.

Trains a One vs. Rest SVM classifier on the fisher vector video outputs.
This script is used to experimentally test different parameter settings for the SVMs.

"""

import os, sys, collections, random, string
import numpy as np
from tempfile import TemporaryFile
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
import sklearn.metrics as metrics
import classify_library
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class_index_file = "./class_index.npz"
training_output = './UCF101_Fishers/train'
testing_output = './UCF101_Fishers/test'

class_index_file_loaded = np.load(class_index_file)
class_index = class_index_file_loaded['class_index'][()]
index_class = class_index_file_loaded['index_class'][()]


# In[7]:


training = [filename for filename in os.listdir(training_output) if filename.endswith('.fisher.npz')]
testing = [filename for filename in os.listdir(testing_output) if filename.endswith('.fisher.npz')]


training_dict = classify_library.toDict(training)
testing_dict = classify_library.toDict(testing)


#GET THE TRAINING AND TESTING DATA.


X_train_vids, X_test_vids = classify_library.limited_input(training_dict, testing_dict, 101, 24)
X_train, Y_train = classify_library.make_FV_matrix(X_train_vids,training_output, class_index)
X_test, Y_test = classify_library.make_FV_matrix(X_test_vids,testing_output, class_index)

training_PCA = classify_library.limited_input1(training_dict,1)



#Experiments with PCA
pca_dim = 1000
pca = PCA(n_components=pca_dim)
pca.fit(X_train)
X_train_PCA = pca.transform(X_train)
X_test_PCA = pca.transform(X_test)
estimator = OneVsRestClassifier(LinearSVC(random_state=0, C=100, loss='l1', penalty='l2'))
classifier = estimator.fit(X_train_PCA, Y_train)
metrics = classify_library.metric_scores(classifier, X_test_PCA, Y_test, verbose=True)
print metrics


do_learning_curve = False
if do_learning_curve:
    X_full = np.vstack([X_train_PCA, X_test_PCA])
    Y_full = np.hstack([Y_train, Y_test])
    title= "Learning Curves (Linear SVM, C: %d, loss: %s, penalty: %s, PCA dim: %d)" % (100,'l1','l2',pca_dim)
    cv = cross_validation.ShuffleSplit(X_full.shape[0], n_iter=4,test_size=0.2, random_state=0)
    estimator = OneVsRestClassifier(LinearSVC(random_state=0, C=100, loss='l1', penalty='l2'))
    plot_learning_curve(estimator, title, X_full, Y_full, (0.7, 1.01), cv=cv, n_jobs=1)
    plt.show()

