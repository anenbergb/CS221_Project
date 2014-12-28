
## Trains a One vs. Rest SVM classifier on the fisher vector video outputs.
## Outputs the optimal choice of hyperparameters in the GridSearch_output file

import os, sys, collections, random, string
import numpy as np
from tempfile import TemporaryFile
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
import sklearn.metrics as metrics
from sklearn.decomposition import PCA

import classify_library
class_index_file = "/Users/Bryan/CS/CS_Research/code/CS221/class_index.npz"
class_index_file_loaded = np.load(class_index_file)
class_index = class_index_file_loaded['class_index'][()]
index_class = class_index_file_loaded['index_class'][()]


# In[7]:

training_output = '/Users/Bryan/CS/CS_Research/code/CS221/UCF101_Fishers/train'
testing_output = '/Users/Bryan/CS/CS_Research/code/CS221/UCF101_Fishers/test'

training = [filename for filename in os.listdir(training_output) if filename.endswith('.fisher.npz')]
testing = [filename for filename in os.listdir(testing_output) if filename.endswith('.fisher.npz')]


training_dict = classify_library.toDict(training)
testing_dict = classify_library.toDict(testing)

####################################################################
####################################################################
################################## Script starts




X_train_vids = classify_library.limited_input1(training_dict, 1000)
X_test_vids = classify_library.limited_input1(testing_dict, 1000)


#GET THE TRAINING AND TESTING DATA.
X_train, Y_train = classify_library.make_FV_matrix(X_train_vids,training_output, class_index)
X_test, Y_test = classify_library.make_FV_matrix(X_test_vids,testing_output, class_index)

#PCA reduction
training_PCA = classify_library.limited_input1(training_dict,40)
X_PCA, _ = classify_library.make_FV_matrix(training_PCA,training_output, class_index)

n_components = 1000
pca = PCA(n_components=n_components)
pca.fit(X_PCA)
X_train_PCA = pca.transform(X_train)
X_test_PCA = pca.transform(X_test)

#Exhaustive Grid Search

C = [1, 10, 50, 100, 1000]
loss = ['l1', 'l2']
penalty = ['l2']
kernel = ['poly', 'rbf', 'sigmoid']
gamma = [0.01, 0.001, 0.0001]

#Optimize the linear classifier first

f = open('./GridSearch_output', 'w')
f.write('PCA components: %d\n' % (n_components))
write2 = 'X_train: %d, X_test: %d\n' % (X_train_PCA.shape[0], X_test_PCA.shape[0])
f.write(write2)
f.write("Scores: mAP, accuracy_score, avg_Precision, avg_Recall\n\n")

for c in C:
    #Optimize the linear kernel first
    for lo in loss:
        for pen in penalty:
            classifier = OneVsRestClassifier(LinearSVC(random_state=0, C=c, loss=lo, penalty=pen)).fit(X_train_PCA, Y_train)
            Scores = classify_library.metric_scores(classifier, X_test_PCA, Y_test)
            setting = "Settings: Linear SVM, C: %d, loss: %s, penalty: %s\n" % (c,lo,pen)
            score = "Scores: (%f, %f, %f, %f)\n" % (Scores[0], Scores[1], Scores[2], Scores[3])
            f.write(setting)
            f.write(score)
            f.write('\n')
    #Optimize the non-linear kernels
    for ker in kernel:
        for gam in gamma:
            classifier = OneVsRestClassifier(svm.SVC(random_state=0, C=c, kernel=ker, gamma=gam)).fit(X_train_PCA, Y_train)
            Scores = classify_library.metric_scores(classifier, X_test_PCA, Y_test)
            setting = "Settings: SVM kernel: %s, C: %d, gamma: %f\n" % (ker,c,gam)
            score = "Scores: (%f, %f, %f, %f)\n" % (Scores[0], Scores[1], Scores[2], Scores[3])
            f.write(setting)
            f.write(score)
            f.write('\n')
            f.write('\n')

f.close()




