
# coding: utf-8

## Trains a One vs. Rest SVM classifier on the fisher vector video outputs.

import os, sys, collections, random, string
import numpy as np
from tempfile import TemporaryFile
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

class_index_file = "/Users/Bryan/CS/CS_Research/code/CS221/class_index.npz"
class_index_file_loaded = np.load(class_index_file)
class_index = class_index_file_loaded['class_index'][()]
index_class = class_index_file_loaded['index_class'][()]


# In[7]:

training_output = '/Users/Bryan/CS/CS_Research/code/CS221/UCF101_Fishers/train'
testing_output = '/Users/Bryan/CS/CS_Research/code/CS221/UCF101_Fishers/test'

################################################################
# Useful Helper functions                                      #
################################################################


#Transforms a list of videos into a dictionary of video name (in lower case) to list of videos.
def toDict(videos):
    videos_by_class = dict()
    for video in videos:
        #we assume each of the videos has the following format: 
        # v_BasketballDunk_g02_c02.fisher.npz
        name = string.lower(video.split('_')[1])
        if name not in videos_by_class:
            videos_by_class[name] = []
        videos_by_class[name].append(video)
    return videos_by_class


# videos: dictionary of class_name to fisher.npz files
# percentage: percent of the data that is used for testing. default is 20%
# Returns two lists:
#   1. Training data (which will always include the first few entries from each category since it is assumed
#        that these videos helped contruct the gmm)
#   2. Testing data.
def split_inputData(videos, percentage):
    training_data = []
    testing_data = []
    for category in videos:
        num = len(videos[category])
        num_testing = int(round(percentage*num))
        # always add the first two elements to the training data
        training_data += videos[category][0:2]
        remaining_vids = videos[category][2:]
        random.shuffle(remaining_vids)
        testing_data += remaining_vids[0:num_testing]
        training_data += remaining_vids[num_testing:]
    return (training_data, testing_data)



#Returns:
# 1. np.ndarray where of verticallys stacked fisher vectors.
# 2. np.array of class labels
# Inputs:
# videos:  is a list of fisher.npz files
# fisher_path: path to the fisher vector directory
# class_index: dictionary from video name to the class.
def make_FV_matrix(videos, fisher_path, class_index):
    matrix = []
    target = []
    for video in videos:
        vid_path = os.path.join(fisher_path,video)
        matrix.append(np.load(vid_path)['fish'])
        name = string.lower(video.split('_')[1])
        target.append(class_index[name])
    X = np.vstack(matrix)
    Y = np.array(target)
    return (X,Y)


#Given a dictionary of 'Class name' to list of .fisher files,
#Returns
# 1. list of .fisher files of 'K' from each class.
# 2. Class number of each .fisher file
def cut_inputData(vid_dict, K):
    vids = []
    for k,v in vid_dict.iteritems():
        vids.extend(v[:K])
    return vids

#Given a dictionary of 'Class name' to list of .fisher files,
#Returns
# 1. list of .fisher files of 'K' from each class.
# 2. Class number of each .fisher file
def cut_inputData2(vid_dict, K, class_index):
    vids = []
    targets = []
    for k,v in vid_dict.iteritems():
        vids.extend(v[:K])
        targets.extend(K*[class_index[k]])
    return (vids,targets)



#Returns the Mean Average Precision (mAP) to evaluate the performance of a run
#Arguments:
# 1. classifier such as sklearn.multiclass.OneVsRestClassifier
# 2. X_test: data to classify
# 3. Y_test: class labels.
# Returns: (mAP, [aps])
def metric_mAP(classifier, X_test, Y_test, verbose=False):
    estimators = classifier.estimators_
    classes = classifier.classes_
    aps = []
    for estimator,class_num in zip(estimators, classes):
        aps.append(metric_AP(estimator, class_num, X_test, Y_test, verbose=verbose))
    map_val = np.mean(aps)
    if verbose: print "mean AP = %.3f" % map_val
    return map_val

#Average Precision
def metric_AP(estimator, class_num, X_test, Y_test, verbose=False):
    
    scores = estimator.decision_function(X_test)
    #Sorted list of (original_index,score) tuples.
    scores_sorted = sorted(enumerate(scores), key=lambda x:x[1], reverse=True)
    # collect the positive results in the dataset
    positive_ranks = [i for i,score in enumerate(scores_sorted) if Y_test[score[0]]==class_num]
    # accumulate trapezoids with this basis
    recall_step = 1.0 / len(positive_ranks)
    ap = 0
    for ntp,rank in enumerate(positive_ranks):
       # ntp = nb of true positives so far
       # rank = nb of retrieved items so far
       # y-size on left side of trapezoid:
       precision_0 = ntp/float(rank) if rank > 0 else 1.0
       # y-size on right side of trapezoid:
       precision_1 = (ntp + 1) / float(rank + 1)
       ap += (precision_1 + precision_0) * recall_step / 2.0
    if verbose: print "class %d, AP = %.3f" % (class_num, ap)
    return ap

#For a sklearn.multiclass.OneVsRestClassifier, calculate the mAP (mean interpolated average precision),
# accuracy score, and average Precision
def metric_scores(classifier, X_test, Y_test, verbose=False):
    mAP = metric_mAP(classifier, X_test, Y_test,verbose=verbose)
    X_test_predictions = classifier.predict(X_test)
    accuracy_score = metrics.accuracy_score(Y_test, X_test_predictions)
    avg_Precision = metrics.precision_score(Y_test, X_test_predictions, average='macro')
    avg_Recall = metrics.recall_score(Y_test, X_test_predictions, average='macro')
    return (mAP, accuracy_score, avg_Precision, avg_Recall)

#Need to get T training and all testing videos for a limited number 'C' classes.
def limited_input(training_dict, testing_dict, C, T):
    tkeys_overC = [k for k in training_dict.keys() if len(training_dict[k]) >= T]
    sampleClasses = random.sample(tkeys_overC,C)
    
    train_vids = []
    test_vids = []
    for vid_class in sampleClasses:
        train_vids.extend(training_dict[vid_class][:T])
        test_vids.extend(testing_dict[vid_class])
    return (train_vids, test_vids)

#Need to get T training videos for all classes in the input dictionary
def limited_input1(input_dict, T):
    vids = []
    for k,v in input_dict.iteritems():
        if len(v) <= T:
            vids.extend(v)
        else:
            vids.extend(random.sample(v,T))
    return vids



#Helper methods for plotting metrics.
#plot_confusion_matrix(Y_test, X_test_predictions)
def plot_confusion_matrix(y_test, y_pred):# Compute confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    print(cm)

    # Show confusion matrix in a separate window
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


#Returns a PCA matrix.
#Arguments:
# 1. inputX: input data matrix
# 2. n_samples: Number of rows from the inputX matrix to sample to construct PCA matrix.
# 3. pca_dim: Number of PCA components to retain. This will be the reduced feature dimension
# of the input matrix.
# Returns the PCA transform matrix.
# use pca_transform as np.dot(inputX, pca_transform)
#X_train_PCA = np.dot(X_train, pca_transform)
#X_test_PCA = np.dot(X_test, pca_transform)
def train_PCA(inputX, n_samples, pca_dim):
    n_samples = min(n_samples, inputX.shape[0])
    sample_indices = np.random.choice(inputX.shape[0], n_samples)
    sample = inputX[sample_indices]
    mean = sample.mean(axis = 0) #for each row
    sample = sample - mean
    cov = np.dot(sample.T, sample)
    #eigvecs are normalized.
    orig_comps = inputX.shape[1]
    eigvals, eigvecs = np.linalg.eig(cov)
    perm = eigvals.argsort()                   # sort by increasing eigenvalue 
    pca_transform = eigvecs[:, perm[orig_comps-pca_dim:orig_comps]]   # eigenvectors for the 64 last eigenvalues
    return pca_transform


from sklearn import cross_validation
from sklearn.learning_curve import learning_curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt