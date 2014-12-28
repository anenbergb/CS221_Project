
# coding: utf-8

## This file implements a basic linear classifier approach to detecting action categories.

# In[26]:

import cv2, ffmpeg, subprocess
import numpy as np
import os, sys, collections, random, string
from sklearn.decomposition import PCA
from scipy import ndimage
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
import sklearn.metrics as metrics
#Personally made libraries
import classify_library
from tempfile import TemporaryFile

get_ipython().magic(u'matplotlib inline')


# In[4]:

orig_dir = "/Users/Bryan/CS/CS_Research/data/UCF101"
#playing a video saved in disk
video_pwd = os.path.join(orig_dir,"v_ApplyEyeMakeup_g01_c03.avi")

tmp_frames = "/Users/Bryan/CS/CS_Research/code/CS221/tmp_frames"


ucf101_path = "/Users/Bryan/CS/CS_Research/data/UCF101"
trainlists = '/Users/Bryan/CS/CS_Research/data/class_attributes_UCF101/ucfTrainTestlist/'
trainlist01 = os.path.join(trainlists, 'trainlist01.txt')
testlist01 = os.path.join(trainlists, 'testlist01.txt')
training_output = '/Users/Bryan/CS/CS_Research/code/CS221/tmp_frames/train'
testing_output = '/Users/Bryan/CS/CS_Research/code/CS221/tmp_frames/test'



# In[ ]:

def extract_frames(vidlist,vidDir,outputDir):
    f = open(vidlist, 'r')
    vids = f.readlines()
    f.close()
    vids = [video.rstrip() for video in vids]
    vids = [line.split()[0].split('/')[1] for line in vids] 
    for vid in vids:
        videoName = os.path.join(vidDir,vid)
        frameName = os.path.join(outputDir, vid.split('.')[0]+".jpeg")
        ffmpeg.extract_frame(videoName,frameName)
extract_frames(trainlist01, ucf101_path,training_output)
extract_frames(testlist01, ucf101_path,testing_output)


## Now lets we must construct training and testing matrices. We will also reduce the dimensions of the videos by projecting into the PCA basis.

# In[15]:

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

#Returns:
# 1. np.ndarray where of verticallys stacked fisher vectors.
# 2. np.array of class labels
# Inputs:
# videos:  is a list of video jpg files
# fisher_path: path to the fisher vector directory
# class_index: dictionary from video name to the class.
def make_frame_matrix(videos, vid_dir, class_index):
    matrix = []
    target = []
    for video in videos:
        vid_path = os.path.join(vid_dir,video)
        im = rgb2gray(ndimage.imread(vid_path))
        imVec = np.array(im.flatten())
        matrix.append(imVec)
        name = string.lower(video.split('_')[1])
        target.append(class_index[name])
    X = np.vstack(matrix)
    Y = np.array(target)
    return (X,Y)



# In[16]:

class_index_file = "./class_index.npz"
class_index_file_loaded = np.load(class_index_file)
class_index = class_index_file_loaded['class_index'][()]
index_class = class_index_file_loaded['index_class'][()]

training = [filename for filename in os.listdir(training_output) if filename.endswith('.jpeg')]
testing = [filename for filename in os.listdir(testing_output) if filename.endswith('.jpeg')]

training_dict = classify_library.toDict(training)
training_PCA = classify_library.limited_input1(training_dict,1)

X_train, Y_train = make_frame_matrix(training,training_output,class_index)
X_test, Y_test = make_frame_matrix(testing,testing_output,class_index)


### Reduced PCA dimension to 1000

# In[18]:

X_PCA, _ = make_frame_matrix(training_PCA, training_output, class_index)
pca = PCA(n_components=1000)
pca.fit(X_PCA)
X_train_PCA = pca.transform(X_train)
X_test_PCA = pca.transform(X_test)


# In[22]:

classifier = OneVsRestClassifier(LinearSVC(random_state=0, C=1, loss='l2', penalty='l2')).fit(X_train_PCA, Y_train)
classify_library.metric_scores(classifier, X_test_PCA, Y_test, verbose=True)


# In[27]:

baseline_file = "./baseline"
np.savez(baseline_file, X_train_PCA=X_train_PCA, X_test_PCA=X_test_PCA, Y_train=Y_train, Y_test=Y_test)


# In[ ]:



