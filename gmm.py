"""
Can execute this as a script to populate the GMM or load it as a module

The IDTF features are temporarily saved at the GMM_dir
Pca reduction on each descriptor is set to false by default.
"""
import computeIDTF, IDT_feature, computeFV
import numpy as np
import sys, os, random
from yael import ynumpy
from tempfile import TemporaryFile
import argparse

GMM_dir = "./GMM_IDTFs"


def populate_gmms(sample_vids, GMM_OUT, k_gmm, sample_size=1500000, PCA=False):
    """
    sample_size is the number of IDTFs that we sample from the total_lines number of IDTFs
    that were computed previously.

    GMM_OUT is the output file to save the list of GMMs.
    Saves the GMMs in the GMM_OUT file as the gmm_list attribute.

    Returns the list of gmms.
    """
    #total_lines = 2488317
    total_lines = total_IDTF_lines()
    print total_lines
    sample_size = min(total_lines,sample_size)
    sample_indices = random.sample(xrange(total_lines),sample_size)
    sample_indices.sort()

    sample_descriptors = IDT_feature.list_descriptors_sampled(GMM_dir, sample_vids, sample_indices)
    bm_list = IDT_feature.bm_descriptors(sample_descriptors)
    #Construct gmm models for each of the different descriptor types.
    
    gmm_list = [gmm_model(bm, k_gmm, PCA=PCA) for bm in bm_list]
    np.savez(GMM_OUT, gmm_list=gmm_list)
    
    return gmm_list


def gmm_model(sample, k_gmm, PCA=False):
    """
    Returns a tuple: (gmm,mean,pca_transform)
    gmm is the ynumpy gmm model fro the sample data. 
    pca_tranform is None if PCA is True.
    Reduces the dimensions of the sample (by 50%) if PCA is true
    """

    print "Building GMM model"
    # until now sample was in uint8. Convert to float32
    sample = sample.astype('float32')
    # compute mean and covariance matrix for the PCA
    mean = sample.mean(axis = 0) #for each row
    sample = sample - mean
    pca_transform = None
    if PCA:
        cov = np.dot(sample.T, sample)

        #decide to keep 1/2 of the original components, so vid_trajs_bm.shape[1]/2
        #compute PCA matrix and keep only 1/2 of the dimensions.
        orig_comps = sample.shape[1]
        pca_dim = orig_comps/2
        #eigvecs are normalized.
        eigvals, eigvecs = np.linalg.eig(cov)
        perm = eigvals.argsort() # sort by increasing eigenvalue 
        pca_transform = eigvecs[:, perm[orig_comps-pca_dim:orig_comps]]   # eigenvectors for the 64 last eigenvalues
        # transform sample with PCA (note that numpy imposes line-vectors,
        # so we right-multiply the vectors)
        sample = np.dot(sample, pca_transform)
    # train GMM
    gmm = ynumpy.gmm_learn(sample, k_gmm)
    toReturn = (gmm,mean,pca_transform)
    return toReturn

def total_IDTF_lines():
    """
    Returns the total number of IDTFs (features) computed
    for all of the videos. Each line in a .feature file is an IDTF, so this
    is the total number of lines in the GMM_dir
    """ 
    videos = [filename for filename in os.listdir(GMM_dir) if filename.endswith('.features')]
    total_lines = sum([sum(1 for line in open(os.path.join(GMM_dir, vid))) for vid in videos])
    return total_lines


def computeIDTFs(training_samples, VID_DIR):
    """
    Computes the IDTFs specifically used for constructing the GMM
    training_samples is a list of videos located at the VID_DIR directory.
    The IDT features are output in the GMM_dir.
    """
    for video in training_samples:
        videoLocation = os.path.join(VID_DIR,video)
        featureOutput = os.path.join(GMM_dir,os.path.basename(video)[:-4]+".features")
        print "Computing IDTF for %s" % (video)
        computeIDTF.extract(videoLocation, featureOutput)
        print "complete."  


def sampleVids(vid_list):
    """
    vid_list is a text file of video names and their corresponding
    class.
    This function reads the video names and creates a list with one video
    from each class.
    """
    f = open(vid_list, 'r')
    videos = f.readlines()
    f.close()
    videos = [video.rstrip() for video in videos]
    vid_dict = {}
    for line in videos:
        l = line.split()
        key = int(l[1])
        if key not in vid_dict:
            vid_dict[key] = []
        vid_dict[key].append(l[0].split('/')[1])
    
    samples = []
    for k,v in vid_dict.iteritems():
        samples.extend(v[:1])
    return samples

#python gmm.py 120 UCF101_dir train_list
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("k_gmm", help="number of GMM modes", type=int)
    parser.add_argument("videos", help="Directory of the input videos", type=str)
    parser.add_argument("input_list", help="List of input videos from which to sample", type=str)
    parser.add_argument("gmm_out", help="Output file to save the list of gmms", type=str)

   # parser.add_argument("-p", "--pca", type=float, help="percent of original descriptor components to retain after PCA")
    parser.add_argument("-p", "--pca", action="store_true",
        help="Reduce each descriptor dimension by 50 percent using PCA")
    args = parser.parse_args()

    print args.k_gmm
    print args.videos
    print args.input_list
    print args.gmm_out

    VID_DIR = args.videos
    input_list = args.input_list

    vid_samples = sampleVids(input_list)

    #computeIDTFs(vid_samples, VID_DIR)
    features = []
    for vid in vid_samples:
        features.append(vid[:-4]+".features")
    populate_gmms(features,args.gmm_out,args.k_gmm,PCA=args.pca)

