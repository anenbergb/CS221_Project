import os, sys, collections
import numpy as np
from yael import ynumpy
import IDT_feature
from tempfile import TemporaryFile
import argparse
import computeFV


"""
computes a Fisher vector given an input stream of IDTFs

Usage:
	stream_of_IDTFs | python computeFVstream.py fisher_path gmm_list
"""


#The input is a stream of IDTFs associated with a single video.
if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("fisher_path", help="File to save the output Fisher Vector", type=str)
   parser.add_argument("gmm_list", help="File of saved list of GMMs", type=str)
   args = parser.parse_args()
   gmm_list = np.load(args.gmm_list+".npz")['gmm_list']
   points = [] # a list of IDT features.
   for line in sys.stdin:
      points.append(IDT_feature.IDTFeature(line))
   video_desc = IDT_feature.vid_descriptors(points)
   computeFV.create_fisher_vector(gmm_list, video_desc, args.fisher_path)




