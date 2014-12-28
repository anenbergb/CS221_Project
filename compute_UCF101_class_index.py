import numpy as np
import subprocess, os, string, sys
from tempfile import TemporaryFile
import argparse


"""
class_index:
Computes a map from class name to class index.
dict{ class_name: class_index}

index_class:
Opposite mapping from the class_index map.
dict{ class_index: class_name}

Example usage:
python compute_UCF101_class_index.py /Users/Bryan/CS/CS_Research/data/class_attributes_UCF101/Class_Index.txt
"""


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("class_index", help="File of class index", type=str)
  parser.add_argument("class_out", help="File of saved class index map", type=str)

  args = parser.parse_args()

  try:
    f = open(args.class_index, 'r')
    videos = f.readlines()
    f.close()
    videos = [video.strip() for video in videos]
    class_index = {}
    index_class = {}
    for video in videos:
      video_s = video.split()
      class_index[string.lower(str(video_s[1]))] = int(video_s[0])
      index_class[int(video_s[0])] = string.lower(str(video_s[1]))

    np.savez(args.class_out, class_index=class_index, index_class=index_class)
  except IOError:
    sys.exit(0)