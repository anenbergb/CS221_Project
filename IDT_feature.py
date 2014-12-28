#
# Handles a improved trajectory feature point
# Check http://lear.inrialpes.fr/people/wang/improved_trajectories
#

import numpy as np
import os
TRAJ_DIM=30
HOG_DIM=96
HOF_DIM=108
MBHX_DIM=96
MBHY_DIM=96

#Represents a single IDTF feature
class IDTFeature(object):
    def __init__(self, line):
        ll = line.strip().split()
        self.frameNum = int(ll[0])
        self.mean_x = float(ll[1])
        self.mean_y = float(ll[2])
        self.var_x = float(ll[3])
        self.var_y = float(ll[4])
        self.length = float(ll[5])
        self.scale = float(ll[6])
        self.x_pos = float(ll[7])
        self.y_pos = float(ll[8])
        self.t_pos = float(ll[9])
        traj_start =10
        hog_start = traj_start + TRAJ_DIM
        hof_start = hog_start + HOG_DIM
        mbhx_start = hof_start + HOF_DIM
        mbhy_start = mbhx_start + MBHX_DIM
        mbhy_end = mbhy_start + MBHY_DIM
        self.traj = [float(l) for l in ll[traj_start:hog_start]]
        self.hog = [float(l) for l in ll[hog_start:hof_start]]
        self.hof = [float(l) for l in ll[hof_start:mbhx_start]]
        self.mbhx = [float(l) for l in ll[mbhx_start:mbhy_start]]
        self.mbhy = [float(l) for l in ll[mbhy_start:mbhy_end]]


#Populates an np.ndarray for each of the descriptors in an IDTF feature
# traj, hog, hof, mbhx, mbhy
class vid_descriptors(object):
	#input is a list of IDTFs objects (as specified by the above IDTFeature class) which represent the features of a video.
	def __init__(self, IDTFeatures):

		trajs = []
		hogs = []
		hofs = []
		mbhxs = []
		mbhys = []
		for feature in IDTFeatures:
			trajs.append(np.ndarray(shape=(1,TRAJ_DIM), buffer=np.array(feature.traj),dtype=float))
			hogs.append(np.ndarray(shape=(1,HOG_DIM), buffer=np.array(feature.hog),dtype=float))
			hofs.append(np.ndarray(shape=(1,HOF_DIM), buffer=np.array(feature.hof),dtype=float))
			mbhxs.append(np.ndarray(shape=(1,MBHX_DIM), buffer=np.array(feature.mbhx),dtype=float))
			mbhys.append(np.ndarray(shape=(1,MBHY_DIM), buffer=np.array(feature.mbhy),dtype=float))

		self.traj = np.vstack(trajs)
		self.hog = np.vstack(hogs)
		self.hof = np.vstack(hofs)
		self.mbhx = np.vstack(mbhxs)
		self.mbhy = np.vstack(mbhys)


################################################################
# Useful Helper functions                                      #
################################################################



#Returns a tuple (trajs, hogs, hofs, mbhxs, mbhys) where each element is a list of of np.ndarray of type of descriptor.
#Each np.ndarray in the list is a matrix concatenating together all of the descriptors of that particular type for a
#given video.
# So the length of each list will be the number of videos names provided in the vid_features input list
#
# directory: Directory where the input videos are located
# vid_features: a list of names of videos.
def populate_descriptors(directory, vid_features):
  vid_trajs = []
  vid_hogs = []
  vid_hofs = []
  vid_mbhxs = []
  vid_mbhys = []
  for vid_feature in vid_features:
    vid_desc = vid_descriptors(read_IDTF_file(directory,vid_feature))
    vid_trajs.append(vid_desc.traj)
    vid_hogs.append(vid_desc.hog)
    vid_hofs.append(vid_desc.hof)
    vid_mbhxs.append(vid_desc.mbhx)
    vid_mbhys.append(vid_desc.mbhy)
  return (vid_trajs, vid_hogs, vid_hofs, vid_mbhxs, vid_mbhys)


#returns a list of vid_descriptors objects.
def list_descriptors(directory, vid_features):
    vid_descs = []
    for vid_feature in vid_features:
      vid_descs.append(vid_descriptors(read_IDTF_file(directory,vid_feature)))
    return vid_descs

#returns a list of vid_descriptors objects, at the specified indices
def list_descriptors_sampled(directory, vid_features, validIndices):
    vid_descs = []
    current_line = 0
    VI_index = 0 #Index in the validIndices sorted list

    def read_file(vid, CL, VI):
        points = []
        with open(os.path.join(directory,vid), 'r') as f:
            for line in f:
                if VI < len(validIndices):
                    if CL == validIndices[VI]:
                        points.append(IDTFeature(line))
                        VI+=1
                CL += 1
        toReturn = (points, CL, VI)
        return toReturn

    for vid_feature in vid_features:
        print vid_feature
        points, current_line, VI_index = read_file(vid_feature, current_line, VI_index)
        vid_descs.append(vid_descriptors(points))
    return vid_descs


#Provided a list of vid_descriptors objects, concatenates the
# each np.ndarray descriptor type (e.g. each traj, hogs, hofs, ...)
# into a large np.ndarray matrix.
# Returns a list of 5 large np.ndarray matrices.
def bm_descriptors(descriptors_list):
    vid_trajs = []
    vid_hogs = []
    vid_hofs = []
    vid_mbhxs = []
    vid_mbhys = []
    for desc in descriptors_list:
        vid_trajs.append(desc.traj)
        vid_hogs.append(desc.hog)
        vid_hofs.append(desc.hof)
        vid_mbhxs.append(desc.mbhx)
        vid_mbhys.append(desc.mbhy)
    #make each of the descriptor lists into a big matrix.
    bm_list = []
    #The indices of the elements in the list are as follows:
    # bm_list[0] >>> trajs
    # bm_list[1] >>> hogs
    # bm_list[2] >>> hofs
    # bm_list[3] >>> mbhxs
    # bm_list[4] >>> mbhxs
    bm_list.append(np.vstack(vid_trajs))
    bm_list.append(np.vstack(vid_hogs))
    bm_list.append(np.vstack(vid_hofs))
    bm_list.append(np.vstack(vid_mbhxs))
    bm_list.append(np.vstack(vid_mbhys))
    return bm_list



# Parses a video's IDTF file and returns a set of points where each point is
# is an IDTF feature.
def read_IDTF_file(directory,vid_feature):
    points = []
    with open(os.path.join(directory,vid_feature), 'r') as f:
      #counter = 0
      for line in f:
        #if counter == 20: break
        points.append(IDTFeature(line))
        #counter += 1
    return points
