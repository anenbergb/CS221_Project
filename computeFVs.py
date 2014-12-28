import argparse
import computeIDTF, os, subprocess, ThreadPool
import classify_library

"""
Uses multi-threading to extract IDTFs and compute the Fisher Vectors (FVs) for
each of the videos in the input list (vid_in). The Fisher Vectors are output
in the output_dir
"""


#This is is the function that each worker will compute.
def processVideo(vid,vid_path,output_dir,gmm_list):
    """
    gmm_list is the file of the saved list of GMMs
    """
    videoLocation = os.path.join(vid_path,vid)
    outputName = os.path.join(output_dir, vid.split('.')[0]+".fisher")
    computeIDTF.extractFV(videoLocation, outputName,gmm_listf)


#python computeFVs.py videos vid_in vid_out
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("vid_path", help="Directory of the input videos", type=str)
    parser.add_argument("vid_in", help="list of input videos in .txt file", type=str)
    parser.add_argument("output_dir", help="output directory to save FVs (.fisher files)", type=str)
    parser.add_argument("gmm_list", help="File of saved list of GMMs", type=str)

    args = parser.parse_args()

    f = open(args.vid_in, 'r')
    input_videos = f.readlines()
    f.close()
    input_videos = [line.split()[0].split('/')[1] for line in [video.rstrip() for video in input_videos]]
    
    ###Just to prevent overwriting already processed vids
    completed_vids = [filename.split('.')[0] for filename in os.listdir(args.output_dir) if filename.endswith('.npz')]
    overlap = [vid for vid in input_videos if vid.split('.')[0] in completed_vids]
    


    #Multi-threaded FV construction.
    numThreads = 10
    pool = ThreadPool.ThreadPool(numThreads)
    for vid in input_videos:
        if vid not in overlap:
            pool.add_task(processVideo,vid,args.vid_path,args.output_dir,args.gmm_list)
    pool.wait_completion()
