
import cv2
import mediapipe as mp
import numpy as np
from BallDetection import BallDetector
# from tqdm import tqdm
from utils import postprocess
from tqdm import tqdm
import numpy as np
import argparse,cv2,os
from itertools import groupby
from scipy.spatial import distance
from tensorflow.keras.models import load_model
from pathlib import Path

def main():
    # video_path = "Video/DjokovicForehandDemo1.mp4"
    video_path = "Video/DjokovicSinner_2024AO.1.mp4"
    video = cv2.VideoCapture(video_path)
    amount_of_frames = video.get(cv2.CAP_PROP_FRAME_COUNT) # to get the total number of the frames
    print('amount_of_frames', amount_of_frames)

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')

    # out = cv2.VideoWriter('Video/DjokovicForehandDemo1_BT_TrackNet.1.mp4', fourcc, 60.0, (w, h)) # more fps, more quickly to run
    out = cv2.VideoWriter('Video/video3_TN.mp4', fourcc, 30.0, (w, h)) # more fps, more quickly to run


    # ball_detector = BallDetector('TrackNet/Weights.pth', out_channels=2)
    model = load_model(args.saved_model_path)
   
    balls, dists = infer_model(frames, model)

    while video.isOpened():
        ret, frames = video.read()
        
        if frames is None:
            print("No more successfully read this video!")
            break

            write_track(frames, ball_track, args.output_video_path, fps)  
    
            
        cv2.imshow("Frames", frames)
        out.write(frames)
        if cv2.waitKey(1) == ord("q"):
            break

    print("The total number of ball tracked:", len(balls))

    out.release()
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #  model = load_model(args.saved_model_path)
    
    # frames, fps = read_video(args.input_video_path)
    # ball_track, dists = infer_model(frames, model)
    # ball_track = remove_outliers(ball_track, dists)    
    
    # if args.extrapolation:
    #     subtracks = split_track(ball_track)
    #     for r in subtracks:
    #         ball_subtrack = ball_track[r[0]:r[1]]
    #         ball_subtrack = interpolation(ball_subtrack)
    #         ball_track[r[0]:r[1]] = ball_subtrack
        
    # write_track(frames, ball_track, args.output_video_path, fps)  
    main()

# amount_of_frames 191
# len(balls): 176
# Successfully tracked rate: 176/191=0.9263