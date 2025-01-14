## Try to reduce the running time by combining the inferring the model and read-write video


# import mediapipe as mp
import numpy as np
from utils import heatMap
from tqdm import tqdm
import argparse,cv2,os
from itertools import groupby
from scipy.spatial import distance
from keras.models import load_model
from pathlib import Path
from infer import infer_model_1, infer_model, write_track, read_video

def main():
    # video_path = "Video/DjokovicForehandDemo1.mp4"
    video_path = "media/DjokovicSinner_2024AO.mp4"
    video = cv2.VideoCapture(video_path)
    amount_of_frames = video.get(cv2.CAP_PROP_FRAME_COUNT) # to get the total number of the frames
    print('amount_of_frames', amount_of_frames)

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')

    # out = cv2.VideoWriter('Video/DjokovicForehandDemo1_BT_TrackNet.1.mp4', fourcc, 60.0, (w, h)) # more fps, more quickly to run
    out = cv2.VideoWriter('media/video3_TN.mp4', fourcc, 30.0, (w, h)) # more fps, more quickly to run


    
    model_path = "models/tracknet.2.keras"
    model = load_model(model_path)
    balls = []
    # balls, dists = infer_model_1(video, model)

    films = []
    while video.isOpened():
        ret, film = video.read()
        if ret:
            films.append(film)
        
        if film is None:
            print("No more successfully read this video!")
            break

    n = len(films)
    print("The length of the films: ", n)
    balls, dists = infer_model_1(films, model)
    print("The number of balls:", len(balls))

    # write_track(frames, ball_track, args.output_video_path, fps)  
    trace = 6
    
    for num in tqdm(range(2, n)):
        print("num:", num)
        frame = films[num]
        for i in range(trace):
            if (num-i > 0):
                if balls[num-i][0]:
                    x = int(balls[num-i][0])
                    y = int(balls[num-i][1])
                    frame = cv2.circle(frame, (x,y), radius=0, color = (0, 225, 165) , thickness=10-i)
                else:
                    break
        out.write(frame) 
            
        # cv2.imshow("Films", np.array(films))
        # out.write(films)
        # if cv2.waitKey(1) == ord("q"):
        #     break

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
