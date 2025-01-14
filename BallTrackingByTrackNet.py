## Tracking a tenis by TrackNet
# The frames need to include the full tennis court because the training data included the full court.

import cv2
import mediapipe as mp
import numpy as np
from BallDetection import BallDetector
# from tqdm import tqdm

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


    ball_detector = BallDetector('TrackNet/Weights.pth', out_channels=2)

    ball = None
    balls = [] # Collect the tracked balls
    # kernel = np.ones((4,4),np.uint8)

    while video.isOpened():
        ret, frames = video.read()
        
        if frames is None:
            print("No more successfully read this video!")
            break
            
        ballPre = ball
       
        # ball_detector.detect_ball(blurFrame)
        ball_detector.detect_ball(frames) # No dilate, eroded and GaussianBlur processing

        if ball_detector.xy_coordinates[-1][0] is not None:
            ball = ball_detector.xy_coordinates[-1]
            balls.append(ball)
            cv2.circle(frames, ball, 4, (0,0,255), 3)
            cv2.circle(frames, ballPre1, 3, (0,0,255), 2)
           
        # print('Balls:', len(balls))

        
        cv2.imshow("Frames", frames)
        out.write(frames)
        if cv2.waitKey(1) == ord("q"):
            break

    print("The total number of ball tracked:", len(balls))

    out.release()
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# amount_of_frames 191
# len(balls): 176
# Successfully tracked rate: 176/191=0.9263
