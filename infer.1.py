## This file is used to test the trained model works or not, and compute the recall  of model: TrackNet + SCCE , or TrackNetU (U_net+Softmax)  + SCCE

from pathlib import Path
import tensorflow as tf
from infer import infer_model, write_track, read_video, remove_outliers,infer_model_1
from keras.models import load_model
import argparse,cv2,os
from utils import WBCE_loss

if __name__ == '__main__':
    root = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--saved_model_path', type=str, default = os.path.join(root, 'models/tracknetUsc.0.keras'), help= 'path to model')
    parser.add_argument('--input_video_path', type=str, help='path to input video')
    parser.add_argument('--output_video_path', type=str, help='path to output video')
    args = parser.parse_args()

    video_path = "media/M-S_24.mp4"
    # video_path = "media/D-S_24AO.mp4"
    model = load_model(args.saved_model_path) # Tracknet2 (U_Net + foftmax) +  SSCE loss function
    
    frames, fps = read_video(video_path)
    print("fps: ", fps)

    print("total frames:", len(frames))
    
    balls, dists = infer_model(frames, model)
    # balls = remove_outliers(balls, dists)  
    
    balls = remove_outliers(balls, dists)    
    output_path = "media/output.mp4"
    write_track(frames, balls, output_path, fps)    

    n = len(balls)
    tracked_balls = []
    
    for  ball in balls:
        if ball[0]:
            tracked_balls.append(ball)
    
    print("The total number of tracked_ball", len(tracked_balls))
    recall = len(tracked_balls)/n
    print(f"recall: {recall}")
  
    
# recall: 0.9424083769633508 at epochs = 400, tracknet.4.keras   TrackNet(softmax)  + SCCE
# recall: 0.8767772511848341 at epochs = 500, tracknet.7.keras (U_Net + softmax) + SCCE
# recall: 0.9528795811518325 at epochs = 400, tracknetUsc.0.keras (U_Net + softmax) + SCCE
