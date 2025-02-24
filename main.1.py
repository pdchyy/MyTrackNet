## This file is used to test the trained model works or not, and compute the recall of Tracknet2(U_Net + Sigmoid) + WBCE_loss

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
    parser.add_argument('--saved_model_path', type=str, default = os.path.join(root, 'models/tracknet2wb.4.keras'), help= 'path to model')
    parser.add_argument('--input_video_path', type=str, help='path to input video')
    parser.add_argument('--output_video_path', type=str, help='path to output video')
    args = parser.parse_args()

    video_path = "media/D-S_24AO.mp4"

    model = load_model(args.saved_model_path, custom_objects={"WBCE_loss": WBCE_loss}) # WBCE_loss is the customized loss function
    
    frames, fps = read_video(video_path)

    print("total frames:", len(frames))
    
    balls, dists = infer_model_1(frames, model) # For WBCE_loss 
    
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
  
    

    
# total frames: 191
# recall: 0.9424083769633508 at epochs = 400, tracknet2wb.4.keras, TrackNet2(U_Net + sigmoid) + WBCE
