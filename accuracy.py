import numpy as np
import cv2
from scipy.spatial import distance
import tensorflow as tf
from utils import heatMap, heatMap_1, binary_heatMap, get_input, get_output, WBCE_loss, generate_binary_heatmap


def validate(model, validation_data, n_classes=256, input_height=360, input_width=640, output_height=720, output_width=1280, min_dist=5): 
    """originally n_classes=256 for sparceCategoricalCrossEntropy, 
        n_classes=1 for WBCE_loss
    """

    tp = [0, 0, 0, 0]
    fp = [0, 0, 0, 0]
    tn = [0, 0, 0, 0]
    fn = [0, 0, 0, 0]
    losses = []
    num_samples = len(validation_data[0]) # validation_data is a dictionary

    print("num_samples ", num_samples)

    for iter in range(num_samples):
        
        path, path_prev, path_preprev, path_gt, x_gt, y_gt, status, vis = validation_data[0][iter], validation_data[1][iter], validation_data[2][iter], validation_data[3][iter], validation_data[4][iter], validation_data[5][iter], validation_data[6][iter], validation_data[7][iter]
        
        imgs = get_input(input_height, input_width, path, path_prev, path_preprev) # combine 3 frames 
        
        prediction = model.predict(np.array([imgs]), verbose=0)[0]
        x_pred, y_pred = heatMap(prediction, n_classes, input_height, input_width, output_height, output_width)
        
        vis = int(vis)
        
        if x_pred:
            if vis != 0:
                dist = distance.euclidean((x_pred, y_pred), (float(x_gt), float(y_gt)))
                if dist < min_dist: # min_dis = 5 because the diameter of the ball is 5 pixels.
                    tp[vis] += 1
                else:
                    fp[vis] += 1
            else:
                fp[vis] += 1
        if not x_pred:
            if vis != 0:
                fn[vis] += 1
            else:
                tn[vis] += 1

        eps = 1e-15
        precision = sum(tp) / (sum(tp) + sum(fp) + eps)
        vc1 = tp[1] + fp[1] + tn[1] + fn[1] # The ball can be easily identified.
        vc2 = tp[2] + fp[2] + tn[2] + fn[2] # The ball is in the frame, but not easily be identified.
        vc3 = tp[3] + fp[3] + tn[3] + fn[3] # the ball is occluded by other objects.
        recall = sum(tp) / (vc1 + vc2 + vc3 + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        y_true = get_output(input_height, input_width, path_gt)
        scce = tf.keras.losses.SparseCategoricalCrossentropy()
        loss = scce(y_true, prediction).numpy()
        # loss = WBCE_loss(output , y_pred).numpy() # for WBCE_loss
        losses.append(loss.item())

        if iter % 842 == 0: # 842 = 5894/7
            print('The number of validated samples  ', iter, "/", num_samples)
            print("Sample Prediction: GT (", x_gt, y_gt, ") Pred (", x_pred, y_pred, ") Visibility:",vis)
            print("tp tn fp fn ", sum(tp), sum(tn), sum(fp), sum(fn))
            print('precision = {}'.format(precision))
            print('recall = {}'.format(recall))
            print('f1 = {}'.format(f1))
            print("Validation loss:",np.mean(losses))

    return np.mean(losses), precision, recall, f1


# This is for WBCE_loss validation
def validate_1(model, validation_data, input_height=360, input_width=640, output_height=720, output_width=1280, min_dist=5): 
# def validate_1(model, validation_data, input_height=360, input_width=640, output_height=360, output_width=640, min_dist=5): 
    """for WBCE_loss
    """

    tp = [0, 0, 0, 0]
    fp = [0, 0, 0, 0]
    tn = [0, 0, 0, 0]
    fn = [0, 0, 0, 0]
    losses = []
    num_samples = len(validation_data[0]) # validation_data is a dictionary

    print("num_samples ", num_samples)

    for iter in range(num_samples):
        
        path, path_prev, path_preprev, path_gt, x_gt, y_gt, status, vis = validation_data[0][iter], validation_data[1][iter], validation_data[2][iter], validation_data[3][iter], validation_data[4][iter], validation_data[5][iter], validation_data[6][iter], validation_data[7][iter]
        
        imgs = get_input(input_height, input_width, path, path_prev, path_preprev) # combine 3 frames 
        
        prediction = model.predict(np.array([imgs]), verbose=0)[0]
    
        x_pred, y_pred = heatMap_1(prediction, input_height, input_width, output_height, output_width)
     
        vis = int(vis)
        
        if x_pred:
            if vis != 0:
                dist = distance.euclidean((x_pred, y_pred), (float(x_gt), float(y_gt)))
                if dist < min_dist: # min_dis = 5 because the diameter of the ball is 5 pixels.
                    tp[vis] += 1
                else:
                    fp[vis] += 1
            else:
                fp[vis] += 1
        if not x_pred:
            if vis != 0:
                fn[vis] += 1
            else:
                tn[vis] += 1

        eps = 1e-15
        precision = sum(tp) / (sum(tp) + sum(fp) + eps)
        vc1 = tp[1] + fp[1] + tn[1] + fn[1] # The ball can be easily identified.
        vc2 = tp[2] + fp[2] + tn[2] + fn[2] # The ball is in the frame, but not easily be identified.
        vc3 = tp[3] + fp[3] + tn[3] + fn[3] # the ball is occluded by other objects.
        recall = sum(tp) / (vc1 + vc2 + vc3 + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        y_true = get_output(input_height, input_width,path_gt)
        y_true = np.reshape(y_true, (input_width*input_height))
        loss = WBCE_loss(y_true , y_pred).numpy()# for WBCE_loss
        losses.append(loss.item())

        if iter % 842 == 0: # 842 = 5894/7
            print('The number of validated samples  ', iter, "/", num_samples)
            print("Sample Prediction: GT (", x_gt, y_gt, ") Pred (", x_pred, y_pred, ") Visibility:",vis)
            print("tp tn fp fn ", sum(tp), sum(tn), sum(fp), sum(fn))
            print('precision = {}'.format(precision))
            print('recall = {}'.format(recall))
            print('f1 = {}'.format(f1))
            print("Validation loss:",np.mean(losses))

    return np.mean(losses), precision, recall, f1
