import numpy as np
import cv2
from scipy.spatial import distance
import tensorflow as tf
from utils import postprocess,postprocess_for_binary_heatmap, get_input, get_output


def validate(model, validation_data, n_classes=256, input_height=360, input_width=640, output_height=720, output_width=1280, min_dist=5):

    tp = [0, 0, 0, 0]
    fp = [0, 0, 0, 0]
    tn = [0, 0, 0, 0]
    fn = [0, 0, 0, 0]
    losses = []
    num_samples = len(validation_data[0])

    print("num_samples ", num_samples)

    for iter in range(num_samples):
        
        path, path_prev, path_preprev, path_gt, x_gt, y_gt, status, vis = validation_data[0][iter], validation_data[1][iter], validation_data[2][iter], validation_data[3][iter], validation_data[4][iter], validation_data[5][iter], validation_data[6][iter], validation_data[7][iter]
        
        imgs = get_input(input_height, input_width, path, path_prev, path_preprev)
        
        prediction = model.predict(np.array([imgs]), verbose=0)[0]
        
        x_pred, y_pred = postprocess(prediction, n_classes, input_height, input_width, output_height, output_width)
        
        vis = int(vis)
        
        if x_pred:
            if vis != 0:
                dist = distance.euclidean((x_pred, y_pred), (float(x_gt), float(y_gt)))
                if dist < min_dist:
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
        vc1 = tp[1] + fp[1] + tn[1] + fn[1]
        vc2 = tp[2] + fp[2] + tn[2] + fn[2]
        vc3 = tp[3] + fp[3] + tn[3] + fn[3]
        recall = sum(tp) / (vc1 + vc2 + vc3 + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        output = get_output(input_height, input_width, path_gt)
        scce = tf.keras.losses.SparseCategoricalCrossentropy()
        loss = scce(output, prediction).numpy()
        losses.append(loss.item())

        if iter % 842 == 0:
            print('iteration ', iter, "/", num_samples)
            print("Sample Prediction: GT (", x_gt, y_gt, ") Pred (", x_pred, y_pred, ") Visibility:",vis)
            print("tp tn fp fn ", sum(tp), sum(tn), sum(fp), sum(fn))
            print('precision = {}'.format(precision))
            print('recall = {}'.format(recall))
            print('f1 = {}'.format(f1))
            print("Validation loss:",np.mean(losses))

    return np.mean(losses), precision, recall, f1