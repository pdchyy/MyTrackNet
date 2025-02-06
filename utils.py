import numpy as np
import cv2
import time
from keras import ops
import tensorflow as tf
import keras.backend as K

def heatMap(prediction, n_classes, model_height, model_width, output_height, output_width):
    """ Use the cv2.threshold and HoughCircles to get the ball centre"""
    # start_time = time.time()
    prediction = prediction.reshape((model_height, model_width, n_classes)).argmax(axis=2) # loss= sparceCategoricalCrossEntropy
    print("prediction.shape:", prediction.shape)
    prediction = prediction.astype(np.uint8)

    feature_map = cv2.resize(prediction, (output_width, output_height))
    ret, feature = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(feature, cv2.HOUGH_GRADIENT, dp=1,
                               minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7)
    x, y = None, None
    if circles is not None:
        if len(circles) == 1:
            x = int(circles[0][0][0])
            y = int(circles[0][0][1])
    
    # run_time = time.time() - start_time
    # print( "The heatMap running time is ", run_time)
    return x, y

## This is for WBCE_loss
def heatMap_1(prediction, model_height, model_width, output_height, output_width):
    """ Use the cv2.threshold and HoughCircles to get the ball centre"""
    # start_time = time.time()
    print("prediction_0.shape:", prediction.shape)
    # prediction = prediction.reshape((model_height, model_width, 3)).argmax(axis=2)
    prediction = prediction.reshape((model_height, model_width)) # the final index is 1 for 1-frame-out and 3 for 3-frames-out.
    print("prediction.shape:", prediction.shape)
    # prediction = prediction.reshape((model_height, model_width)) # loss= WBCE_loss
    prediction = prediction.astype(np.uint8)

    feature_map = cv2.resize(prediction, (output_width, output_height))
    # feature_map = cv2.resize(prediction, (model_width, model_height))
    ret, feature = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
    # ret, feature = cv2.threshold(prediction, 127, 255, cv2.THRESH_BINARY)

    print("featuer.shape:", feature.shape)
    circles = cv2.HoughCircles(feature, cv2.HOUGH_GRADIENT, dp=1,
                               minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7)
    print("cirles:", circles)
    x, y = None, None
    if circles is not None:
        if len(circles) == 1:
            x = int(circles[0][0][0])
            y = int(circles[0][0][1])
    
    # run_time = time.time() - start_time
    # print( "The heatMap running time is ", run_time)
    return x, y


def binary_heatMap(prediction, ratio=2):

    prediction = prediction > 0.5
    prediction = prediction.astype('float32')
    h_pred = prediction*255
    h_pred = h_pred.astype('uint8')
    cx_pred, cy_pred = None, None
    if np.amax(h_pred) <= 0:
        return cx_pred, cy_pred
    else:
        (cnts, _) = cv2.findContours(h_pred[0].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(ctr) for ctr in cnts]
        max_area_idx = 0
        max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
        for i in range(len(rects)):
            area = rects[i][2] * rects[i][3]
            if area > max_area:
                max_area_idx = i
                max_area = area
        target = rects[max_area_idx]
        (cx_pred, cy_pred) = (int(ratio*(target[0] + target[2] / 2)), int(ratio*(target[1] + target[3] / 2)))
    return cx_pred, cy_pred
    

def get_input(height, width, path, path_prev, path_preprev):

    img = cv2.imread(path)
    img = cv2.resize(img, (width, height))

    img_prev = cv2.imread(path_prev)
    img_prev = cv2.resize(img_prev, (width, height))

    img_preprev = cv2.imread(path_preprev)
    img_preprev = cv2.resize(img_preprev, (width, height))

    imgs = np.concatenate((img, img_prev, img_preprev), axis=2)

    imgs = imgs.astype(np.float32) / 255.0

    imgs = np.rollaxis(imgs, 2, 0)

    return np.array(imgs)


def get_output(height, width, path_gt):
    img = cv2.imread(path_gt)
    img = cv2.resize(img, (width, height))
    img = img[:, :, 0]
    # img = np.reshape(img, (width * height))
    # img = np.catenate((img, img,img), axis=2)
    # img = np.reshape(img, (3, width , height)) 
    img = np.reshape(img, (width, height))
    return img


def WBCE_loss(y_true, y_pred): 
    """" Weighted binary crossentropy loss function"""
    # print("type(y_true0):",type(y_true))
    # print("y_true.shape:", y_true.shape)
    # y_true = tf.reshape(y_true, (y_true.shape[1], y_true.shape[2], y_true.shape[3]))
    # # for i in range(y_true.shape[0]):
    #     for j in range(y_true.shape[1]):
    #         # for k in range(y_true.shape[2]):
    #         indices = tf.constant([[i,j]])
    #         updates = tf.constant([1])
    #         print("y_true[i,j]", y_true[i,j])
    #         if (y_true[i,j] > tf.constant([1.0])):
    #             y_true = tf.tensor_scatter_nd_update(y_true, indices, updates)
    # print("type(y_true):",type(y_true))
    # print("y_true:", y_true)
    # y_true = np.where(y_true < 1, y_true,1) # To make VC2 and VC3 are VC1
    # y_true  = tf.convert_to_tensor(y_true)

    # T_fn = lambda: 0.0
    # F_fn = lambda: 1.0
    # y_true = tf.cond(y_true < 1, T_fn, F_fn )


    # cv2.imshow("y_true", y_true)
	# # Wait for the user to press a key
    # cv2.waitKey(0)

	# # Close all windows
    # cv2.destroyAllWindows()
    # print("rank of y_pred:", tf.rank(y_pred))
    # print("rank of y-true:", tf.rank(y_true))
    loss = (-1)*(ops.square(1 - y_pred) * y_true * ops.log(ops.clip(y_pred, 1e-7, 1)) + ops.square(y_pred) * (1 - y_true) * ops.log(ops.clip(1 - y_pred, 1e-7, 1)))
    # loss = (-1)* (y_true * ops.log(ops.clip(y_pred, 1e-7, 1)) +  (1 - y_true) * ops.log(ops.clip(1 - y_pred, 1e-7, 1)))
    return ops.mean(loss)



def WBCELoss(y_pred, y, reduce=True):
    """ Weighted Binary Cross Entropy loss function defined in TrackNetV2 paper.

        Args:
            y_pred (torch.Tensor): Predicted values with shape (N, 1, H, W)
            y (torch.Tensor): Ground truth values with shape (N, 1, H, W)
            reduce (bool): Whether to reduce the loss to a single value or not

        Returns:
            (torch.Tensor): Loss value with shape (1,) if reduce, else (N, 1)
    """
    
    loss = (-1)*(torch.square(1 - y_pred) * y * torch.log(torch.clamp(y_pred, 1e-7, 1))\
            + torch.square(y_pred) * (1 - y) * torch.log(torch.clamp(1 - y_pred, 1e-7, 1)))
    if reduce:
        return torch.mean(loss)
    else:
        return torch.mean(torch.flatten(loss, start_dim=1), 1)
