import numpy as np
import cv2
import time
from keras import ops
import tensorflow as tf
import keras.backend as K

def heatMap(prediction, n_classes, model_height, model_width, output_height, output_width):
    """ Use the cv2.threshold and HoughCircles to get the ball centre"""

    prediction = prediction.reshape((model_height, model_width, n_classes)).argmax(axis=2) # loss= sparceCategoricalCrossEntropy
    
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
    
    return x, y

## This is for WBCE_loss
def heatMap_1(prediction, model_height, model_width, output_height, output_width):
    """ Use the cv2.threshold and HoughCircles to get the ball centre"""
  
    prediction = prediction > 0.5
    prediction = prediction.astype('float32')
    prediction = prediction * 255
    prediction = prediction.astype('uint8')
    prediction = prediction.reshape((model_height, model_width)) # loss= WBCE_loss
    prediction = prediction.astype(np.uint8)

    feature_map = cv2.resize(prediction, (output_width, output_height))
   
    ret, feature = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)

    circles = cv2.HoughCircles(feature, cv2.HOUGH_GRADIENT, dp=1,
                               minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7)
    print("cirles:", circles)
    x, y = None, None
    if circles is not None:
        if len(circles) == 1:
            x = float(circles[0][0][0])
            y = float(circles[0][0][1])
        
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
    # For WBCE_loss + Binary heatMap to fit the output of the model  #########################
    img = img/255
    img = img > 0.5
    img = img.astype('float32')
    ##########################################################################################
    img = np.reshape(img, (width* height))
    return img

def generate_binary_heatmap(cx, cy, r, mag):
        height = 360
        width = 640
        
        if vis > 0:
            cx = int(float(cx.strip()))
            cy = int(float(cy.strip()))

        else:
            return np.zeros((1, height, width))
        
        x, y = np.meshgrid(np.linspace(1, width, width), np.linspace(1, height, height))
        heatmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2) # ?
        heatmap[heatmap <= r**2] = 1
        heatmap[heatmap > r**2] = 0
        y = heatmap*mag
        y = np.reshape(y, (1, height, width))
        return y


def WBCE_loss(y_true, y_pred): 
    """" Weighted binary crossentropy loss function"""
	
    if y_pred is None:
        y_pred = np.array(0.0)
    else:
        tf.cast(y_pred, tf.float32)
    
    loss = (-1)*(ops.square(1 - y_pred) * y_true * ops.log(ops.clip(y_pred, 1e-07, 1)) + ops.square(y_pred) * (1 - y_true) * ops.log(ops.clip(1 - y_pred, 1e-07, 1)))
    # loss = (-1)* (y_true * ops.log(ops.clip(y_pred, 1e-7, 1)) +  (1 - y_true) * ops.log(ops.clip(1 - y_pred, 1e-7, 1))) # Binary CrossEntropy loss
    return ops.mean(loss)


def BCE_loss(y_true, y_pred): 
    """" binary crossentropy loss function, it can not be used for for imbalanced binary heatMap"""
    
    if y_pred is None:
        # y_pred = np.array(0.0, dtype=np.float32)
        y_pred = tf.constant([0.0], tf.float32)
    else:
        tf.cast(y_pred, tf.float32)

    loss = (-1)* y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-07, 1)) + (1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-07, 1)) # Binary CrossEntropy loss
    # loss = (-1)* (y_true * ops.log(ops.clip(y_pred, 1e-7, 1)) +  (1 - y_true) * ops.log(ops.clip(1 - y_pred, 1e-7, 1))) # Binary CrossEntropy loss
    return ops.mean(loss)


