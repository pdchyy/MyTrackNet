import numpy as np
import cv2


def postprocess(prediction, n_classes, model_height, model_width, output_height, output_width):

    prediction = prediction.reshape((model_height, model_width, n_classes)).argmax(axis=2)
    prediction = prediction.astype(np.uint8)

    feature_map = cv2.resize(prediction, (output_width, output_height))
    ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1,
                               minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7)
    x, y = None, None
    if circles is not None:
        if len(circles) == 1:
            x = int(circles[0][0][0])
            y = int(circles[0][0][1])

    return x, y


def postprocess_for_binary_heatmap(prediction, ratio=2):

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
    img = np.reshape(img, (width * height))
    return img
