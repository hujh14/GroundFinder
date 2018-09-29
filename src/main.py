import os
import json
import cv2
import math
import numpy as np
from scipy.optimize import minimize

import vis
from utils import *

np.set_printoptions(suppress=True)

def car_loss(car_pred, car_gt, cam_params, img_shape):
    losses = []
    for label in car_gt:
        point = car_pred[label]
        pixel = project(cam_params, point, img_shape)
        pixel_gt = car_gt[label]

        loss = (pixel[0] - pixel_gt[0])**2 + (pixel[1] - pixel_gt[1])**2
        losses.append(loss)
    return np.sum(losses)

def total_loss(camera_params, car_params_all, car_gt_all, img_shape, default_car=DEFAULT_CAR):
    cam_params = transform(camera_params)
    losses = []
    for car_params, car_gt in zip(car_params_all, car_gt_all):
        car_pred = move_car(default_car, car_params)
        loss = car_loss(car_pred, car_gt, cam_params, img_shape)
        losses.append(loss)
    print losses
    return np.sum(losses)

def decode_params(x):
    # x[1] = x[1]*10 # Scale focal parameter
    camera_params = x[:4]
    other_params = x[4:]
    car_params_all = []
    for i in range(0, len(other_params), 3):
        car_params = other_params[i:i+3]
        car_params_all.append(car_params)
    return camera_params, car_params_all

def total_loss_func(x, car_gt_all, img_shape, default_car):
    camera_params, car_params_all = decode_params(x)
    loss = total_loss(camera_params, car_params_all, car_gt_all, img_shape, default_car)
    return loss

def main(img, cars_gt, maxiter=1e6):
    ''' camera_params: height, focal, pitch, roll
    car_params: posx, posy, theta
    x0 = [camera params, car params]
    '''

    camera_params = [10,400,0,0]
    car_params_all = [10,0,0] * len(cars_gt)
    x0 = camera_params + car_params_all

    cons = [{'type': 'ineq', 'fun': lambda x:  x[0]}, # height > 0
        {'type': 'ineq', 'fun': lambda x:  x[1]}, # focal > 0
        {'type': 'ineq', 'fun': lambda x:  x[2]}, # pitch > 0
        {'type': 'ineq', 'fun': lambda x:  math.pi/2 - x[2]}] # pitch < pi/2
    cons.append({'type': 'ineq', 'fun': lambda x:  x[4]}) # posx > 0
    cons.append({'type': 'ineq', 'fun': lambda x:  x[7]}) # posx > 0
    # cons.append({'type': 'ineq', 'fun': lambda x:  x[10]}) # posx > 0

    options = {"maxiter":maxiter}
    result = minimize(total_loss_func, x0, args=(cars_gt, img.shape, DEFAULT_CAR), constraints = tuple(cons), method='COBYLA', options=options)
    print result

    camera_params, car_params_all = decode_params(result.x)
    print "camera_params =", list(camera_params)
    print "car_params_all =", [list(p) for p in car_params_all]
    return camera_params, car_params_all


if __name__ == "__main__":
    img = cv2.imread("../data/parking_lot.jpg", cv2.IMREAD_COLOR)
    cm = cv2.imread("../data/parking_lot_cm.png", cv2.IMREAD_GRAYSCALE)
    camera_params, car_params_all = main(img, CARS_GT, maxiter=1e5)

    # camera_params = [6.8574390102580356, 399.82839300101642, 0.86950537352213109, -0.015051591866106547]
    # car_params_all = [[-1.18892403,  6.24860725,  6.79912993], [ 1.15937309,  5.89879014, -2.23985101]]

    total_loss(camera_params, car_params_all, CARS_GT, img.shape)
    drawn = vis.visualize_all(img, camera_params, car_params_all, category_mask=None)

    # cv2.imshow("", img)
    # cv2.waitKey(0)
    cv2.imshow("", drawn)
    cv2.waitKey(0)






