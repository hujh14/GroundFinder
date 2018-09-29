import os
import cv2
import math
import numpy as np
from random import randint
import matplotlib.pyplot as plt

from utils import *

def visualize_all(img, camera_params, car_params_all, category_mask=None):
    drawn = img.copy()
    if category_mask is not None:
        drawn = vis_ground(drawn, camera_params, category_mask)
    drawn = vis_cars(drawn, camera_params, car_params_all)

    cv2.imwrite("../data/vis_all.jpg", drawn)
    return drawn

def vis_cars(img, camera_params, car_params_all):
    cam_params = transform(camera_params)

    plt.axis([0, 40, -20, 20])
    drawn = img.copy()

    for car_params in car_params_all:
        car_pred = move_car(DEFAULT_CAR, car_params)

        xs = []
        ys = []
        keypoints = {}
        for label in car_pred:
            point = car_pred[label]
            xs.append(point[0])
            ys.append(point[1])
            pixel = project(cam_params, point, img.shape)
            keypoints[label] = pixel

        color = random_color()
        plt.plot(xs, ys, 'o', color=color)
        drawn = draw_keypoints(drawn, keypoints, color=color)

    plt.savefig('../data/vis_above.jpg')
    cv2.imwrite("../data/vis_cars.jpg", drawn)
    return drawn

def vis_ground(img, camera_params, category_mask):
    floor = category_mask == 3
    road = category_mask == 6
    sidewalk = category_mask == 11
    earth = category_mask == 13
    seven = category_mask == 7
    ground_mask = np.logical_or.reduce([floor, road, sidewalk, earth, seven])

    # cv2.imwrite(os.path.join(DATA_DIR, "ground_mask.png"), np.array(ground_mask, dtype='uint8') * 255)
    checkboard = draw_checkboard(img, camera_params)
    img[ground_mask] = checkboard[ground_mask]
    return img

def draw_keypoints(img, keypoints, color=None):
    drawn = np.copy(img)
    if type(keypoints) is list:
        for k in keypoints:
            drawn = draw_keypoints(drawn, k)
        return drawn
    else:
        if color is None:
            color = random_color()
        radius = 5
        color = np.multiply(color, 255)
        color = (color[2], color[1], color[0])
        for label in keypoints:
            pos = keypoints[label]
            pos = (int(pos[0]), int(pos[1]))
            cv2.circle(drawn, pos, radius, color, -1)
            cv2.putText(drawn, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255), 2)
        return drawn
            
def draw_checkboard(img, camera_params):
    cam_params = transform(camera_params)
    R_inv = np.linalg.inv(cam_params[0])
    cam_params = (R_inv, cam_params[1], cam_params[2])

    h,w = img.shape[:2]
    checkboard = cv2.resize(img, (w/3, h/3))
    for y in range(checkboard.shape[0]):
        for x in range(checkboard.shape[1]):
            pixel = [x*3, y*3]
            point = inv_project_to_ground(cam_params, pixel, img.shape)
            white = (math.floor(point[0]) + math.floor(point[1])) % 2 == 0
            if white:
                checkboard[y,x] = [150,150,150]
            else:
                checkboard[y,x] = [0,0,0]
    checkboard = cv2.resize(checkboard, (w, h))
    return checkboard

def random_color():
    return np.random.rand(3)

if __name__ == "__main__":
    img = cv2.imread("../data/parking_lot.jpg", cv2.IMREAD_COLOR)
    cm = cv2.imread("../data/parking_lot_cm.png", cv2.IMREAD_GRAYSCALE)

    camera_params = [6.8574390102580356, 399.82839300101642, 0.86950537352213109, -0.015051591866106547]
    car_params_all = [[-1.18892403,  6.24860725,  6.79912993], [ 1.15937309,  5.89879014, -2.23985101]]

    drawn = visualize_all(img, camera_params, car_params_all, category_mask=cm)
    cv2.imshow("", drawn)
    cv2.waitKey(0)

