import os
import cv2
import math
import numpy as np

from data import *

DATA_DIR = "../data/"

def move_car(default_car, car_params):
    x, y, theta = car_params
    eulerAngles = np.array([0, 0, -theta])
    R = eulerAnglesToRotationMatrix(eulerAngles)
    t = np.array([x, y, 0])

    new_car = {}
    for label in default_car:
        point = default_car[label]
        new_car[label] = np.matmul(R, point) + t
    return new_car

def inv_project_to_ground(cam_params, pixel, img_shape):
    R_inv, t, focal = cam_params

    # Camera to world coordinates
    u,v = pixel
    x = (u - img_shape[1]/2) / focal
    y = (v - img_shape[0]/2) / focal
    point_cam = np.array([x,y,1])
    point = np.matmul(R_inv, point_cam)

    c = -(np.linalg.norm(t) / point[2]) # Extend ray so z = -height
    point = point * c
    point += t
    return point

def project(cam_params, point, img_shape):
    R, t, focal = cam_params

    # World to camera coordinates
    point_cam = np.matmul(R, point - t)
    x = point_cam[0] / point_cam[2]
    y = point_cam[1] / point_cam[2]
    u = x * focal + img_shape[1]/2
    v = y * focal + img_shape[0]/2
    return u,v

def transform(camera_params):
    height, focal, pitch, roll = camera_params
    eulerAngles = np.array([0, -(math.pi/2 + pitch), roll + 0.5*math.pi])
    R = eulerAnglesToRotationMatrix(eulerAngles)
    t = np.array([0,0,height])
    cam_params = (R, t, focal)
    return cam_params

def eulerAnglesToRotationMatrix(theta) :
    '''
    Rotation about x-axis, y-axis, and z-axis.
    Order is important!!
    '''
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])     
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R


if __name__ == "__main__":
    import vis
    img = cv2.imread("../data/parking_lot.jpg", cv2.IMREAD_COLOR)
    drawn = vis.draw_keypoints(img, CARS_GT)
    cv2.imshow("drawn", drawn)
    cv2.waitKey(0)

    cv2.imwrite("../data/keypoints.jpg", drawn)


    # camera_params = [6,400,math.pi/4, 0]
    # car_params0 = [-0.4*math.pi, 6.5, 6.5]
    # car_params1 = [0.35*math.pi, 6, -2]
    # car_params_all = [car_params0, car_params1]
