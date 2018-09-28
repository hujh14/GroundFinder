import os
import json
import cv2
import math
import numpy as np
from scipy.optimize import minimize

np.set_printoptions(suppress=True)

cars = []
car = {}
car["front"] = (200,290)
car["back"] = (340,333)
car["tire_fl"] = (191,333)
car["tire_bl"] = (277,364)
cars.append(car)
car = {}
car["front"] = (882,287)
car["back"] = (730,336)
car["tire_fr"] = (902,333)
car["tire_br"] = (800,367)
cars.append(car)

DEFAULT_CAR = {}
# DEFAULT_CAR["front"] = (0,0,0)
DEFAULT_CAR["front"] = (4.5,0,0.5)
DEFAULT_CAR["back"] = (0,0,0.5)
DEFAULT_CAR["tire_fr"] = (3.5,-1,0.3)
DEFAULT_CAR["tire_fl"] = (3.5,1,0.3)
DEFAULT_CAR["tire_br"] = (0.5,-1,0.3)
DEFAULT_CAR["tire_bl"] = (0.5,1,0.3)

def draw_keypoints(img, keypoints):
    img = np.copy(img)
    if type(keypoints) is list:
        for k in keypoints:
            img = draw_keypoints(img, k)
        return img
    else:
        radius = 5
        for label in keypoints:
            pos = keypoints[label]
            pos = (int(pos[0]), int(pos[1]))
            cv2.circle(img, pos, radius, (0,0,255), -1)
            cv2.putText(img, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255), 2)
        return img

def eulerAnglesToRotationMatrix(theta) :
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

def point3D_to_pixel(point3D, camera_params, img_shape):
    height = camera_params[0]
    focal = camera_params[1]
    pitch = camera_params[2]
    roll = camera_params[3]
    # World coords to camera coords
    # Rotate about y and z axis
    eulerAngles = np.array([0, -(math.pi/2 + pitch), roll + 0.5*math.pi])
    rotationMatrix = eulerAnglesToRotationMatrix(eulerAngles)
    tvec = np.array([0,0,height])
    point3D_camera = np.matmul(rotationMatrix, point3D - tvec)

    # Camera coords to pixel coords
    x_ = point3D_camera[0] / point3D_camera[2]
    y_ = point3D_camera[1] / point3D_camera[2]
    u = x_ * focal + img_shape[1]/2
    v = y_ * focal + img_shape[0]/2
    return u,v

def move_car_by_params(car, car_params):
    theta = car_params[0]
    x = car_params[1]
    y = car_params[2]

    rot = np.array([[np.cos(theta), np.sin(theta),0],
                    [-np.sin(theta), np.cos(theta), 0],
                    [0,0,1]])
    tvec = np.array([x, y, 0])

    new_car = {}
    for label in car:
        point3D = car[label]
        point = np.matmul(rot, point3D) + tvec
        new_car[label] = point
    return new_car

def car_loss(default_car, car_gt, img_shape, camera_params, car_params):
    loss = 0
    car_pred = move_car_by_params(default_car, car_params)
    for label in car_gt:
        pixel_gt = car_gt[label]
        point3D = car_pred[label]
        pixel = point3D_to_pixel(point3D, camera_params, img_shape)
        loss += (pixel[0] - pixel_gt[0])**2 + (pixel[1] - pixel_gt[1])**2
    return loss

def total_loss(x, cars, default_car, img_shape):
    camera_params = x[:4]
    car_params = x[4:]

    loss = 0
    for i, car in enumerate(cars):
        car_param = car_params[i*3:i*3+3]
        loss += car_loss(default_car, car, img_shape, camera_params, car_param)
    print loss
    return loss

def main(img, cars):
    # Camera params: height, focal, pitch, roll
    # Car params: theta, posx, posy
    # x0 = [camera params, car params....]
    camera_params = [10,400,0,0]
    car_params = []
    for car in cars:
        car_params.extend([0,0,0])
    # car_params0 = [-0.4*math.pi, 6.5, 6.5]
    # car_params1 = [0.35*math.pi, 6, -2]
    # car_params = car_params0 + car_params1

    cons = [{'type': 'ineq', 'fun': lambda x:  x[0]}, # height > 0
        {'type': 'ineq', 'fun': lambda x:  x[1]}, # focal > 0
        {'type': 'ineq', 'fun': lambda x:  x[2]}, # pitch > 0
        {'type': 'ineq', 'fun': lambda x:  x[5]}] # theta > 0
        # {'type': 'ineq', 'fun': lambda x:  x[8]}] # theta > 0

    x0 = camera_params + car_params
    options = {"maxiter":1e6}
    result = minimize(total_loss, x0, args=(cars, DEFAULT_CAR, img.shape), constraints = tuple(cons), method='COBYLA', options=options)
    print result

    x = result.x
    camera_params = x[:4]
    car_params = x[4:]
    print "camera_params =", list(camera_params)
    for i, car in enumerate(cars):
        car_param = car_params[i*3:i*3+3]
        print "car_params =",  list(car_param)


def test(img, camera_params, car_params, car_gt=None):
    new_car = move_car_by_params(DEFAULT_CAR, car_params)

    keypoints = {}
    for label in new_car:
        point3D = new_car[label]
        pixel = point3D_to_pixel(point3D, camera_params, img.shape)
        keypoints[label] = pixel

    if car_gt is not None:
        loss = car_loss(DEFAULT_CAR, car_gt, img.shape, camera_params, car_params)
        print loss
        # keypoints = [keypoints, car_gt]

    drawn = draw_keypoints(img, keypoints)
    cv2.imshow("keypoints", drawn)
    cv2.waitKey(0)
    return drawn

img = cv2.imread("../data/parking_lot.jpg", cv2.IMREAD_COLOR)
# main(img, cars)

# camera_params = [6,400,math.pi/4, 0]
# car_params0 = [-0.4*math.pi, 6.5, 6.5]
# car_params1 = [0.35*math.pi, 6, -2]
# camera_params = [6, 400, 0.7853981633974483, 0]
# car_params0 = [-1.2566370614359172, 6.5, 6.5]
# car_params1 = [1.0995574287564276, 6, -2]

camera_params = [6.9469942114550385, 399.56204356940839, 0.86482525165273449, -0.0021812117818751906]
car_params0 = [-1.2000574320405821, 6.2593574635076195, 6.872753076790354]
car_params1 = [1.1266967349285399, 6.0420404161815586, -2.2944568988883547]
img = test(img, camera_params, car_params0, cars[0])
img = test(img, camera_params, car_params1, cars[1])


# for car in cars:
#     img = draw_keypoints(img, car)
# cv2.imshow("keypoints", img)
# cv2.waitKey(0)






