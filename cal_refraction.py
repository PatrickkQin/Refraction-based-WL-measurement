"""
This algorithm is used to calculate refraction.
"""

import numpy as np


def __cal_xy__(f, l, xq, yq, s, t, p, Zq):
    if Zq == None:
        Zq = 0
    Xq = (np.sin(p)*(l+Zq*np.sin(t))*(xq*np.sin(s)+yq*np.cos(s)) + np.cos(p)*(l*np.sin(t)+Zq)*(xq*np.cos(s)-yq*np.sin(s)) - Zq*f*np.cos(t)*np.sin(p))/(xq*np.cos(t)*np.sin(s) + yq*np.cos(t)*np.cos(s) + f*np.sin(t))
    Yq = (-np.cos(p)*(l+Zq*np.sin(t))*(xq*np.sin(s)+yq*np.cos(s)) + np.sin(p)*(l*np.sin(t)+Zq)*(xq*np.cos(s)-yq*np.sin(s)) + Zq*f*np.cos(t)*np.cos(p))/(xq*np.cos(t)*np.sin(s) + yq*np.cos(t)*np.cos(s) + f*np.sin(t))
    return Xq, Yq


def __distance__(x1, y1, x2, y2):
    dis = ((x1-x2)**2 + (y1-y2)**2)**(1/2)
    return dis


def __refrac__(H, D, a, n_air, n_water):
    theta_3 = np.arctan(H/(D+a))
    theta_2 = np.arccos(n_air/(n_water*((H/(D+a))**2 + 1)**(1/2)))
    h = a*np.sin(theta_3)*np.sin(theta_2) / np.sin(theta_2 - theta_3)
    return h


def refraction(f, l, s, t, p, x_1, y_1, x_2, y_2, width, height, X_cam, Y_cam, Z_cam, n_air, n_water):
    # 0.Adjust the coordinates to the center of the image as the origin and rotate the y-axis direction
    # Note: The coordinate system of the input image is that the upper left corner is the origin,
    # horizontal from left to right is x+, and vertical from top to bottom is y+.
    # The coordinate system needed is with the center of the image as the coordinate origin,
    # y+ vertically from the center up and x+ horizontally from the center to the right
    # So it is necessary to set X = x - w/2; Y = -y + h/2
    xq_1 = x_1 - (width / 2)
    yq_1 = (height / 2) - y_1
    xq_2 = x_2 - (width / 2)
    yq_2 = (height / 2) - y_2

    # 1. Calculate the straight-line distance first
    Xq_1, Yq_1 = __cal_xy__(f, l, xq_1, yq_1, s, t, p, Zq = None)
    Xq_2, Yq_2 = __cal_xy__(f, l, xq_2, yq_2, s, t, p, Zq = None)
    # print("The coordinates of point1 are:", Xq_1, Yq_1)
    # print("The coordinates of point2 are:", Xq_2, Yq_2)

    dis_1_cam = __distance__(Xq_1, Yq_1, X_cam, Y_cam)
    dis_2_cam = __distance__(Xq_2, Yq_2, X_cam, Y_cam)
    # print("dis_1_cam, dis_2_cam = " ,dis_1_cam, dis_2_cam)

    # 2. Calculate water depth
    h_waterlogging = __refrac__(Z_cam, dis_1_cam, dis_2_cam-dis_1_cam, n_air, n_water)
    # print("waterlogging level is ", h_waterlogging, "m.")

    return h_waterlogging
