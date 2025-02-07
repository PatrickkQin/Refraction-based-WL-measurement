"""
This algorithm is used to calibrate the camera.
"""

import numpy as np

def calibration(height, width, omiga, selected_points, input_object):
    # Enter the width of the rectangle in m
    if omiga == None:
        omiga = 2

    # four corner of the rectangular
    xa = selected_points[0][0] - int(width/2)
    ya = -selected_points[0][1] + int(height/2)
    xb = selected_points[1][0] - int(width/2)
    yb = -selected_points[1][1] + int(height/2)
    xc = selected_points[2][0] - int(width/2)
    yc = -selected_points[2][1] + int(height/2)
    xd = selected_points[3][0] - int(width/2)
    yd = -selected_points[3][1] + int(height/2)
    # The input here is to pay attention to the order, you need to convert the coordinate system,
    # from [↓y, →x, the upper-left corner origin] to [↑y, →x, the center origin].

    # Calculate 16 small parameters
    alpha_ab = xb - xa
    beta_ab = yb - ya
    x_ab = xa*yb - xb*ya

    alpha_ac = xc - xa
    beta_ac = yc - ya
    x_ac = xa*yc - xc*ya

    alpha_bd = xd - xb
    beta_bd = yd - yb
    x_bd = xb*yd - xd*yb

    alpha_cd = xd - xc
    beta_cd = yd - yc
    x_cd = xc*yd - xd*yc

    # pan angle p, tilt angle t, swing angle s
    # Pan angle corresponds to yaw, tilt angle corresponds to pitch, and swing angle corresponds to roll.
    tans = (-beta_ab*beta_ac*x_bd*alpha_cd + beta_ac*alpha_bd*beta_ab*x_cd + beta_cd*x_ab*beta_bd*alpha_ac - beta_ab*x_cd*beta_bd*alpha_ac
            - beta_cd*beta_bd*x_ac*alpha_ab - beta_ac*x_ab*alpha_bd*beta_cd + beta_ab*x_ac*beta_bd*alpha_cd + beta_cd*beta_ac*x_bd*alpha_ab)/\
           (-beta_ab*x_ac*alpha_bd*alpha_cd + beta_ac*x_ab*alpha_bd*alpha_cd - beta_ac*alpha_bd*alpha_ab*x_cd - alpha_ac*x_bd*beta_cd*alpha_ab
            - alpha_cd*x_ab*beta_bd*alpha_ac + beta_ab*alpha_ac*x_bd*alpha_cd + alpha_ab*x_cd*beta_bd*alpha_ac + alpha_bd*x_ac*beta_cd*alpha_ab)
    print("tan(s) = ", tans)
    s = np.arctan(tans)
    print("s = ", s)

    sint = -((((alpha_bd*x_ac - alpha_ac*x_bd) * np.sin(s) + (beta_bd*x_ac - beta_ac*x_bd) * np.cos(s)) * ((alpha_cd*x_ab - alpha_ab*x_cd) * np.sin(s) + (beta_cd*x_ab - beta_ab*x_cd) * np.cos(s))) / \
           (((alpha_cd*x_ab - alpha_ab*x_cd) * np.cos(s) + (beta_ab*x_cd - beta_cd*x_ab) * np.sin(s)) * ((beta_bd*x_ac - beta_ac*x_bd) * np.sin(s) + (alpha_ac*x_bd - alpha_bd*x_ac) * np.cos(s))))**(1/2)

    print("sin(t) = ", sint)
    t = np.arcsin(sint)
    print("t = ", t)

    tanp = (sint*((beta_bd*x_ac - beta_ac*x_bd)* np.sin(s) + (alpha_ac*x_bd - alpha_bd*x_ac) * np.cos(s)))/((alpha_bd*x_ac - alpha_ac*x_bd) * np.sin(s) + (beta_bd*x_ac - beta_ac*x_bd) * np.cos(s))
    p = np.arctan(tanp)
    if p < 0:
        p = p + np.pi
    print("tan(p) = ", tanp)
    print("p = ", p)

    print("The three angles are:", s/np.pi*180, t/np.pi*180, p/np.pi*180)

    # Find the focal length f
    f = (x_bd * np.cos(p) * np.cos(t))/(beta_bd * np.sin(p) * np.cos(s) - beta_bd * np.cos(p) * np.sin(t) * np.sin(s) + alpha_bd * np.sin(p) * np.sin(s) + alpha_bd * np.cos(p) * np.sin(t) * np.cos(s))
    print("The focal length of the camera is:", f)

    # Find the Euclidean distance between the camera and the world coordinate system
    if input_object == "vertical":
        l = omiga * (f*sint + xa*np.cos(t)*np.sin(s) + ya * np.cos(t)*np.cos(s)) * (f * sint + xc*np.cos(t)*np.sin(s) + yc*np.cos(t)*np.cos(s))/\
            (-(f*sint + xa*np.cos(t)*np.sin(s) + ya*np.cos(t)*np.cos(s)) * (xc*np.cos(p)*np.sin(s) - xc* np.sin(p) * sint * np.cos(s) + yc*np.cos(p) * np.cos(s) +yc* np.sin(p)* sint * np.sin(s))
             + (f* sint + xc*np.cos(t)*np.sin(s) + yc*np.cos(t)*np.cos(s)) * (xa * np.cos(p)* np.sin(s) -xa*np.sin(p)*sint*np.cos(s) + ya * np.cos(p)*np.cos(s) + ya*np.sin(p) * sint * np.sin(s)))
    elif input_object == "parallel":
        l = omiga * (xc*np.cos(t)*np.sin(s)+yc*np.cos(t)*np.cos(s)+f*sint) * (xd*np.cos(t)*np.sin(s)+yd*np.cos(t)*np.cos(s)+f*sint)/\
        (-(xd*np.cos(t)*np.sin(s) + yd*np.cos(t)*np.cos(s) + f*sint)*(np.sin(p)*(xc*np.sin(s)+yc*np.cos(s))+sint*np.cos(p)*(xc*np.cos(s)-yc*np.sin(s)))
         + (xc*np.cos(t)*np.sin(s) + yc*np.cos(t)*np.cos(s) + f*sint)*(np.sin(p)*(xd*np.sin(s) + yd*np.cos(s)) + sint*np.cos(p)*(xd*np.cos(s)-yd*np.sin(s))))
    # print("The distance between camera and the origin of the world coordinate system is", l)
    print("The distance from the camera to the origin is", l)

    # You can get the coordinates of the camera in the world coordinate system
    # todo Note that, according to the algorithm, the origin of the world coordinate system is the point on the ground
    #  corresponding to the center of the pixel of the picture taken by the camera (i.e., Z = 0)
    x_cam = l * np.sin(p) * np.cos(t)
    y_cam = -l * np.cos(p) * np.cos(t)
    z_cam = -l * sint
    print("The camera's world coordinate system is: (", x_cam, y_cam, z_cam, ")")


    def __map2world__(x_map, y_map, s, t, p, f, l):
           """
           Design a function that calculates the world coordinate system from the image coordinate system.
           Note that it is assumed here, that the Z-axis height of the selected point is equal to 0
           :param x_map:The x-coordinate in the image
           :param y_map:The y-coordinate in the image
           :return:X- and Y-coordinates in the world coordinate system
           """
           zq = 0
           xq = (np.sin(p)*(l+zq*np.sin(t))*(x_map*np.sin(s) + y_map*np.cos(s)) + np.cos(p)*(l*np.sin(t)+zq)*(x_map*np.cos(s)-y_map*np.sin(s))-zq*f*np.cos(t)*np.sin(p))/\
                (x_map*np.cos(t)*np.sin(s) + y_map*np.cos(t)*np.cos(s) + f*np.sin(t))
           yq = (-np.cos(p)*(l+zq*np.sin(t))*(x_map*np.sin(s) + y_map*np.cos(s)) + np.sin(p) *(l*np.sin(t) + zq)*(x_map* np.cos(s) - y_map*np.sin(s)) + zq*f*np.cos(t)*np.cos(p))/\
                (x_map*np.cos(t)*np.sin(s) + y_map*np.cos(t)*np.cos(s) + f*np.sin(t))
           return xq, yq, zq


    # Calculate the world coordinates between the four points
    Xa, Ya, Za = __map2world__(xa, ya, s, t, p, f, l)
    Xb, Yb, Zb = __map2world__(xb, yb, s, t, p, f, l)
    Xc, Yc, Zc = __map2world__(xc, yc, s, t, p, f, l)
    Xd, Yd, Zd = __map2world__(xd, yd, s, t, p, f, l)
    print("The world coordinates of the four points are: (", Xa, Ya, Za, "), (", Xb, Yb, Zb, "), (", Xc, Yc, Zc, "), (", Xd, Yd, Zd, ")。")
    print("X variance = ", abs(Xa - Xb), "Y variance = ", abs(Ya-Yc))

    return s, t, p, f, l, x_cam, y_cam, z_cam
