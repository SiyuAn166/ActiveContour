import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy import linalg
from external_energy import external_energy
from internal_energy_matrix import get_matrix
from scipy.interpolate import *
import warnings

def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        #save point
        xs.append(x)
        ys.append(y)

        #display point
        cv2.circle(img, (x, y), 3, 128, -1)
        cv2.imshow('image', img)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    #point initialization
    #img_path = 'images/star.png'
    img_path = '../images/star.png'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img.copy(), (5, 5), 1, 1)

    xs = []
    ys = []
    cv2.imshow('image', img)

    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #selected points are in xs and ys

    tck, u = splprep([xs, ys], s=0, per=True)
    new_points_x, new_points_y = splev(np.linspace(0, 1, 1000), tck)

    alpha = 4e-2 # dist
    beta = 1     # stiffness
    gamma = 1
    # step
    kappa = 1e-1
    
    w_line = 5e-1
    w_edge = 9e-1
    w_term = 5e-1 #curvature
    # alpha = 0.04
    # beta = 1
    # gamma = 1
    # kappa = 0.1
    num_points = len(new_points_x)

    #get matrix
    M = get_matrix(alpha, beta, gamma, num_points)



    #get external energy
    # w_line = 0.5
    # w_edge = 0.6
    # w_term = 0.5
    E = external_energy(img, w_line, w_edge, w_term)
    #print(E)
    fy, fx = np.gradient(E)

    step=0
    while step<9000:
        try:
            new_points_x = M @ (gamma * new_points_x - kappa * fx[new_points_y.astype(int), new_points_x.astype(int)])
            new_points_y = M @ (gamma * new_points_y - kappa * fy[new_points_y.astype(int), new_points_x.astype(int)])
            # print(new_points_x)
        except Exception as e:
            print('error')
        step+=1

    #print(new_points_x)
    #print(new_points_y)
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.plot(new_points_x, new_points_y, '--', lw=3, color="red")
    plt.show()

